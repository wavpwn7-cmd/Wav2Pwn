from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the independent Wav2Pwn dataset experiment.")
    parser.add_argument("--dataset", choices=["librispeech", "commonvoice", "voxpopuli"], required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing partial outputs and rerun from scratch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from src.wav2pwn_diversity.attacks import optimize_attack, save_waveform
    from src.wav2pwn_diversity.config import load_yaml
    from src.wav2pwn_diversity.data import sample_dataset_items
    from src.wav2pwn_diversity.families import build_family_map
    from src.wav2pwn_diversity.logging_utils import append_csv, get_logger, write_csv, write_json
    from src.wav2pwn_diversity.metrics import compute_table3_metrics
    from src.wav2pwn_diversity.models import load_asr_bundle, load_audio, load_model_registry, transcribe_waveform
    from src.wav2pwn_diversity.probing import build_clean_transcript_frame, build_probing_similarity_outputs
    from src.wav2pwn_diversity.transfer import aggregate_transfer_rows
    from src.wav2pwn_diversity.visualize import plot_family_tsne, plot_heatmap

    config = load_yaml(args.config)
    dataset_config = config["datasets"][args.dataset]
    model_registry_path = PROJECT_ROOT / config["paths"]["model_registry"]
    model_specs = load_model_registry(model_registry_path)

    dataset_name = dataset_config["name"]
    dataset_slug = args.dataset
    results_dir = PROJECT_ROOT / config["paths"]["root_results_dir"] / dataset_slug
    artifacts_dir = PROJECT_ROOT / config["paths"]["root_artifacts_dir"] / dataset_slug
    log_file = PROJECT_ROOT / config["paths"]["root_logs_dir"] / f"{dataset_slug}.log"
    clean_transcripts_path = results_dir / f"clean_transcripts_{dataset_slug}.csv"
    transfer_rows_path = results_dir / f"transfer_rows_{dataset_slug}.csv"
    probing_matrix_path = results_dir / f"probing_matrix_{dataset_slug}.csv"
    probing_pairs_path = results_dir / f"probing_pairs_{dataset_slug}.csv"
    transfer_pairs_path = results_dir / f"transfer_pairs_{dataset_slug}.csv"
    transfer_matrix_path = results_dir / f"transfer_matrix_{dataset_slug}.csv"
    tasr_matrix_path = results_dir / f"transfer_tasr_matrix_{dataset_slug}.csv"
    uasr_matrix_path = results_dir / f"transfer_uasr_matrix_{dataset_slug}.csv"
    family_metrics_path = results_dir / f"family_metrics_{dataset_slug}.json"
    skipped_models_path = results_dir / f"skipped_models_{dataset_slug}.json"
    resume_state_path = results_dir / f"resume_state_{dataset_slug}.json"
    logger = get_logger(f"run_dataset_experiment.{dataset_slug}", log_file=log_file)
    resume_enabled = not args.no_resume

    def write_resume_state(phase: str, **extra: object) -> None:
        payload = {
            "dataset": dataset_slug,
            "dataset_name": dataset_name,
            "phase": phase,
            "sample_count": len(items),
            "resume_enabled": resume_enabled,
        }
        payload.update(extra)
        write_json(resume_state_path, payload)

    logger.info("Sampling %s items for dataset %s independently.", dataset_config["sample_count"], dataset_name)
    items = sample_dataset_items(dataset_slug, dataset_config, seed=int(config["seed"]))
    results_dir.mkdir(parents=True, exist_ok=True)
    if not resume_enabled:
        for stale_path in [
            clean_transcripts_path,
            transfer_rows_path,
            probing_matrix_path,
            probing_pairs_path,
            transfer_pairs_path,
            transfer_matrix_path,
            tasr_matrix_path,
            uasr_matrix_path,
            family_metrics_path,
            skipped_models_path,
            resume_state_path,
        ]:
            if stale_path.exists():
                stale_path.unlink()
    write_resume_state("sampling_complete")

    logger.info("Loading up to %s ASR models.", len(model_specs))
    bundles = {}
    loaded_specs = []
    skipped_models = []
    for spec in model_specs:
        try:
            bundles[spec.id] = load_asr_bundle(spec, device=config["runtime"]["device"])
            loaded_specs.append(spec)
            logger.info("Loaded model %s (%s).", spec.id, spec.hf_name)
        except Exception as exc:
            skipped_models.append({"model_id": spec.id, "hf_name": spec.hf_name, "error": str(exc)})
            logger.warning(
                "Skipping model %s (%s) because it is not loadable as a CTC ASR checkpoint in the current environment: %s",
                spec.id,
                spec.hf_name,
                exc,
            )

    if len(bundles) < 2:
        raise RuntimeError(
            "Fewer than two ASR models could be loaded. "
            "Please replace unsupported pretraining-only checkpoints in configs/models.yaml."
        )

    family_map = build_family_map([spec.__dict__ for spec in loaded_specs])
    if skipped_models:
        write_json(skipped_models_path, skipped_models)
    write_resume_state("models_loaded", loaded_model_ids=list(bundles.keys()), skipped_models=skipped_models)

    clean_transcripts: dict[str, list[dict]] = {}
    existing_clean_df = pd.DataFrame()
    if resume_enabled and clean_transcripts_path.exists():
        existing_clean_df = pd.read_csv(clean_transcripts_path, keep_default_na=False)
        if not existing_clean_df.empty:
            existing_clean_df = existing_clean_df.drop_duplicates(["model_id", "sample_index"], keep="last")
            logger.info(
                "Resuming from existing clean transcripts at %s with %s cached rows.",
                clean_transcripts_path,
                len(existing_clean_df),
            )

    for model_id, bundle in bundles.items():
        if not existing_clean_df.empty:
            model_df = existing_clean_df[existing_clean_df["model_id"] == model_id].sort_values("sample_index")
            if len(model_df) == len(items):
                clean_transcripts[model_id] = model_df.to_dict(orient="records")
                logger.info("Skipping clean transcription for model %s because cached results are complete.", model_id)
                continue

        logger.info("Transcribing clean audio for model %s.", model_id)
        rows = []
        clean_progress = tqdm(
            enumerate(items),
            total=len(items),
            desc=f"Clean ASR [{model_id}]",
            unit="sample",
            dynamic_ncols=True,
        )
        for sample_index, item in clean_progress:
            waveform, sample_rate = load_audio(item.audio_path)
            rows.append(
                {
                    "sample_index": sample_index,
                    "audio_path": item.audio_path,
                    "ground_truth": item.ground_truth,
                    "transcription": transcribe_waveform(bundle, waveform, sample_rate),
                }
            )
        clean_transcripts[model_id] = rows
        updated_rows = []
        for cached_model_id, cached_rows in clean_transcripts.items():
            for row in cached_rows:
                updated_rows.append(
                    {
                        "model_id": cached_model_id,
                        "sample_index": row["sample_index"],
                        "audio_path": row["audio_path"],
                        "transcription": row["transcription"],
                        "ground_truth": row.get("ground_truth"),
                    }
                )
        write_csv(clean_transcripts_path, updated_rows)
        write_resume_state(
            "clean_transcripts_partial",
            completed_clean_models=sorted(clean_transcripts.keys()),
            cached_clean_rows=len(updated_rows),
        )

    transcript_frame = build_clean_transcript_frame(clean_transcripts)
    probing_matrix, probing_pairs, probing_embedding_matrix = build_probing_similarity_outputs(transcript_frame)
    probing_matrix.to_csv(probing_matrix_path)
    probing_pairs.to_csv(probing_pairs_path, index=False)
    write_resume_state(
        "probing_complete",
        completed_clean_models=sorted(clean_transcripts.keys()),
        probing_models=list(probing_matrix.index),
    )

    attack_config = config["experiment"]
    expanded_target_phrase = " ".join([attack_config["target_phrase"]] * int(attack_config["target_repeat"]))
    existing_transfer_df = pd.DataFrame()
    if resume_enabled and transfer_rows_path.exists():
        existing_transfer_df = pd.read_csv(transfer_rows_path, keep_default_na=False)
        if not existing_transfer_df.empty:
            existing_transfer_df = existing_transfer_df.drop_duplicates(
                ["source_model", "target_model", "attack_mode", "sample_index"],
                keep="last",
            )
            logger.info(
                "Resuming from existing transfer rows at %s with %s cached rows.",
                transfer_rows_path,
                len(existing_transfer_df),
            )

    completed_attack_steps: set[tuple[str, int, str]] = set()
    if not existing_transfer_df.empty:
        target_count = len(bundles)
        grouped_counts = (
            existing_transfer_df.groupby(["source_model", "sample_index", "attack_mode"])["target_model"]
            .nunique()
            .reset_index(name="target_count")
        )
        for row in grouped_counts.itertuples(index=False):
            if row.target_count == target_count:
                completed_attack_steps.add((row.source_model, int(row.sample_index), row.attack_mode))

    target_model_ids = list(bundles.keys())
    attack_modes = list(attack_config["attack_modes"])
    total_source_steps = len(items) * len(attack_modes)

    for source_model_id, bundle in bundles.items():
        logger.info("Generating adversarial examples from source model %s.", source_model_id)
        attack_progress = tqdm(
            enumerate(items),
            total=len(items),
            desc=f"Attack+Transfer [{source_model_id}]",
            unit="sample",
            dynamic_ncols=True,
        )
        for sample_index, item in attack_progress:
            pending_modes = [
                attack_mode
                for attack_mode in attack_config["attack_modes"]
                if (source_model_id, sample_index, attack_mode) not in completed_attack_steps
            ]
            if not pending_modes:
                attack_progress.set_postfix(mode="cached", transfer_pct="100.0%")
                continue

            waveform, sample_rate = load_audio(item.audio_path)
            source_clean_prediction = clean_transcripts[source_model_id][sample_index]["transcription"]
            for attack_mode in attack_config["attack_modes"]:
                if (source_model_id, sample_index, attack_mode) in completed_attack_steps:
                    completed_transfers = (
                        (sample_index * len(attack_modes) + attack_modes.index(attack_mode) + 1) * len(target_model_ids)
                    )
                    total_transfers = total_source_steps * len(target_model_ids)
                    attack_progress.set_postfix(
                        mode=f"{attack_mode} (cached)",
                        transfer_pct=f"{(completed_transfers / total_transfers) * 100:5.1f}%",
                    )
                    continue

                targeted = attack_mode == "targeted"
                attack_result = optimize_attack(
                    model=bundle.model,
                    processor=bundle.processor,
                    waveform=waveform,
                    reference_text=source_clean_prediction,
                    epsilon=float(attack_config["epsilon"]),
                    alpha=float(attack_config["alpha"]),
                    num_iterations=int(attack_config["iterations"]),
                    targeted=targeted,
                    target_text=expanded_target_phrase if targeted else None,
                    device=bundle.device,
                    method=attack_config["attack_method"],
                    l2_weight=float(attack_config["l2_weight"]),
                    random_start=bool(attack_config["random_start"]),
                )

                adv_waveform = attack_result.adversarial_waveform
                if config["runtime"]["save_audio"]:
                    adv_dir = artifacts_dir / source_model_id / attack_mode
                    adv_dir.mkdir(parents=True, exist_ok=True)
                    save_waveform(str(adv_dir / f"{sample_index:06d}.wav"), adv_waveform)

                step_rows = []
                for target_model_id, target_bundle in bundles.items():
                    target_clean_prediction = clean_transcripts[target_model_id][sample_index]["transcription"]
                    adv_prediction = transcribe_waveform(target_bundle, adv_waveform, sample_rate)
                    step_rows.append(
                        {
                            "dataset": dataset_name,
                            "sample_index": sample_index,
                            "audio_path": item.audio_path,
                            "source_model": source_model_id,
                            "target_model": target_model_id,
                            "attack_mode": attack_mode,
                            "clean_prediction": target_clean_prediction,
                            "adv_prediction": adv_prediction,
                        }
                    )
                append_csv(transfer_rows_path, step_rows)
                completed_attack_steps.add((source_model_id, sample_index, attack_mode))
                write_resume_state(
                    "attack_transfer_partial",
                    completed_clean_models=sorted(clean_transcripts.keys()),
                    completed_attack_steps=len(completed_attack_steps),
                    total_attack_steps=len(bundles) * len(items) * len(attack_modes),
                    current_source_model=source_model_id,
                    current_sample_index=sample_index,
                    current_attack_mode=attack_mode,
                )
                completed_transfers = (sample_index * len(attack_modes) + attack_modes.index(attack_mode) + 1) * len(target_model_ids)
                total_transfers = total_source_steps * len(target_model_ids)
                attack_progress.set_postfix(
                    mode=attack_mode,
                    transfer_pct=f"{(completed_transfers / total_transfers) * 100:5.1f}%",
                )

    if not transfer_rows_path.exists():
        raise RuntimeError(f"No transfer rows were written for dataset {dataset_slug}.")
    transfer_rows_df = pd.read_csv(transfer_rows_path, keep_default_na=False)
    transfer_rows_df = transfer_rows_df.drop_duplicates(
        ["source_model", "target_model", "attack_mode", "sample_index"],
        keep="last",
    )
    write_csv(transfer_rows_path, transfer_rows_df.to_dict(orient="records"))

    transfer_pairs, transfer_matrix, tasr_matrix, uasr_matrix = aggregate_transfer_rows(
        transfer_rows_df.to_dict(orient="records"),
        target_phrase=attack_config["target_phrase"],
        cer_threshold=float(attack_config["cer_threshold"]),
        wer_threshold=float(attack_config["wer_threshold"]),
    )
    transfer_pairs.to_csv(transfer_pairs_path, index=False)
    transfer_matrix.to_csv(transfer_matrix_path)
    tasr_matrix.to_csv(tasr_matrix_path)
    uasr_matrix.to_csv(uasr_matrix_path)

    transfer_feature_matrix = transfer_matrix.values
    table3_metrics = compute_table3_metrics(dataset_name, transfer_matrix, family_map, transfer_feature_matrix)
    write_json(family_metrics_path, table3_metrics)

    plot_heatmap(transfer_matrix, f"{dataset_name} Transfer Similarity", str(results_dir / f"transfer_similarity_{dataset_slug}"))
    plot_heatmap(probing_matrix, f"{dataset_name} Probing Similarity", str(results_dir / f"probing_similarity_{dataset_slug}"))
    plot_family_tsne(
        transfer_feature_matrix,
        list(transfer_matrix.index),
        family_map,
        f"{dataset_name} Family Clustering (Transfer Features)",
        str(results_dir / f"family_clustering_{dataset_slug}"),
        perplexity=float(attack_config["tsne_perplexity"]),
    )
    plot_family_tsne(
        probing_embedding_matrix.values,
        list(probing_embedding_matrix.index),
        family_map,
        f"{dataset_name} t-SNE / UMAP-style Probing Visualization",
        str(results_dir / f"probing_clustering_{dataset_slug}"),
        perplexity=float(attack_config["tsne_perplexity"]),
    )

    write_csv(clean_transcripts_path, transcript_frame.to_dict(orient="records"))
    if skipped_models:
        write_json(skipped_models_path, skipped_models)
    write_resume_state(
        "complete",
        completed_clean_models=sorted(clean_transcripts.keys()),
        completed_attack_steps=len(completed_attack_steps),
        total_attack_steps=len(bundles) * len(items) * len(attack_modes),
        loaded_model_ids=list(bundles.keys()),
        skipped_models=skipped_models,
        final_outputs={
            "clean_transcripts": str(clean_transcripts_path),
            "probing_matrix": str(probing_matrix_path),
            "transfer_matrix": str(transfer_matrix_path),
            "family_metrics": str(family_metrics_path),
        },
    )
    logger.info("Finished independent experiment for dataset %s.", dataset_name)


if __name__ == "__main__":
    main()
