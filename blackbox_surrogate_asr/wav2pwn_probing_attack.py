from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_CANDIDATES = [
    "facebook/wav2vec2-base-100h",
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-robust-ft-libri-960h",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run probing-guided Wav2Pwn-style attack by selecting the best aligned proxy per sample."
    )
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels_3datasets_900.json")
    parser.add_argument("--target-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--candidate-models", nargs="+", default=DEFAULT_CANDIDATES)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--target-phrase", default="OPEN THE DOOR")
    parser.add_argument("--target-repeat", type=int, default=1)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--attack-method", choices=["pgd", "adam"], default="adam")
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--l2-weight", type=float, default=0.0001)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="artifacts/wav2pwn_attack")
    parser.add_argument("--output-json", default="results/wav2pwn_attack.json")
    parser.add_argument("--output-csv", default="results/wav2pwn_attack.csv")
    parser.add_argument("--log-file", default="logs/wav2pwn_attack.log")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def render_progress(current: int, total: int) -> str:
    width = 30
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:6.2f}% ({current}/{total})"


def main() -> None:
    args = parse_args()
    from src.asr_blackbox.attack import pgd_ctc_attack, save_waveform
    from src.asr_blackbox.data import load_pseudo_label_dataset
    from src.asr_blackbox.logging_utils import get_logger, write_json
    from src.asr_blackbox.metrics import compute_text_metrics, targeted_success, untargeted_success
    from src.asr_blackbox.models import load_asr_bundle, load_audio, transcribe_waveform

    logger = get_logger("wav2pwn_attack", log_file=args.log_file)
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    target_bundle = load_asr_bundle(args.target_model, device=args.device)
    candidate_bundles = {name: load_asr_bundle(name, device=args.device) for name in args.candidate_models}
    expanded_target_phrase = " ".join([args.target_phrase] * max(args.target_repeat, 1))

    end_index = min(args.start_index + args.num_samples, len(dataset))
    selected_indices = list(range(args.start_index, end_index))
    if not selected_indices:
        raise ValueError("No samples selected for Wav2Pwn attack.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total_start = time.perf_counter()
    probing_query_count = 0

    if str(target_bundle.device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(target_bundle.device)

    for offset, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[int(sample_index)]
        waveform, sample_rate = load_audio(sample["audio_path"])

        target_clean_prediction = transcribe_waveform(target_bundle, waveform, sample_rate)
        probing_query_count += 1

        probe_rows = []
        for model_name, bundle in candidate_bundles.items():
            proxy_prediction = transcribe_waveform(bundle, waveform, sample_rate)
            alignment = compute_text_metrics(target_clean_prediction, proxy_prediction)
            probe_rows.append(
                {
                    "model_name": model_name,
                    "proxy_prediction": proxy_prediction,
                    "alignment_cer": alignment.cer,
                    "alignment_wer": alignment.wer,
                    "alignment_score": alignment.cer + alignment.wer,
                }
            )

        probe_rows.sort(key=lambda item: (item["alignment_score"], item["alignment_cer"], item["alignment_wer"]))
        best_probe = probe_rows[0]
        selected_bundle = candidate_bundles[best_probe["model_name"]]

        result = pgd_ctc_attack(
            model=selected_bundle.model,
            processor=selected_bundle.processor,
            waveform=waveform,
            reference_text=best_probe["proxy_prediction"],
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_iterations=args.iterations,
            targeted=args.targeted,
            target_text=expanded_target_phrase if args.targeted else None,
            device=selected_bundle.device,
            method=args.attack_method,
            l2_weight=args.l2_weight,
            random_start=args.random_start,
        )

        adv_audio_path = output_dir / f"{sample['sample_id']}_adv.wav"
        save_waveform(str(adv_audio_path), result.adversarial_waveform)
        adv_prediction_proxy = transcribe_waveform(selected_bundle, result.adversarial_waveform, sample_rate)
        text_metrics = compute_text_metrics(best_probe["proxy_prediction"], adv_prediction_proxy)

        row = {
            "sample_index": sample_index,
            "sample_id": sample["sample_id"],
            "dataset_name": sample.get("dataset_name"),
            "audio_path": sample["audio_path"],
            "target_model": args.target_model,
            "selected_proxy_model": best_probe["model_name"],
            "target_clean_prediction": target_clean_prediction,
            "selected_proxy_clean_prediction": best_probe["proxy_prediction"],
            "selected_proxy_adv_prediction": adv_prediction_proxy,
            "probe_alignment_cer": round(best_probe["alignment_cer"], 6),
            "probe_alignment_wer": round(best_probe["alignment_wer"], 6),
            "probe_alignment_score": round(best_probe["alignment_score"], 6),
            "target_phrase": args.target_phrase,
            "expanded_target_phrase": expanded_target_phrase,
            "targeted": args.targeted,
            "attack_method": args.attack_method,
            "epsilon": args.epsilon,
            "alpha": args.alpha,
            "iterations": args.iterations,
            "l2_weight": args.l2_weight,
            "random_start": args.random_start,
            "attack_generation_time": round(result.elapsed_seconds, 4),
            "average_optimization_iterations": result.num_iterations,
            "final_loss": round(result.final_loss, 6),
            "cer_from_proxy_clean": round(text_metrics.cer, 6),
            "wer_from_proxy_clean": round(text_metrics.wer, 6),
            "target_success_relaxed": targeted_success(adv_prediction_proxy, args.target_phrase, relaxed=True) if args.targeted else False,
            "untarget_success": untargeted_success(
                best_probe["proxy_prediction"],
                adv_prediction_proxy,
                args.cer_threshold,
                args.wer_threshold,
            ),
            "candidate_rankings": " || ".join(
                f"{probe['model_name']}|cer={probe['alignment_cer']:.4f}|wer={probe['alignment_wer']:.4f}"
                for probe in probe_rows
            ),
            "adv_audio_path": str(adv_audio_path),
        }
        rows.append(row)

        progress = render_progress(offset, len(selected_indices))
        print(f"\r{progress}", end="", flush=True)
        logger.info(
            "Generated Wav2Pwn attack %s | sample_id=%s | selected_proxy=%s | alignment_score=%.6f",
            progress,
            sample["sample_id"],
            best_probe["model_name"],
            best_probe["alignment_score"],
        )

    print()

    memory_mb = 0.0
    if str(target_bundle.device).startswith("cuda"):
        import torch

        memory_mb = float(torch.cuda.max_memory_allocated(target_bundle.device) / (1024 ** 2))

    summary = {
        "method": "Wav2Pwn (with Probing)",
        "num_samples": len(rows),
        "target_model": args.target_model,
        "candidate_models": args.candidate_models,
        "start_index": args.start_index,
        "end_index_exclusive": end_index,
        "targeted": args.targeted,
        "target_phrase": args.target_phrase,
        "attack_method": args.attack_method,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "iterations": args.iterations,
        "l2_weight": args.l2_weight,
        "random_start": args.random_start,
        "targeted_success_count": sum(1 for row in rows if row["target_success_relaxed"]),
        "untargeted_success_count": sum(1 for row in rows if row["untarget_success"]),
        "avg_probe_alignment_score": round(sum(row["probe_alignment_score"] for row in rows) / len(rows), 6),
        "avg_cer_from_proxy_clean": round(sum(row["cer_from_proxy_clean"] for row in rows) / len(rows), 6),
        "avg_wer_from_proxy_clean": round(sum(row["wer_from_proxy_clean"] for row in rows) / len(rows), 6),
        "total_attack_time": round(time.perf_counter() - total_start, 4),
        "gpu_peak_memory_mb": round(memory_mb, 4),
        "device": str(target_bundle.device),
        "blackbox_query_count_for_attack": probing_query_count,
        "surrogate_training_time_seconds": 0.0,
        "output_dir": str(output_dir.resolve()),
    }
    payload = {"summary": summary, "samples": rows}
    write_json(args.output_json, payload)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved Wav2Pwn attack summary to %s", args.output_json)


if __name__ == "__main__":
    main()
