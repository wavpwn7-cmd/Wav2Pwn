from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run white-box PGD on the surrogate CTC model.")
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels.json")
    parser.add_argument("--surrogate-model", default="checkpoints/best_model")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--target-phrase", default="OPEN THE DOOR")
    parser.add_argument("--target-repeat", type=int, default=1)
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--attack-method", choices=["pgd", "adam"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--alpha", type=float, default=0.0002)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--output-audio", default="artifacts/adv.wav")
    parser.add_argument("--output-json", default="results/attack_cost.json")
    parser.add_argument("--output-dir", default="artifacts/pgd_attack")
    parser.add_argument("--output-csv", default="results/pgd_attack_batch.csv")
    parser.add_argument("--log-file", default="logs/pgd_attack.log")
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

    logger = get_logger("pgd_attack", log_file=args.log_file)
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    surrogate = load_asr_bundle(args.surrogate_model, device=args.device)
    expanded_target_phrase = " ".join([args.target_phrase] * max(args.target_repeat, 1))

    start_index = args.sample_index if args.start_index is None else args.start_index
    end_index = min(start_index + args.num_samples, len(dataset))
    selected_indices = list(range(start_index, end_index))
    if not selected_indices:
        raise ValueError("No samples selected for PGD attack.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    total_start = time.perf_counter()

    if str(surrogate.device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(surrogate.device)

    for offset, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[int(sample_index)]
        waveform, sample_rate = load_audio(sample["audio_path"])
        clean_prediction = transcribe_waveform(surrogate, waveform, sample_rate)

        result = pgd_ctc_attack(
            model=surrogate.model,
            processor=surrogate.processor,
            waveform=waveform,
            reference_text=clean_prediction,
            epsilon=args.epsilon,
            alpha=args.alpha,
            num_iterations=args.iterations,
            targeted=args.targeted,
            target_text=expanded_target_phrase if args.targeted else None,
            device=surrogate.device,
            method=args.attack_method,
            l2_weight=args.l2_weight,
            random_start=args.random_start,
        )

        adv_audio_path = output_dir / f"{sample['sample_id']}_adv.wav"
        save_waveform(str(adv_audio_path), result.adversarial_waveform)
        adv_prediction = transcribe_waveform(surrogate, result.adversarial_waveform, sample_rate)
        text_metrics = compute_text_metrics(clean_prediction, adv_prediction)
        row_target_success = targeted_success(adv_prediction, args.target_phrase, relaxed=True) if args.targeted else False
        row_untarget_success = untargeted_success(
            clean_prediction,
            adv_prediction,
            cer_threshold=args.cer_threshold,
            wer_threshold=args.wer_threshold,
        )

        row = {
            "sample_index": sample_index,
            "sample_id": sample["sample_id"],
            "dataset_name": sample.get("dataset_name"),
            "audio_path": sample["audio_path"],
            "clean_prediction": clean_prediction,
            "adversarial_prediction": adv_prediction,
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
            "cer_from_clean": round(text_metrics.cer, 6),
            "wer_from_clean": round(text_metrics.wer, 6),
            "target_success_relaxed": row_target_success,
            "untarget_success": row_untarget_success,
            "adv_audio_path": str(adv_audio_path),
        }
        rows.append(row)

        progress = render_progress(offset, len(selected_indices))
        print(f"\r{progress}", end="", flush=True)
        logger.info(
            "Generated attack %s | sample_id=%s | final_loss=%.6f | adv_audio=%s",
            progress,
            sample["sample_id"],
            result.final_loss,
            adv_audio_path,
        )

    print()

    memory_mb = 0.0
    if str(surrogate.device).startswith("cuda"):
        import torch

        memory_mb = float(torch.cuda.max_memory_allocated(surrogate.device) / (1024 ** 2))

    summary = {
        "num_samples": len(rows),
        "start_index": start_index,
        "end_index_exclusive": end_index,
        "targeted": args.targeted,
        "target_phrase": args.target_phrase,
        "expanded_target_phrase": expanded_target_phrase,
        "attack_method": args.attack_method,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "iterations": args.iterations,
        "l2_weight": args.l2_weight,
        "random_start": args.random_start,
        "targeted_success_count": sum(1 for row in rows if row["target_success_relaxed"]),
        "untargeted_success_count": sum(1 for row in rows if row["untarget_success"]),
        "avg_cer_from_clean": round(sum(row["cer_from_clean"] for row in rows) / len(rows), 6),
        "avg_wer_from_clean": round(sum(row["wer_from_clean"] for row in rows) / len(rows), 6),
        "total_attack_time": round(time.perf_counter() - total_start, 4),
        "gpu_peak_memory_mb": round(memory_mb, 4),
        "device": str(surrogate.device),
        "output_dir": str(output_dir.resolve()),
    }
    payload = {"summary": summary, "samples": rows}
    write_json(args.output_json, payload)

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Saved batch PGD summary to %s", args.output_json)
    logger.info("Saved batch PGD rows to %s", args.output_csv)


if __name__ == "__main__":
    main()
