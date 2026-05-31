from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch transfer evaluation on the black-box target model.")
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels_3datasets_900.json")
    parser.add_argument("--target-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--adversarial-dir", required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--target-phrase", default="OPEN THE DOOR")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--relaxed-substring-match", action="store_true")
    parser.add_argument("--output-json", default="results/transfer_batch.json")
    parser.add_argument("--output-csv", default="results/transfer_batch.csv")
    parser.add_argument("--log-file", default="logs/transfer_batch.log")
    parser.add_argument("--attack-metadata-json", default=None)
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
    from src.asr_blackbox.data import load_pseudo_label_dataset
    from src.asr_blackbox.logging_utils import get_logger, write_csv, write_json
    from src.asr_blackbox.metrics import compute_text_metrics, targeted_success, untargeted_success
    from src.asr_blackbox.models import load_asr_bundle, load_audio, transcribe_waveform

    logger = get_logger("transfer_batch", log_file=args.log_file)
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    target = load_asr_bundle(args.target_model, device=args.device)

    attack_metadata = {}
    if args.attack_metadata_json:
        import json

        attack_metadata = json.loads(Path(args.attack_metadata_json).read_text())

    end_index = min(args.start_index + args.num_samples, len(dataset))
    selected_indices = list(range(args.start_index, end_index))
    if not selected_indices:
        raise ValueError("No samples selected for transfer evaluation.")

    rows: list[dict] = []
    total_start = time.perf_counter()

    for offset, sample_index in enumerate(selected_indices, start=1):
        sample = dataset[int(sample_index)]
        adv_path = Path(args.adversarial_dir) / f"{sample['sample_id']}_adv.wav"
        if not adv_path.exists():
            raise FileNotFoundError(f"Missing adversarial audio: {adv_path}")

        clean_waveform, clean_sr = load_audio(sample["audio_path"])
        adv_waveform, adv_sr = load_audio(str(adv_path))

        clean_transcription = transcribe_waveform(target, clean_waveform, clean_sr)
        adv_transcription = transcribe_waveform(target, adv_waveform, adv_sr)

        clean_metrics = compute_text_metrics(sample.get("ground_truth") or sample["transcription"], clean_transcription)
        shift_metrics = compute_text_metrics(clean_transcription, adv_transcription)
        tasr_success = targeted_success(adv_transcription, args.target_phrase, relaxed=args.relaxed_substring_match)
        uasr_success = untargeted_success(clean_transcription, adv_transcription, args.cer_threshold, args.wer_threshold)

        row = {
            "sample_id": sample["sample_id"],
            "sample_index": sample_index,
            "clean_transcript": clean_transcription,
            "adversarial_transcript": adv_transcription,
            "target_transcript": args.target_phrase,
            "clean_cer": round(clean_metrics.cer, 6),
            "clean_wer": round(clean_metrics.wer, 6),
            "transcription_shift_cer": round(shift_metrics.cer, 6),
            "transcription_shift_wer": round(shift_metrics.wer, 6),
            "tasr_success": int(tasr_success),
            "uasr_success": int(uasr_success),
            "query_count": 2,
            "adversarial_audio_path": str(adv_path),
        }
        rows.append(row)

        progress = render_progress(offset, len(selected_indices))
        print(f"\r{progress}", end="", flush=True)
        logger.info(
            "Transfer eval %s | sample_id=%s | tasr=%s | uasr=%s",
            progress,
            sample["sample_id"],
            tasr_success,
            uasr_success,
        )

    print()

    blackbox_eval_queries = 2 * len(rows)
    attack_summary = attack_metadata.get("summary", {})
    method_name = attack_summary.get("method", "Unknown Method")
    total_query_count = blackbox_eval_queries + int(attack_summary.get("blackbox_query_count_for_attack", 0))
    total_attack_time = float(attack_summary.get("total_attack_time", 0.0))
    training_time = float(attack_summary.get("surrogate_training_time_seconds", 0.0))
    total_gpu_seconds = total_attack_time + training_time

    summary = {
        "Method": method_name,
        "Attack Cost": {
            "total_query_count": total_query_count,
            "blackbox_query_count_for_attack": int(attack_summary.get("blackbox_query_count_for_attack", 0)),
            "blackbox_query_count_for_transfer_eval": blackbox_eval_queries,
            "surrogate_training_time_seconds": round(training_time, 4),
            "attack_optimization_time_seconds": round(total_attack_time, 4),
            "total_gpu_seconds": round(total_gpu_seconds, 4),
            "gpu_peak_memory_mb": attack_summary.get("gpu_peak_memory_mb"),
        },
        "Clean CER": round(sum(row["clean_cer"] for row in rows) / len(rows), 6),
        "TASR": round(sum(row["tasr_success"] for row in rows) / len(rows), 6),
        "UASR": round(sum(row["uasr_success"] for row in rows) / len(rows), 6),
        "avg_transcription_shift_cer": round(sum(row["transcription_shift_cer"] for row in rows) / len(rows), 6),
        "avg_transcription_shift_wer": round(sum(row["transcription_shift_wer"] for row in rows) / len(rows), 6),
        "num_samples": len(rows),
        "targeted_mode": args.targeted,
        "target_phrase": args.target_phrase,
        "runtime_seconds": round(time.perf_counter() - total_start, 4),
    }

    write_json(args.output_json, {"summary": summary, "samples": rows, "attack_summary": attack_summary})
    write_csv(args.output_csv, rows, list(rows[0].keys()))
    logger.info("Saved transfer batch summary to %s", args.output_json)


if __name__ == "__main__":
    main()
