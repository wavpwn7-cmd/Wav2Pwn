from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a black-box HuBERT target and build pseudo-labels.")
    parser.add_argument("--librispeech-root", default="../dataset/LibriSpeech", help="Path to the LibriSpeech root directory to sample from.")
    parser.add_argument("--commonvoice-root", default="../dataset/CommonVoice_wav", help="Path to the CommonVoice_wav root directory to sample from.")
    parser.add_argument("--voxpopuli-root", default="../dataset/voxpopuli", help="Path to the voxpopuli root directory to sample from.")
    parser.add_argument("--samples-per-dataset", type=int, default=300)
    parser.add_argument(
        "--allow-librispeech-test-clean",
        action="store_true",
        help="Allow sampling from LibriSpeech/test-clean when that is the only available LibriSpeech subset.",
    )
    parser.add_argument("--output-json", default="artifacts/pseudo_labels.json")
    parser.add_argument("--output-stats-json", default="results/query_stats.json")
    parser.add_argument("--output-csv", default="results/query_records.csv")
    parser.add_argument("--log-file", default="logs/query.log")
    parser.add_argument("--target-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--seed", type=int, default=13)
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
    from src.asr_blackbox.data import build_pseudo_label_records, sample_multiple_datasets, summarize_dataset_counts
    from src.asr_blackbox.logging_utils import get_logger, write_json
    from src.asr_blackbox.models import load_asr_bundle, load_audio, transcribe_waveform

    logger = get_logger("query_target", log_file=args.log_file)
    bundle = load_asr_bundle(args.target_model, device=args.device)
    dataset_specs = [
        ("LibriSpeech", args.librispeech_root, args.samples_per_dataset),
        ("CommonVoice_wav", args.commonvoice_root, args.samples_per_dataset),
        ("voxpopuli", args.voxpopuli_root, args.samples_per_dataset),
    ]
    sampled_items = sample_multiple_datasets(
        dataset_specs,
        args.seed,
        allow_librispeech_test_clean=args.allow_librispeech_test_clean,
    )
    dataset_counts = summarize_dataset_counts(sampled_items)
    total_samples = len(sampled_items)
    logger.info("Sampling plan: %s", dataset_counts)

    transcriptions: list[str] = []
    query_times: list[float] = []
    sample_rows: list[dict] = []
    total_query_start = time.perf_counter()

    if str(bundle.device).startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats(bundle.device)

    for index, item in enumerate(sampled_items, start=1):
        waveform, sample_rate = load_audio(item.audio_path)
        start_time = time.perf_counter()
        prediction = transcribe_waveform(bundle, waveform, sample_rate)
        elapsed = time.perf_counter() - start_time
        transcriptions.append(prediction)
        query_times.append(elapsed)
        sample_rows.append(
            {
                "sample_id": f"query_{index - 1:06d}",
                "dataset_name": item.dataset_name,
                "audio_path": item.audio_path,
                "query_seconds": round(elapsed, 4),
            }
        )
        progress = render_progress(index, total_samples)
        print(f"\r{progress}", end="", flush=True)
        if index % 25 == 0 or index == total_samples:
            logger.info(
                "Queried %s | dataset=%s | last_query_seconds=%.4f",
                progress,
                item.dataset_name,
                elapsed,
            )
    print()

    total_query_seconds = time.perf_counter() - total_query_start
    avg_query_seconds = total_query_seconds / max(total_samples, 1)
    gpu_memory_mb = 0.0
    if str(bundle.device).startswith("cuda"):
        import torch

        gpu_memory_mb = float(torch.cuda.max_memory_allocated(bundle.device) / (1024 ** 2))

    records = build_pseudo_label_records(sampled_items, transcriptions, query_times=query_times)
    output_path = Path(args.output_json)
    write_json(output_path, records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.output_csv).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sample_rows)

    stats = {
        "target_model": args.target_model,
        "device": str(bundle.device),
        "samples_per_dataset": args.samples_per_dataset,
        "total_samples": total_samples,
        "dataset_counts": dataset_counts,
        "total_query_count": total_samples,
        "total_query_seconds": round(total_query_seconds, 4),
        "average_query_seconds": round(avg_query_seconds, 4),
        "gpu_peak_memory_mb": round(gpu_memory_mb, 4),
        "log_file": str(Path(args.log_file).resolve()),
        "pseudo_label_json": str(output_path.resolve()),
        "query_records_csv": str(Path(args.output_csv).resolve()),
    }
    write_json(args.output_stats_json, stats)
    logger.info("Saved %s pseudo-labeled utterances to %s", len(records), output_path)
    logger.info("Saved query records to %s", args.output_csv)
    logger.info("Saved query stats to %s", args.output_stats_json)
    logger.info("Query summary: %s", stats)


if __name__ == "__main__":
    main()
