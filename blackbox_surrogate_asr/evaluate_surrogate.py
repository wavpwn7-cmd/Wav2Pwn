from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate how closely the surrogate imitates HuBERT pseudo-labels.")
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels.json")
    parser.add_argument("--surrogate-model", default="checkpoints/best_model")
    parser.add_argument("--output-json", default="results/surrogate_eval.json")
    parser.add_argument("--output-csv", default="results/surrogate_eval.csv")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.asr_blackbox.data import load_pseudo_label_dataset
    from src.asr_blackbox.logging_utils import get_logger, write_csv, write_json
    from src.asr_blackbox.metrics import compute_text_metrics
    from src.asr_blackbox.models import load_asr_bundle, load_audio, transcribe_waveform

    logger = get_logger("evaluate_surrogate")
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    surrogate = load_asr_bundle(args.surrogate_model, device=args.device)

    rows = []
    for sample in dataset:
        waveform, sample_rate = load_audio(sample["audio_path"])
        prediction = transcribe_waveform(surrogate, waveform, sample_rate)
        metrics = compute_text_metrics(sample["transcription"], prediction)
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "hubert_transcription": sample["transcription"],
                "surrogate_transcription": prediction,
                "wer": metrics.wer,
                "cer": metrics.cer,
            }
        )

    summary = {
        "num_samples": len(rows),
        "avg_wer": sum(row["wer"] for row in rows) / max(len(rows), 1),
        "avg_cer": sum(row["cer"] for row in rows) / max(len(rows), 1),
    }
    write_json(args.output_json, {"summary": summary, "samples": rows})
    write_csv(args.output_csv, rows, list(rows[0].keys()) if rows else ["sample_id"])
    logger.info("Surrogate evaluation complete: %s", summary)


if __name__ == "__main__":
    main()
