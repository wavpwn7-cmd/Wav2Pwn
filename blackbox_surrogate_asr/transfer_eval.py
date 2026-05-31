from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transfer surrogate-generated adversarial examples to the black-box target model.")
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels.json")
    parser.add_argument("--target-model", default="facebook/hubert-large-ls960-ft")
    parser.add_argument("--adversarial-audio", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--target-phrase", default="OPEN THE DOOR")
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--relaxed-substring-match", action="store_true")
    parser.add_argument("--output-json", default="results/results_metrics.json")
    parser.add_argument("--output-csv", default="results/results_metrics.csv")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.asr_blackbox.data import load_pseudo_label_dataset
    from src.asr_blackbox.logging_utils import get_logger, write_csv, write_json
    from src.asr_blackbox.metrics import compute_text_metrics, targeted_success, untargeted_success
    from src.asr_blackbox.models import load_asr_bundle, load_audio, transcribe_waveform

    logger = get_logger("transfer_eval")
    dataset = load_pseudo_label_dataset(args.pseudo_labels)
    sample = dataset[int(args.sample_index)]
    target = load_asr_bundle(args.target_model, device=args.device)

    clean_waveform, clean_sr = load_audio(sample["audio_path"])
    adv_waveform, adv_sr = load_audio(args.adversarial_audio)

    clean_transcription = transcribe_waveform(target, clean_waveform, clean_sr)
    adv_transcription = transcribe_waveform(target, adv_waveform, adv_sr)

    clean_metrics = compute_text_metrics(sample.get("ground_truth") or sample["transcription"], clean_transcription)
    shift_metrics = compute_text_metrics(clean_transcription, adv_transcription)
    tasr_success = targeted_success(adv_transcription, args.target_phrase, relaxed=args.relaxed_substring_match)
    uasr_success = untargeted_success(clean_transcription, adv_transcription, args.cer_threshold, args.wer_threshold)

    row = {
        "sample_id": sample["sample_id"],
        "clean_transcript": clean_transcription,
        "adversarial_transcript": adv_transcription,
        "target_transcript": args.target_phrase,
        "clean_cer": clean_metrics.cer,
        "clean_wer": clean_metrics.wer,
        "transcription_shift_cer": shift_metrics.cer,
        "transcription_shift_wer": shift_metrics.wer,
        "tasr_success": int(tasr_success),
        "uasr_success": int(uasr_success),
        "attack_time": None,
        "query_count": 2,
    }

    summary = {
        "Attack Cost": {"total_query_count": 2},
        "Clean CER": clean_metrics.cer,
        "TASR": float(tasr_success),
        "UASR": float(uasr_success),
        "targeted_mode": args.targeted,
    }
    write_json(args.output_json, {"summary": summary, "samples": [row]})
    write_csv(args.output_csv, [row], list(row.keys()))
    logger.info("Transfer evaluation complete: %s", summary)


if __name__ == "__main__":
    main()
