from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor


def calculate_weighted_accuracy(predicted: str, target: str, tolerance_factor: float = 1.0) -> float:
    predicted_clean = predicted.replace(" ", "").replace("|", "")
    target_clean = target.replace(" ", "").replace("|", "")
    matcher = SequenceMatcher(None, predicted_clean, target_clean)
    similarity = matcher.ratio()
    max_length = max(len(predicted_clean), len(target_clean), 1)
    accuracy = (similarity * (len(target_clean) + (tolerance_factor ** 0.5)) / max_length) * 100.0
    return min(accuracy, 100.0)


def run_model_eval(
    *,
    default_model_name: str | None,
    paper_name: str,
) -> None:
    parser = argparse.ArgumentParser(description=f"Evaluate one adversarial audio file on {paper_name}.")
    parser.add_argument("--audio-path", required=True, help="Path to the adversarial audio file to test.")
    parser.add_argument("--target-text", default="HELLO WORLD")
    parser.add_argument(
        "--model-name",
        default=default_model_name,
        help="Override the Hugging Face model or local checkpoint path.",
    )
    args = parser.parse_args()

    if not args.model_name:
        raise RuntimeError(
            f"No default checkpoint is defined for {paper_name}. "
            "Please pass --model-name with the CTC-capable checkpoint used in your paper experiments."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        processor = AutoProcessor.from_pretrained(args.model_name)
        model = AutoModelForCTC.from_pretrained(args.model_name).to(device).eval()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {args.model_name} as a CTC ASR checkpoint for {paper_name}. "
            "If the paper used a local fine-tuned checkpoint for this model family, rerun with --model-name "
            "pointing to that checkpoint."
        ) from exc

    audio_path = Path(args.audio_path)
    waveform, sample_rate = sf.read(audio_path, always_2d=True)
    waveform = waveform.mean(axis=1)

    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0].strip()
    success_rate = calculate_weighted_accuracy(transcription, args.target_text)

    print(f"Model       : {paper_name}")
    print(f"Checkpoint  : {args.model_name}")
    print(f"Audio       : {audio_path}")
    print(f"Predicted   : {transcription}")
    print(f"Target      : {args.target_text}")
    print(f"SuccessRate : {success_rate:.2f}%")


def main(*, default_model_name: str | None, paper_name: str) -> None:
    run_model_eval(default_model_name=default_model_name, paper_name=paper_name)
