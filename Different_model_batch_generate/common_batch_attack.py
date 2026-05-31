from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoModelForCTC, AutoProcessor


DEFAULT_TARGET_TEXT = "TURN OFF SECURITY CAMERA"


def _load_audio(path: Path, target_sr: int = 16000) -> tuple[torch.Tensor, int]:
    waveform, sample_rate = sf.read(path, always_2d=True)
    waveform = torch.from_numpy(waveform.T).float()
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        import torchaudio

        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
        sample_rate = target_sr
    return waveform.squeeze(0), sample_rate


def _iter_audio_files(root_dir: Path) -> list[Path]:
    audio_files = []
    for pattern in ("*.flac", "*.wav"):
        audio_files.extend(root_dir.rglob(pattern))
    return sorted(audio_files)


def _write_summary_row(csv_path: Path, row: list[str | int | float]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "audio_file",
                    "original_transcription",
                    "adversarial_transcription",
                    "target_text",
                    "steps_used",
                    "success",
                ]
            )
        writer.writerow(row)


def run_batch_attack(
    *,
    default_model_name: str,
    model_slug: str,
    paper_name: str,
) -> None:
    parser = argparse.ArgumentParser(
        description=f"Batch targeted attack generator for {paper_name} ({default_model_name})."
    )
    parser.add_argument("--model-name", default=default_model_name)
    parser.add_argument("--root-dir", default="dataset/LibriSpeech_wav")
    parser.add_argument("--output-dir", default=f"outputs/{model_slug}")
    parser.add_argument("--target-text", default=DEFAULT_TARGET_TEXT)
    parser.add_argument("--num-files", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--regularization-weight", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForCTC.from_pretrained(args.model_name).to(device).eval()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    summary_csv = output_dir / "attack_results.csv"

    audio_files = _iter_audio_files(root_dir)
    if not audio_files:
        raise RuntimeError(f"No .flac or .wav files found under {root_dir}")

    rng = random.Random(args.seed)
    selected_files = audio_files if len(audio_files) <= args.num_files else rng.sample(audio_files, args.num_files)

    print(f"Model: {args.model_name}")
    print(f"Paper label: {paper_name}")
    print(f"Using device: {device}")
    print(f"Processing {len(selected_files)} files from {root_dir}")

    for audio_path in selected_files:
        waveform, sample_rate = _load_audio(audio_path)
        inputs = processor(
            waveform.numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        target_ids = processor.tokenizer(args.target_text, return_tensors="pt").input_ids.to(device)
        perturbation = torch.zeros_like(input_values, requires_grad=True, device=device)
        optimizer = torch.optim.Adam([perturbation], lr=args.learning_rate)

        original_logits = model(input_values=input_values, attention_mask=attention_mask).logits
        original_transcription = processor.batch_decode(torch.argmax(original_logits, dim=-1))[0].strip()

        adversarial_transcription = ""
        steps_used = args.steps
        for step in range(args.steps):
            perturbed_input = torch.clamp(input_values + perturbation, -1.0, 1.0)
            outputs = model(input_values=perturbed_input, attention_mask=attention_mask)
            logits = outputs.logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1).permute(1, 0, 2)

            input_lengths = torch.full(
                (log_probs.size(1),),
                fill_value=log_probs.size(0),
                dtype=torch.long,
                device=device,
            )
            target_lengths = torch.full(
                (target_ids.size(0),),
                fill_value=target_ids.size(1),
                dtype=torch.long,
                device=device,
            )

            ctc_loss = torch.nn.functional.ctc_loss(
                log_probs,
                target_ids,
                input_lengths,
                target_lengths,
                zero_infinity=True,
            )
            l2_penalty = torch.norm(perturbation, p=2)
            loss = ctc_loss + args.regularization_weight * l2_penalty

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
            optimizer.step()

            with torch.no_grad():
                perturbation.data = torch.clamp(perturbation.data, -args.epsilon, args.epsilon)

            adversarial_transcription = processor.batch_decode(torch.argmax(logits, dim=-1))[0].strip()
            if adversarial_transcription.upper() == args.target_text.upper():
                steps_used = step + 1
                break

        perturbed_audio = torch.clamp(input_values + perturbation, -1.0, 1.0).detach().cpu().numpy().squeeze()
        output_subdir = output_dir / audio_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)
        sf.write(output_subdir / f"{audio_path.stem}_perturbed.wav", perturbed_audio, sample_rate)

        success = int(adversarial_transcription.upper() == args.target_text.upper())
        _write_summary_row(
            summary_csv,
            [
                str(audio_path),
                original_transcription,
                adversarial_transcription,
                args.target_text,
                steps_used,
                success,
            ],
        )
        print(f"[{paper_name}] {audio_path.name} -> success={success}, steps={steps_used}")


def main(*, default_model_name: str, model_slug: str, paper_name: str) -> None:
    run_batch_attack(
        default_model_name=default_model_name,
        model_slug=model_slug,
        paper_name=paper_name,
    )
