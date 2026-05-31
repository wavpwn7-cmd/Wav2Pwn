from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a LibriSpeech directory from FLAC to WAV while preserving directory structure."
    )
    parser.add_argument("--input-root", default="../dataset/LibriSpeech", help="Path to the source LibriSpeech directory.")
    parser.add_argument("--output-root", default="../dataset/LibriSpeech_wav", help="Path to the converted LibriSpeech WAV directory.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output WAV sample rate.")
    parser.add_argument("--log-file", default="logs/convert_librispeech.log")
    return parser.parse_args()


def render_progress(current: int, total: int) -> str:
    width = 30
    ratio = current / max(total, 1)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {ratio * 100:6.2f}% ({current}/{total})"


def main() -> None:
    args = parse_args()
    from src.asr_blackbox.logging_utils import get_logger

    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'soundfile'. Install it with: "
            "pip install soundfile"
        ) from exc

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger = get_logger("convert_librispeech_flac_to_wav", log_file=args.log_file)
    flac_files = sorted(input_root.rglob("*.flac"))
    if not flac_files:
        raise SystemExit(f"No FLAC files found under {input_root}")

    logger.info("Found %s FLAC files under %s", len(flac_files), input_root)
    start_time = time.perf_counter()

    for index, flac_path in enumerate(flac_files, start=1):
        rel_path = flac_path.relative_to(input_root)
        wav_path = output_root / rel_path.with_suffix(".wav")
        wav_path.parent.mkdir(parents=True, exist_ok=True)

        waveform, sample_rate = sf.read(flac_path, always_2d=True)
        if sample_rate != args.sample_rate:
            import torchaudio
            import torch

            tensor = torch.from_numpy(waveform.T).float()
            tensor = torchaudio.functional.resample(tensor, sample_rate, args.sample_rate)
            waveform = tensor.T.cpu().numpy()
            sample_rate = args.sample_rate

        sf.write(wav_path, waveform, sample_rate)

        progress = render_progress(index, len(flac_files))
        print(f"\r{progress}", end="", flush=True)
        if index % 100 == 0 or index == len(flac_files):
            logger.info("Converted %s", progress)

    print()

    copied_files = 0
    for path in input_root.rglob("*"):
        if path.is_dir() or path.suffix.lower() == ".flac":
            continue
        rel_path = path.relative_to(input_root)
        dst = output_root / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst)
        copied_files += 1

    elapsed = time.perf_counter() - start_time
    logger.info("Finished conversion in %.2f seconds", elapsed)
    logger.info("Copied %s metadata/non-audio files", copied_files)
    logger.info("Converted WAV directory: %s", output_root)


if __name__ == "__main__":
    main()
