from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert CommonVoice MP3 files to WAV while preserving the directory tree."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("CommonVoice"),
        help="Root directory containing .mp3 files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("CommonVoice_wav"),
        help="Root directory where .wav files will be written.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=None,
        help="Optional output sample rate. Defaults to the original file sample rate.",
    )
    parser.add_argument(
        "--subtype",
        default="PCM_16",
        help="WAV subtype for soundfile.write, e.g. PCM_16 or FLOAT.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N files. Useful for testing.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose .wav output already exists.",
    )
    return parser.parse_args()


def maybe_resample(audio, original_sr: int, target_sr: int):
    if original_sr == target_sr:
        return audio

    import scipy.signal

    axis = 0 if getattr(audio, "ndim", 1) == 1 else 0
    num_samples = round(audio.shape[axis] * target_sr / original_sr)
    return scipy.signal.resample(audio, num_samples, axis=axis)


def iter_mp3_files(source_dir: Path):
    return sorted(source_dir.rglob("*.mp3"))


def main() -> int:
    args = parse_args()

    if not args.source_dir.exists():
        raise SystemExit(f"Source directory not found: {args.source_dir}")

    mp3_files = iter_mp3_files(args.source_dir)
    if args.limit is not None:
        mp3_files = mp3_files[: args.limit]

    total = len(mp3_files)
    if total == 0:
        print(f"No .mp3 files found under {args.source_dir}")
        return 0

    print(f"Found {total} mp3 files under {args.source_dir}")
    print(f"Writing wav files to {args.output_dir}")

    converted = 0
    skipped = 0
    failed = 0

    for index, mp3_path in enumerate(mp3_files, start=1):
        relative_path = mp3_path.relative_to(args.source_dir)
        wav_path = (args.output_dir / relative_path).with_suffix(".wav")

        if args.skip_existing and wav_path.exists():
            skipped += 1
            if index % 1000 == 0 or index == total:
                print(f"[{index}/{total}] skipped existing: {wav_path}")
            continue

        wav_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio, sample_rate = sf.read(mp3_path)
            output_sr = args.sample_rate or sample_rate
            if args.sample_rate is not None and args.sample_rate != sample_rate:
                audio = maybe_resample(audio, sample_rate, args.sample_rate)
            sf.write(wav_path, audio, output_sr, subtype=args.subtype)
            converted += 1
            if index <= 5 or index % 1000 == 0 or index == total:
                print(f"[{index}/{total}] converted: {mp3_path} -> {wav_path}")
        except Exception as exc:
            failed += 1
            print(f"[{index}/{total}] failed: {mp3_path} ({exc})")

    print(
        f"Done. converted={converted}, skipped={skipped}, failed={failed}, total={total}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
