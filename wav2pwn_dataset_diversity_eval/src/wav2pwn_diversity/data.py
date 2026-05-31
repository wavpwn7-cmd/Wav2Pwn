from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path


LIBRISPEECH_EXTENSIONS = {".flac", ".wav"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


@dataclass
class AudioItem:
    dataset_name: str
    audio_path: str
    ground_truth: str | None = None


def _discover_librispeech(root: str | Path, allow_test_clean: bool) -> list[AudioItem]:
    root_path = Path(root)
    transcripts: dict[str, str] = {}
    for transcript_file in root_path.rglob("*.trans.txt"):
        with transcript_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                utterance_id, text = line.strip().split(" ", 1)
                transcripts[utterance_id] = text

    items = []
    for file_path in root_path.rglob("*"):
        if file_path.suffix.lower() not in LIBRISPEECH_EXTENSIONS:
            continue
        if not allow_test_clean and "test-clean" in file_path.parts:
            continue
        items.append(
            AudioItem(
                dataset_name="LibriSpeech",
                audio_path=str(file_path),
                ground_truth=transcripts.get(file_path.stem),
            )
        )
    return sorted(items, key=lambda item: item.audio_path)


def _discover_commonvoice(root: str | Path) -> list[AudioItem]:
    items = []
    for file_path in Path(root).rglob("*"):
        if file_path.suffix.lower() in AUDIO_EXTENSIONS:
            items.append(AudioItem(dataset_name="CommonVoice_wav", audio_path=str(file_path)))
    return sorted(items, key=lambda item: item.audio_path)


def _discover_voxpopuli(root: str | Path) -> list[AudioItem]:
    root_path = Path(root)
    transcripts: dict[str, str] = {}
    for manifest_name in ("manifest.tsv", "dev.tsv", "test.tsv"):
        manifest_path = root_path / manifest_name
        if not manifest_path.exists():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            header = next(handle, None)
            if header is None:
                continue
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 5:
                    continue
                transcripts[parts[2]] = parts[3]

    items = []
    for file_path in (root_path / "wav").rglob("*.wav"):
        rel_path = file_path.relative_to(root_path).as_posix()
        items.append(
            AudioItem(
                dataset_name="voxpopuli",
                audio_path=str(file_path),
                ground_truth=transcripts.get(rel_path),
            )
        )
    return sorted(items, key=lambda item: item.audio_path)


def discover_dataset_items(dataset_slug: str, dataset_config: dict) -> list[AudioItem]:
    normalized = dataset_slug.lower()
    if normalized == "librispeech":
        return _discover_librispeech(dataset_config["root"], allow_test_clean=dataset_config.get("allow_test_clean", False))
    if normalized == "commonvoice":
        return _discover_commonvoice(dataset_config["root"])
    if normalized == "voxpopuli":
        return _discover_voxpopuli(dataset_config["root"])
    raise ValueError(f"Unsupported dataset slug: {dataset_slug}")


def sample_dataset_items(dataset_slug: str, dataset_config: dict, seed: int) -> list[AudioItem]:
    items = discover_dataset_items(dataset_slug, dataset_config)
    sample_count = int(dataset_config["sample_count"])
    if sample_count > len(items):
        raise ValueError(
            f"Requested {sample_count} samples for {dataset_slug}, but only found {len(items)} at {dataset_config['root']}."
        )
    rng = random.Random(seed)
    return rng.sample(items, sample_count)
