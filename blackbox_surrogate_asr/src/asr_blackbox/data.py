from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from datasets import Dataset

from .models import load_audio


LIBRISPEECH_EXTENSIONS = {".flac", ".wav"}
AUDIO_EXTENSIONS = {".flac", ".wav", ".mp3", ".m4a", ".ogg"}


@dataclass
class QuerySample:
    sample_id: str
    dataset_name: str
    audio_path: str
    transcription: str
    duration: float
    ground_truth: str | None = None


@dataclass
class AudioItem:
    dataset_name: str
    audio_path: str
    ground_truth: str | None = None


def discover_librispeech_samples(
    root: str | Path,
    excluded_subdirs: tuple[str, ...] = ("test-clean", "train-clean", "train-clean-100"),
) -> list[AudioItem]:
    root_path = Path(root)
    transcripts: dict[str, str] = {}
    for transcript_file in root_path.rglob("*.trans.txt"):
        if any(part in excluded_subdirs for part in transcript_file.parts):
            continue
        with transcript_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                utterance_id, text = line.strip().split(" ", 1)
                transcripts[utterance_id] = text

    items: list[AudioItem] = []
    for file_path in root_path.rglob("*"):
        if file_path.suffix.lower() in LIBRISPEECH_EXTENSIONS:
            if any(part in excluded_subdirs for part in file_path.parts):
                continue
            items.append(
                AudioItem(
                    dataset_name="LibriSpeech",
                    audio_path=str(file_path),
                    ground_truth=transcripts.get(file_path.stem),
                )
            )
    return sorted(items, key=lambda item: item.audio_path)


def discover_commonvoice_samples(root: str | Path) -> list[AudioItem]:
    root_path = Path(root)
    items: list[AudioItem] = []
    for file_path in root_path.rglob("*"):
        if file_path.suffix.lower() in AUDIO_EXTENSIONS:
            items.append(
                AudioItem(
                    dataset_name="CommonVoice_wav",
                    audio_path=str(file_path),
                )
            )
    return sorted(items, key=lambda item: item.audio_path)


def discover_voxpopuli_samples(root: str | Path) -> list[AudioItem]:
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
                wav_path = parts[2]
                normalized_text = parts[3]
                transcripts[wav_path] = normalized_text

    items: list[AudioItem] = []
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


def sample_dataset_items(dataset_name: str, root: str | Path, num_samples: int, seed: int) -> list[AudioItem]:
    normalized = dataset_name.lower()
    if normalized == "librispeech":
        samples = discover_librispeech_samples(root)
    elif normalized == "commonvoice_wav":
        samples = discover_commonvoice_samples(root)
    elif normalized == "voxpopuli":
        samples = discover_voxpopuli_samples(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_samples > len(samples):
        raise ValueError(f"Requested {num_samples} samples but only found {len(samples)} in {root}.")
    rng = random.Random(seed)
    return rng.sample(samples, num_samples)


def sample_dataset_items_with_options(
    dataset_name: str,
    root: str | Path,
    num_samples: int,
    seed: int,
    allow_librispeech_test_clean: bool = False,
) -> list[AudioItem]:
    normalized = dataset_name.lower()
    if normalized == "librispeech":
        excluded = ("train-clean", "train-clean-100")
        if not allow_librispeech_test_clean:
            excluded = ("test-clean", "train-clean", "train-clean-100")
        samples = discover_librispeech_samples(root, excluded_subdirs=excluded)
    elif normalized == "commonvoice_wav":
        samples = discover_commonvoice_samples(root)
    elif normalized == "voxpopuli":
        samples = discover_voxpopuli_samples(root)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if num_samples > len(samples):
        raise ValueError(
            f"Requested {num_samples} samples but only found {len(samples)} in {root} "
            f"for dataset {dataset_name}."
        )
    rng = random.Random(seed)
    return rng.sample(samples, num_samples)


def sample_multiple_datasets(
    dataset_specs: list[tuple[str, str | Path, int]],
    seed: int,
    allow_librispeech_test_clean: bool = False,
) -> list[AudioItem]:
    combined: list[AudioItem] = []
    for index, (dataset_name, root, num_samples) in enumerate(dataset_specs):
        combined.extend(
            sample_dataset_items_with_options(
                dataset_name,
                root,
                num_samples,
                seed + index,
                allow_librispeech_test_clean=allow_librispeech_test_clean,
            )
        )
    return combined


def build_pseudo_label_records(audio_items: Iterable[AudioItem], transcriptions: list[str], query_times: list[float] | None = None) -> list[dict]:
    records = []
    for idx, (item, transcription) in enumerate(zip(audio_items, transcriptions, strict=True)):
        waveform, sample_rate = load_audio(item.audio_path)
        record = {
            "sample_id": f"query_{idx:06d}",
            "dataset_name": item.dataset_name,
            "audio_path": item.audio_path,
            "transcription": transcription,
            "duration": round(waveform.numel() / sample_rate, 4),
            "ground_truth": item.ground_truth,
        }
        if query_times is not None:
            record["query_seconds"] = round(query_times[idx], 4)
        records.append(record)
    return records


def summarize_dataset_counts(audio_items: Iterable[AudioItem]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in audio_items:
        counts[item.dataset_name] += 1
    return dict(counts)


def load_pseudo_label_dataset(path: str | Path) -> Dataset:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return Dataset.from_list(payload)


def train_val_split(dataset: Dataset, validation_ratio: float, seed: int) -> dict[str, Dataset]:
    return dataset.train_test_split(test_size=validation_ratio, seed=seed)


class CTCCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch
