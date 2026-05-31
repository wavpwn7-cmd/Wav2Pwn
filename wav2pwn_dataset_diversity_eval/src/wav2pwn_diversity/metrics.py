from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from jiwer import cer, wer
from sklearn.metrics import silhouette_score

from .families import family_pairs


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    if isinstance(text, float) and pd.isna(text):
        return ""
    return " ".join(str(text).upper().strip().split())


def targeted_success(prediction: str, target_phrase: str, relaxed: bool = True) -> bool:
    prediction_norm = normalize_text(prediction)
    target_norm = normalize_text(target_phrase)
    if relaxed:
        return target_norm in prediction_norm
    return prediction_norm == target_norm


def untargeted_success(clean_prediction: str, adv_prediction: str, cer_threshold: float, wer_threshold: float) -> bool:
    clean_norm = normalize_text(clean_prediction)
    adv_norm = normalize_text(adv_prediction)
    return cer(clean_norm, adv_norm) >= cer_threshold or wer(clean_norm, adv_norm) >= wer_threshold


@dataclass
class TextMetrics:
    wer: float
    cer: float


def compute_text_metrics(reference: str, hypothesis: str) -> TextMetrics:
    reference_norm = normalize_text(reference)
    hypothesis_norm = normalize_text(hypothesis)
    return TextMetrics(wer=wer(reference_norm, hypothesis_norm), cer=cer(reference_norm, hypothesis_norm))


def pairwise_similarity_from_transcripts(transcript_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_ids = sorted(transcript_frame["model_id"].unique())
    audio_groups = {
        model_id: transcript_frame[transcript_frame["model_id"] == model_id]
        .sort_values("sample_index")
        .reset_index(drop=True)
        for model_id in model_ids
    }

    cer_matrix = pd.DataFrame(np.eye(len(model_ids)), index=model_ids, columns=model_ids, dtype=float)
    sim_matrix = pd.DataFrame(np.eye(len(model_ids)), index=model_ids, columns=model_ids, dtype=float)

    for source_id, target_id in product(model_ids, model_ids):
        if source_id == target_id:
            continue
        source_rows = audio_groups[source_id]
        target_rows = audio_groups[target_id]
        cers = []
        for source_text, target_text in zip(source_rows["transcription"], target_rows["transcription"], strict=True):
            cers.append(compute_text_metrics(source_text, target_text).cer)
        mean_cer = float(np.mean(cers))
        cer_matrix.loc[source_id, target_id] = mean_cer
        sim_matrix.loc[source_id, target_id] = max(0.0, 1.0 - mean_cer)
    return cer_matrix, sim_matrix


def compute_table3_metrics(
    dataset_name: str,
    similarity_matrix: pd.DataFrame,
    family_map: dict[str, str],
    feature_matrix: np.ndarray,
) -> dict:
    model_ids = list(similarity_matrix.index)
    intra_pairs, inter_pairs = family_pairs(model_ids, family_map)

    def average_similarity(pairs: list[tuple[str, str]]) -> float:
        if not pairs:
            return 0.0
        values = []
        for model_a, model_b in pairs:
            values.append(similarity_matrix.loc[model_a, model_b])
            values.append(similarity_matrix.loc[model_b, model_a])
        return float(np.mean(values))

    intra_similarity = average_similarity(intra_pairs)
    inter_similarity = average_similarity(inter_pairs)
    similarity_gap = intra_similarity - inter_similarity

    off_diagonal_values = []
    for source_id, target_id in product(model_ids, model_ids):
        if source_id != target_id:
            off_diagonal_values.append(similarity_matrix.loc[source_id, target_id])
    transfer_variance = float(np.var(off_diagonal_values)) if off_diagonal_values else 0.0

    labels = [family_map[model_id] for model_id in model_ids]
    unique_labels = sorted(set(labels))
    silhouette = 0.0
    if len(unique_labels) > 1 and len(feature_matrix) > len(unique_labels):
        silhouette = float(silhouette_score(feature_matrix, labels, metric="cosine"))

    return {
        "dataset": dataset_name,
        "intra_similarity": intra_similarity,
        "inter_similarity": inter_similarity,
        "similarity_gap": similarity_gap,
        "transfer_variance": transfer_variance,
        "silhouette_score": silhouette,
    }
