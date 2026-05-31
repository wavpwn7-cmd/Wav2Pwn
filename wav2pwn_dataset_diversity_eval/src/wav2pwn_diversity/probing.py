from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .metrics import compute_text_metrics, pairwise_similarity_from_transcripts


def build_clean_transcript_frame(model_transcripts: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for model_id, model_rows in model_transcripts.items():
        for row in model_rows:
            rows.append(
                {
                    "model_id": model_id,
                    "sample_index": row["sample_index"],
                    "audio_path": row["audio_path"],
                    "transcription": row["transcription"],
                    "ground_truth": row.get("ground_truth"),
                }
            )
    return pd.DataFrame(rows)


def build_probing_similarity_outputs(transcript_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cer_matrix, similarity_matrix = pairwise_similarity_from_transcripts(transcript_frame)

    pair_rows = []
    for source_id, target_id in product(similarity_matrix.index, similarity_matrix.columns):
        pair_rows.append(
            {
                "source_model": source_id,
                "target_model": target_id,
                "probing_similarity": float(similarity_matrix.loc[source_id, target_id]),
                "probing_cer": float(cer_matrix.loc[source_id, target_id]),
            }
        )
    pair_df = pd.DataFrame(pair_rows)

    signature_matrix = similarity_matrix.values
    embedding_similarity = cosine_similarity(signature_matrix)
    embedding_df = pd.DataFrame(embedding_similarity, index=similarity_matrix.index, columns=similarity_matrix.columns)
    return similarity_matrix, pair_df, embedding_df


def select_best_target_proxy(
    target_model_id: str,
    target_clean_transcripts: list[dict],
    candidate_model_transcripts: dict[str, list[dict]],
) -> pd.DataFrame:
    rows = []
    target_map = {row["sample_index"]: row["transcription"] for row in target_clean_transcripts}
    for candidate_id, candidate_rows in candidate_model_transcripts.items():
        cers = []
        wers = []
        for row in candidate_rows:
            metrics = compute_text_metrics(target_map[row["sample_index"]], row["transcription"])
            cers.append(metrics.cer)
            wers.append(metrics.wer)
        rows.append(
            {
                "target_model": target_model_id,
                "candidate_model": candidate_id,
                "avg_alignment_cer": float(np.mean(cers)),
                "avg_alignment_wer": float(np.mean(wers)),
                "alignment_score": float(np.mean(cers) + np.mean(wers)),
            }
        )
    return pd.DataFrame(rows).sort_values(["alignment_score", "avg_alignment_cer", "avg_alignment_wer"])
