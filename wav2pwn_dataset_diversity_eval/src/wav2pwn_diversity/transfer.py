from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from .metrics import targeted_success, untargeted_success


def aggregate_transfer_rows(
    transfer_rows: list[dict],
    target_phrase: str,
    cer_threshold: float,
    wer_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for row in transfer_rows:
        tasr = int(targeted_success(row["adv_prediction"], target_phrase, relaxed=True))
        uasr = int(untargeted_success(row["clean_prediction"], row["adv_prediction"], cer_threshold, wer_threshold))
        similarity = 0.5 * (tasr + uasr)
        rows.append(
            {
                "source_model": row["source_model"],
                "target_model": row["target_model"],
                "attack_mode": row["attack_mode"],
                "tasr": tasr,
                "uasr": uasr,
                "transfer_similarity": similarity,
                "sample_index": row["sample_index"],
            }
        )

    pair_df = pd.DataFrame(rows)
    grouped = pair_df.groupby(["source_model", "target_model"], as_index=False).agg(
        tasr=("tasr", "mean"),
        uasr=("uasr", "mean"),
        transfer_similarity=("transfer_similarity", "mean"),
    )

    transfer_matrix = grouped.pivot(index="source_model", columns="target_model", values="transfer_similarity").fillna(1.0)
    tasr_matrix = grouped.pivot(index="source_model", columns="target_model", values="tasr").fillna(1.0)
    uasr_matrix = grouped.pivot(index="source_model", columns="target_model", values="uasr").fillna(1.0)

    ordered_models = sorted(set(pair_df["source_model"]).union(set(pair_df["target_model"])))
    transfer_matrix = transfer_matrix.reindex(index=ordered_models, columns=ordered_models, fill_value=1.0)
    tasr_matrix = tasr_matrix.reindex(index=ordered_models, columns=ordered_models, fill_value=1.0)
    uasr_matrix = uasr_matrix.reindex(index=ordered_models, columns=ordered_models, fill_value=1.0)
    np.fill_diagonal(transfer_matrix.values, 1.0)
    np.fill_diagonal(tasr_matrix.values, 1.0)
    np.fill_diagonal(uasr_matrix.values, 1.0)
    return grouped, transfer_matrix, tasr_matrix, uasr_matrix
