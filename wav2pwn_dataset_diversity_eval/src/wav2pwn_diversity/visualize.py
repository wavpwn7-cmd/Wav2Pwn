from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def _base_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.titlesize": 18,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def plot_heatmap(matrix: pd.DataFrame, title: str, output_prefix: str) -> None:
    _base_style()
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png")
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()


def plot_family_tsne(feature_matrix, model_ids: list[str], family_map: dict[str, str], title: str, output_prefix: str, perplexity: float) -> None:
    _base_style()
    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)
    tsne = TSNE(n_components=2, random_state=13, perplexity=min(perplexity, max(2, len(model_ids) - 1)))
    coords = tsne.fit_transform(feature_matrix)

    plt.figure(figsize=(9, 7))
    family_labels = [family_map[model_id] for model_id in model_ids]
    palette = sns.color_palette("Set2", n_colors=len(sorted(set(family_labels))))
    family_to_color = {family: palette[idx] for idx, family in enumerate(sorted(set(family_labels)))}

    for idx, model_id in enumerate(model_ids):
        family = family_map[model_id]
        plt.scatter(coords[idx, 0], coords[idx, 1], s=180, color=family_to_color[family], edgecolor="black")
        plt.text(coords[idx, 0], coords[idx, 1], model_id, fontsize=10, ha="left", va="bottom")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black", markersize=10, label=family)
        for family, color in family_to_color.items()
    ]
    plt.legend(handles=handles, title="Family", loc="best")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}.png")
    plt.savefig(f"{output_prefix}.pdf")
    plt.close()
