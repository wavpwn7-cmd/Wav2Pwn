from __future__ import annotations

from itertools import combinations


def build_family_map(model_entries: list[dict]) -> dict[str, str]:
    return {entry["id"]: entry["family"] for entry in model_entries}


def same_family(model_a: str, model_b: str, family_map: dict[str, str]) -> bool:
    return family_map[model_a] == family_map[model_b]


def family_pairs(model_ids: list[str], family_map: dict[str, str]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    intra, inter = [], []
    for model_a, model_b in combinations(model_ids, 2):
        if same_family(model_a, model_b, family_map):
            intra.append((model_a, model_b))
        else:
            inter.append((model_a, model_b))
    return intra, inter
