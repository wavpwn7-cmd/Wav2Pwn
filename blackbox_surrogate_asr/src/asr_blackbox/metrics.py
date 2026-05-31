from __future__ import annotations

from dataclasses import dataclass

from jiwer import cer, wer


def normalize_text(text: str) -> str:
    return " ".join(text.upper().strip().split())


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
    return TextMetrics(
        wer=wer(reference_norm, hypothesis_norm),
        cer=cer(reference_norm, hypothesis_norm),
    )

