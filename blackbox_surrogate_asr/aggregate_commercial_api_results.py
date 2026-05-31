from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate completed commercial-API transcripts into Table IV style metrics."
    )
    parser.add_argument("--manifest-csv", default="results/commercial_api_query_manifest.csv")
    parser.add_argument("--output-csv", default="results/commercial_api_table4.csv")
    parser.add_argument("--output-json", default="results/commercial_api_table4.json")
    parser.add_argument("--output-md", default="results/commercial_api_table4.md")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--relaxed-substring-match", action="store_true")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Manifest CSV is missing required columns: {missing}")


def _non_empty(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["api_transcript"].fillna("").str.strip() != ""].copy()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).upper().strip().split())


def _edit_distance(seq_a: list[str], seq_b: list[str]) -> int:
    rows = len(seq_a) + 1
    cols = len(seq_b) + 1
    dp = [[0] * cols for _ in range(rows)]

    for row in range(rows):
        dp[row][0] = row
    for col in range(cols):
        dp[0][col] = col

    for row in range(1, rows):
        for col in range(1, cols):
            substitution_cost = 0 if seq_a[row - 1] == seq_b[col - 1] else 1
            dp[row][col] = min(
                dp[row - 1][col] + 1,
                dp[row][col - 1] + 1,
                dp[row - 1][col - 1] + substitution_cost,
            )
    return dp[-1][-1]


@dataclass
class _TextMetrics:
    wer: float
    cer: float


def _compute_text_metrics(reference: str, hypothesis: str) -> _TextMetrics:
    reference_norm = _normalize_text(reference)
    hypothesis_norm = _normalize_text(hypothesis)

    reference_words = reference_norm.split()
    hypothesis_words = hypothesis_norm.split()
    reference_chars = list(reference_norm)
    hypothesis_chars = list(hypothesis_norm)

    wer_denominator = max(len(reference_words), 1)
    cer_denominator = max(len(reference_chars), 1)
    return _TextMetrics(
        wer=_edit_distance(reference_words, hypothesis_words) / wer_denominator,
        cer=_edit_distance(reference_chars, hypothesis_chars) / cer_denominator,
    )


def _targeted_success(prediction: str, target_phrase: str, relaxed: bool = True) -> bool:
    prediction_norm = _normalize_text(prediction)
    target_norm = _normalize_text(target_phrase)
    if relaxed:
        return target_norm in prediction_norm
    return prediction_norm == target_norm


def _untargeted_success(clean_prediction: str, adv_prediction: str, cer_threshold: float, wer_threshold: float) -> bool:
    shift_metrics = _compute_text_metrics(clean_prediction, adv_prediction)
    return shift_metrics.cer >= cer_threshold or shift_metrics.wer >= wer_threshold


def _render_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    args = parse_args()

    from src.asr_blackbox.logging_utils import write_json

    manifest_df = pd.read_csv(PROJECT_ROOT / args.manifest_csv, keep_default_na=False)
    _require_columns(
        manifest_df,
        [
            "target_group",
            "target_id",
            "display_name",
            "provider",
            "product",
            "api_version",
            "model_name",
            "sample_index",
            "sample_id",
            "audio_variant",
            "reference_transcript",
            "target_phrase",
            "api_transcript",
        ],
    )
    manifest_df = _non_empty(manifest_df)
    if manifest_df.empty:
        raise ValueError("No completed transcripts found. Fill in api_transcript before aggregation.")

    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    grouped = manifest_df.groupby(
        ["target_group", "target_id", "display_name", "provider", "product", "api_version", "model_name"],
        dropna=False,
    )
    for group_key, target_df in grouped:
        clean_df = target_df[target_df["audio_variant"] == "clean"].copy()
        adv_df = target_df[target_df["audio_variant"] == "adversarial"].copy()

        merged = clean_df.merge(
            adv_df,
            on=[
                "target_group",
                "target_id",
                "display_name",
                "provider",
                "product",
                "api_version",
                "model_name",
                "sample_index",
                "sample_id",
                "reference_transcript",
                "target_phrase",
            ],
            suffixes=("_clean", "_adv"),
        )
        if merged.empty:
            continue

        clean_cers = []
        shift_cers = []
        shift_wers = []
        tasr_hits = 0
        uasr_hits = 0

        for row in merged.itertuples(index=False):
            clean_metrics = _compute_text_metrics(row.reference_transcript, row.api_transcript_clean)
            shift_metrics = _compute_text_metrics(row.api_transcript_clean, row.api_transcript_adv)
            tasr_hit = _targeted_success(
                row.api_transcript_adv,
                row.target_phrase,
                relaxed=args.relaxed_substring_match,
            )
            uasr_hit = _untargeted_success(
                row.api_transcript_clean,
                row.api_transcript_adv,
                args.cer_threshold,
                args.wer_threshold,
            )

            clean_cers.append(clean_metrics.cer)
            shift_cers.append(shift_metrics.cer)
            shift_wers.append(shift_metrics.wer)
            tasr_hits += int(tasr_hit)
            uasr_hits += int(uasr_hit)

            sample_rows.append(
                {
                    "target_group": row.target_group,
                    "target_id": row.target_id,
                    "display_name": row.display_name,
                    "sample_index": int(row.sample_index),
                    "sample_id": row.sample_id,
                    "reference_transcript": row.reference_transcript,
                    "clean_prediction": row.api_transcript_clean,
                    "adversarial_prediction": row.api_transcript_adv,
                    "clean_cer": round(clean_metrics.cer, 6),
                    "shift_cer": round(shift_metrics.cer, 6),
                    "shift_wer": round(shift_metrics.wer, 6),
                    "tasr_success": int(tasr_hit),
                    "uasr_success": int(uasr_hit),
                }
            )

        target_group, target_id, display_name, provider, product, api_version, model_name = group_key
        rows.append(
            {
                "Type": target_group,
                "Black-box API": display_name,
                "provider": provider,
                "product": product,
                "api_version": api_version,
                "model_name": model_name,
                "num_samples": len(merged),
                "TASR (%)": round(100.0 * tasr_hits / len(merged), 2),
                "UASR (%)": round(100.0 * uasr_hits / len(merged), 2),
                "CER (%)": round(100.0 * sum(shift_cers) / len(merged), 2),
                "WER (%)": round(100.0 * sum(shift_wers) / len(merged), 2),
                "Clean CER (%)": round(100.0 * sum(clean_cers) / len(merged), 2),
                "target_id": target_id,
            }
        )

    if not rows:
        raise ValueError("No complete clean/adversarial row pairs were found in the manifest.")

    summary_df = pd.DataFrame(rows).sort_values(["Type", "Black-box API"]).reset_index(drop=True)
    output_csv = PROJECT_ROOT / args.output_csv
    output_json = PROJECT_ROOT / args.output_json
    output_md = PROJECT_ROOT / args.output_md

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    output_md.write_text(_render_table(summary_df), encoding="utf-8")
    write_json(
        output_json,
        {
            "summary_rows": rows,
            "sample_rows": sample_rows,
            "settings": {
                "cer_threshold": args.cer_threshold,
                "wer_threshold": args.wer_threshold,
                "relaxed_substring_match": args.relaxed_substring_match,
            },
        },
    )
    print(_render_table(summary_df))


if __name__ == "__main__":
    main()
