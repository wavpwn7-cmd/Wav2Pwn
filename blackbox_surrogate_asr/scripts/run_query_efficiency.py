from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate query-budget experiments and export plots.")
    parser.add_argument("--input-csv", required=True, help="CSV with one row per query budget.")
    parser.add_argument("--output-dir", default="results/query_efficiency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from src.asr_blackbox.logging_utils import write_csv, write_json
    from src.asr_blackbox.plotting import plot_metric

    df = pd.read_csv(args.input_csv).sort_values("queries")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(df, "queries", "transfer_success", str(output_dir / "query_budget_vs_transfer_success.png"), "Query Budget vs Transfer Success")
    plot_metric(df, "queries", "surrogate_wer", str(output_dir / "query_budget_vs_surrogate_wer.png"), "Query Budget vs Surrogate WER")
    plot_metric(df, "queries", "tasr", str(output_dir / "query_budget_vs_tasr.png"), "Query Budget vs TASR")
    plot_metric(df, "queries", "uasr", str(output_dir / "query_budget_vs_uasr.png"), "Query Budget vs UASR")

    summary_rows = df.to_dict(orient="records")
    write_json(output_dir / "summary.json", summary_rows)
    write_csv(output_dir / "summary.csv", summary_rows, list(df.columns))


if __name__ == "__main__":
    main()
