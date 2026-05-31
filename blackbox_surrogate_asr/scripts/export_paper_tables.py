from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-ready result tables.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", default="results/paper_tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    table = df.rename(
        columns={
            "queries": "Queries",
            "clean_cer": "Clean CER",
            "tasr": "TASR",
            "uasr": "UASR",
            "avg_attack_time": "Avg Attack Time",
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / "paper_ready_table.csv", index=False)
    with (output_dir / "paper_ready_table.md").open("w", encoding="utf-8") as handle:
        handle.write(table.to_markdown(index=False))


if __name__ == "__main__":
    main()

