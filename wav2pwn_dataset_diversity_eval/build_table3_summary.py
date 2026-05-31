from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the cross-dataset Table III summary.")
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    from src.wav2pwn_diversity.config import load_yaml
    from src.wav2pwn_diversity.logging_utils import write_json

    config = load_yaml(args.config)
    results_root = PROJECT_ROOT / config["paths"]["root_results_dir"]
    rows = []
    for dataset_slug in ("librispeech", "commonvoice", "voxpopuli"):
        metric_path = results_root / dataset_slug / f"family_metrics_{dataset_slug}.json"
        if not metric_path.exists():
            raise FileNotFoundError(f"Missing family metrics for {dataset_slug}: {metric_path}")
        rows.append(json.loads(metric_path.read_text(encoding="utf-8")))

    df = pd.DataFrame(rows)
    df.to_csv(results_root / "table3_summary.csv", index=False)
    write_json(results_root / "table3_summary.json", rows)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main()
