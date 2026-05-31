from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Expand attack outputs into a commercial-API query manifest."
    )
    parser.add_argument("--attack-csv", required=True, help="Attack CSV with clean and adversarial audio paths.")
    parser.add_argument("--targets-config", default="configs/commercial_api_targets.yaml")
    parser.add_argument("--output-csv", default="results/commercial_api_query_manifest.csv")
    parser.add_argument("--output-json", default="results/commercial_api_query_manifest.json")
    return parser.parse_args()


def _require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Attack CSV is missing required columns: {missing}")


def _preview_table(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return df.to_string(index=False)


def main() -> None:
    args = parse_args()

    from src.asr_blackbox.config import load_config
    from src.asr_blackbox.logging_utils import write_json

    attack_df = pd.read_csv(PROJECT_ROOT / args.attack_csv, keep_default_na=False)
    _require_columns(
        attack_df,
        [
            "sample_index",
            "sample_id",
            "audio_path",
            "adv_audio_path",
            "target_phrase",
        ],
    )

    targets_payload = load_config(PROJECT_ROOT / args.targets_config)
    manifest_rows: list[dict[str, object]] = []

    for group in targets_payload["target_groups"]:
        group_name = group["name"]
        for target in group["targets"]:
            for attack_row in attack_df.itertuples(index=False):
                base_row = {
                    "target_group": group_name,
                    "target_id": target["id"],
                    "display_name": target["display_name"],
                    "provider": target["provider"],
                    "product": target["product"],
                    "api_version": target["api_version"],
                    "model_name": target["model_name"],
                    "reproducibility_status": target["reproducibility_status"],
                    "sample_index": int(attack_row.sample_index),
                    "sample_id": attack_row.sample_id,
                    "reference_transcript": getattr(attack_row, "target_clean_prediction", ""),
                    "target_phrase": attack_row.target_phrase,
                    "attack_csv": args.attack_csv,
                    "notes": target.get("notes", ""),
                }
                manifest_rows.append(
                    {
                        **base_row,
                        "audio_variant": "clean",
                        "audio_path": getattr(attack_row, "audio_path"),
                        "api_transcript": "",
                        "request_id": "",
                    }
                )
                manifest_rows.append(
                    {
                        **base_row,
                        "audio_variant": "adversarial",
                        "audio_path": getattr(attack_row, "adv_audio_path"),
                        "api_transcript": "",
                        "request_id": "",
                    }
                )

    manifest_df = pd.DataFrame(manifest_rows)
    output_csv = PROJECT_ROOT / args.output_csv
    output_json = PROJECT_ROOT / args.output_json
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_csv, index=False)
    write_json(output_json, manifest_rows)
    print(_preview_table(manifest_df.head(20)))
    print(f"\nWrote {len(manifest_df)} rows to {output_csv}")


if __name__ == "__main__":
    main()
