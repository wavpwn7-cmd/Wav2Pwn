from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final three-method comparison tables.")
    parser.add_argument("--output-dir", default="results/final_comparison")
    parser.add_argument("--direct-step5-json", default="results/direct_transfer_step5_10.json")
    parser.add_argument("--wav2pwn-step5-json", default="results/wav2pwn_step5_10.json")
    parser.add_argument("--surrogate-step5-dir", default="results/step5_transfer_targeted_10")
    parser.add_argument("--surrogate-attack-json", default="results/step4_dual_attack_10_adam/targeted.json")
    parser.add_argument("--query-stats-json", default="results/query_stats_3datasets_900.json")
    parser.add_argument("--train-log", default="logs/train_surrogate_3datasets_900_50epoch.log")
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_transfer_dir(path: Path) -> dict:
    rows = []
    for json_path in sorted(path.glob("query_*.json")):
        payload = _load_json(json_path)
        rows.append(payload["samples"][0])
    if not rows:
        raise ValueError(f"No transfer result files found under {path}")
    return {
        "clean_cer": sum(row["clean_cer"] for row in rows) / len(rows),
        "tasr": sum(row["tasr_success"] for row in rows) / len(rows),
        "uasr": sum(row["uasr_success"] for row in rows) / len(rows),
        "avg_shift_cer": sum(row["transcription_shift_cer"] for row in rows) / len(rows),
        "avg_shift_wer": sum(row["transcription_shift_wer"] for row in rows) / len(rows),
        "num_samples": len(rows),
    }


def _load_train_runtime_seconds(log_path: Path) -> float:
    marker = "'train_runtime': '"
    train_runtime = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if marker in line:
            start = line.index(marker) + len(marker)
            end = line.index("'", start)
            train_runtime = float(line[start:end])
    if train_runtime is None:
        raise ValueError(f"Could not parse train_runtime from {log_path}")
    return train_runtime


def build_rows(project_root: Path, args: argparse.Namespace) -> list[dict]:
    direct_transfer = _load_json(project_root / args.direct_step5_json)["summary"]
    wav2pwn = _load_json(project_root / args.wav2pwn_step5_json)["summary"]

    query_stats = _load_json(project_root / args.query_stats_json)
    surrogate_attack = _load_json(project_root / args.surrogate_attack_json)["summary"]
    surrogate_transfer = _aggregate_transfer_dir(project_root / args.surrogate_step5_dir)
    train_runtime_seconds = _load_train_runtime_seconds(project_root / args.train_log)

    surrogate_total_query_count = int(query_stats["total_query_count"]) + 2 * int(surrogate_transfer["num_samples"])
    surrogate_total_gpu_seconds = (
        float(query_stats["total_query_seconds"]) + train_runtime_seconds + float(surrogate_attack["total_attack_time"])
    )

    rows = [
        {
            "method": "Direct Transfer (No Probing)",
            "attack_cost_seconds": round(float(direct_transfer["Attack Cost"]["total_gpu_seconds"]), 4),
            "total_query_count": int(direct_transfer["Attack Cost"]["total_query_count"]),
            "surrogate_training_time_seconds": round(
                float(direct_transfer["Attack Cost"]["surrogate_training_time_seconds"]), 4
            ),
            "attack_optimization_time_seconds": round(
                float(direct_transfer["Attack Cost"]["attack_optimization_time_seconds"]), 4
            ),
            "clean_cer": round(float(direct_transfer["Clean CER"]), 6),
            "tasr": round(float(direct_transfer["TASR"]), 6),
            "uasr": round(float(direct_transfer["UASR"]), 6),
        },
        {
            "method": "Surrogate Model",
            "attack_cost_seconds": round(surrogate_total_gpu_seconds, 4),
            "total_query_count": surrogate_total_query_count,
            "surrogate_training_time_seconds": round(train_runtime_seconds, 4),
            "attack_optimization_time_seconds": round(float(surrogate_attack["total_attack_time"]), 4),
            "clean_cer": round(float(surrogate_transfer["clean_cer"]), 6),
            "tasr": round(float(surrogate_transfer["tasr"]), 6),
            "uasr": round(float(surrogate_transfer["uasr"]), 6),
        },
        {
            "method": "Wav2Pwn (with Probing)",
            "attack_cost_seconds": round(float(wav2pwn["Attack Cost"]["total_gpu_seconds"]), 4),
            "total_query_count": int(wav2pwn["Attack Cost"]["total_query_count"]),
            "surrogate_training_time_seconds": round(
                float(wav2pwn["Attack Cost"]["surrogate_training_time_seconds"]), 4
            ),
            "attack_optimization_time_seconds": round(
                float(wav2pwn["Attack Cost"]["attack_optimization_time_seconds"]), 4
            ),
            "clean_cer": round(float(wav2pwn["Clean CER"]), 6),
            "tasr": round(float(wav2pwn["TASR"]), 6),
            "uasr": round(float(wav2pwn["UASR"]), 6),
        },
    ]
    return rows


def _to_markdown_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in df.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *body])


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(PROJECT_ROOT, args)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "final_results.csv", index=False)

    with (output_dir / "final_results.md").open("w", encoding="utf-8") as handle:
        handle.write(_to_markdown_table(df))

    latex_df = df.rename(
        columns={
            "method": "Method",
            "attack_cost_seconds": "Attack Cost (s)",
            "total_query_count": "Total Queries",
            "surrogate_training_time_seconds": "Training Time (s)",
            "attack_optimization_time_seconds": "Attack Time (s)",
            "clean_cer": "Clean CER",
            "tasr": "TASR",
            "uasr": "UASR",
        }
    )
    with (output_dir / "final_results.tex").open("w", encoding="utf-8") as handle:
        handle.write(latex_df.to_latex(index=False, float_format=lambda value: f"{value:.4f}"))

    summary = {
        "final_results_csv": str((output_dir / "final_results.csv").resolve()),
        "final_results_md": str((output_dir / "final_results.md").resolve()),
        "final_results_tex": str((output_dir / "final_results.tex").resolve()),
        "rows": rows,
    }
    with (output_dir / "final_results.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(_to_markdown_table(df))


if __name__ == "__main__":
    main()
