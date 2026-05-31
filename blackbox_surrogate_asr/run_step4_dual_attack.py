from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run both targeted and untargeted PGD attacks in one command."
    )
    parser.add_argument("--pseudo-labels", default="artifacts/pseudo_labels_3datasets_900.json")
    parser.add_argument("--surrogate-model", default="checkpoints/best_model")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--target-phrase", default="OPEN THE DOOR")
    parser.add_argument("--target-repeat", type=int, default=1)
    parser.add_argument("--attack-method", choices=["pgd", "adam"], default="pgd")
    parser.add_argument("--epsilon", type=float, default=0.002)
    parser.add_argument("--alpha", type=float, default=0.0002)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--l2-weight", type=float, default=0.0)
    parser.add_argument("--random-start", action="store_true")
    parser.add_argument("--cer-threshold", type=float, default=0.5)
    parser.add_argument("--wer-threshold", type=float, default=0.5)
    parser.add_argument("--output-root", default="artifacts/step4_dual_attack")
    parser.add_argument("--results-root", default="results/step4_dual_attack")
    parser.add_argument("--logs-root", default="logs/step4_dual_attack")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def build_command(args: argparse.Namespace, targeted: bool) -> list[str]:
    attack_name = "targeted" if targeted else "untargeted"
    command = [
        sys.executable,
        "pgd_attack.py",
        "--pseudo-labels",
        args.pseudo_labels,
        "--surrogate-model",
        args.surrogate_model,
        "--start-index",
        str(args.start_index),
        "--num-samples",
        str(args.num_samples),
        "--attack-method",
        args.attack_method,
        "--epsilon",
        str(args.epsilon),
        "--alpha",
        str(args.alpha),
        "--iterations",
        str(args.iterations),
        "--l2-weight",
        str(args.l2_weight),
        "--cer-threshold",
        str(args.cer_threshold),
        "--wer-threshold",
        str(args.wer_threshold),
        "--output-dir",
        str(Path(args.output_root) / attack_name),
        "--output-json",
        str(Path(args.results_root) / f"{attack_name}.json"),
        "--output-csv",
        str(Path(args.results_root) / f"{attack_name}.csv"),
        "--log-file",
        str(Path(args.logs_root) / f"{attack_name}.log"),
        "--device",
        args.device,
    ]
    if args.random_start:
        command.append("--random-start")
    if targeted:
        command.extend(
            [
                "--targeted",
                "--target-phrase",
                args.target_phrase,
                "--target-repeat",
                str(args.target_repeat),
            ]
        )
    return command


def main() -> None:
    args = parse_args()

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.results_root).mkdir(parents=True, exist_ok=True)
    Path(args.logs_root).mkdir(parents=True, exist_ok=True)

    commands = [
        ("targeted", build_command(args, targeted=True)),
        ("untargeted", build_command(args, targeted=False)),
    ]

    for attack_name, command in commands:
        print(f"\nRunning {attack_name} attack...")
        print(" ".join(command))
        completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"{attack_name} attack failed with exit code {completed.returncode}")

    print("\nStep 4 dual attack completed.")
    print(f"Targeted results:   {Path(args.results_root) / 'targeted.json'}")
    print(f"Untargeted results: {Path(args.results_root) / 'untargeted.json'}")


if __name__ == "__main__":
    main()
