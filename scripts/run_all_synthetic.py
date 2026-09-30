"""Run the synthetic SimplexUQ benchmark suite."""
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="d[1-6]_*.yaml",
        help="Glob pattern under configs/synthetic/ for benchmark configs",
    )
    parser.add_argument(
        "--python",
        default="python",
        help="Python executable used to launch run_synthetic.py",
    )
    args = parser.parse_args()

    config_dir = Path("configs/synthetic")
    configs = sorted(config_dir.glob(args.pattern))
    if not configs:
        raise SystemExit(f"No configs matched {config_dir / args.pattern}")

    for cfg in configs:
        print(f"\n{'=' * 72}\nRunning {cfg}\n{'=' * 72}")
        subprocess.run([args.python, "scripts/run_synthetic.py", "--config", str(cfg)], check=True)


if __name__ == "__main__":
    main()
