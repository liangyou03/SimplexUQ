"""Run stratification-sensitivity sweeps across all real benchmark tasks."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


CORE_METHODS = ["global", "partition", "twostage", "fullcp", "jackknife_plus"]
ALL_METHODS = [
    "global",
    "partition",
    "twostage",
    "fullcp",
    "jackknife_plus",
    "oneshot",
    "trainres",
    "weighted",
]
DEFAULT_STRATA = ["boundary", "entropy", "dominant", "kmeans"]
TASKS = ["cifar10", "topics", "affectivetext", "samson", "utkface", "pbmc"]


def build_pbmc_config(
    base_config: Path,
    out_dir: Path,
    strata_method: str,
    methods: list[str],
) -> Path:
    with open(base_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["experiment"] = f"pbmc_sensitivity_{base_config.stem}_{strata_method}_fixed"
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["strata_method"] = strata_method
    cfg["evaluation"]["fixed_strata"] = True
    cfg["conformal"]["methods"] = methods

    out_path = out_dir / f"{cfg['experiment']}.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return out_path


def command_matrix(
    mode: str,
    strata: list[str],
    methods: list[str],
    config_dir: Path,
) -> list[list[str]]:
    py = ["uv", "run", "--extra", "bio", "python"]
    commands: list[list[str]] = []

    if mode == "smoke":
        n_rep = "5"
        softmax_extra = ["--max_samples", "2000"]
        utk_extra = ["--max_samples", "2000"]
        pbmc_base = Path("configs/real/exp2_1_smoke_test.yaml")
    else:
        n_rep = "50"
        softmax_extra = []
        utk_extra = []
        pbmc_base = Path("configs/real/exp2_1_bulk_deconv.yaml")

    method_args = ["--methods", *methods]

    for s in strata:
        tag = f"strata_{s}_fixed"

        commands.append([
            *py, "scripts/run_softmax.py",
            "--dataset", "cifar10",
            "--model", "resnet18",
            "--device", "cpu",
            "--n_rep", n_rep,
            "--n_strata", "5",
            "--strata", s,
            "--fixed-strata",
            "--tag", tag,
            *method_args,
            *softmax_extra,
        ])

        commands.append([
            *py, "scripts/run_topics.py",
            "--K", "10",
            "--n_rep", n_rep,
            "--n_strata", "5",
            "--strata", s,
            "--fixed-strata",
            "--tag", tag,
            *method_args,
        ])

        commands.append([
            *py, "scripts/run_affective_text.py",
            "--n_rep", n_rep,
            "--n_strata", "5",
            "--strata", s,
            "--fixed-strata",
            "--tag", tag,
            *method_args,
        ])

        commands.append([
            *py, "scripts/run_hyperspectral.py",
            "--dataset", "samson",
            "--unmix", "nmf",
            "--n_rep", n_rep,
            "--n_strata", "5",
            "--strata", s,
            "--fixed-strata",
            "--tag", tag,
            *method_args,
        ])

        commands.append([
            *py, "scripts/run_age_ldl.py",
            "--data-dir", "data/raw/UTKFace",
            "--pred-method", "image_knn",
            "--n_rep", n_rep,
            "--n_strata", "5",
            "--strata", s,
            "--fixed-strata",
            "--tag", tag,
            *method_args,
            *utk_extra,
        ])

        pbmc_cfg = build_pbmc_config(pbmc_base, config_dir, s, methods)
        commands.append([
            *py, "scripts/run_bulk_deconv.py",
            "--config", str(pbmc_cfg),
        ])

    return commands


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--method-set", choices=["core", "all"], default="core")
    parser.add_argument("--strata", nargs="+", default=DEFAULT_STRATA)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--manifest", default="results/tables/real_strata_sensitivity_manifest.json")
    args = parser.parse_args()

    methods = CORE_METHODS if args.method_set == "core" else ALL_METHODS
    config_dir = Path("temp/strata_sensitivity_configs")
    config_dir.mkdir(parents=True, exist_ok=True)

    commands = command_matrix(args.mode, args.strata, methods, config_dir)

    manifest = {
        "mode": args.mode,
        "method_set": args.method_set,
        "methods": methods,
        "strata": args.strata,
        "commands": [" ".join(cmd) for cmd in commands],
        "completed": [],
        "failed": [],
    }

    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return

    for i, cmd in enumerate(commands, start=1):
        print(f"[{i}/{len(commands)}] {' '.join(cmd)}", flush=True)
        try:
            subprocess.run(cmd, check=True)
            manifest["completed"].append(" ".join(cmd))
        except subprocess.CalledProcessError as exc:
            manifest["failed"].append({"command": " ".join(cmd), "returncode": exc.returncode})
            with open(args.manifest, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
            if not args.continue_on_error:
                raise
        with open(args.manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
