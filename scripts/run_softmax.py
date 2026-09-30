"""Exp 2.2 — Classification softmax calibration on CIFAR-10/100.

Softmax output ∈ Δ^{K-1}, one-hot label ∈ Δ^{K-1}.
Tests whether global conformal creates disparity across easy vs hard classes.

Usage:
    python scripts/run_softmax.py --dataset cifar10
    python scripts/run_softmax.py --dataset cifar100 --n_strata 10
"""
import argparse
import json
import logging
import numpy as np
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.simplex import aitchison_dist
from src.utils.strata import (
    precompute_fixed_strata,
    stratify_by_boundary,
    stratify_by_entropy,
)
from src.utils.seed import get_rng
from src.methods import (
    full_conformal,
    global_split_conformal,
    jackknife_plus_conformal,
    oneshot_conformal,
    partition_conformal,
    trainres_conformal,
    twostage_conformal,
    weighted_conformal,
)
from src.methods._knn_sigma import knn_sigma_hat, knn_sigma_leave_one_out
from src.metrics.coverage import (
    coverage_variance,
    marginal_coverage,
    max_disparity,
    stratified_coverage,
    worst_stratum_coverage,
)
from src.metrics.sscv import size_stratified_coverage_violation
from src.metrics.setsize import mean_radius, mean_volume_ratio, volume_ratio_by_strata

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_METHODS = [
    "global",
    "partition",
    "twostage",
    "jackknife_plus",
    "weighted",
    "oneshot",
    "trainres",
]


def get_softmax_predictions(dataset: str, model_name: str = "resnet50",
                             device: str = "cuda"):
    """Train or load a classifier, return softmax predictions on test set.

    Returns:
        Y: one-hot labels (n, K)
        U: softmax predictions (n, K)
        class_names: list of class names
    """
    # Check for cached predictions
    cache_path = Path(f"data/processed/{dataset}_{model_name}_softmax.npz")
    if cache_path.exists():
        log.info(f"Loading cached predictions from {cache_path}")
        data = np.load(cache_path)
        return data["Y"], data["U"], list(data["class_names"])

    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as T

    # Load dataset
    if dataset == "cifar10":
        transform = T.Compose([T.Resize(224), T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
        testset = torchvision.datasets.CIFAR10(
            root="data/raw", train=False, download=True, transform=transform)
        trainset = torchvision.datasets.CIFAR10(
            root="data/raw", train=True, download=True, transform=transform)
        K = 10
        class_names = testset.classes
    elif dataset == "cifar100":
        transform = T.Compose([T.Resize(224), T.ToTensor(),
                               T.Normalize([0.485, 0.456, 0.406],
                                           [0.229, 0.224, 0.225])])
        testset = torchvision.datasets.CIFAR100(
            root="data/raw", train=False, download=True, transform=transform)
        trainset = torchvision.datasets.CIFAR100(
            root="data/raw", train=True, download=True, transform=transform)
        K = 100
        class_names = testset.classes
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    log.info(f"Training/loading {model_name} on {dataset}...")

    # Use pretrained model + finetune last layer
    if model_name == "resnet50":
        model = torchvision.models.resnet50(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, K)
    elif model_name == "resnet18":
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        model.fc = nn.Linear(model.fc.in_features, K)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)

    # Quick finetune (5 epochs, enough for reasonable softmax)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                               shuffle=True, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        total_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        log.info(f"  Epoch {epoch+1}/5, loss={total_loss/len(trainloader):.4f}")

    # Get test predictions
    model.eval()
    testloader = torch.utils.data.DataLoader(testset, batch_size=256,
                                              shuffle=False, num_workers=4)
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())

    U = np.concatenate(all_probs)  # (n, K) softmax predictions
    labels = np.concatenate(all_labels)  # (n,) integer labels

    # One-hot encode labels (these are vertices of the simplex)
    Y = np.zeros((len(labels), K))
    Y[np.arange(len(labels)), labels] = 1.0
    # Add tiny smoothing to avoid log(0) in Aitchison distance
    Y = (Y + 1e-8)
    Y = Y / Y.sum(axis=1, keepdims=True)

    acc = (np.argmax(U, axis=1) == labels).mean()
    log.info(f"Test accuracy: {acc:.4f}")

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, Y=Y, U=U, class_names=np.array(class_names))
    log.info(f"Cached predictions to {cache_path}")

    return Y, U, class_names


def compute_weight_vectors(R_cal, U_cal, U_test, k=20):
    sigma_cal = knn_sigma_leave_one_out(U_cal, R_cal, k=k)
    sigma_test = knn_sigma_hat(U_cal, R_cal, U_test, k=k)
    weights_cal = 1.0 / np.maximum(sigma_cal, 1e-8)
    weights_test = 1.0 / np.maximum(sigma_test, 1e-8)
    weights_cal /= np.mean(weights_cal)
    weights_test /= np.mean(weights_test)
    return weights_cal, weights_test


def evaluate_result(
    res,
    U_test,
    strata_test,
    alpha,
    runtime_sec,
    compute_volume=False,
    volume_score="tv",
    volume_n_mc=50000,
    volume_max_points=None,
    rep=0,
):
    result = dict(
        marginal_coverage=float(marginal_coverage(res.covered)),
        max_disparity=float(max_disparity(res.covered, strata_test, alpha)),
        worst_stratum_coverage=float(worst_stratum_coverage(res.covered, strata_test)),
        mean_radius=float(mean_radius(res.radius)),
        sscv=float(size_stratified_coverage_violation(res.covered, res.radius, alpha)),
        coverage_variance=float(coverage_variance(res.covered, strata_test)),
        runtime_sec=float(runtime_sec),
        stratified_coverage={
            str(k): float(v) for k, v in stratified_coverage(res.covered, strata_test).items()
        },
    )
    if compute_volume:
        result["mean_volume_ratio"] = float(
            mean_volume_ratio(
                U_test,
                res.radius,
                score=volume_score,
                n_mc=volume_n_mc,
                max_points=volume_max_points,
                rng=np.random.default_rng(rep),
            )
        )
        result["volume_ratio_by_strata"] = {
            str(k): float(v)
            for k, v in volume_ratio_by_strata(
                U_test,
                res.radius,
                strata_test,
                score=volume_score,
                n_mc=volume_n_mc,
                max_points=volume_max_points,
                rng=np.random.default_rng(rep),
            ).items()
        }
    return result


def run_experiment(
    Y,
    U,
    alpha,
    n_rep,
    cal_frac,
    n_strata,
    rng,
    methods,
    compute_volume=False,
    volume_score="tv",
    volume_n_mc=50000,
    volume_max_points=None,
    strata_method="entropy",
    fixed_strata=True,
    strata_seed=2026,
):
    """Run conformal with repeated splits."""
    # Use L1 distance instead of Aitchison for one-hot labels
    # (Aitchison is ill-defined at simplex vertices)
    R = np.sum(np.abs(Y - U), axis=1) / 2.0  # total variation distance

    n = len(R)
    n_cal = int(n * cal_frac)

    all_results = {m: [] for m in methods}
    fixed_labels = None
    if fixed_strata:
        fixed_labels = precompute_fixed_strata(U, strata_method, n_strata, seed=strata_seed)
    elif strata_method not in {"boundary", "entropy"}:
        raise ValueError("Non-fixed softmax strata must be 'boundary' or 'entropy'.")

    for rep in range(n_rep):
        perm = rng.permutation(n)
        idx_cal, idx_test = perm[:n_cal], perm[n_cal:]

        R_cal, R_test = R[idx_cal], R[idx_test]
        U_cal, U_test = U[idx_cal], U[idx_test]

        if fixed_labels is not None:
            strata_cal = fixed_labels[idx_cal]
            strata_test = fixed_labels[idx_test]
        else:
            strata_fn = stratify_by_boundary if strata_method == "boundary" else stratify_by_entropy
            strata_cal = strata_fn(U_cal, n_strata)
            strata_test = strata_fn(U_test, n_strata)
        weights_cal, weights_test = compute_weight_vectors(R_cal, U_cal, U_test)

        for m in methods:
            start = time.perf_counter()
            if m == "global":
                res = global_split_conformal(R_cal, R_test, alpha)
            elif m == "partition":
                res = partition_conformal(R_cal, R_test, alpha,
                                          strata_cal, strata_test)
            elif m == "twostage":
                res = twostage_conformal(R_cal, R_test, alpha,
                                         U_cal, U_test)
            elif m == "jackknife_plus":
                res = jackknife_plus_conformal(R_cal, R_test, alpha, U_cal=U_cal, U_test=U_test)
            elif m == "weighted":
                res = weighted_conformal(R_cal, R_test, alpha, weights_cal, weights_test)
            elif m == "oneshot":
                res = oneshot_conformal(R_cal, R_test, alpha, U_cal, U_test)
            elif m == "trainres":
                train_perm = rng.permutation(n)
                idx_train = train_perm[:n_cal]
                res = trainres_conformal(
                    R_cal, R_test, alpha, U_cal, U_test, R[idx_train], U[idx_train]
                )
            elif m == "fullcp":
                res = full_conformal(R_cal, R_test, alpha, U_cal, U_test)
            else:
                continue

            runtime_sec = time.perf_counter() - start
            all_results[m].append(
                evaluate_result(
                    res,
                    U_test,
                    strata_test,
                    alpha,
                    runtime_sec,
                    compute_volume=compute_volume,
                    volume_score=volume_score,
                    volume_n_mc=volume_n_mc,
                    volume_max_points=volume_max_points,
                    rep=rep,
                )
            )

        if (rep + 1) % 50 == 0:
            log.info(f"  Rep {rep + 1}/{n_rep}")

    return all_results


def maybe_subsample(Y, U, max_samples, rng):
    if max_samples is None or max_samples >= len(Y):
        return Y, U
    idx = rng.choice(len(Y), size=max_samples, replace=False)
    return Y[idx], U[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n_rep", type=int, default=200)
    parser.add_argument("--cal_frac", type=float, default=0.4)
    parser.add_argument("--n_strata", type=int, default=5)
    parser.add_argument(
        "--strata",
        choices=["entropy", "boundary", "dominant", "kmeans", "random"],
        default="entropy",
    )
    parser.add_argument("--fixed-strata", dest="fixed_strata", action="store_true")
    parser.add_argument(
        "--separate-strata",
        dest="fixed_strata",
        action="store_false",
        help="Diagnostic only: fit calibration/test strata separately.",
    )
    parser.set_defaults(fixed_strata=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--compute-volume", action="store_true")
    parser.add_argument("--volume-score", choices=["tv", "aitchison"], default="tv")
    parser.add_argument("--volume-n-mc", type=int, default=50000)
    parser.add_argument("--volume-max-points", type=int, default=None)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        choices=DEFAULT_METHODS + ["fullcp"],
    )
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    rng = get_rng(args.seed)

    # Get predictions
    Y, U, class_names = get_softmax_predictions(args.dataset, args.model, args.device)
    Y, U = maybe_subsample(Y, U, args.max_samples, rng)
    K = Y.shape[1]
    log.info(f"Dataset: {args.dataset}, K={K}, n={len(Y)}")

    # Residual diagnostics
    R = np.sum(np.abs(Y - U), axis=1) / 2.0
    log.info(f"Residuals: mean={R.mean():.4f}, std={R.std():.4f}")

    # Per-class residuals
    true_labels = np.argmax(Y, axis=1)
    for k in range(min(K, 10)):
        mask = true_labels == k
        log.info(f"  {class_names[k]:12s}: n={mask.sum()}, "
                 f"R_mean={R[mask].mean():.4f}, R_std={R[mask].std():.4f}")

    # Run
    all_results = run_experiment(
        Y,
        U,
        args.alpha,
        args.n_rep,
        args.cal_frac,
        args.n_strata,
        rng,
        args.methods,
        compute_volume=args.compute_volume,
        volume_score=args.volume_score,
        volume_n_mc=args.volume_n_mc,
        volume_max_points=args.volume_max_points,
        strata_method=args.strata,
        fixed_strata=args.fixed_strata,
        strata_seed=args.seed,
    )

    # Aggregate
    log.info("\n" + "=" * 60)
    log.info(f"RESULTS — Softmax calibration ({args.dataset})")
    log.info("=" * 60)

    summary = {}
    scalar_keys = [
        "marginal_coverage",
        "max_disparity",
        "worst_stratum_coverage",
        "mean_radius",
        "sscv",
        "coverage_variance",
        "runtime_sec",
        "mean_volume_ratio",
    ]
    for m in args.methods:
        if not all_results[m]:
            continue
        reps = all_results[m]
        s = {}
        for key in scalar_keys:
            if key in reps[0]:
                vals = [r[key] for r in reps]
                s[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        strata_keys = set()
        for r in reps:
            strata_keys.update(r["stratified_coverage"].keys())
        s["stratified_coverage"] = {
            k: {
                "mean": float(np.mean([r["stratified_coverage"][k] for r in reps if k in r["stratified_coverage"]])),
                "std": float(np.std([r["stratified_coverage"][k] for r in reps if k in r["stratified_coverage"]])),
                "n_reps": int(sum(k in r["stratified_coverage"] for r in reps)),
            }
            for k in sorted(strata_keys, key=int)
        }
        if "volume_ratio_by_strata" in reps[0]:
            vol_keys = set()
            for r in reps:
                vol_keys.update(r["volume_ratio_by_strata"].keys())
            s["volume_ratio_by_strata"] = {
                k: {
                    "mean": float(np.mean([r["volume_ratio_by_strata"][k] for r in reps if k in r["volume_ratio_by_strata"]])),
                    "std": float(np.std([r["volume_ratio_by_strata"][k] for r in reps if k in r["volume_ratio_by_strata"]])),
                    "n_reps": int(sum(k in r["volume_ratio_by_strata"] for r in reps)),
                }
                for k in sorted(vol_keys, key=int)
            }
        summary[m] = s
        log.info(
            f"  {m:12s}  cov={s['marginal_coverage']['mean']:.3f}±{s['marginal_coverage']['std']:.3f}  "
            f"disp={s['max_disparity']['mean']:.3f}±{s['max_disparity']['std']:.3f}  "
            f"worst={s['worst_stratum_coverage']['mean']:.3f}  "
            f"sscv={s['sscv']['mean']:.3f}"
        )

    # Save
    out_dir = Path(args.output_dir) / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_file = out_dir / f"exp2_2_softmax_{args.dataset}{suffix}.json"
    with open(out_file, "w") as f:
        json.dump(dict(summary=summary, dataset=args.dataset, K=K,
                       class_names=class_names, config=vars(args), raw=all_results),
                  f, indent=2)
    log.info(f"Saved to {out_file}")


if __name__ == "__main__":
    main()
