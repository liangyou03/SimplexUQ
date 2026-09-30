"""Tests for the paired SimplexTasks-12 multivalid runner."""
from pathlib import Path

import numpy as np

from scripts.run_multivalid_benchmark import (
    aggregate_task,
    compute_real_scores,
    load_benchmark_config,
    paired_split,
    render_summary_markdown,
)


def test_config_registers_all_twelve_tasks():
    cfg = load_benchmark_config(
        Path("configs/multivalid/simplextasks12_batchmvp.yaml")
    )

    assert len(cfg["tasks"]) == 12
    assert {task["kind"] for task in cfg["tasks"]} == {"synthetic", "real"}


def test_compute_real_scores_supports_tv_and_aitchison():
    Y = np.array([[1.0, 0.0], [0.25, 0.75]])
    U = np.array([[0.8, 0.2], [0.5, 0.5]])

    tv = compute_real_scores(Y, U, "tv")
    aitchison = compute_real_scores(Y + 1e-6, U + 1e-6, "aitchison")

    np.testing.assert_allclose(tv, [0.2, 0.25])
    assert np.all(np.isfinite(aitchison))
    assert np.all(aitchison >= 0.0)


def test_paired_split_is_reproducible_and_disjoint():
    first = paired_split(100, 0.4, np.random.default_rng(8))
    second = paired_split(100, 0.4, np.random.default_rng(8))

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert len(first[0]) == 40
    assert not set(first[0]).intersection(first[1])


def test_aggregation_and_markdown_include_all_methods():
    repetitions = []
    for value in (0.1, 0.2):
        def metrics(disparity):
            return {
                "marginal_coverage": 0.9,
                "max_disparity": disparity,
                "worst_stratum_coverage": 0.9 - disparity,
                "mean_radius": 1.0,
                "sscv": disparity,
                "coverage_variance": 0.0,
                "overlap_max_disparity": disparity,
                "overlap_worst_coverage": 0.9 - disparity,
                "runtime_sec": 0.01,
                "n_patches": 0,
                "n_groups": 5,
                "max_calibration_violation": 0.0,
            }
        repetitions.append(
            {
                "Global": metrics(value),
                "Mondrian": metrics(value / 2),
                "BatchMVP": metrics(value / 3),
            }
        )

    summary = aggregate_task(repetitions)
    markdown = render_summary_markdown({"toy": {"summary": summary}})

    assert np.isclose(summary["Global"]["max_disparity"]["mean"], 0.15)
    assert "BatchMVP" in markdown
    assert "toy" in markdown
