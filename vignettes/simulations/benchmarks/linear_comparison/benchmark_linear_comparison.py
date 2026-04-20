"""
Benchmarking script: linear regression baselines for interventional effect prediction.

Two configs are evaluated, each running one model per trial:

  1. lasso — candidate predictors are all non-end-node graph variables plus
     additional independent noise variables.  Lasso selects a sparse subset;
     the interventional prediction is made by setting all selected features to
     1 (vs 0 baseline), so the predicted effect for each end node equals the
     sum of its non-zero Lasso coefficients.

  2. known_start — uses only the true start (ligand) nodes as predictors for
     each end node via ordinary least squares.  The interventional prediction
     sets all start nodes to 1 (vs 0 baseline), giving a predicted effect equal
     to the sum of the fitted start-node coefficients.

Both are validated against the analytical ground-truth SEM effect of
do(all_start_nodes=1) minus do(all_start_nodes=0) on the end nodes.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

from causomic.simulation.proteomics_simulator import simulate_data
from causomic.simulation.random_network import (
    generate_structured_dag,
    ground_truth_interventional_effect,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Defines one benchmarking scenario.

    Each config is swept over `seeds` — one trial per seed.
    The `model` field selects which linear model is evaluated: "lasso" or
    "known_start".
    """
    name: str
    model: str  # "lasso" or "known_start"

    # DAG generation params
    dag_params: dict = field(default_factory=lambda: dict(
        n_start=20,
        n_end=8,
        max_mediators=3,
        shared_mediator_prob=0.3,
        confounder_prob=0.00,
    ))

    # Simulation params
    n_samples: int = 200

    # Lasso params (ignored when model="known_start")
    n_noise_vars: int = 50   # additional independent variables added for Lasso
    lasso_alpha: float = 0.1  # regularization strength

    # Seeds to run
    seeds: list[int] = field(default_factory=lambda: list(range(10)))


# ---------------------------------------------------------------------------
# Interventional metric computation
# ---------------------------------------------------------------------------

def compute_interventional_metrics(
    config_name: str,
    seed: int,
    model_name: str,
    ci_series: pd.Series,
    gt_series: pd.Series,
) -> tuple[dict[str, float], list[dict]]:
    """Compute summary metrics and per-node detail rows.

    Parameters
    ----------
    ci_series : pd.Series
        Model-predicted effect (do=1 minus do=0), indexed by output node.
    gt_series : pd.Series
        Ground-truth SEM effect, aligned to the same index.

    Returns
    -------
    summary : dict
        int_rmse, int_mae, int_pearson_r, int_direction_accuracy, int_n_outcomes.
    per_node_rows : list[dict]
        One dict per output node with config, seed, model, node, ci_result,
        gt_result, difference, correct_direction.
    """
    ci = ci_series.values.astype(float)
    gt = gt_series.values.astype(float)
    diff = ci - gt

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    if len(ci) >= 2 and np.std(ci) > 0 and np.std(gt) > 0:
        corr = float(np.corrcoef(ci, gt)[0, 1])
    else:
        corr = float("nan")

    # Direction accuracy: fraction of end nodes where predicted sign matches GT.
    # Nodes where gt == 0 are excluded (no direction to be right or wrong about).
    nonzero_mask = gt != 0
    if nonzero_mask.sum() > 0:
        direction_acc = float(
            (np.sign(ci[nonzero_mask]) == np.sign(gt[nonzero_mask])).mean()
        )
    else:
        direction_acc = float("nan")

    summary = {
        "int_rmse": rmse,
        "int_mae": mae,
        "int_pearson_r": corr,
        "int_direction_accuracy": direction_acc,
        "int_n_outcomes": len(ci),
    }

    per_node_rows = [
        {
            "config": config_name,
            "seed": seed,
            "model": model_name,
            "node": node,
            "ci_result": float(ci_val),
            "gt_result": float(gt_val),
            "difference": float(ci_val - gt_val),
            "correct_direction": (
                float("nan") if gt_val == 0
                else float(np.sign(ci_val) == np.sign(gt_val))
            ),
        }
        for node, ci_val, gt_val in zip(ci_series.index, ci, gt)
    ]

    return summary, per_node_rows


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(cfg: BenchmarkConfig, seed: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one simulation trial and evaluate the model specified by cfg.model."""

    if cfg.model not in ("lasso", "known_start"):
        raise ValueError(f"Unknown model {cfg.model!r}. Must be 'lasso' or 'known_start'.")

    # 1. Ground-truth DAG
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)
    start_nodes = roles["start"]
    end_nodes = roles["end"]

    # 2. Simulate data from the SEM
    sim = simulate_data(
        gt_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    # 3. Ground-truth interventional effect: do(start=1) vs do(start=0) on end nodes.
    #    ground_truth_interventional_effect returns "effect" as E[node|do(...)] - E[node|baseline],
    #    so the combined effect of going from 0 to 1 is the difference of the two.
    gt_do1 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes={n: 1 for n in start_nodes},
        output_nodes=end_nodes,
    )
    gt_do0 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes={n: 0 for n in start_nodes},
        output_nodes=end_nodes,
    )
    gt_effect = {k: gt_do1["effect"][k] - gt_do0["effect"][k] for k in end_nodes}
    gt_series = pd.Series(gt_effect, dtype=float)

    row: dict[str, Any] = {
        "config": cfg.name,
        "model": cfg.model,
        "seed": seed,
        "n_nodes": gt_dag.number_of_nodes(),
        "n_edges": gt_dag.number_of_edges(),
        "n_start": len(start_nodes),
        "n_end": len(end_nodes),
    }
    per_node_rows: list[dict] = []

    if cfg.model == "lasso":
        # ── Lasso with noise variables ────────────────────────────────────────
        # Noise variables are truly independent of the causal graph — adding them
        # makes feature selection non-trivial and creates a more realistic setting
        # where the model cannot assume all inputs are causally relevant.
        rng = np.random.default_rng(seed + 10_000)
        noise_df = pd.DataFrame(
            rng.normal(0, 1, size=(cfg.n_samples, cfg.n_noise_vars)),
            columns=[f"NOISE{i}" for i in range(cfg.n_noise_vars)],
        )

        # Candidate features: all non-end graph nodes + noise variables
        non_end_cols = [c for c in protein_df.columns if c not in end_nodes]
        lasso_feat_df = pd.concat(
            [
                protein_df[non_end_cols].reset_index(drop=True),
                noise_df.reset_index(drop=True),
            ],
            axis=1,
        )
        X_lasso = lasso_feat_df.values

        effects: dict[str, float] = {}
        n_selected: dict[str, int] = {}
        for end_node in end_nodes:
            y = protein_df[end_node].values
            lasso_model = Lasso(alpha=cfg.lasso_alpha, max_iter=10_000, fit_intercept=True)
            lasso_model.fit(X_lasso, y)
            # Interventional prediction: set all selected features to 1 vs 0.
            # For a linear model y = intercept + sum(coef_i * x_i), the effect of
            # going from x=0 to x=1 for all selected predictors equals sum(coef_i).
            effects[end_node] = float(np.sum(lasso_model.coef_))
            n_selected[end_node] = int(np.sum(lasso_model.coef_ != 0))

        row["avg_n_selected"] = float(np.mean(list(n_selected.values())))

    else:
        # ── Known-start OLS ───────────────────────────────────────────────────
        # Uses only the true start nodes as predictors — an oracle baseline that
        # knows which variables to intervene on.
        X_start = protein_df[start_nodes].values
        X_design = np.column_stack([np.ones(len(X_start)), X_start])

        effects = {}
        for end_node in end_nodes:
            y = protein_df[end_node].values
            coeffs, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
            # Effect = sum of all start-node coefficients (do(start=1) vs do(start=0))
            effects[end_node] = float(np.sum(coeffs[1:]))

    ci_series = pd.Series(effects, dtype=float).reindex(end_nodes)
    metrics, nodes = compute_interventional_metrics(
        cfg.name, seed, cfg.model, ci_series, gt_series
    )
    row.update(metrics)
    per_node_rows.extend(nodes)

    return row, per_node_rows


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    configs: list[BenchmarkConfig], verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all configs across all seeds.

    Returns
    -------
    results : pd.DataFrame
        One row per trial with summary metrics. A `model` column identifies
        which linear model was evaluated.
    node_results : pd.DataFrame
        Long-format per-node results with a `model` column.
    """
    rows: list[dict] = []
    all_per_node: list[dict] = []
    total = sum(len(c.seeds) for c in configs)
    done = 0

    for cfg in configs:
        for seed in cfg.seeds:
            done += 1
            if verbose:
                print(f"[{done}/{total}] config={cfg.name!r}  seed={seed} ...", flush=True)
            try:
                row, per_node_rows = run_trial(cfg, seed)
                rows.append(row)
                all_per_node.extend(per_node_rows)
                if verbose:
                    msg = (
                        f"         rmse={row['int_rmse']:.4f}"
                        f"  dir_acc={row['int_direction_accuracy']:.3f}"
                    )
                    if "avg_n_selected" in row:
                        msg += f"  n_sel={row['avg_n_selected']:.1f}"
                    print(msg)
            except Exception as exc:
                print(f"  [ERROR] config={cfg.name!r} seed={seed}: {exc}")

    return pd.DataFrame(rows), pd.DataFrame(all_per_node)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-config mean ± std."""
    int_metrics = ["int_rmse", "int_mae", "int_pearson_r", "int_direction_accuracy"]
    metrics = [m for m in int_metrics if m in results.columns]
    agg = (
        results.groupby("config")[metrics]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# Benchmark configuration
# ---------------------------------------------------------------------------

_DAG_PARAMS = dict(
    n_start=20,
    n_end=8,
    max_mediators=3,
    shared_mediator_prob=0.3,
    confounder_prob=0.5,
)

_SEEDS = list(range(100))

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig(
        name="lasso",
        model="lasso",
        dag_params=_DAG_PARAMS,
        n_samples=200,
        n_noise_vars=50,
        lasso_alpha=0.1,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="known_start",
        model="known_start",
        dag_params=_DAG_PARAMS,
        n_samples=200,
        seeds=_SEEDS,
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results, node_results = run_benchmark(BENCHMARK_CONFIGS, verbose=True)

    print("\n" + "=" * 70)
    print("PER-TRIAL RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    print(summarize(results).to_string())

    results_path = OUTPUT_DIR / "benchmark_results.csv"
    results.to_csv(results_path, index=False)
    print(f"\nTrial results saved to {results_path}")

    if not node_results.empty:
        nodes_path = OUTPUT_DIR / "benchmark_per_node.csv"
        node_results.to_csv(nodes_path, index=False)
        print(f"Per-node results saved to {nodes_path}")
