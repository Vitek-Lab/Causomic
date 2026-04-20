"""
Benchmarking script: effect of missing real edges and mediated shortcuts on
posterior network accuracy.

Sweeps:
  1. p_missing_real in {0, 0.2, 0.5}  — fraction of ground-truth edges absent
     from the INDRA prior (no mediated shortcuts).
  2. p_missing_real in {0.2, 0.5} combined with p_mediated_shortcut=0.3  —
     tests whether spurious shortcut edges partially compensate for the
     missing direct evidence.

Fixed settings for all configs:
  - Scoring function : BICGaussIndraPriors
  - Noise level      : moderate (fake_node_multiplier=2.0, fake_edge_multiplier=5.0)
  - Replicates       : 200 samples per simulation
  - Seeds            : 30 random DAGs per configuration

To add new parameter sweeps, add entries to BENCHMARK_CONFIGS at the bottom.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch

from causomic.causal_model.LVM import LVM
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    SparseHillClimb,
)
from causomic.network import estimate_posterior_dag
from causomic.simulation.proteomics_simulator import simulate_data
from causomic.simulation.random_network import (
    generate_indra_data,
    generate_structured_dag,
    ground_truth_interventional_effect,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Defines one benchmarking scenario.

    Each config is swept over `seeds` — one trial per seed.
    """
    name: str

    # DAG generation params
    dag_params: dict = field(default_factory=lambda: dict(
        n_start=20,
        n_end=10,
        max_mediators=3,
        shared_mediator_prob=0.4,
        confounder_prob=0.0,
        end_node_alpha=0.8,
    ))

    # INDRA noise multipliers (relative to real graph size)
    fake_node_multiplier: float = 1.0   # n_fake_nodes = n_real_nodes * multiplier
    fake_edge_multiplier: float = 2.0   # n_fake_edges = n_real_edges * multiplier

    # INDRA missingness / shortcut params
    p_missing_real: float = 0.0         # fraction of real edges dropped from priors
    p_mediated_shortcut: float = 0.0    # prob of spurious direct edge for each mediated pair

    # Simulation params
    n_samples: int = 200

    # Posterior estimation params
    scoring_function: type = BICGaussIndraPriors
    prior_strength: float = 5.0
    n_bootstrap: int = 100
    edge_probability: float = 0.5
    add_high_corr_edges_to_priors: bool = False
    corr_threshold: float = 0.5

    # Seeds to run
    seeds: list[int] = field(default_factory=lambda: list(range(30)))

    # Interventional benchmark
    run_interventional: bool = True
    lvm_num_steps: int = 1000
    lvm_predictive_samples: int = 200


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    posterior,
    gt_dag: nx.DiGraph,
    all_nodes: set[str],
) -> dict[str, float]:
    """Compute Precision, Recall, F1, Accuracy against gt_dag.

    The edge universe is all ordered pairs of nodes in all_nodes (excluding
    self-loops), giving a well-defined TN count for Accuracy.
    """
    pred_edges = set((str(u), str(v)) for u, v in posterior.directed.edges())
    gt_edges = set((str(u), str(v)) for u, v in gt_dag.edges())

    universe = {
        (u, v)
        for u, v in itertools.permutations(all_nodes, 2)
    }

    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    tn = len(universe - pred_edges - gt_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    accuracy = (tp + tn) / len(universe) if universe else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_pred_edges": len(pred_edges),
        "n_gt_edges": len(gt_edges),
    }


# ---------------------------------------------------------------------------
# Interventional metric computation
# ---------------------------------------------------------------------------

def compute_interventional_metrics(
    config_name: str,
    seed: int,
    ci_series: pd.Series,
    gt_series: pd.Series,
) -> tuple[dict[str, float], list[dict]]:
    """Compute summary metrics and per-node detail rows for interventional predictions.

    Parameters
    ----------
    config_name, seed : str, int
        Identifiers written into every per-node row.
    ci_series : pd.Series
        Model-predicted effect (intervention1 - intervention0), indexed by output node.
    gt_series : pd.Series
        Ground-truth effect from the SEM, aligned to the same index.

    Returns
    -------
    summary : dict
        int_rmse, int_mae, int_pearson_r, int_direction_accuracy, int_n_outcomes.
    per_node_rows : list[dict]
        One dict per output node with config, seed, node, ci_result, gt_result,
        difference, correct_direction.
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


def _run_interventional(
    cfg: BenchmarkConfig,
    seed: int,
    sim: dict,
    roles: dict,
    gt_dag: nx.DiGraph,
    posterior,
) -> tuple[dict[str, float], list[dict]]:
    """Fit LVM, run paired interventions, and return metrics vs ground truth.

    Interventions are always do(start_nodes=1) vs do(start_nodes=0).
    The reported effect is E[outcome | do=1] - E[outcome | do=0].
    """
    model_input = pd.DataFrame(sim["Protein_data"])
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=cfg.lvm_num_steps, verbose=False)
    lvm.fit(model_input, posterior)

    torch.manual_seed(seed)
    lvm.intervention(intervention0, outcome, predictive_samples=cfg.lvm_predictive_samples)
    int0 = lvm.intervention_samples

    torch.manual_seed(seed)
    lvm.intervention(intervention1, outcome, predictive_samples=cfg.lvm_predictive_samples)
    int1 = lvm.intervention_samples

    ci_results = (int0 - int1).mean(axis=0)

    gt_int0 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes=intervention0,
        output_nodes=outcome,
    )
    gt_int1 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes=intervention1,
        output_nodes=outcome,
    )
    gt_results = {k: gt_int0["effect"][k] - gt_int1["effect"][k] for k in gt_int1["effect"]}

    ci_series = pd.Series(ci_results, index=outcome, dtype=float)
    gt_series = pd.Series(gt_results, dtype=float).reindex(outcome)

    return compute_interventional_metrics(cfg.name, seed, ci_series, gt_series)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(cfg: BenchmarkConfig, seed: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one simulation + estimation trial and return (metrics_row, per_node_rows)."""

    # 1. Ground-truth DAG (seed controls the DAG structure)
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * cfg.fake_node_multiplier))
    n_fake_edges = max(1, round(n_real_edges * cfg.fake_edge_multiplier))

    # 2. INDRA-style priors with missingness and optional mediated shortcuts
    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=cfg.p_missing_real,
        p_mediated_shortcut=cfg.p_mediated_shortcut,
        preferential_attachment=True,
    )

    # 3. Augment simulation graph: add spurious nodes (isolated, no edges)
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]
    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)

    # 4. Simulate proteomics data
    sim = simulate_data(
        augmented_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    # 5. Estimate posterior DAG
    posterior, _bootstraps = estimate_posterior_dag(  # noqa: F841
        protein_df,
        indra_priors=indra_df,
        prior_strength=cfg.prior_strength,
        scoring_function=cfg.scoring_function,
        search_algorithm=SparseHillClimb,
        n_bootstrap=cfg.n_bootstrap,
        add_high_corr_edges_to_priors=cfg.add_high_corr_edges_to_priors,
        corr_threshold=cfg.corr_threshold,
        edge_probability=cfg.edge_probability,
        convert_to_probability=True,
        return_bootstrap_dags=True,
    )

    # 6. Graph metrics — universe = all nodes seen during estimation
    all_nodes = set(str(n) for n in protein_df.columns)
    graph_metrics = compute_metrics(posterior, gt_dag, all_nodes)

    row = {
        "config": cfg.name,
        "seed": seed,
        "p_missing_real": cfg.p_missing_real,
        "p_mediated_shortcut": cfg.p_mediated_shortcut,
        "n_missing_edges": len(_missing),
        "n_real_nodes": n_real_nodes,
        "n_real_edges": n_real_edges,
        "n_fake_nodes": n_fake_nodes,
        "n_fake_edges": n_fake_edges,
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 7. Interventional metrics (optional)
    if cfg.run_interventional:
        try:
            int_metrics, per_node_rows = _run_interventional(
                cfg, seed, sim, roles, gt_dag, posterior
            )
            row.update(int_metrics)
        except Exception as exc:
            print(f"  [WARN] interventional benchmark failed seed={seed}: {exc}")
            row.update({
                "int_rmse": float("nan"),
                "int_mae": float("nan"),
                "int_pearson_r": float("nan"),
                "int_direction_accuracy": float("nan"),
                "int_n_outcomes": float("nan"),
            })

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
        One row per trial with graph + interventional summary metrics.
    node_results : pd.DataFrame
        Long-format per-node interventional results (empty if no config has
        run_interventional=True).
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
                        f"         precision={row['precision']:.3f}  "
                        f"recall={row['recall']:.3f}  "
                        f"f1={row['f1']:.3f}  "
                        f"accuracy={row['accuracy']:.3f}  "
                        f"n_missing={row['n_missing_edges']}"
                    )
                    if "int_rmse" in row and not np.isnan(row["int_rmse"]):
                        msg += (
                            f"  |  int_rmse={row['int_rmse']:.4f}"
                            f"  int_dir_acc={row['int_direction_accuracy']:.3f}"
                        )
                    print(msg)
            except Exception as exc:
                print(f"  [ERROR] config={cfg.name!r} seed={seed}: {exc}")

    return pd.DataFrame(rows), pd.DataFrame(all_per_node)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-config mean ± std for graph and interventional metrics."""
    graph_metrics = ["precision", "recall", "f1", "accuracy"]
    int_metrics = ["int_rmse", "int_mae", "int_pearson_r", "int_direction_accuracy"]
    metrics = graph_metrics + [m for m in int_metrics if m in results.columns]
    agg = (
        results.groupby("config")[metrics]
        .agg(["mean", "std"], skipna=True)
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# Benchmark configurations  ← edit / extend here
# ---------------------------------------------------------------------------

_DAG_PARAMS = dict(
    n_start=20,
    n_end=10,
    max_mediators=3,
    shared_mediator_prob=0.4,
    confounder_prob=0.0,
    end_node_alpha=0.8,
)

_SEEDS = list(range(30))

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [

    # ── Sweep 1: missing real edges only ────────────────────────────────────
    BenchmarkConfig(
        name="missing_20pct",
        dag_params=_DAG_PARAMS,
        fake_node_multiplier=2.0,
        fake_edge_multiplier=5.0,
        p_missing_real=0.2,
        p_mediated_shortcut=0.0,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="missing_50pct",
        dag_params=_DAG_PARAMS,
        fake_node_multiplier=2.0,
        fake_edge_multiplier=5.0,
        p_missing_real=0.5,
        p_mediated_shortcut=0.0,
        seeds=_SEEDS,
    ),

    # ── Sweep 2: missing edges + mediated shortcuts (p_shortcut=0.3) ────────
    BenchmarkConfig(
        name="missing_20pct_shortcut_30pct",
        dag_params=_DAG_PARAMS,
        fake_node_multiplier=2.0,
        fake_edge_multiplier=5.0,
        p_missing_real=0.2,
        p_mediated_shortcut=0.3,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="missing_50pct_shortcut_30pct",
        dag_params=_DAG_PARAMS,
        fake_node_multiplier=2.0,
        fake_edge_multiplier=5.0,
        p_missing_real=0.5,
        p_mediated_shortcut=0.3,
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

    # Save trial-level results
    out_path = "missing_edge_benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nTrial results saved to {out_path}")

    # Save per-node interventional results
    if not node_results.empty:
        node_out_path = "missing_edge_interventional_nodes.csv"
        node_results.to_csv(node_out_path, index=False)
        print(f"Per-node interventional results saved to {node_out_path}")
