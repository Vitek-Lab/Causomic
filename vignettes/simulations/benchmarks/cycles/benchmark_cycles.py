"""
Benchmark: Effect of cycles on posterior network recovery.

Tests three cyclic graph configurations — cycle anchored at a start node, a pure
mediator cycle, and a cycle anchored at an end node — against a no-cycle baseline.
For each trial the benchmark reports standard graph-structure metrics (precision,
recall, F1, accuracy) plus cycle-specific metrics (cycle_edge_recall,
cycle_edge_precision) that isolate how well the method recovers the feedback edges.

Interventional metrics are disabled for cyclic configs because ground-truth
expected values cannot be computed analytically via topological sort.

Configurations
--------------
- no_cycle       : baseline structured DAG, standard simulation
- cycle_start    : L0 → CYS0 → CYS1 → L0 anchored at a start node
- cycle_mediators: CYM0 → CYM1 → CYM2 → CYM0 pure mediator loop
- cycle_end      : R0 → CYE0 → CYE1 → R0 anchored at an end node

Each config is swept over SEEDS (default: 10 seeds).

Usage
-----
    python benchmark_cycles.py

Output CSVs are written to the same directory as the script:
    benchmark_cycles_results.csv        — one row per trial
    benchmark_cycles_interventional.csv — per-node interventional rows (baseline only)
"""

from __future__ import annotations

import itertools
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch

from causomic.causal_model.LVM import LVM
from causomic.graph_construction.prior_data_reconciliation import (
    AICGaussIndraPriors,
    SparseHillClimb,
)
from causomic.network import estimate_posterior_dag
from causomic.simulation.cyclic_network import generate_cyclic_graph, simulate_cyclic_data
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
    """One benchmarking scenario, swept over ``seeds``."""

    name: str

    # Passed to generate_cyclic_graph (or generate_structured_dag for baseline)
    dag_params: dict = field(default_factory=lambda: dict(
        n_start=20,
        n_end=8,
        max_mediators=3,
        shared_mediator_prob=0.3,
        confounder_prob=0.0,
    ))

    # Cycle-specific params — empty dict means no cycles (baseline)
    cycle_params: dict = field(default_factory=dict)

    # Cyclic simulation params
    threshold: float = 20.0
    max_iterations: int = 100

    # INDRA noise multipliers (relative to graph size)
    fake_node_multiplier: float = 1.0
    fake_edge_multiplier: float = 3.0

    # Simulation
    n_samples: int = 200

    # Posterior estimation
    scoring_function: type = AICGaussIndraPriors
    prior_strength: float = 5.0
    n_bootstrap: int = 100
    edge_probability: float = 0.5

    seeds: list[int] = field(default_factory=lambda: list(range(10)))

    # Enable interventional metrics only for the non-cyclic baseline
    # (ground_truth_interventional_effect requires topological order)
    run_interventional: bool = False
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
    """Precision, Recall, F1, Accuracy for graph structure recovery."""
    pred_edges = set((str(u), str(v)) for u, v in posterior.directed.edges())
    gt_edges   = set((str(u), str(v)) for u, v in gt_dag.edges())

    universe = {(u, v) for u, v in itertools.permutations(all_nodes, 2)}

    tp = len(pred_edges & gt_edges)
    fp = len(pred_edges - gt_edges)
    fn = len(gt_edges - pred_edges)
    tn = len(universe - pred_edges - gt_edges)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / len(universe) if universe else 0.0

    return {
        "precision": precision, "recall": recall,
        "f1": f1,               "accuracy": accuracy,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "n_pred_edges": len(pred_edges),
        "n_gt_edges":   len(gt_edges),
    }


def compute_cycle_edge_metrics(
    posterior,
    gt_dag: nx.DiGraph,
    cycle_nodes: set[str],
) -> dict[str, float]:
    """Precision and Recall restricted to cycle edges (both endpoints in cycle).

    Cycle edges are those where *both* source and target belong to the cycle.
    These are the feedback-loop edges that standard DAG methods are least
    expected to recover correctly.

    Parameters
    ----------
    posterior : NxMixedGraph
        Estimated posterior graph from estimate_posterior_dag.
    gt_dag : nx.DiGraph
        Ground-truth causal graph (with cycles).
    cycle_nodes : set of str
        All node names that participate in any cycle.

    Returns
    -------
    dict with cycle_edge_recall, cycle_edge_precision, n_cycle_gt_edges,
    n_cycle_pred_edges.
    """
    gt_cycle_edges = {
        (str(u), str(v)) for u, v in gt_dag.edges()
        if str(u) in cycle_nodes and str(v) in cycle_nodes
    }
    pred_cycle_edges = {
        (str(u), str(v)) for u, v in posterior.directed.edges()
        if str(u) in cycle_nodes and str(v) in cycle_nodes
    }

    tp = len(pred_cycle_edges & gt_cycle_edges)
    fp = len(pred_cycle_edges - gt_cycle_edges)
    fn = len(gt_cycle_edges - pred_cycle_edges)

    cycle_recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    cycle_precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

    return {
        "cycle_edge_recall":    cycle_recall,
        "cycle_edge_precision": cycle_precision,
        "n_cycle_gt_edges":     len(gt_cycle_edges),
        "n_cycle_pred_edges":   len(pred_cycle_edges),
    }


# ---------------------------------------------------------------------------
# Interventional helpers (baseline only)
# ---------------------------------------------------------------------------

def compute_interventional_metrics(
    config_name: str,
    seed: int,
    ci_series: pd.Series,
    gt_series: pd.Series,
) -> tuple[dict[str, float], list[dict]]:
    ci  = ci_series.values.astype(float)
    gt  = gt_series.values.astype(float)
    diff = ci - gt

    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))

    if len(ci) >= 2 and np.std(ci) > 0 and np.std(gt) > 0:
        corr = float(np.corrcoef(ci, gt)[0, 1])
    else:
        corr = float("nan")

    nonzero = gt != 0
    direction_acc = (
        float((np.sign(ci[nonzero]) == np.sign(gt[nonzero])).mean())
        if nonzero.sum() > 0 else float("nan")
    )

    summary = {
        "int_rmse": rmse, "int_mae": mae,
        "int_pearson_r": corr, "int_direction_accuracy": direction_acc,
        "int_n_outcomes": len(ci),
    }
    per_node_rows = [
        {
            "config": config_name, "seed": seed, "node": node,
            "ci_result": float(ci_val), "gt_result": float(gt_val),
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
    model_input  = pd.DataFrame(sim["Protein_data"])
    outcome      = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=cfg.lvm_num_steps, verbose=False)
    lvm.fit(model_input, posterior)

    torch.manual_seed(seed)
    lvm.intervention(intervention0, outcome,
                     predictive_samples=cfg.lvm_predictive_samples)
    int0 = lvm.intervention_samples

    torch.manual_seed(seed)
    lvm.intervention(intervention1, outcome,
                     predictive_samples=cfg.lvm_predictive_samples)
    int1 = lvm.intervention_samples

    ci_results = (int0 - int1).mean(axis=0)

    gt_int0 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"], intervention0, outcome)
    gt_int1 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"], intervention1, outcome)
    gt_results = {k: gt_int0["effect"][k] - gt_int1["effect"][k]
                  for k in gt_int1["effect"]}

    ci_series = pd.Series(ci_results, index=outcome, dtype=float)
    gt_series = pd.Series(gt_results, dtype=float).reindex(outcome)

    return compute_interventional_metrics(cfg.name, seed, ci_series, gt_series)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(cfg: BenchmarkConfig, seed: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one trial for the given config and seed.

    For cyclic configs, ``generate_cyclic_graph`` and ``simulate_cyclic_data``
    are used.  For the baseline config (empty ``cycle_params``), the standard
    ``generate_structured_dag`` and ``simulate_data`` are used.
    """
    is_cyclic = bool(cfg.cycle_params)

    # 1. Ground-truth graph
    if is_cyclic:
        gt_dag, roles = generate_cyclic_graph(
            **cfg.dag_params, **cfg.cycle_params, seed=seed
        )
    else:
        gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * cfg.fake_node_multiplier))
    n_fake_edges = max(1, round(n_real_edges * cfg.fake_edge_multiplier))

    # Collect all cycle nodes (across locations) for cycle-edge metrics
    all_cycle_nodes: set[str] = set()
    if is_cyclic:
        for node_list in roles.get("cycle_nodes", {}).values():
            all_cycle_nodes.update(str(n) for n in node_list)

    # 2. INDRA-style priors
    # generate_indra_data works directly with cyclic graphs: it iterates over
    # edges without requiring acyclicity, and marks cycle edges ground_truth=True.
    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=0.0,
    )

    # 3. Augmented graph: spurious INDRA nodes added as isolated nodes
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]
    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)

    # 4. Simulate data
    if is_cyclic:
        sim = simulate_cyclic_data(
            augmented_dag,
            roles,
            threshold=cfg.threshold,
            max_iterations=cfg.max_iterations,
            n=cfg.n_samples,
            add_feature_var=False,
            seed=seed,
        )
    else:
        sim = simulate_data(
            augmented_dag,
            n=cfg.n_samples,
            add_feature_var=False,
            add_error=True,
            seed=seed,
        )

    protein_df = pd.DataFrame(sim["Protein_data"])

    # 5. Exclude confounder edges from priors (C0, C1, … pattern; not CYS/CYM/CYE)
    is_conf_src = indra_df["source"].str.match(r"^C\d")
    is_conf_tgt = indra_df["target"].str.match(r"^C\d")
    filtered_indra = indra_df[~is_conf_src & ~is_conf_tgt].reset_index(drop=True)

    # 6. Estimate posterior DAG
    posterior, _bootstraps = estimate_posterior_dag(
        protein_df,
        indra_priors=filtered_indra,
        prior_strength=cfg.prior_strength,
        scoring_function=cfg.scoring_function,
        search_algorithm=SparseHillClimb,
        n_bootstrap=cfg.n_bootstrap,
        add_high_corr_edges_to_priors=False,
        edge_probability=cfg.edge_probability,
        convert_to_probability=True,
        return_bootstrap_dags=True,
    )

    # 7. Graph structure metrics
    all_nodes = set(str(n) for n in protein_df.columns)
    graph_metrics = compute_metrics(posterior, gt_dag, all_nodes)

    # 8. Cycle-specific metrics (NaN for baseline)
    if is_cyclic and all_cycle_nodes:
        cycle_metrics = compute_cycle_edge_metrics(posterior, gt_dag, all_cycle_nodes)
    else:
        cycle_metrics = {
            "cycle_edge_recall":    float("nan"),
            "cycle_edge_precision": float("nan"),
            "n_cycle_gt_edges":     0,
            "n_cycle_pred_edges":   0,
        }

    n_cycle_nodes = len(all_cycle_nodes)

    row: dict[str, Any] = {
        "config":        cfg.name,
        "seed":          seed,
        "n_real_nodes":  n_real_nodes,
        "n_real_edges":  n_real_edges,
        "n_fake_nodes":  n_fake_nodes,
        "n_fake_edges":  n_fake_edges,
        "n_cycle_nodes": n_cycle_nodes,
        **graph_metrics,
        **cycle_metrics,
    }
    per_node_rows: list[dict] = []

    # 9. Interventional metrics (baseline only — requires DAG topological order)
    if cfg.run_interventional:
        try:
            int_metrics, per_node_rows = _run_interventional(
                cfg, seed, sim, roles, gt_dag, posterior
            )
            row.update(int_metrics)
        except Exception as exc:
            print(f"  [WARN] interventional failed seed={seed}: {exc}")
            row.update({
                "int_rmse": float("nan"), "int_mae": float("nan"),
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
    """Run all configs, return (trial_results, per_node_results)."""
    rows: list[dict] = []
    all_per_node: list[dict] = []
    total = sum(len(c.seeds) for c in configs)
    done  = 0

    for cfg in configs:
        for seed in cfg.seeds:
            done += 1
            if verbose:
                print(f"[{done}/{total}] config={cfg.name!r}  seed={seed} ...",
                      flush=True)
            try:
                row, pn = run_trial(cfg, seed)
                rows.append(row)
                all_per_node.extend(pn)
                if verbose:
                    msg = (
                        f"         precision={row['precision']:.3f}  "
                        f"recall={row['recall']:.3f}  "
                        f"f1={row['f1']:.3f}  "
                        f"accuracy={row['accuracy']:.3f}"
                    )
                    if not np.isnan(row.get("cycle_edge_recall", float("nan"))):
                        msg += (
                            f"  |  cycle_recall={row['cycle_edge_recall']:.3f}"
                            f"  cycle_prec={row['cycle_edge_precision']:.3f}"
                        )
                    print(msg)
            except Exception as exc:
                import traceback
                print(f"  [ERROR] config={cfg.name!r} seed={seed}: {exc}")
                traceback.print_exc()

    return pd.DataFrame(rows), pd.DataFrame(all_per_node)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean ± std across seeds."""
    graph_cols = ["precision", "recall", "f1", "accuracy"]
    cycle_cols = ["cycle_edge_recall", "cycle_edge_precision"]
    int_cols   = ["int_rmse", "int_mae", "int_pearson_r", "int_direction_accuracy"]

    metric_cols = (
        graph_cols
        + [c for c in cycle_cols if c in results.columns]
        + [c for c in int_cols   if c in results.columns]
    )
    agg = (
        results.groupby("config")[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

_DAG_PARAMS_BASE = dict(
    n_start=20,
    n_end=8,
    max_mediators=3,
    shared_mediator_prob=0.3,
    confounder_prob=0.0,
)

_SEEDS = list(range(10))

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [

    # ── Baseline: no cycles ───────────────────────────────────────────────
    BenchmarkConfig(
        name="no_cycle",
        dag_params=_DAG_PARAMS_BASE,
        cycle_params={},                   # empty → uses standard DAG pipeline
        fake_node_multiplier=1.0,
        fake_edge_multiplier=3.0,
        seeds=_SEEDS,
        run_interventional=False,          # set True to compare interventional effects
    ),

    # ── Cycle anchored at a start node ───────────────────────────────────
    # Structure: L0 → CYS0 → CYS1 → L0  (plus L0's normal downstream edges)
    BenchmarkConfig(
        name="cycle_start",
        dag_params=_DAG_PARAMS_BASE,
        cycle_params=dict(
            add_cycle_in_start=True,
            cycle_size=3,
        ),
        fake_node_multiplier=1.0,
        fake_edge_multiplier=3.0,
        seeds=_SEEDS,
        run_interventional=False,
    ),

    # ── Pure mediator cycle ───────────────────────────────────────────────
    # Structure: CYM0 → CYM1 → CYM2 → CYM0, attached to main causal path
    BenchmarkConfig(
        name="cycle_mediators",
        dag_params=_DAG_PARAMS_BASE,
        cycle_params=dict(
            add_cycle_in_mediators=True,
            cycle_size=3,
        ),
        fake_node_multiplier=1.0,
        fake_edge_multiplier=3.0,
        seeds=_SEEDS,
        run_interventional=False,
    ),

    # ── Cycle anchored at an end node ────────────────────────────────────
    # Structure: R0 → CYE0 → CYE1 → R0  (plus R0's normal upstream edges)
    BenchmarkConfig(
        name="cycle_end",
        dag_params=_DAG_PARAMS_BASE,
        cycle_params=dict(
            add_cycle_in_end=True,
            cycle_size=3,
        ),
        fake_node_multiplier=1.0,
        fake_edge_multiplier=3.0,
        seeds=_SEEDS,
        run_interventional=False,
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))

    results, node_results = run_benchmark(BENCHMARK_CONFIGS, verbose=True)

    print("\n" + "=" * 70)
    print("PER-TRIAL RESULTS")
    print("=" * 70)
    print(results.to_string(index=False))

    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    print(summarize(results).to_string())

    results_path = os.path.join(out_dir, "benchmark_cycles_results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nTrial results saved to {results_path}")

    if not node_results.empty:
        node_path = os.path.join(out_dir, "benchmark_cycles_interventional.csv")
        node_results.to_csv(node_path, index=False)
        print(f"Per-node interventional results saved to {node_path}")
