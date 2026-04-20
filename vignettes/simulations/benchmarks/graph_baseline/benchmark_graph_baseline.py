"""
Benchmarking script: pure data-driven causal discovery via Hill Climb (no INDRA prior).

Uses pgmpy's HillClimbSearch with BIC scoring to learn a graph entirely from
observational data, with no biological prior knowledge.  This serves as a
baseline to quantify how much the INDRA prior improves graph recovery and
interventional prediction in benchmark_posterior_accuracy.py.

Three sample-size configurations are evaluated (low / mid / high replicates),
mirroring the sweep in benchmark_posterior_accuracy.py.  Uncorrelated noise
variables are added to the feature matrix (same pattern as
linear_comparison/benchmark_linear_comparison.py) to make structure learning
non-trivial.

After learning a graph, the learned structure is used to fit an LVM and
evaluate interventional effect predictions against the ground-truth SEM.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pyro
import torch
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators.StructureScore import BICGauss

from causomic.causal_model.LVM import LVM
from causomic.graph_construction.repair import convert_to_y0_graph
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

    # Simulation params
    n_samples: int = 200

    # Noise variables added to the HC input to make feature selection non-trivial.
    # These are truly independent of the causal graph.
    n_noise_vars: int = 20

    # Seeds to run
    seeds: list[int] = field(default_factory=lambda: list(range(30)))

    # Interventional benchmark
    run_interventional: bool = True
    lvm_num_steps: int = 1000
    lvm_predictive_samples: int = 200


# ---------------------------------------------------------------------------
# Graph metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    learned_edges: set[tuple[str, str]],
    gt_dag: nx.DiGraph,
    all_nodes: set[str],
) -> dict[str, float]:
    """Compute Precision, Recall, F1, Accuracy against gt_dag.

    Parameters
    ----------
    learned_edges : set of (str, str)
        Directed edges from the HC-learned graph, restricted to real nodes.
    gt_dag : nx.DiGraph
        Ground-truth DAG.
    all_nodes : set[str]
        The universe of real (non-noise) nodes — defines the edge universe.
    """
    pred_edges = learned_edges
    gt_edges = {(str(u), str(v)) for u, v in gt_dag.edges()}

    # Full universe of possible directed edges (no self-loops)
    universe = {(u, v) for u, v in itertools.permutations(all_nodes, 2)}

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

    # Direction accuracy: fraction of nodes where predicted sign matches GT sign.
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
# Interventional runner
# ---------------------------------------------------------------------------

def _run_interventional(
    cfg: BenchmarkConfig,
    seed: int,
    sim: dict,
    roles: dict,
    gt_dag: nx.DiGraph,
    hc_real_edges: set[tuple[str, str]],
    protein_df: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    """Convert HC graph to y0 format, fit LVM, run paired interventions.

    Interventions are always do(start_nodes=1) vs do(start_nodes=0).
    The reported effect is E[outcome | do=1] - E[outcome | do=0].

    Parameters
    ----------
    hc_real_edges : set of (str, str)
        HC-learned edges restricted to real (non-noise) nodes.
    protein_df : pd.DataFrame
        Simulated protein data (real nodes only, no noise columns).
    """
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

    # Convert HC edges to y0 NxMixedGraph expected by LVM
    if len(hc_real_edges) == 0:
        # HC learned no edges among real nodes — cannot run LVM
        nan_metrics = {
            "int_rmse": float("nan"),
            "int_mae": float("nan"),
            "int_pearson_r": float("nan"),
            "int_direction_accuracy": float("nan"),
            "int_n_outcomes": float("nan"),
        }
        return nan_metrics, []

    edges_df = pd.DataFrame(list(hc_real_edges), columns=["source", "target"])
    posterior = convert_to_y0_graph(edges_df)

    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=cfg.lvm_num_steps, verbose=False)
    lvm.fit(protein_df, posterior)

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
    """Run one simulation + HC discovery trial and return (metrics_row, per_node_rows)."""

    # 1. Ground-truth DAG
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)
    start_nodes = roles["start"]
    end_nodes = roles["end"]

    # 2. Simulate data
    sim = simulate_data(
        gt_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    # 3. Add uncorrelated noise variables to make structure learning non-trivial.
    #    These are truly independent of the causal graph — HC must ignore them.
    rng = np.random.default_rng(seed + 10_000)
    noise_df = pd.DataFrame(
        rng.normal(0, 1, size=(cfg.n_samples, cfg.n_noise_vars)),
        columns=[f"NOISE{i}" for i in range(cfg.n_noise_vars)],
    )
    hc_input_df = pd.concat(
        [protein_df.reset_index(drop=True), noise_df.reset_index(drop=True)],
        axis=1,
    )

    # 4. Learn graph from data alone using Hill Climb + BIC
    hc_dag = HillClimbSearch(hc_input_df).estimate(
        scoring_method=BICGauss(hc_input_df),
        show_progress=True,
        max_indegree=3,
    )

    # 5. Extract edges restricted to real (non-noise) nodes for graph metrics and LVM
    real_nodes = set(protein_df.columns)
    hc_real_edges = {
        (str(u), str(v))
        for u, v in hc_dag.edges()
        if u in real_nodes and v in real_nodes
    }

    # 6. Graph metrics — universe is all ordered real-node pairs (no noise)
    graph_metrics = compute_metrics(hc_real_edges, gt_dag, real_nodes)

    row: dict[str, Any] = {
        "config": cfg.name,
        "seed": seed,
        "n_nodes": gt_dag.number_of_nodes(),
        "n_edges": gt_dag.number_of_edges(),
        "n_start": len(start_nodes),
        "n_end": len(end_nodes),
        "n_noise_vars": cfg.n_noise_vars,
        "n_hc_edges_total": len(list(hc_dag.edges())),
        "n_hc_edges_real": len(hc_real_edges),
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 7. Interventional metrics (optional)
    if cfg.run_interventional:
        try:
            int_metrics, per_node_rows = _run_interventional(
                cfg, seed, sim, roles, gt_dag, hc_real_edges, protein_df
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
        Long-format per-node interventional results (empty if run_interventional=False).
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
                        f"accuracy={row['accuracy']:.3f}"
                    )
                    if "int_rmse" in row and not np.isnan(row.get("int_rmse", float("nan"))):
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
        .agg(["mean", "std"])
        .round(4)
    )
    agg.columns = ["_".join(c) for c in agg.columns]
    return agg


# ---------------------------------------------------------------------------
# Benchmark configurations
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

_N_LOW  =  50
_N_MID  = 200
_N_HIGH = 500

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig(
        name="hillclimb_low_replicates",
        dag_params=_DAG_PARAMS,
        n_samples=_N_LOW,
        n_noise_vars=20,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="hillclimb_mid_replicates",
        dag_params=_DAG_PARAMS,
        n_samples=_N_MID,
        n_noise_vars=20,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="hillclimb_high_replicates",
        dag_params=_DAG_PARAMS,
        n_samples=_N_HIGH,
        n_noise_vars=20,
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
        nodes_path = OUTPUT_DIR / "benchmark_interventional_nodes.csv"
        node_results.to_csv(nodes_path, index=False)
        print(f"Per-node results saved to {nodes_path}")
