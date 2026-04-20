"""
Head-to-head comparison: BICGaussIndraPriors vs. pure Hill Climb (BICGauss, no prior).

Both methods receive identical data (same GT DAG, same simulation, same INDRA-derived
spurious nodes as distractors) so results are directly comparable.

3 seeds × 2 methods × 200 samples.  Each trial is written to CSV immediately after
completion so the file is always readable even if the run is interrupted.
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
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    SparseHillClimb,
)
from causomic.graph_construction.repair import convert_to_y0_graph
from causomic.network import estimate_posterior_dag
from causomic.simulation.proteomics_simulator import simulate_data
from causomic.simulation.random_network import (
    generate_indra_data,
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
    `method` controls which graph-learning approach is used:
      "hillclimb"  — pgmpy HillClimbSearch + BICGauss, no prior knowledge.
      "bic_prior"  — estimate_posterior_dag with BICGaussIndraPriors.
    """
    name: str
    method: str = "hillclimb"   # "hillclimb" | "bic_prior"

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
    fake_node_multiplier: float = 2.0
    fake_edge_multiplier: float = 5.0

    # Simulation params
    n_samples: int = 200

    # Prior-method params (ignored when method="hillclimb")
    prior_strength: float = 5.0
    n_bootstrap: int = 100
    edge_probability: float = 0.5

    # Seeds to run
    seeds: list[int] = field(default_factory=lambda: list(range(3)))

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
    posterior,
    real_protein_df: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    """Fit LVM on a pre-built posterior, run paired interventions, return metrics.

    Interventions are always do(start_nodes=1) vs do(start_nodes=0).
    The reported effect is E[outcome | do=1] - E[outcome | do=0].

    Parameters
    ----------
    posterior : y0 NxMixedGraph
        Already-built posterior graph (from either HC or estimate_posterior_dag).
    real_protein_df : pd.DataFrame
        Simulated protein data restricted to real nodes only.
    """
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=cfg.lvm_num_steps, verbose=False)
    lvm.fit(real_protein_df, posterior)

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
    """Run one simulation trial for the configured method."""

    # 1. Ground-truth DAG
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)
    start_nodes = roles["start"]
    end_nodes = roles["end"]

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * cfg.fake_node_multiplier))
    n_fake_edges = max(1, round(n_real_edges * cfg.fake_edge_multiplier))

    # 2. INDRA pipeline — always run so both methods see the same spurious nodes.
    #    HC ignores indra_df; bic_prior uses it as the prior.
    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=0.0,
        p_mediated_shortcut=0.1,
        preferential_attachment=True,
    )
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]

    # 3. Augment GT DAG with spurious isolated nodes (no edges added)
    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)

    # 4. Simulate data on augmented DAG — spurious columns act as distractors
    sim = simulate_data(
        augmented_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    # 5. Restrict to real nodes for metrics and LVM
    real_nodes = set(str(n) for n in gt_dag.nodes())
    real_protein_df = protein_df[[c for c in protein_df.columns if c in real_nodes]]

    # 6. Learn graph — method-specific
    posterior = None
    method_row: dict[str, Any] = {}

    if cfg.method == "hillclimb":
        hc_dag = HillClimbSearch(protein_df).estimate(
            scoring_method=BICGauss(protein_df),
            show_progress=False,
            max_indegree=3,
        )
        hc_real_edges = {
            (str(u), str(v))
            for u, v in hc_dag.edges()
            if u in real_nodes and v in real_nodes
        }
        graph_metrics = compute_metrics(hc_real_edges, gt_dag, real_nodes)
        method_row = {
            "n_learned_edges_total": len(list(hc_dag.edges())),
            "n_learned_edges_real": len(hc_real_edges),
        }
        if hc_real_edges:
            edges_df = pd.DataFrame(list(hc_real_edges), columns=["source", "target"])
            posterior = convert_to_y0_graph(edges_df)

    elif cfg.method == "bic_prior":
        posterior, _ = estimate_posterior_dag(
            protein_df,
            indra_priors=indra_df,
            prior_strength=cfg.prior_strength,
            scoring_function=BICGaussIndraPriors,
            search_algorithm=SparseHillClimb,
            n_bootstrap=cfg.n_bootstrap,
            edge_probability=cfg.edge_probability,
            convert_to_probability=True,
            return_bootstrap_dags=True,
        )
        pred_edges = {
            (str(u), str(v))
            for u, v in posterior.directed.edges()
            if u in real_nodes and v in real_nodes
        }
        graph_metrics = compute_metrics(pred_edges, gt_dag, real_nodes)
        method_row = {"n_learned_edges_real": len(pred_edges)}

    else:
        raise ValueError(f"Unknown method {cfg.method!r}")

    # 7. Assemble row
    row: dict[str, Any] = {
        "config": cfg.name,
        "method": cfg.method,
        "seed": seed,
        "n_nodes": n_real_nodes,
        "n_edges": n_real_edges,
        "n_start": len(start_nodes),
        "n_end": len(end_nodes),
        "n_spurious_nodes": len(spurious_nodes),
        **method_row,
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 8. Interventional metrics (optional)
    if cfg.run_interventional and posterior is not None:
        try:
            int_metrics, per_node_rows = _run_interventional(
                cfg, seed, sim, roles, gt_dag, posterior, real_protein_df
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
    elif cfg.run_interventional:
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
    configs: list[BenchmarkConfig],
    results_path: Path,
    node_results_path: Path,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all configs across all seeds, writing each completed trial immediately.

    Each row is appended to `results_path` (and `node_results_path` if non-empty)
    as soon as it finishes, so results are recoverable if the run is interrupted.
    """
    rows: list[dict] = []
    all_per_node: list[dict] = []
    total = sum(len(c.seeds) for c in configs)
    done = 0

    for cfg in configs:
        for seed in cfg.seeds:
            done += 1
            if verbose:
                print(
                    f"[{done}/{total}] config={cfg.name!r}  method={cfg.method!r}"
                    f"  seed={seed} ...",
                    flush=True,
                )
            try:
                row, per_node_rows = run_trial(cfg, seed)
                rows.append(row)
                all_per_node.extend(per_node_rows)

                # Incremental save — append header only on first write
                pd.DataFrame([row]).to_csv(
                    results_path,
                    mode="a",
                    header=not results_path.exists() or results_path.stat().st_size == 0,
                    index=False,
                )
                if per_node_rows:
                    pd.DataFrame(per_node_rows).to_csv(
                        node_results_path,
                        mode="a",
                        header=not node_results_path.exists()
                              or node_results_path.stat().st_size == 0,
                        index=False,
                    )

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

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [
    BenchmarkConfig(
        name="bic_prior",
        method="bic_prior",
        dag_params=_DAG_PARAMS,
        n_samples=200,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="hillclimb",
        method="hillclimb",
        dag_params=_DAG_PARAMS,
        n_samples=200,
        seeds=_SEEDS,
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_path = OUTPUT_DIR / "baseline_benchmark_results.csv"
    nodes_path = OUTPUT_DIR / "baseline_benchmark_interventional_nodes.csv"

    # Clear any previous run so incremental writes start fresh
    results_path.unlink(missing_ok=True)
    nodes_path.unlink(missing_ok=True)

    results, node_results = run_benchmark(
        BENCHMARK_CONFIGS,
        results_path=results_path,
        node_results_path=nodes_path,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("SUMMARY (mean ± std across seeds)")
    print("=" * 70)
    print(summarize(results).to_string())
    print(f"\nTrial results saved to {results_path}")
    if not node_results.empty:
        print(f"Per-node results saved to {nodes_path}")
