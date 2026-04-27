"""
Combined benchmarking script: Causomic vs DAGMA.

Runs both methods on identical simulated data (same seed → same DAG, same
INDRA noise, same protein matrix) and reports graph-structure and
interventional-prediction metrics.

**New in this version**: every trial saves the learned graph (edge list),
ground-truth graph, and node role metadata to disk so you never lose
intermediate artefacts again.

Usage — full sweep (all configs × all seeds):
    python benchmark_combined.py

Usage — single trial (for cluster jobs / debugging):
    python benchmark_combined.py --method dagma --seed 0 --n_samples 200
    python benchmark_combined.py --method causomic --seed 0 --n_samples 200
"""

from __future__ import annotations

import argparse
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
from dagma.linear import DagmaLinear

from causomic.causal_model.LVM import LVM
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    BICGaussNoPriors,
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

OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_outputs"
GRAPH_DIR = OUTPUT_DIR / "graphs"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """One benchmarking scenario.  Swept over `seeds`."""

    name: str
    method: str  # "causomic" or "dagma"

    # DAG generation
    dag_params: dict = field(default_factory=lambda: dict(
        n_start=20,
        n_end=10,
        max_mediators=3,
        shared_mediator_prob=0.4,
        confounder_prob=0.0,
        end_node_alpha=0.8,
    ))

    # INDRA noise multipliers
    fake_node_multiplier: float = 2.0
    fake_edge_multiplier: float = 5.0

    # Simulation
    n_samples: int = 200

    # Causomic-specific
    scoring_function: type = BICGaussIndraPriors
    prior_strength: float = 5.0
    n_bootstrap: int = 100
    edge_probability: float = 0.5
    add_high_corr_edges_to_priors: bool = False
    corr_threshold: float = 0.5

    # DAGMA-specific
    dagma_lambda1: float = 0.02
    dagma_threshold: float = 0.2

    # LVM / interventional
    run_interventional: bool = True
    lvm_num_steps: int = 1000
    lvm_predictive_samples: int = 200

    # Seeds
    seeds: list[int] = field(default_factory=lambda: list(range(30)))


# ---------------------------------------------------------------------------
# Graph I/O helpers
# ---------------------------------------------------------------------------

def _save_edge_list(
    edges: set[tuple[str, str]],
    path: Path,
    extra_cols: dict[str, Any] | None = None,
) -> None:
    """Save edges as a CSV with source, target, plus optional metadata columns."""
    rows = [{"source": u, "target": v} for u, v in sorted(edges)]
    if extra_cols:
        for r in rows:
            r.update(extra_cols)
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_node_roles(roles: dict, spurious_nodes: list, path: Path) -> None:
    """Save a CSV mapping every node to its role (start/end/mediator/spurious)."""
    rows = []
    for role_name, node_list in roles.items():
        for n in node_list:
            rows.append({"node": str(n), "role": role_name})
    for n in spurious_nodes:
        rows.append({"node": str(n), "role": "spurious"})
    pd.DataFrame(rows).to_csv(path, index=False)


def _save_gt_graph(gt_dag: nx.DiGraph, path: Path) -> None:
    """Save ground-truth DAG as a CSV edge list."""
    edges = {(str(u), str(v)) for u, v in gt_dag.edges()}
    _save_edge_list(edges, path)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_graph_metrics(
    learned_edges: set[tuple[str, str]],
    gt_dag: nx.DiGraph,
    all_nodes: set[str],
) -> dict[str, float]:
    """Precision, Recall, F1, Accuracy against gt_dag."""
    pred_edges = learned_edges
    gt_edges = {(str(u), str(v)) for u, v in gt_dag.edges()}
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


def compute_interventional_metrics(
    config_name: str,
    seed: int,
    ci_series: pd.Series,
    gt_series: pd.Series,
) -> tuple[dict[str, float], list[dict]]:
    """RMSE, MAE, Pearson r, direction accuracy + per-node detail."""
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


# ---------------------------------------------------------------------------
# Graph learning: method dispatch
# ---------------------------------------------------------------------------

def _learn_graph_dagma(
    cfg: BenchmarkConfig,
    protein_df: pd.DataFrame,
) -> tuple[set[tuple[str, str]], nx.DiGraph]:
    """Run DAGMA and return (learned_edges_set, learned_nx_digraph)."""
    X = protein_df.values
    node_names = list(protein_df.columns)

    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=cfg.dagma_lambda1)

    W_est[np.abs(W_est) < cfg.dagma_threshold] = 0

    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(node_names)
    d = len(node_names)
    for i in range(d):
        for j in range(d):
            if W_est[i, j] != 0:
                learned_dag.add_edge(node_names[j], node_names[i])

    learned_edges = {(str(u), str(v)) for u, v in learned_dag.edges()}
    return learned_edges, learned_dag


def _learn_graph_causomic(
    cfg: BenchmarkConfig,
    protein_df: pd.DataFrame,
    indra_df: pd.DataFrame,
) -> tuple[set[tuple[str, str]], object]:
    """Run Causomic hill-climb and return (learned_edges_set, posterior)."""
    posterior, _bootstraps = estimate_posterior_dag(
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

    learned_edges = {(str(u), str(v)) for u, v in posterior.directed.edges()}
    return learned_edges, posterior


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
    protein_df: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    """Fit LVM on learned graph, run paired interventions, return metrics."""
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

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
# Shared simulation setup
# ---------------------------------------------------------------------------

def _build_simulation(
    cfg: BenchmarkConfig, seed: int
) -> tuple[nx.DiGraph, dict, nx.DiGraph, pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Generate ground-truth DAG, INDRA noise, and simulated protein data.

    Returns
    -------
    gt_dag, roles, indra_dag, indra_df, protein_df, spurious_nodes
    """
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * cfg.fake_node_multiplier))
    n_fake_edges = max(1, round(n_real_edges * cfg.fake_edge_multiplier))

    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=0.0,
        p_mediated_shortcut=0.1,
        preferential_attachment=True,
    )
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]

    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)

    sim = simulate_data(
        augmented_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    return gt_dag, roles, indra_dag, indra_df, protein_df, spurious_nodes, sim


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(cfg: BenchmarkConfig, seed: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one simulation + graph learning + (optional) interventional trial."""

    # 1. Shared simulation
    gt_dag, roles, indra_dag, indra_df, protein_df, spurious_nodes, sim = \
        _build_simulation(cfg, seed)

    real_nodes = set(str(n) for n in gt_dag.nodes())
    all_nodes = set(str(c) for c in protein_df.columns)

    # 2. Save ground-truth graph and node roles (once per seed, shared by methods)
    trial_dir = GRAPH_DIR / f"seed{seed}_n{cfg.n_samples}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    gt_path = trial_dir / "gt_edges.csv"
    if not gt_path.exists():
        _save_gt_graph(gt_dag, gt_path)

    roles_path = trial_dir / "node_roles.csv"
    if not roles_path.exists():
        _save_node_roles(roles, spurious_nodes, roles_path)

    indra_path = trial_dir / "indra_priors.csv"
    if not indra_path.exists():
        indra_df.to_csv(indra_path, index=False)

    # 3. Learn graph
    if cfg.method == "dagma":
        learned_edges, learned_obj = _learn_graph_dagma(cfg, protein_df)
        # Filter to real nodes for metrics
        learned_real_edges = {
            (u, v) for u, v in learned_edges
            if u in real_nodes and v in real_nodes
        }
        # Build posterior for LVM from DAGMA edges
        posterior = None
        if learned_real_edges:
            edges_df = pd.DataFrame(
                list(learned_real_edges), columns=["source", "target"]
            )
            posterior = convert_to_y0_graph(edges_df)
        # For interventional: use only real-node columns
        lvm_protein_df = protein_df[[c for c in protein_df.columns if c in real_nodes]]

    elif cfg.method == "causomic":
        learned_edges, posterior = _learn_graph_causomic(cfg, protein_df, indra_df)
        learned_real_edges = {
            (u, v) for u, v in learned_edges
            if u in real_nodes and v in real_nodes
        }
        lvm_protein_df = protein_df

    else:
        raise ValueError(f"Unknown method: {cfg.method!r}")

    # 4. Save learned graph
    learned_path = trial_dir / f"{cfg.method}_learned_edges.csv"
    _save_edge_list(
        learned_edges, learned_path,
        extra_cols={"method": cfg.method, "config": cfg.name},
    )

    # Also save the real-node-only version for convenience
    learned_real_path = trial_dir / f"{cfg.method}_learned_real_edges.csv"
    _save_edge_list(
        learned_real_edges, learned_real_path,
        extra_cols={"method": cfg.method, "config": cfg.name},
    )

    # 5. Graph metrics (evaluated on real nodes only)
    graph_metrics = compute_graph_metrics(learned_real_edges, gt_dag, real_nodes)

    row: dict[str, Any] = {
        "config": cfg.name,
        "method": cfg.method,
        "seed": seed,
        "n_samples": cfg.n_samples,
        "n_real_nodes": gt_dag.number_of_nodes(),
        "n_real_edges": gt_dag.number_of_edges(),
        "n_fake_nodes": len(spurious_nodes),
        "n_fake_edges": len([
            e for e in indra_dag.edges()
            if e not in gt_dag.edges()
        ]),
        "n_learned_edges_total": len(learned_edges),
        "n_learned_edges_real": len(learned_real_edges),
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 6. Interventional metrics
    _nan_int = {
        "int_rmse": float("nan"),
        "int_mae": float("nan"),
        "int_pearson_r": float("nan"),
        "int_direction_accuracy": float("nan"),
        "int_n_outcomes": float("nan"),
    }

    if cfg.run_interventional:
        if posterior is not None:
            try:
                int_metrics, per_node_rows = _run_interventional(
                    cfg, seed, sim, roles, gt_dag, posterior, lvm_protein_df,
                )
                row.update(int_metrics)
            except Exception as exc:
                print(f"  [WARN] interventional failed "
                      f"method={cfg.method} seed={seed}: {exc}")
                row.update(_nan_int)
        else:
            print(f"  [WARN] no edges learned, skipping interventional "
                  f"method={cfg.method} seed={seed}")
            row.update(_nan_int)
    else:
        row.update(_nan_int)

    return row, per_node_rows


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    configs: list[BenchmarkConfig], verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all configs × seeds.  Returns (summary_df, per_node_df)."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    all_per_node: list[dict] = []
    total = sum(len(c.seeds) for c in configs)
    done = 0

    for cfg in configs:
        for seed in cfg.seeds:
            done += 1
            if verbose:
                print(
                    f"[{done}/{total}] method={cfg.method}  "
                    f"config={cfg.name!r}  seed={seed} ...",
                    flush=True,
                )
            try:
                row, per_node_rows = run_trial(cfg, seed)
                rows.append(row)
                all_per_node.extend(per_node_rows)
                if verbose:
                    msg = (
                        f"  prec={row['precision']:.3f}  "
                        f"rec={row['recall']:.3f}  "
                        f"f1={row['f1']:.3f}  "
                        f"acc={row['accuracy']:.3f}"
                    )
                    if not np.isnan(row.get("int_rmse", float("nan"))):
                        msg += (
                            f"  |  rmse={row['int_rmse']:.4f}"
                            f"  dir_acc={row['int_direction_accuracy']:.3f}"
                        )
                    print(msg)
            except Exception as exc:
                print(f"  [ERROR] method={cfg.method} config={cfg.name!r} "
                      f"seed={seed}: {exc}")

    return pd.DataFrame(rows), pd.DataFrame(all_per_node)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean ± std for all metrics."""
    graph_metrics = ["precision", "recall", "f1", "accuracy",
                     "n_learned_edges_real", "n_pred_edges"]
    int_metrics = ["int_rmse", "int_mae", "int_pearson_r", "int_direction_accuracy"]
    metrics = graph_metrics + [m for m in int_metrics if m in results.columns]
    agg = (
        results.groupby(["method", "config"])[metrics]
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

_N_LOW = 50
_N_MID = 200
_N_HIGH = 500

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [

    # ── DAGMA: sample-size sweep ────────────────────────────────────────
    BenchmarkConfig(
        name="dagma_low",
        method="dagma",
        dag_params=_DAG_PARAMS,
        n_samples=_N_LOW,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="dagma_mid",
        method="dagma",
        dag_params=_DAG_PARAMS,
        n_samples=_N_MID,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="dagma_high",
        method="dagma",
        dag_params=_DAG_PARAMS,
        n_samples=_N_HIGH,
        seeds=_SEEDS,
    ),

    # ── Causomic (BICGaussIndraPriors): sample-size sweep ───────────────
    BenchmarkConfig(
        name="causomic_indra_low",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussIndraPriors,
        n_samples=_N_LOW,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="causomic_indra_mid",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussIndraPriors,
        n_samples=_N_MID,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="causomic_indra_high",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussIndraPriors,
        n_samples=_N_HIGH,
        seeds=_SEEDS,
    ),

    # ── Causomic (BICGaussNoPriors): ablation ───────────────────────────
    BenchmarkConfig(
        name="causomic_noprior_low",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussNoPriors,
        n_samples=_N_LOW,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="causomic_noprior_mid",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussNoPriors,
        n_samples=_N_MID,
        seeds=_SEEDS,
    ),
    BenchmarkConfig(
        name="causomic_noprior_high",
        method="causomic",
        dag_params=_DAG_PARAMS,
        scoring_function=BICGaussNoPriors,
        n_samples=_N_HIGH,
        seeds=_SEEDS,
    ),
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _run_single(method: str, seed: int, n_samples: int) -> None:
    """CLI mode: run one trial and save results."""
    matching = [
        c for c in BENCHMARK_CONFIGS
        if c.method == method and c.n_samples == n_samples
    ]
    if not matching:
        raise ValueError(
            f"No config found for method={method!r}, n_samples={n_samples}"
        )
    cfg = matching[0]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {method}: seed={seed}, n_samples={n_samples}", flush=True)
    row, per_node_rows = run_trial(cfg, seed)

    tag = f"{method}_n{n_samples}_s{seed}"
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / f"results_{tag}.csv", index=False)
    if per_node_rows:
        pd.DataFrame(per_node_rows).to_csv(
            OUTPUT_DIR / f"nodes_{tag}.csv", index=False
        )

    print(f"  prec={row['precision']:.3f}  rec={row['recall']:.3f}  "
          f"f1={row['f1']:.3f}  acc={row['accuracy']:.3f}")
    if not np.isnan(row.get("int_rmse", float("nan"))):
        print(f"  int_rmse={row['int_rmse']:.4f}  "
              f"int_dir_acc={row['int_direction_accuracy']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default=None,
                        choices=["dagma", "causomic"],
                        help="Run a single method (requires --seed, --n_samples)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None)
    args = parser.parse_args()

    if args.method is not None:
        # Single-trial mode
        if args.seed is None or args.n_samples is None:
            parser.error("--method requires --seed and --n_samples")
        _run_single(args.method, args.seed, args.n_samples)
    else:
        # Full sweep
        results, node_results = run_benchmark(BENCHMARK_CONFIGS, verbose=True)

        print("\n" + "=" * 70)
        print("SUMMARY (mean ± std across seeds)")
        print("=" * 70)
        print(summarize(results).to_string())

        # Save everything
        results.to_csv(OUTPUT_DIR / "all_results.csv", index=False)
        if not node_results.empty:
            node_results.to_csv(
                OUTPUT_DIR / "all_interventional_nodes.csv", index=False
            )
        summarize(results).to_csv(OUTPUT_DIR / "summary.csv")

        print(f"\nAll outputs saved to {OUTPUT_DIR}/")
        print(f"Learned graphs saved to {GRAPH_DIR}/")