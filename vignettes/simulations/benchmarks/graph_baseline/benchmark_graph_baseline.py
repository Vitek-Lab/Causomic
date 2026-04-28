"""
Unified benchmark — DAGMA + Causomic (with/without INDRA priors).
 
Generates one ground-truth DAG per (seed, n_samples) pair, then runs three
structure-learning methods on the same simulated data:
 
    1. DAGMA            (continuous optimization, no priors)
    2. Causomic-NoPrior (bootstrap hill-climb, BICGaussNoPriors)
    3. Causomic-Prior   (bootstrap hill-climb, BICGaussIndraPriors)
 
Each method is evaluated on graph recovery (Precision/Recall/F1/Accuracy)
and interventional effect prediction (RMSE/MAE/Pearson r/direction accuracy).
 
Outputs are organised into subfolders:
    data/       protein_df, indra_df
    graphs/     ground-truth, INDRA, augmented, and per-method learned DAGs
    results/    per-method summary CSVs and per-node interventional CSVs
 
Usage (single job):
    python benchmark_unified.py --seed 0 --n_samples 200
 
Usage (SLURM array — see run_benchmarks.sh).
"""
 
from __future__ import annotations
 
import argparse
import itertools
import warnings
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
 
OUTPUT_DIR = Path(__file__).resolve().parent
 
# ---------------------------------------------------------------------------
# DAG / simulation parameters (shared across all methods)
# ---------------------------------------------------------------------------
 
DAG_PARAMS = dict(
    n_start=20,
    n_end=10,
    max_mediators=3,
    shared_mediator_prob=0.4,
    confounder_prob=0.0,
    end_node_alpha=0.8,
)
 
FAKE_NODE_MULTIPLIER = 2.0
FAKE_EDGE_MULTIPLIER = 5.0
 
# Causomic-specific
CAUSOMIC_PRIOR_STRENGTH = 5.0
CAUSOMIC_N_BOOTSTRAP = 100
CAUSOMIC_EDGE_PROBABILITY = 0.5
 
# LVM / interventional (shared)
LVM_NUM_STEPS = 1000
LVM_PREDICTIVE_SAMPLES = 200
 
# DAGMA-specific
DAGMA_LAMBDA1 = 0.02
DAGMA_THRESHOLD = 0.3
 
 
# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
 
def _ensure_dirs(base: Path, tag: str) -> dict[str, Path]:
    """Create and return paths for data/, graphs/, results/ subdirectories."""
    dirs = {}
    for name in ("data", "graphs", "results"):
        d = base / name
        d.mkdir(parents=True, exist_ok=True)
        dirs[name] = d
    return dirs
 
 
# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
 
def compute_graph_metrics(
    pred_edges: set[tuple[str, str]],
    gt_dag: nx.DiGraph,
    all_nodes: set[str],
) -> dict[str, float]:
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
# Interventional evaluation (shared across methods)
# ---------------------------------------------------------------------------
 
def _nan_interventional() -> dict[str, float]:
    return {
        "int_rmse": float("nan"),
        "int_mae": float("nan"),
        "int_pearson_r": float("nan"),
        "int_direction_accuracy": float("nan"),
        "int_n_outcomes": float("nan"),
    }
 
 
def run_interventional(
    config_name: str,
    seed: int,
    sim: dict,
    roles: dict,
    gt_dag: nx.DiGraph,
    posterior,
    fit_df: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    """Fit LVM on *fit_df* using *posterior*, run paired do-calculus, compare to GT."""
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}
 
    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=LVM_NUM_STEPS, verbose=False)
    lvm.fit(fit_df, posterior)
 
    torch.manual_seed(seed)
    lvm.intervention(intervention0, outcome, predictive_samples=LVM_PREDICTIVE_SAMPLES)
    int0 = lvm.intervention_samples
 
    torch.manual_seed(seed)
    lvm.intervention(intervention1, outcome, predictive_samples=LVM_PREDICTIVE_SAMPLES)
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
 
    return compute_interventional_metrics(config_name, seed, ci_series, gt_series)
 
 
# ---------------------------------------------------------------------------
# Method runners
# ---------------------------------------------------------------------------
 
def run_dagma(
    seed: int,
    n_samples: int,
    protein_df: pd.DataFrame,
    real_nodes: set[str],
    gt_dag: nx.DiGraph,
    sim: dict,
    roles: dict,
    dirs: dict[str, Path],
    tag: str,
) -> tuple[dict[str, Any], list[dict]]:
    config_name = f"dagma_n{n_samples}"
 
    X = protein_df.values
    node_names = list(protein_df.columns)
 
    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=DAGMA_LAMBDA1)
    W_est[np.abs(W_est) < DAGMA_THRESHOLD] = 0
 
    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(node_names)
    d = len(node_names)
    for i in range(d):
        for j in range(d):
            if W_est[i, j] != 0:
                learned_dag.add_edge(node_names[j], node_names[i])
 
    nx.write_graphml(learned_dag, dirs["graphs"] / f"learned_dag_dagma_{tag}.graphml")
 
    learned_real_edges = {
        (str(u), str(v))
        for u, v in learned_dag.edges()
        if str(u) in real_nodes and str(v) in real_nodes
    }
    graph_metrics = compute_graph_metrics(learned_real_edges, gt_dag, real_nodes)
 
    row: dict[str, Any] = {
        "config": config_name,
        "method": "dagma",
        "seed": seed,
        "n_samples": n_samples,
        **graph_metrics,
    }
    per_node_rows: list[dict] = []
 
    # Interventional — use full learned DAG (including spurious nodes)
    all_learned_edges = {(str(u), str(v)) for u, v in learned_dag.edges()}
    posterior = None
    if all_learned_edges:
        edges_df = pd.DataFrame(list(all_learned_edges), columns=["source", "target"])
        posterior = convert_to_y0_graph(edges_df)
 
    if posterior is not None:
        try:
            int_metrics, per_node_rows = run_interventional(
                config_name, seed, sim, roles, gt_dag, posterior, protein_df,
            )
            row.update(int_metrics)
        except Exception as exc:
            print(f"  [WARN] dagma interventional failed seed={seed}: {exc}")
            row.update(_nan_interventional())
    else:
        row.update(_nan_interventional())
 
    return row, per_node_rows
 
 
def run_causomic(
    seed: int,
    n_samples: int,
    protein_df: pd.DataFrame,
    indra_df: pd.DataFrame,
    real_nodes: set[str],
    gt_dag: nx.DiGraph,
    sim: dict,
    roles: dict,
    scoring_function: type,
    method_label: str,
    dirs: dict[str, Path],
    tag: str,
) -> tuple[dict[str, Any], list[dict]]:
    config_name = f"{method_label}_n{n_samples}"
 
    posterior, _bootstraps = estimate_posterior_dag(
        protein_df,
        indra_priors=indra_df,
        prior_strength=CAUSOMIC_PRIOR_STRENGTH,
        scoring_function=scoring_function,
        search_algorithm=SparseHillClimb,
        n_bootstrap=CAUSOMIC_N_BOOTSTRAP,
        add_high_corr_edges_to_priors=False,
        corr_threshold=0.5,
        edge_probability=CAUSOMIC_EDGE_PROBABILITY,
        convert_to_probability=True,
        return_bootstrap_dags=True,
    )
 
    # Save learned graph
    learned_nx = nx.DiGraph()
    learned_nx.add_edges_from(posterior.directed.edges())
    nx.write_graphml(learned_nx, dirs["graphs"] / f"learned_dag_{method_label}_{tag}.graphml")
 
    # Graph metrics — restricted to real nodes
    pred_edges = {
        (str(u), str(v))
        for u, v in posterior.directed.edges()
        if str(u) in real_nodes and str(v) in real_nodes
    }
    all_nodes = set(str(n) for n in protein_df.columns)
    graph_metrics = compute_graph_metrics(pred_edges, gt_dag, real_nodes)
 
    row: dict[str, Any] = {
        "config": config_name,
        "method": method_label,
        "seed": seed,
        "n_samples": n_samples,
        **graph_metrics,
    }
    per_node_rows: list[dict] = []
 
    # Interventional
    try:
        int_metrics, per_node_rows = run_interventional(
            config_name, seed, sim, roles, gt_dag, posterior, protein_df,
        )
        row.update(int_metrics)
    except Exception as exc:
        print(f"  [WARN] {method_label} interventional failed seed={seed}: {exc}")
        row.update(_nan_interventional())
 
    return row, per_node_rows
 
 
# ---------------------------------------------------------------------------
# Main trial: shared data generation → three method runs
# ---------------------------------------------------------------------------
 
def run_trial(seed: int, n_samples: int) -> tuple[list[dict], list[dict]]:
    tag = f"n{n_samples}_s{seed}"
    dirs = _ensure_dirs(OUTPUT_DIR, tag)
 
    # ── 1. Ground-truth DAG ──────────────────────────────────────────────
    gt_dag, roles = generate_structured_dag(**DAG_PARAMS, seed=seed)
    start_nodes = roles["start"]
    end_nodes = roles["end"]
 
    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * FAKE_NODE_MULTIPLIER))
    n_fake_edges = max(1, round(n_real_edges * FAKE_EDGE_MULTIPLIER))
 
    # ── 2. INDRA noise ───────────────────────────────────────────────────
    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=0.0,
        p_mediated_shortcut=0.1,
        preferential_attachment=True,
    )
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]
 
    # ── 3. Augmented DAG (GT + isolated spurious nodes) ──────────────────
    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)
 
    # ── 4. Simulate data ─────────────────────────────────────────────────
    sim = simulate_data(
        augmented_dag,
        n=n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])
    real_nodes = set(str(n) for n in gt_dag.nodes())
 
    # ── 5. Save shared artifacts ─────────────────────────────────────────
    nx.write_graphml(gt_dag, dirs["graphs"] / f"gt_dag_{tag}.graphml")
    nx.write_graphml(indra_dag, dirs["graphs"] / f"indra_dag_{tag}.graphml")
    nx.write_graphml(augmented_dag, dirs["graphs"] / f"augmented_dag_{tag}.graphml")
    protein_df.to_csv(dirs["data"] / f"protein_df_{tag}.csv", index=False)
    indra_df.to_csv(dirs["data"] / f"indra_df_{tag}.csv", index=False)
 
    # Shared metadata for every row
    shared = {
        "n_nodes": n_real_nodes,
        "n_edges": n_real_edges,
        "n_start": len(start_nodes),
        "n_end": len(end_nodes),
        "n_spurious_nodes": len(spurious_nodes),
    }
 
    all_rows: list[dict] = []
    all_per_node: list[dict] = []
 
    # ── 6. DAGMA ─────────────────────────────────────────────────────────
    print(f"  Running DAGMA ...", flush=True)
    row, per_node = run_dagma(
        seed, n_samples, protein_df, real_nodes,
        gt_dag, sim, roles, dirs, tag,
    )
    row.update(shared)
    all_rows.append(row)
    all_per_node.extend(per_node)
 
    # ── 7. Causomic — BICGaussNoPriors ───────────────────────────────────
    print(f"  Running Causomic (BICGaussNoPriors) ...", flush=True)
    row, per_node = run_causomic(
        seed, n_samples, protein_df, indra_df, real_nodes,
        gt_dag, sim, roles,
        scoring_function=BICGaussNoPriors,
        method_label="causomic_no_prior",
        dirs=dirs, tag=tag,
    )
    row.update(shared)
    all_rows.append(row)
    all_per_node.extend(per_node)
 
    # ── 8. Causomic — BICGaussIndraPriors ────────────────────────────────
    print(f"  Running Causomic (BICGaussIndraPriors) ...", flush=True)
    row, per_node = run_causomic(
        seed, n_samples, protein_df, indra_df, real_nodes,
        gt_dag, sim, roles,
        scoring_function=BICGaussIndraPriors,
        method_label="causomic_indra_prior",
        dirs=dirs, tag=tag,
    )
    row.update(shared)
    all_rows.append(row)
    all_per_node.extend(per_node)
 
    return all_rows, all_per_node
 
 
# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    args = parser.parse_args()
 
    print(f"=== Unified benchmark: seed={args.seed}, n_samples={args.n_samples} ===",
          flush=True)
 
    rows, per_node_rows = run_trial(args.seed, args.n_samples)
 
    # Write results — one file per job
    tag = f"n{args.n_samples}_s{args.seed}"
    results_dir = OUTPUT_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
 
    results_path = results_dir / f"results_{tag}.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)
 
    if per_node_rows:
        nodes_path = results_dir / f"nodes_{tag}.csv"
        pd.DataFrame(per_node_rows).to_csv(nodes_path, index=False)
 
    for r in rows:
        print(f"  [{r['method']}] precision={r['precision']:.3f}  "
              f"recall={r['recall']:.3f}  f1={r['f1']:.3f}  "
              f"accuracy={r['accuracy']:.3f}", end="")
        if not np.isnan(r.get("int_rmse", float("nan"))):
            print(f"  |  int_rmse={r['int_rmse']:.4f}  "
                  f"int_dir_acc={r['int_direction_accuracy']:.3f}", end="")
        print()
 
    print(f"Saved to {results_path}")
 