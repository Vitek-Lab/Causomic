"""
DAGMA baseline benchmark — single trial.

Usage:
    python benchmark_graph_baseline.py --seed 0 --n_samples 1000
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
from causomic.graph_construction.repair import convert_to_y0_graph
from causomic.simulation.proteomics_simulator import simulate_data
from causomic.simulation.random_network import (
    generate_indra_data,
    generate_structured_dag,
    ground_truth_interventional_effect,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# DAG params (fixed across all runs)
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
LVM_NUM_STEPS = 1000
LVM_PREDICTIVE_SAMPLES = 200


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    learned_edges: set[tuple[str, str]],
    gt_dag: nx.DiGraph,
    all_nodes: set[str],
) -> dict[str, float]:
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
# Interventional runner
# ---------------------------------------------------------------------------

def _run_interventional(
    seed: int,
    n_samples: int,
    sim: dict,
    roles: dict,
    gt_dag: nx.DiGraph,
    posterior,
    real_protein_df: pd.DataFrame,
) -> tuple[dict[str, float], list[dict]]:
    config_name = f"dagma_n{n_samples}"
    outcome = roles["end"]
    intervention0 = {node: 1 for node in roles["start"]}
    intervention1 = {node: 0 for node in roles["start"]}

    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=LVM_NUM_STEPS, verbose=False)
    lvm.fit(real_protein_df, posterior)

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
# Single trial
# ---------------------------------------------------------------------------

def run_trial(seed: int, n_samples: int) -> tuple[dict[str, Any], list[dict]]:
    config_name = f"dagma_n{n_samples}"

    # 1. Ground-truth DAG
    gt_dag, roles = generate_structured_dag(**DAG_PARAMS, seed=seed)
    start_nodes = roles["start"]
    end_nodes = roles["end"]

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * FAKE_NODE_MULTIPLIER))
    n_fake_edges = max(1, round(n_real_edges * FAKE_EDGE_MULTIPLIER))

    # 2. INDRA noise
    indra_dag, indra_df, _missing = generate_indra_data(
        gt_dag,
        num_incorrect_nodes=n_fake_nodes,
        num_incorrect_edges=n_fake_edges,
        p_missing_real=0.0,
        p_mediated_shortcut=0.1,
        preferential_attachment=True,
    )
    spurious_nodes = [n for n in indra_dag.nodes() if n not in gt_dag.nodes()]

    # 3. Augment GT DAG with spurious isolated nodes
    augmented_dag = gt_dag.copy()
    for xn in spurious_nodes:
        augmented_dag.add_node(xn)

    # 4. Simulate data
    sim = simulate_data(
        augmented_dag,
        n=n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])

    # 5. Real nodes
    real_nodes = set(str(n) for n in gt_dag.nodes())
    real_protein_df = protein_df[[c for c in protein_df.columns if c in real_nodes]]

    # 6. DAGMA
    X = protein_df.values
    node_names = list(protein_df.columns)

    model = DagmaLinear(loss_type="l2")
    W_est = model.fit(X, lambda1=0.02)

    THRESHOLD = 0.3
    W_est[np.abs(W_est) < THRESHOLD] = 0

    learned_dag = nx.DiGraph()
    learned_dag.add_nodes_from(node_names)
    d = len(node_names)
    for i in range(d):
        for j in range(d):
            if W_est[i, j] != 0:
                learned_dag.add_edge(node_names[j], node_names[i])

    learned_real_edges = {
        (str(u), str(v))
        for u, v in learned_dag.edges()
        if str(u) in real_nodes and str(v) in real_nodes
    }
    graph_metrics = compute_metrics(learned_real_edges, gt_dag, real_nodes)

    posterior = None
    if learned_real_edges:
        edges_df = pd.DataFrame(list(learned_real_edges), columns=["source", "target"])
        posterior = convert_to_y0_graph(edges_df)

    # 7. Assemble row
    row: dict[str, Any] = {
        "config": config_name,
        "method": "dagma",
        "seed": seed,
        "n_samples": n_samples,
        "n_nodes": n_real_nodes,
        "n_edges": n_real_edges,
        "n_start": len(start_nodes),
        "n_end": len(end_nodes),
        "n_spurious_nodes": len(spurious_nodes),
        "n_learned_edges_total": len(list(learned_dag.edges())),
        "n_learned_edges_real": len(learned_real_edges),
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 8. Interventional
    if posterior is not None:
        try:
            int_metrics, per_node_rows = _run_interventional(
                seed, n_samples, sim, roles, gt_dag, posterior, real_protein_df
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
    else:
        row.update({
            "int_rmse": float("nan"),
            "int_mae": float("nan"),
            "int_pearson_r": float("nan"),
            "int_direction_accuracy": float("nan"),
            "int_n_outcomes": float("nan"),
        })

    return row, per_node_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n_samples", type=int, required=True)
    args = parser.parse_args()

    print(f"Running DAGMA: seed={args.seed}, n_samples={args.n_samples}", flush=True)

    row, per_node_rows = run_trial(args.seed, args.n_samples)

    # Write results — one file per job to avoid write contention
    tag = f"n{args.n_samples}_s{args.seed}"
    results_path = OUTPUT_DIR / f"results_{tag}.csv"
    nodes_path = OUTPUT_DIR / f"nodes_{tag}.csv"

    pd.DataFrame([row]).to_csv(results_path, index=False)
    if per_node_rows:
        pd.DataFrame(per_node_rows).to_csv(nodes_path, index=False)

    print(f"  precision={row['precision']:.3f}  recall={row['recall']:.3f}  "
          f"f1={row['f1']:.3f}  accuracy={row['accuracy']:.3f}")
    if not np.isnan(row.get("int_rmse", float("nan"))):
        print(f"  int_rmse={row['int_rmse']:.4f}  "
              f"int_dir_acc={row['int_direction_accuracy']:.3f}")
    print(f"Saved to {results_path}")