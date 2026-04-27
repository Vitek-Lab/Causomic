"""
Benchmarking script for DECI (causica) against ground-truth simulated SEMs.

Uses the *same* simulation infrastructure as the Causomic benchmark
(generate_structured_dag, simulate_data, ground_truth_interventional_effect)
but replaces Causomic's posterior estimation + LVM pipeline with Microsoft's
DECI model from the causica package.

Evaluates:
  - Graph structure: Precision, Recall, F1, Accuracy vs ground-truth DAG.
  - Interventional effects: RMSE, MAE, Pearson r, direction accuracy of
    E[Y|do(start=1)] - E[Y|do(start=0)] vs the linear SEM ground truth.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from tensordict import TensorDict

from causica.lightning.data_modules.basic_data_module import BasicDECIDataModule
from causica.lightning.data_modules.deci_data_module import DECIDataModule
from causica.lightning.modules.deci_module import DECIModule
from causica.lightning.callbacks import AuglagLRCallback
from causica.distributions.noise.joint import ContinuousNoiseDist
from causica.datasets.tensordict_utils import tensordict_from_pandas
from causica.datasets.variable_types import VariableTypeEnum

# Causomic simulation utilities (unchanged — we only use the data generation)
from causomic.simulation.proteomics_simulator import simulate_data
from causomic.simulation.random_network import (
    generate_indra_data,
    generate_structured_dag,
    ground_truth_interventional_effect,
)

warnings.filterwarnings("ignore")

# Number of graph samples used when extracting the posterior adjacency matrix
# and when computing interventional expectations.
NUM_GRAPH_SAMPLES = 50
NUM_MC_SAMPLES = 2000


# ---------------------------------------------------------------------------
# Fallback DataModule — used if BasicDECIDataModule is unavailable or its
# constructor signature differs across causica versions.
# ---------------------------------------------------------------------------

class PandasDECIDataModule(DECIDataModule):
    """Minimal DECIDataModule wrapping a pandas DataFrame of continuous variables."""

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = 256,
        train_frac: float = 0.8,
    ):
        super().__init__()
        self._df = df
        self._batch_size = batch_size

        td = tensordict_from_pandas(df)            # {col: [N, 1]} TensorDict
        n = len(df)
        n_train = int(n * train_frac)
        idx = torch.randperm(n)
        self._dataset_train = td[idx[:n_train]]
        self._dataset_test = td[idx[n_train:]]

        self._variable_shapes = {col: torch.Size([1]) for col in df.columns}
        self._variable_types = {
            col: VariableTypeEnum.CONTINUOUS for col in df.columns
        }

    # --- required abstract properties ---
    @property
    def dataset_name(self) -> str:
        return "pandas_data"

    @property
    def dataset_train(self) -> TensorDict:
        return self._dataset_train

    @property
    def dataset_test(self) -> TensorDict:
        return self._dataset_test

    @property
    def variable_shapes(self) -> dict[str, torch.Size]:
        return self._variable_shapes

    @property
    def variable_types(self) -> dict[str, VariableTypeEnum]:
        return self._variable_types
    
    @property
    def column_names(self) -> list[str]:
        """The names of the columns in the dataset."""
        return list(self._df.columns)

    # --- dataloaders ---
    def train_dataloader(self):
        from causica.datasets.tensordict_utils import identity
        return torch.utils.data.DataLoader(
            self._dataset_train,
            batch_size=self._batch_size,
            shuffle=True,
            collate_fn=identity,
        )

    def test_dataloader(self):
        from causica.datasets.tensordict_utils import identity
        return torch.utils.data.DataLoader(
            self._dataset_test,
            batch_size=len(self._dataset_test),
            shuffle=False,
            collate_fn=identity,
        )


def make_data_module(
    df: pd.DataFrame, batch_size: int = 256, train_frac: float = 0.8
) -> DECIDataModule:
    """Try BasicDECIDataModule first; fall back to our manual wrapper."""
    try:
        dm = BasicDECIDataModule(df=df, batch_size=batch_size, train_frac=train_frac)
        # Quick sanity check — make sure it exposes the required properties
        _ = dm.variable_shapes
        return dm
    except Exception:
        return PandasDECIDataModule(df, batch_size=batch_size, train_frac=train_frac)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Defines one DECI benchmarking scenario.

    Each config is swept over `seeds` — one trial per seed.
    """
    name: str

    # DAG generation params (passed to generate_structured_dag)
    dag_params: dict = field(default_factory=lambda: dict(
        n_start=30,
        n_end=8,
        max_mediators=3,
        shared_mediator_prob=0.5,
        confounder_prob=0.05,
    ))

    # INDRA noise multipliers (still used to generate the *augmented* simulation
    # graph — DECI does NOT see the INDRA priors, just the simulated data)
    fake_node_multiplier: float = 1.0
    fake_edge_multiplier: float = 3.0

    # Simulation
    n_samples: int = 250

    # DECI hyper-parameters
    noise_dist: ContinuousNoiseDist = ContinuousNoiseDist.GAUSSIAN
    embedding_size: int = 32
    out_dim_g: int = 32
    num_layers_g: int = 2
    num_layers_zeta: int = 2
    prior_sparsity_lambda: float = 0.05
    max_epochs: int = 500
    batch_size: int = 256

    # Graph thresholding
    adj_threshold: float = 0.5

    # Seeds to run
    seeds: list[int] = field(default_factory=lambda: list(range(10)))

    # Interventional benchmark
    run_interventional: bool = True


# ---------------------------------------------------------------------------
# Graph metric computation (unchanged from Causomic benchmark)
# ---------------------------------------------------------------------------

def compute_metrics(
    learned_adj: np.ndarray,
    gt_dag: nx.DiGraph,
    column_names: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Precision / Recall / F1 / Accuracy of DECI's learned graph vs gt_dag."""

    binary = (learned_adj > threshold).astype(int)
    n = len(column_names)

    pred_edges = set()
    for i in range(n):
        for j in range(n):
            if binary[i, j]:
                pred_edges.add((column_names[i], column_names[j]))

    gt_edges = set((str(u), str(v)) for u, v in gt_dag.edges())
    all_nodes = set(str(c) for c in column_names)
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
# Interventional metric computation (unchanged from Causomic benchmark)
# ---------------------------------------------------------------------------

def compute_interventional_metrics(
    config_name: str,
    seed: int,
    ci_series: pd.Series,
    gt_series: pd.Series,
) -> tuple[dict[str, float], list[dict]]:
    """Summary metrics + per-node detail rows for interventional predictions."""
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
# DECI helper: extract posterior mean adjacency
# ---------------------------------------------------------------------------

def get_posterior_adj(deci: DECIModule, n_samples: int = NUM_GRAPH_SAMPLES) -> np.ndarray:
    """Sample graphs from DECI's variational posterior and return the mean adjacency."""
    sems = deci.sem_module().sample(torch.Size([n_samples]))
    adj_matrices = torch.stack([sem.graph for sem in sems])
    return adj_matrices.float().mean(dim=0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# DECI helper: estimate ATE via do-calculus on the learned SEM
# ---------------------------------------------------------------------------

def deci_ate(
    deci: DECIModule,
    column_names: list[str],
    intervention_nodes: dict[str, float],
    outcome_nodes: list[str],
    n_graph_samples: int = NUM_GRAPH_SAMPLES,
    n_mc_samples: int = NUM_MC_SAMPLES,
) -> pd.Series:
    """Estimate E[outcome | do(intervention_nodes)] using DECI's learned SEM.

    Returns a pd.Series indexed by outcome_nodes.
    """
    # Build intervention TensorDict: {node_name: tensor([value])}
    int_td = TensorDict(
        {name: torch.tensor([val], dtype=torch.float32)
         for name, val in intervention_nodes.items()},
        batch_size=torch.Size([]),
    )

    sems = list(deci.sem_module().sample(torch.Size([n_graph_samples])))

    outcome_means = {node: [] for node in outcome_nodes}

    for sem in sems:
        # Apply do-operator: returns a new SEM with mutilated graph
        intervened_sem = sem.do(int_td)

        # Sample from the interventional distribution
        noise = intervened_sem.sample_noise(torch.Size([n_mc_samples]))
        samples = intervened_sem.noise_to_sample(noise)  # TensorDict

        for node in outcome_nodes:
            vals = samples[node].detach().cpu().numpy().flatten()
            outcome_means[node].append(vals.mean())

    return pd.Series(
        {node: np.mean(outcome_means[node]) for node in outcome_nodes},
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Interventional evaluation
# ---------------------------------------------------------------------------

def _run_interventional(
    cfg: BenchmarkConfig,
    seed: int,
    sim: dict,
    roles: dict,
    gt_dag: nx.DiGraph,
    deci: DECIModule,
    column_names: list[str],
) -> tuple[dict[str, float], list[dict]]:
    """Run paired do(start=1) vs do(start=0) and compare to ground truth."""

    outcome = roles["end"]
    intervention0 = {node: 1.0 for node in roles["start"]}
    intervention1 = {node: 0.0 for node in roles["start"]}

    # DECI predictions
    torch.manual_seed(seed)
    e_y_do0 = deci_ate(deci, column_names, intervention0, outcome)
    torch.manual_seed(seed)
    e_y_do1 = deci_ate(deci, column_names, intervention1, outcome)
    ci_results = e_y_do0 - e_y_do1   # same sign convention as Causomic bench

    # Ground-truth from the linear SEM
    gt_int0 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes={n: 1 for n in roles["start"]},
        output_nodes=outcome,
    )
    gt_int1 = ground_truth_interventional_effect(
        gt_dag, sim["Coefficients"],
        intervention_nodes={n: 0 for n in roles["start"]},
        output_nodes=outcome,
    )
    gt_results = {k: gt_int0["effect"][k] - gt_int1["effect"][k]
                  for k in gt_int1["effect"]}

    ci_series = pd.Series(ci_results, dtype=float).reindex(outcome)
    gt_series = pd.Series(gt_results, dtype=float).reindex(outcome)

    return compute_interventional_metrics(cfg.name, seed, ci_series, gt_series)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

def run_trial(cfg: BenchmarkConfig, seed: int) -> tuple[dict[str, Any], list[dict]]:
    """Run one simulation → DECI training → evaluation trial."""

    # 1. Ground-truth DAG
    gt_dag, roles = generate_structured_dag(**cfg.dag_params, seed=seed)

    n_real_nodes = gt_dag.number_of_nodes()
    n_real_edges = gt_dag.number_of_edges()
    n_fake_nodes = max(1, round(n_real_nodes * cfg.fake_node_multiplier))
    n_fake_edges = max(1, round(n_real_edges * cfg.fake_edge_multiplier))

    # 2. Build augmented graph with spurious nodes (DECI won't see INDRA priors,
    #    but we still add the extra nodes to make the data matrix comparable)
    indra_dag, _indra_df, _missing = generate_indra_data(
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

    # 3. Simulate proteomics data
    sim = simulate_data(
        augmented_dag,
        n=cfg.n_samples,
        add_feature_var=False,
        add_error=True,
        seed=seed,
    )
    protein_df = pd.DataFrame(sim["Protein_data"])
    column_names = list(protein_df.columns)

    # 4. Prepare DECI data module
    data_module = make_data_module(protein_df, batch_size=cfg.batch_size)

    # 5. Build DECI model
    deci = DECIModule(
        noise_dist=cfg.noise_dist,
        embedding_size=cfg.embedding_size,
        out_dim_g=cfg.out_dim_g,
        num_layers_g=cfg.num_layers_g,
        num_layers_zeta=cfg.num_layers_zeta,
        prior_sparsity_lambda=cfg.prior_sparsity_lambda,
    )

    # 6. Train
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )
    trainer.fit(deci, datamodule=data_module)

    # 7. Extract learned graph + compute graph metrics
    mean_adj = get_posterior_adj(deci)
    graph_metrics = compute_metrics(mean_adj, gt_dag, column_names, cfg.adj_threshold)

    row: dict[str, Any] = {
        "config": cfg.name,
        "seed": seed,
        "n_real_nodes": n_real_nodes,
        "n_real_edges": n_real_edges,
        "n_fake_nodes": n_fake_nodes,
        "n_fake_edges": n_fake_edges,
        **graph_metrics,
    }
    per_node_rows: list[dict] = []

    # 8. Interventional metrics (optional)
    if cfg.run_interventional:
        try:
            int_metrics, per_node_rows = _run_interventional(
                cfg, seed, sim, roles, gt_dag, deci, column_names,
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
# Benchmark runner (unchanged from Causomic benchmark)
# ---------------------------------------------------------------------------

def run_benchmark(
    configs: list[BenchmarkConfig], verbose: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    all_per_node: list[dict] = []
    total = sum(len(c.seeds) for c in configs)
    done = 0

    for cfg in configs:
        for seed in cfg.seeds:
            done += 1
            if verbose:
                print(f"[{done}/{total}] config={cfg.name!r}  seed={seed} ...",
                      flush=True)
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
# Benchmark configurations  ← edit / extend here
# ---------------------------------------------------------------------------

_DAG_PARAMS = dict(
    n_start=5,
    n_end=3,
    max_mediators=3,
    shared_mediator_prob=0.4,
    confounder_prob=0.0,
    end_node_alpha=0.8,
)

_SEEDS = list(range(2))

_N_LOW  =  50
_N_MID  = 200
_N_HIGH = 5000

BENCHMARK_CONFIGS: list[BenchmarkConfig] = [

    # ── Sample-size sweep (Gaussian noise, moderate sparsity) ───────────
    # BenchmarkConfig(
    #     name="DECI_gaussian_low_replicates",
    #     dag_params=_DAG_PARAMS,
    #     fake_node_multiplier=2.0,
    #     fake_edge_multiplier=5.0,
    #     n_samples=_N_LOW,
    #     noise_dist=ContinuousNoiseDist.GAUSSIAN,
    #     prior_sparsity_lambda=0.1,
    #     max_epochs=500,
    #     seeds=_SEEDS,
    # ),
    # BenchmarkConfig(
    #     name="DECI_gaussian_mid_replicates",
    #     dag_params=_DAG_PARAMS,
    #     fake_node_multiplier=2.0,
    #     fake_edge_multiplier=5.0,
    #     n_samples=_N_MID,
    #     noise_dist=ContinuousNoiseDist.GAUSSIAN,
    #     prior_sparsity_lambda=0.1,
    #     max_epochs=500,
    #     seeds=_SEEDS,
    # ),
    BenchmarkConfig(
        name="DECI_gaussian_high_replicates",
        dag_params=_DAG_PARAMS,
        fake_node_multiplier=2.0,
        fake_edge_multiplier=5.0,
        n_samples=_N_HIGH,
        noise_dist=ContinuousNoiseDist.GAUSSIAN,
        prior_sparsity_lambda=0.1,
        max_epochs=500,
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

    out_path = "DECI_benchmark_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nTrial results saved to {out_path}")

    if not node_results.empty:
        node_out_path = "DECI_benchmark_interventional_nodes.csv"
        node_results.to_csv(node_out_path, index=False)
        print(f"Per-node interventional results saved to {node_out_path}")