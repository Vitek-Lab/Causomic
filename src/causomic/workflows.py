"""Workflow entrypoints for causomic network estimation pipelines.

This module exposes two packaged pipelines plus lightweight file logging
utilities for tracking graph sizes across workflow steps:

- `run_toxicity_detection_workflow`: drug-name + DILI-target use case, resolves
  target/outcome nodes via `query_drug_targets`/`query_effect_nodes`. Chains
  `query_forward_paths` -> `estimate_posterior_dag` -> `repair_confounding`.
- `run_causal_workflow`: generic counterpart for when target/outcome nodes are
  already known gene symbols. Same call order, plus `LVM.fit`/
  `LVM.intervention` appended, returning predicted vs. baseline outcome values.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from y0.graph import NxMixedGraph

from causomic.causal_model import LVM
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    SparseHillClimb,
)
from causomic.graph_construction.utils_nx import (
    query_drug_targets,
    query_effect_nodes,
    query_forward_paths,
)
from causomic.network import estimate_posterior_dag, repair_confounding


def _graph_counts(
    graph_like: Any,
) -> Tuple[Optional[int], Optional[int], Optional[Tuple[int, int]]]:
    """Return graph node/edge counts for supported graph-like objects."""
    if isinstance(graph_like, pd.DataFrame) and {"source", "target"}.issubset(graph_like.columns):
        node_count = len(pd.unique(graph_like[["source", "target"]].values.ravel()))
        edge_count = len(graph_like)
        return node_count, edge_count, None

    if isinstance(graph_like, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
        return graph_like.number_of_nodes(), graph_like.number_of_edges(), None

    if hasattr(graph_like, "directed") and hasattr(graph_like, "undirected"):
        directed = graph_like.directed
        undirected = graph_like.undirected
        node_count = len(set(directed.nodes).union(set(undirected.nodes)))
        directed_edges = directed.number_of_edges()
        undirected_edges = undirected.number_of_edges()
        edge_count = directed_edges + undirected_edges
        return node_count, edge_count, (directed_edges, undirected_edges)

    return None, None, None


def _normalize_drug_name(drug_name: Any) -> str:
    """Normalize `drug_name` input into a stable log-friendly string."""
    if isinstance(drug_name, str):
        return drug_name
    if isinstance(drug_name, (list, tuple, set, np.ndarray)):
        return ",".join(str(d) for d in drug_name)
    return str(drug_name)


def _append_log_line(line: str, log_file: Optional[str]) -> None:
    """Append a single line to the workflow log file if logging is enabled."""
    if log_file is None:
        return

    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line)


def _log_workflow_targets(
    drug_name: Any,
    main_drug_targets: Sequence[str],
    dili_targets: Sequence[str],
    log_file: Optional[str],
) -> None:
    """Log run metadata for the current workflow execution."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drug_value = _normalize_drug_name(drug_name)
    line = (
        f"{timestamp}\tworkflow_inputs\tdrug_name={drug_value}"
        f"\tmain_drug_targets={len(main_drug_targets)}\tdili_targets={len(dili_targets)}\n"
    )
    _append_log_line(line, log_file)


def _log_graph_step(
    step_name: str,
    graph_like: Any,
    log_file: Optional[str],
    drug_name: Optional[Any] = None,
) -> None:
    """Log graph node/edge counts for one workflow step."""

    node_count, edge_count, split_counts = _graph_counts(graph_like)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    drug_info = ""
    if drug_name is not None:
        drug_value = _normalize_drug_name(drug_name)
        drug_info = f"\tdrug_name={drug_value}"

    if node_count is None or edge_count is None:
        line = f"{timestamp}\t{step_name}{drug_info}\tnodes=NA\tedges=NA\n"
    elif split_counts is None:
        line = f"{timestamp}\t{step_name}{drug_info}\tnodes={node_count}\tedges={edge_count}\n"
    else:
        directed_edges, undirected_edges = split_counts
        line = (
            f"{timestamp}\t{step_name}{drug_info}\tnodes={node_count}\tedges={edge_count}"
            f"\tdirected_edges={directed_edges}\tundirected_edges={undirected_edges}\n"
        )

    _append_log_line(line, log_file)


def run_toxicity_detection_workflow(
    input_data,
    indra_graph,
    drug_name,
    drug_target_evidence_count_threshold=2,
    dili_target_evidence_count_threshold=2,
    number_of_mediators=3,
    mediator_evidence_count_threshold=None,
    log_file="graph_step_counts.log",
):
    """Run the toxicity workflow and log graph size diagnostics.

    Parameters
    ----------
    input_data : pd.DataFrame
        Proteomics input matrix where columns are measured proteins.
    indra_graph : nx.DiGraph
        INDRA prior graph with edge evidence attributes.
    drug_name : str or sequence of str
        Drug name(s) used to retrieve candidate targets.
    drug_target_evidence_count_threshold : int, default=2
        Minimum total evidence required for selected drug targets.
    dili_target_evidence_count_threshold : int, default=2
        Minimum evidence required for DILI disease targets.
    number_of_mediators : int, default=3
        Maximum number of mediators to allow in forward-path querying.
    mediator_evidence_count_threshold : list[int] or None, default=None
        Evidence thresholds per mediator level; defaults to [1, 1, 1, 2].
    log_file : str or None, default="graph_step_counts.log"
        Output path for workflow diagnostics log. Set to None to disable logging.

    Notes
    -----
    The same full `input_data` is used both for DAG structure learning
    (`estimate_posterior_dag`) and for the downstream data passed back to the
    caller for model fitting — there is no train/test split. This is
    intentional: `estimate_posterior_dag` already bootstraps over the data
    internally to quantify structure-learning uncertainty, so holding out a
    separate fitting split would only shrink the sample available to the
    (typically data-hungry) causal model fit without buying additional
    protection against overfitting the graph.
    """
    if mediator_evidence_count_threshold is None:
        mediator_evidence_count_threshold = [1, 1, 1, 2]

    measured_proteins = input_data.columns.tolist()

    # Use the full dataset for both graph learning and the downstream fit
    # (see "Notes" above for why this isn't split).
    input_data_graph = input_data.copy()

    # Extra drug targets
    main_drug_targets = query_drug_targets(
        indra_graph, drug_name, target_ev_filter=drug_target_evidence_count_threshold
    )
    main_drug_targets = (
        main_drug_targets.loc[(main_drug_targets["target"].isin(measured_proteins))]
        .drop_duplicates()["target"]
        .values
    )

    if len(main_drug_targets) == 0:
        raise ValueError(
            f"No drug targets found for {drug_name} with evidence count >= {drug_target_evidence_count_threshold}"
        )

    # Extract DILI nodes
    dili_targets = query_effect_nodes(
        indra_graph,
        "Chemical and Drug Induced Liver Injury",
        target_ev_filter=dili_target_evidence_count_threshold,
    )
    dili_targets = (
        dili_targets[(dili_targets["source"].isin(measured_proteins))]
        .drop_duplicates()["source"]
        .values
    )

    mapping = {node: node.replace("-", "") for node in indra_graph.nodes()}
    indra_graph = nx.relabel_nodes(indra_graph, mapping)
    main_drug_targets = [ct.replace("-", "") for ct in main_drug_targets]
    dili_targets = [dt.replace("-", "") for dt in dili_targets]
    _log_workflow_targets(drug_name, main_drug_targets, dili_targets, log_file)

    indra_prior = query_forward_paths(
        graph=indra_graph,
        start_nodes=main_drug_targets,
        end_nodes=dili_targets,
        n_mediators=number_of_mediators,
        med_ev_filter=mediator_evidence_count_threshold,
    )
    _log_graph_step("indra_prior", indra_prior, log_file, drug_name=drug_name)

    indra_nodes = pd.unique(indra_prior[["source", "target"]].values.ravel())

    input_data_graph.columns = input_data_graph.columns.str.replace("-", "")
    input_data_graph = input_data_graph.loc[
        :, input_data_graph.columns.str.replace("-", "").isin(indra_nodes)
    ]

    posterior_network = estimate_posterior_dag(
        input_data_graph,
        indra_prior,
        5,
        BICGaussIndraPriors,
        SparseHillClimb,
        100,
        False,
        0.5,
        0.5,
    )
    _log_graph_step("posterior_network", posterior_network, log_file, drug_name=drug_name)

    repaired_network = repair_confounding(
        input_data_graph,
        posterior_network,
        indra_graph,
        max_conditional=2,
        confounder_evidence=5,
    )
    _log_graph_step("repaired_network", repaired_network, log_file, drug_name=drug_name)

    return indra_prior, posterior_network, repaired_network


def run_causal_workflow(
    data: pd.DataFrame,
    indra_graph: nx.DiGraph,
    target_nodes: Sequence[str],
    outcome_nodes: Sequence[str],
    intervention: Dict[str, float],
    evidence_count_threshold: int = 1,
    number_of_mediators: int = 1,
    mediator_evidence_count_threshold: Optional[Sequence[int]] = None,
    mediator_source_count_threshold: Optional[Sequence[int]] = None,
    prior_strength: float = 5.0,
    n_bootstrap: int = 100,
    corr_threshold: float = 0.5,
    edge_probability: float = 0.5,
    repair_confounders: bool = True,
    max_conditional: int = 2,
    confounder_evidence: int = 5,
    lvm_backend: str = "pyro",
    lvm_kwargs: Optional[Dict[str, Any]] = None,
    compare_value: float = 0.0,
    predictive_samples: int = 100,
    log_file: Optional[str] = "graph_step_counts.log",
) -> Tuple[pd.DataFrame, NxMixedGraph, NxMixedGraph, LVM, pd.DataFrame]:
    """Chain prior extraction through interventional prediction for known nodes.

    Generic counterpart to `run_toxicity_detection_workflow`: instead of
    resolving targets/outcomes from a drug name and a DILI lookup via
    `query_drug_targets`/`query_effect_nodes`, this function takes gene-symbol
    node lists directly. It reuses the same call order as
    `run_toxicity_detection_workflow` — `query_forward_paths` ->
    `estimate_posterior_dag` -> `repair_confounding` — and appends
    `LVM.fit` / `LVM.intervention`, returning predicted vs. baseline values
    for the outcome nodes.

    Parameters
    ----------
    data : pd.DataFrame
        Wide-format proteomics data (genes as columns, samples as rows). The
        full dataset is used for both graph learning and the downstream LVM
        fit — see the "Notes" section of `run_toxicity_detection_workflow`
        for why that's the intended behavior rather than an oversight.
    indra_graph : nx.DiGraph
        INDRA prior graph with edge evidence attributes (e.g. produced by
        `causomic.graph_construction.prepare_graph`).
    target_nodes : sequence of str
        Known upstream gene symbols to use as `start_nodes` for
        `query_forward_paths` and as the intervention target set.
    outcome_nodes : sequence of str
        Known downstream gene symbols to use as `end_nodes` for
        `query_forward_paths` and as the outcome nodes for the final
        `LVM.intervention` call.
    intervention : dict[str, float]
        Intervention values passed to `LVM.intervention`. These are ABSOLUTE
        target-scale values in the same units as `data` (e.g. log2
        abundance), not deltas or fold-changes — see `LVM.intervention` for
        the fold-change conversion recipe.
    evidence_count_threshold : int, default=1
        Evidence threshold used to build a uniform
        `mediator_evidence_count_threshold` when one isn't supplied.
    number_of_mediators : int, default=1
        Maximum number of mediators allowed between target and outcome nodes;
        forwarded to `query_forward_paths` as `n_mediators`.
    mediator_evidence_count_threshold, mediator_source_count_threshold : sequence of int, optional
        Per-depth thresholds forwarded to `query_forward_paths` as
        `med_ev_filter` / `med_src_filter`. Each defaults to a uniform list
        of length `number_of_mediators + 1` (using `evidence_count_threshold`
        and 1, respectively) when not supplied.
    prior_strength, n_bootstrap, corr_threshold, edge_probability
        Forwarded to `estimate_posterior_dag`.
    repair_confounders : bool, default=True
        Whether to run `repair_confounding` on the posterior DAG before
        fitting. Set to False to skip straight from the posterior DAG to
        model fitting (cheaper, but leaves unresolved latent confounding
        exactly as `estimate_posterior_dag` found it).
    max_conditional, confounder_evidence
        Forwarded to `repair_confounding` (only used if `repair_confounders`
        is True).
    lvm_backend : str, default="pyro"
        Backend passed to `LVM(backend=...)`. The "numpyro" backend's
        intervention path is experimental (see `LVM`) and is not recommended
        here.
    lvm_kwargs : dict, optional
        Additional keyword arguments forwarded to the `LVM` constructor
        (e.g. `num_steps`, `seed`).
    compare_value, predictive_samples
        Forwarded to `LVM.intervention`.
    log_file : str or None, default="graph_step_counts.log"
        Output path for workflow diagnostics log. Set to None to disable.

    Returns
    -------
    indra_prior : pd.DataFrame
        Forward-path edges extracted from `indra_graph`.
    posterior_network : NxMixedGraph
        DAG estimated by `estimate_posterior_dag`.
    fitted_network : NxMixedGraph
        The DAG actually used to fit the LVM: equal to `posterior_network`
        if `repair_confounders=False`, otherwise the `repair_confounding`
        output.
    lvm : LVM
        The fitted latent-variable model.
    outcome_summary : pd.DataFrame
        Indexed by outcome node, with columns "baseline" and "intervention"
        (predicted posterior means) and "effect" (their difference).
    """
    if mediator_evidence_count_threshold is None:
        mediator_evidence_count_threshold = [evidence_count_threshold] * (number_of_mediators + 1)
    if mediator_source_count_threshold is None:
        mediator_source_count_threshold = [1] * (number_of_mediators + 1)

    target_nodes = [str(t).replace("-", "") for t in target_nodes]
    outcome_nodes = [str(o).replace("-", "") for o in outcome_nodes]

    mapping = {node: node.replace("-", "") for node in indra_graph.nodes()}
    indra_graph = nx.relabel_nodes(indra_graph, mapping)

    input_data_graph = data.copy()
    input_data_graph.columns = input_data_graph.columns.str.replace("-", "")

    indra_prior = query_forward_paths(
        graph=indra_graph,
        start_nodes=target_nodes,
        end_nodes=outcome_nodes,
        n_mediators=number_of_mediators,
        med_ev_filter=list(mediator_evidence_count_threshold),
        med_src_filter=list(mediator_source_count_threshold),
    )
    _log_graph_step("indra_prior", indra_prior, log_file)

    if indra_prior.empty:
        raise ValueError(
            "No INDRA paths found between target_nodes and outcome_nodes; "
            "try raising number_of_mediators or lowering evidence_count_threshold."
        )

    indra_nodes = pd.unique(indra_prior[["source", "target"]].values.ravel())
    input_data_graph = input_data_graph.loc[:, input_data_graph.columns.isin(indra_nodes)]

    posterior_network = estimate_posterior_dag(
        data=input_data_graph,
        indra_priors=indra_prior,
        prior_strength=prior_strength,
        scoring_function=BICGaussIndraPriors,
        search_algorithm=SparseHillClimb,
        n_bootstrap=n_bootstrap,
        add_high_corr_edges_to_priors=False,
        corr_threshold=corr_threshold,
        edge_probability=edge_probability,
    )
    _log_graph_step("posterior_network", posterior_network, log_file)

    if repair_confounders:
        fitted_network = repair_confounding(
            input_data_graph,
            posterior_network,
            indra_graph,
            max_conditional=max_conditional,
            confounder_evidence=confounder_evidence,
        )
        _log_graph_step("repaired_network", fitted_network, log_file)
    else:
        fitted_network = posterior_network

    # repair_confounding can introduce confounder nodes that aren't measured
    # in `data`; reindexing adds them as all-NaN columns so LVM's model-based
    # missing-data imputation handles them like any other partially-observed
    # node instead of raising a KeyError during fitting.
    fitted_nodes = {str(n) for n in fitted_network.directed.nodes()} | {
        str(n) for n in fitted_network.undirected.nodes()
    }
    fit_data = input_data_graph.reindex(columns=sorted(fitted_nodes))

    lvm = LVM(backend=lvm_backend, **(lvm_kwargs or {}))
    lvm.fit(fit_data, fitted_network)

    lvm.intervention(
        intervention,
        outcome_node=list(outcome_nodes),
        compare_value=compare_value,
        predictive_samples=predictive_samples,
    )

    outcome_summary = pd.DataFrame(
        {
            "baseline": lvm.posterior_samples.mean(),
            "intervention": lvm.intervention_samples.mean(),
        }
    )
    outcome_summary["effect"] = outcome_summary["intervention"] - outcome_summary["baseline"]

    return indra_prior, posterior_network, fitted_network, lvm, outcome_summary
