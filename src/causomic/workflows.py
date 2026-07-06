"""Workflow entrypoints for causomic network estimation pipelines.

This module currently exposes the toxicity detection workflow and lightweight
file logging utilities for tracking graph sizes across workflow steps.
"""

import os
import pickle
from datetime import datetime
from typing import Any, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from causomic.graph_construction.utils_nx import (
    query_effect_nodes,
    query_forward_paths,
    query_drug_targets,
)
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    SparseHillClimb,
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
    """
    if mediator_evidence_count_threshold is None:
        mediator_evidence_count_threshold = [1, 1, 1, 2]

    measured_proteins = input_data.columns.tolist()

    # Randomly split input_data into two equal parts
    # input_data_graph, _ = train_test_split(input_data, test_size=0.5, random_state=42)
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
