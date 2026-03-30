"""Utilities for working with NetworkX graphs produced by INDRA.

This module provides helpers to filter graphs by statement types and
evidence, compute simple path-based queries, and prepare graphs for
downstream analysis. Functions generally accept and return NetworkX
DiGraph objects and, where noted, may modify graphs in-place.

The file intentionally keeps behaviour stable while adding documentation
and small readability improvements.
"""

from typing import Any, Dict, Iterable, List, Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm


def add_evidence_info(graph: nx.DiGraph) -> nx.DiGraph:
    """Compute and attach simple evidence summaries to every edge.

    This function updates the graph in-place by adding an ``evidence``
    attribute for each edge. The attribute is a dict with keys:
      - ``total_evidence``: integer sum of ``evidence_count`` across
        statements (missing/None treated as 0)
      - ``source_evidence``: number of distinct source keys found in the
        statements' ``source_counts`` dicts
      - ``stmt_type``: list of unique statement types found on the edge

    Args:
        graph: DiGraph with edge attribute "statements" containing dicts
            (or similar) describing INDRA statements.

    Returns:
        The same graph object (modified in-place) for convenience.
    """

    for u, v, attrs in graph.edges(data=True):
        stmts = attrs.get("statements", []) or []

        # total evidence count (handles None/missing)
        total_evidence = sum(int(s.get("evidence_count") or 0) for s in stmts)

        # union of all source keys across statements
        source_counts = [
            s.get("source_counts") for s in stmts if isinstance(s.get("source_counts"), dict)
        ]
        source_key_union = (
            set().union(*(sc.keys() for sc in source_counts)) if source_counts else set()
        )
        source_keys = len(source_key_union)

        # unique statement types across statements
        stmt_types = list(set(
            s.get("stmt_type") for s in stmts 
            if s.get("stmt_type") is not None
        ))

        # attach a fresh dict per edge (do not reuse mutable objects)
        new_ev: Dict[str, Any] = {
            "total_evidence": total_evidence,
            "source_evidence": source_keys,
            "stmt_type": stmt_types,
        }
        graph[u][v]["evidence"] = new_ev

    return graph


def filter_graph_by_evidence_count(graph: nx.DiGraph, evidence_count: int) -> nx.DiGraph:
    """Return a subgraph containing only edges whose total evidence is
    at least ``evidence_count``.

    Args:
        graph: DiGraph with edge attribute "evidence" (a dict containing
            "total_evidence"). If edges do not have that attribute, a
            default of 0 is assumed.
        evidence_count: Minimum required total evidence to keep an edge.

    Returns:
        A new DiGraph containing only the edges that meet the threshold
        and their incident nodes.
    """

    edges_to_keep: List[tuple] = []
    for u, v, attrs in graph.edges(data=True):
        ev = attrs.get("evidence", {}).get("total_evidence", 0)
        if ev >= evidence_count:
            edges_to_keep.append((u, v))

    # Build a new graph containing only those edges
    filtered_graph = graph.edge_subgraph(edges_to_keep).copy()

    return filtered_graph

def prepare_graph(
    graph: nx.DiGraph,
    measured_nodes: List[str],
    node_types: List[str],
    stmt_types: List[str],
) -> nx.DiGraph:
    """Prepare a graph for analysis by selecting node namespace, measured
    nodes, and statement types.

    Steps applied (in order):
      1. Keep only nodes whose ``ns`` attribute is in ``node_types``.
      2. Restrict edges to those connecting measured nodes.
      3. Filter edges to only include statements with ``stmt_types``.
      4. Annotate edges with evidence summary using :func:`add_evidence_info`.

    Args:
        graph: Original DIgraph produced by INDRA/other pipeline.
        measured_nodes: List of nodes that were measured (e.g., columns of
            an input dataset).
        node_types: Allowed node namespace types (e.g., ["HGNC"]).
        stmt_types: Allowed statement types to keep on edges.

    Returns:
        A prepared :class:`networkx.DiGraph` suitable for path queries and
        downstream processing.
    """
    allowed_node_types = set(node_types)
    measured_node_set = set(measured_nodes)
    allowed_stmt_types = set(stmt_types)

    prepared_graph = nx.DiGraph()
    node_attrs = graph.nodes

    for u, v, attrs in graph.edges(data=True):
        if u not in measured_node_set or v not in measured_node_set:
            continue
        if node_attrs[u].get("ns") not in allowed_node_types:
            continue
        if node_attrs[v].get("ns") not in allowed_node_types:
            continue

        stmts = attrs.get("statements", [])
        filtered_statements = []
        total_evidence = 0
        source_key_union = set()
        stmt_type_union = set()

        for stmt in stmts:
            if not isinstance(stmt, dict):
                continue

            stmt_type = stmt.get("stmt_type")
            if stmt_type not in allowed_stmt_types:
                continue

            filtered_statements.append(stmt)
            total_evidence += int(stmt.get("evidence_count") or 0)
            stmt_type_union.add(stmt_type)

            source_counts = stmt.get("source_counts")
            if isinstance(source_counts, dict):
                source_key_union.update(source_counts.keys())

        if not filtered_statements:
            continue

        if not prepared_graph.has_node(u):
            prepared_graph.add_node(u, **node_attrs[u])
        if not prepared_graph.has_node(v):
            prepared_graph.add_node(v, **node_attrs[v])

        edge_attrs: Dict[str, Any] = dict(attrs)
        edge_attrs["statements"] = filtered_statements
        edge_attrs["evidence"] = {
            "total_evidence": total_evidence,
            "source_evidence": len(source_key_union),
            "stmt_type": list(stmt_type_union),
        }
        prepared_graph.add_edge(u, v, **edge_attrs)

    return prepared_graph


def query_confounders(
    graph: nx.DiGraph,
    confounders: List[str],
    mediators: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Find common predecessor nodes (potential confounders) for a pair of
    target nodes and return their evidence counts.

    Args:
        graph: DiGraph annotated with edge evidence (see :func:`add_evidence_info`).
        confounders: Two-element list-like containing the two target node IDs
            for which to find shared predecessors.
        mediators: Optional iterable of mediator nodes to restrict the
            predecessor search. If provided, predecessors will be filtered to
            the mediator set; otherwise all predecessors are considered.

    Returns:
        A :class:`pandas.DataFrame` with columns ["source", "target",
        "evidence_count", "source_count"] listing the edges from each
        common confounder to the two targets and their evidence summaries.
    """

    if len(confounders) != 2:
        raise ValueError("confounders must contain exactly two node identifiers")

    pred_c1 = list(graph.predecessors(confounders[0]))
    pred_c2 = list(graph.predecessors(confounders[1]))

    if mediators is not None:
        mediator_set = set(mediators)
        pred_c1 = [i for i in pred_c1 if i in mediator_set and i != confounders[1]]
        pred_c2 = [i for i in pred_c2 if i in mediator_set and i != confounders[0]]
    else:
        pred_c1 = [i for i in pred_c1 if i != confounders[1]]
        pred_c2 = [i for i in pred_c2 if i != confounders[0]]

    common_confounders = list(set(pred_c1) & set(pred_c2))

    confounders_edge_list: List[tuple] = []
    for confounder in common_confounders:
        edge1 = graph[confounder][confounders[0]]
        edge2 = graph[confounder][confounders[1]]
        confounders_edge_list.append(
            (
                confounder,
                confounders[0],
                edge1["evidence"]["total_evidence"],
                edge1["evidence"]["source_evidence"],
            )
        )
        confounders_edge_list.append(
            (
                confounder,
                confounders[1],
                edge2["evidence"]["total_evidence"],
                edge2["evidence"]["source_evidence"],
            )
        )

    confounder_df = pd.DataFrame(
        confounders_edge_list,
        columns=["source", "target", "evidence_count", "source_count"],
    )

    return confounder_df


def edge_ok(G: nx.DiGraph, u: str, v: str, thr: int = 5) -> bool:
    """Return True if edge (u, v) has total evidence >= thr.

    Args:
        G: Graph containing the edge.
        u: source node id.
        v: target node id.
        thr: evidence threshold (inclusive).
    """

    d = G[u][v]  # edge attributes dict
    ev = d.get("evidence", {}).get("total_evidence", 0)
    return ev >= thr


def filtered_paths(
    G: nx.Graph, source: str, target: str, cutoff: Optional[int] = None, thr: int = 1
):
    """Yield simple paths from source to target over edges meeting evidence threshold.

    The function constructs a subgraph view that hides edges failing the
    evidence threshold and then yields paths found by
    :func:`networkx.all_simple_paths`.
    """

    view = nx.subgraph_view(G, filter_edge=lambda u, v: edge_ok(G, u, v, thr=thr))
    # works for Graph/DiGraph/Multi(Di)Graph (paths are node sequences)
    yield from nx.all_simple_paths(view, source, target, cutoff=cutoff)

def query_drug_targets(graph: nx.DiGraph, 
                       drug: str,
                       target_ev_filter: int = 1) -> pd.DataFrame:
    
    """
    Query drug targets from a directed graph and return aggregated evidence data.
    This function retrieves all direct targets of a given drug from a NetworkX directed graph,
    filters them based on a minimum evidence threshold, and returns a consolidated DataFrame
    with aggregated evidence counts and source counts.
    Args:
        graph (nx.DiGraph): A NetworkX directed graph where nodes represent drugs/targets
            and edges contain evidence metadata.
        drug (str): The drug node identifier to query targets for.
        target_ev_filter (int, optional): Minimum total evidence count required for a target
            to be included in results. Defaults to 1.
    Returns:
        pd.DataFrame: A DataFrame containing:
            - source: The drug identifier (str)
            - target: The target identifier (str)
            - evidence_count: Total aggregated evidence count (int)
            - source_count: Total aggregated source count (int)
            Rows are grouped by source-target pairs with summed evidence metrics.
    Raises:
        nx.NodeNotFound: If the drug node does not exist in the graph.
    Notes:
        - Edge data is expected to have an 'evidence' dictionary with keys 'total_evidence'
            and 'source_evidence'.
        - Missing or malformed evidence data defaults to 0.
        - Results are aggregated by unique source-target pairs.
    """
    
    edges_list: List[tuple] = []
    for successor in graph.successors(drug):
        edge = graph[drug][successor]
        ev = edge.get("evidence", {}).get("total_evidence", 0)
        if ev >= target_ev_filter:
            edges_list.append(
                (
                    drug,
                    successor,
                    edge["evidence"]["total_evidence"],
                    edge["evidence"]["source_evidence"],
                )
            )
    
    result_df = pd.DataFrame(
        edges_list, columns=["source", "target", "evidence_count", "source_count"]
    )
    result_df = result_df.groupby(["source", "target"], as_index=False).agg(
        {"evidence_count": "sum", "source_count": "sum"}
    )
    return result_df

def query_effect_nodes(
    graph: nx.DiGraph,
    effect: str,
    target_ev_filter: int = 1) -> pd.DataFrame:
    
    """
    Query effect nodes from a directed graph and return aggregated evidence data.
    This function retrieves all direct predecessors of a given effect node from a NetworkX directed graph,
    filters them based on a minimum evidence threshold, and returns a consolidated DataFrame
    with aggregated evidence counts and source counts.
    Args:
        graph (nx.DiGraph): A NetworkX directed graph where nodes represent effects
            and edges contain evidence metadata.
        effect (str): The effect node identifier to query predecessors for.
        target_ev_filter (int, optional): Minimum total evidence count required for a predecessor
            to be included in results. Defaults to 1.
    Returns:
        pd.DataFrame: A DataFrame containing:
            - source: The predecessor identifier (str)
            - target: The effect identifier (str)
            - evidence_count: Total aggregated evidence count (int)
            - source_count: Total aggregated source count (int)
            Rows are grouped by source-target pairs with summed evidence metrics.
    Raises:
        nx.NodeNotFound: If the effect node does not exist in the graph.
    Notes:
        - Edge data is expected to have an 'evidence' dictionary with keys 'total_evidence'
            and 'source_evidence'.
        - Missing or malformed evidence data defaults to 0.
        - Results are aggregated by unique source-target pairs.
    """

    edges_list: List[tuple] = []
    for predecessor in graph.predecessors(effect):
        edge = graph[predecessor][effect]
        ev = edge.get("evidence", {}).get("total_evidence", 0)
        if ev >= target_ev_filter:
            edges_list.append(
                (
                    predecessor,
                    effect,
                    edge["evidence"]["total_evidence"],
                    edge["evidence"]["source_evidence"],
                )
            )

    result_df = pd.DataFrame(
        edges_list, columns=["source", "target", "evidence_count", "source_count"]
    )
    result_df = result_df.groupby(["source", "target"], as_index=False).agg(
        {"evidence_count": "sum", "source_count": "sum"}
    )
    return result_df
    

def query_forward_paths(
    graph: nx.DiGraph,
    start_nodes: Iterable[str],
    end_nodes: Iterable[str],
    n_mediators: int = 1,
    med_ev_filter: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Search for simple forward paths from any start node to any end node.

    For each mediator depth from 0..n_mediators the function will collect
    paths with exactly that many intermediate nodes between the start and
    end nodes. This corresponds to path lengths of ``mediator_count + 1``
    edges, subject to the corresponding evidence threshold in
    ``med_ev_filter``.

    Args:
        graph: DiGraph annotated with evidence counts on edges.
        start_nodes: Iterable of starting node ids.
        end_nodes: Iterable of target node ids.
        n_mediators: Maximum number of mediator nodes between start and end.
        med_ev_filter: Optional list of integer thresholds with length
            ``n_mediators + 1`` where index ``i`` applies to paths with
            ``i`` mediators. If None, defaults to all ones.

    Returns:
        A pandas.DataFrame with rows for each edge that appears on any
        discovered path. Columns: ["source", "target", "evidence_count",
        "source_count"].
    """

    if med_ev_filter is None:
        med_ev_filter = [1] * (n_mediators + 1)

    if n_mediators < 0:
        raise ValueError("n_mediators must be non-negative")

    if len(med_ev_filter) != (n_mediators + 1):
        raise ValueError("med_ev_filter must have length n_mediators + 1")

    paths = []
    for start in tqdm(list(start_nodes), desc="Processing start nodes"):
        for end in end_nodes:
            for med in range(0, n_mediators + 1):
                thr = med_ev_filter[med]
                all_paths = list(
                    filtered_paths(
                        graph,
                        source=start,
                        target=end,
                        cutoff=med + 1,
                        thr=thr,
                    )
                )
                if all_paths:
                    paths.extend(all_paths)

    # Flatten list of paths for extraction into df (already flattened by extend)
    forward_edge_list: List[tuple] = []
    for path in tqdm(paths, desc="Extracting paths into dataframe"):
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge = graph[u][v]
            forward_edge_list.append(
                (u, v, edge["evidence"]["total_evidence"], edge["evidence"]["source_evidence"])
            )

    forward_df = pd.DataFrame(
        forward_edge_list, columns=["source", "target", "evidence_count", "source_count"]
    )

    return forward_df


def main():

    file = "../../AstraZeneca_project/data/INDRA/indranet_dir_graph_fix_corr_weights.pkl"
    import pickle

    # Replace 'file_path.pkl' with your actual file path
    with open(file, "rb") as f:
        graph = pickle.load(f)

    trog_targets = ["SERPINE1", "CYP3A4", "CTNNB1", "MAPK1"]

    dili_targets = [
        "ABCC2",
        "ALB",
        "CAT",
        "CYP2C19",
        "CYP2C9",
        "CYP2E1",
        "ENO1",
        "GPT",
        "GSR",
        "GSTM1",
        "GSTT1",
        "HLA-A",
        "HMOX1",
        "HPD",
        "KNG1",
        "MTHFR",
        "NAT2",
        "SOD1",
    ]

    input_data = pd.read_csv("data/model_input.csv")

    hgnc_graph = prepare_graph(
        graph, input_data.columns, ["HGNC"], ["IncreaseAmount", "DecreaseAmount"]
    )

    forward_paths = query_forward_paths(
        graph=hgnc_graph,
        start_nodes=trog_targets,
        end_nodes=dili_targets,
        n_mediators=2,
        med_ev_filter=[1, 1, 3],
    )

    print(forward_paths)


if __name__ == "__main__":
    main()
