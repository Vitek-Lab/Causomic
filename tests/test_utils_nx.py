import importlib
import os

import networkx as nx
import pandas as pd

# Ensure src is on path via tests/conftest.py
utils_nx = importlib.import_module("causomic.graph_construction.utils_nx")


def make_sample_graph():
    G = nx.DiGraph()
    # add nodes with namespace attribute
    for n in ["n1", "n2", "n3", "n4"]:
        G.add_node(n, ns="HGNC")

    # edges with statements list (dicts)
    G.add_edge("n1", "n2", statements=[{"stmt_type": "IncreaseAmount", "evidence_count": 2, "source_counts": {"SRC1": 1}}])
    G.add_edge("n2", "n3", statements=[{"stmt_type": "DecreaseAmount", "evidence_count": 5, "source_counts": {"SRC1": 2, "SRC2": 1}}])
    G.add_edge("n3", "n4", statements=[{"stmt_type": "OtherType", "evidence_count": None, "source_counts": {}}])

    return G


def test_filter_graph_by_stmt_types_and_statement_filtering():
    G = make_sample_graph()
    # keep only IncreaseAmount statements
    out = utils_nx.filter_graph_by_stmt_types(G, ["IncreaseAmount"])
    assert out.number_of_nodes() >= 2
    # only edge n1->n2 should remain
    assert out.has_edge("n1", "n2")
    assert not out.has_edge("n2", "n3")
    # statements list filtered
    stmts = out["n1"]["n2"]["statements"]
    assert all(s.get("stmt_type") == "IncreaseAmount" for s in stmts)


def test_add_evidence_info_and_filter_by_evidence_count():
    G = make_sample_graph()
    utils_nx.add_evidence_info(G)
    # evidence should be attached
    assert G["n1"]["n2"].get("evidence") is not None
    # n2->n3 has evidence_count 5 so edge_ok with thr=5 should be True
    assert utils_nx.edge_ok(G, "n2", "n3", thr=5)

    # filter edges with threshold 3 should keep only n2->n3
    f = utils_nx.filter_graph_by_evidence_count(G, 3)
    assert f.number_of_edges() == 1
    assert f.has_edge("n2", "n3")


def test_filter_graph_by_measured_nodes_and_prepare_graph():
    G = make_sample_graph()
    measured = ["n1", "n2"]
    sub = utils_nx.filter_graph_by_measured_nodes(G, measured)
    assert sub.number_of_edges() == 1

    # prepare_graph should run without error (uses node ns attr and stmt types)
    G2 = make_sample_graph()
    prepared = utils_nx.prepare_graph(G2, measured_nodes=["n1", "n2", "n3", "n4"], node_types=["HGNC"], stmt_types=["IncreaseAmount", "DecreaseAmount", "OtherType"])  # noqa: E501
    # evidence attribute should be present after prepare_graph
    assert prepared.edges(data=True)


def test_query_confounders_returns_dataframe():
    G = make_sample_graph()
    # attach evidence info first
    utils_nx.add_evidence_info(G)
    # create a common confounder node that points to n2 and n3
    G.add_edge("c", "n2", statements=[{"stmt_type": "IncreaseAmount", "evidence_count": 3, "source_counts": {"S1": 1}}])
    G.add_edge("c", "n3", statements=[{"stmt_type": "IncreaseAmount", "evidence_count": 4, "source_counts": {"S1": 1}}])
    utils_nx.add_evidence_info(G)
    df = utils_nx.query_confounders(G, ["n2", "n3"]) 
    assert isinstance(df, pd.DataFrame)
    # should contain rows for the confounder 'c'
    assert (df["source"] == "c").any()


def test_filtered_paths_and_query_forward_paths():
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B", {"evidence": {"total_evidence": 2, "source_evidence": 1}}),
        ("B", "C", {"evidence": {"total_evidence": 2, "source_evidence": 1}}),
    ])
    # filtered_paths yields the path A->B->C
    paths = list(utils_nx.filtered_paths(G, "A", "C", cutoff=2, thr=1))
    assert any(path for path in paths if path[0] == "A" and path[-1] == "C")

    # query_forward_paths should return dataframe with the forward edges
    fwd = utils_nx.query_forward_paths(G, start_nodes=["A"], end_nodes=["C"], n_mediators=2, med_ev_filter=[1, 1, 1])
    assert isinstance(fwd, pd.DataFrame)
    assert set(["source", "target"]).issubset(fwd.columns)


def test_query_forward_paths_counts_mediators_not_edges():
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B", {"evidence": {"total_evidence": 2, "source_evidence": 1}}),
        ("B", "C", {"evidence": {"total_evidence": 2, "source_evidence": 1}}),
    ])

    direct_only = utils_nx.query_forward_paths(
        G,
        start_nodes=["A"],
        end_nodes=["C"],
        n_mediators=0,
        med_ev_filter=[1],
    )
    assert direct_only.empty

    one_mediator = utils_nx.query_forward_paths(
        G,
        start_nodes=["A"],
        end_nodes=["C"],
        n_mediators=1,
        med_ev_filter=[1, 1],
    )
    assert set(zip(one_mediator["source"], one_mediator["target"])) == {
        ("A", "B"),
        ("B", "C"),
    }
