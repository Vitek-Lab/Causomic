"""Tests for the pure helper utilities in causomic.workflows.

The public run_toxicity_detection_workflow requires INDRA queries and structure
learning, so it is not exercised here; these tests pin the graph-counting and
drug-name-normalization helpers plus the no-op logging path.
"""

import importlib

import networkx as nx
import pandas as pd

wf = importlib.import_module("causomic.workflows")


def test_graph_counts_dataframe():
    df = pd.DataFrame({"source": ["A", "B"], "target": ["B", "C"]})
    assert wf._graph_counts(df) == (3, 2, None)


def test_graph_counts_networkx():
    g = nx.DiGraph([("A", "B"), ("B", "C")])
    assert wf._graph_counts(g) == (3, 2, None)


def test_graph_counts_mixed_graph_splits_edge_types():
    from y0.graph import NxMixedGraph

    G = NxMixedGraph.from_edges(directed=[("A", "B")], undirected=[("B", "C")])
    nodes, edges, split = wf._graph_counts(G)
    assert nodes == 3
    assert edges == 2
    assert split == (1, 1)


def test_graph_counts_unsupported_returns_none():
    assert wf._graph_counts(42) == (None, None, None)


def test_normalize_drug_name_str():
    assert wf._normalize_drug_name("aspirin") == "aspirin"


def test_normalize_drug_name_list():
    assert wf._normalize_drug_name(["a", "b"]) == "a,b"


def test_normalize_drug_name_tuple():
    assert wf._normalize_drug_name(("a", "b")) == "a,b"


def test_normalize_drug_name_other():
    assert wf._normalize_drug_name(5) == "5"


def test_append_log_line_none_is_noop(tmp_path):
    # log_file=None must not create any file or raise.
    wf._append_log_line("hello", None)
    assert list(tmp_path.iterdir()) == []


def test_append_log_line_writes(tmp_path):
    log_file = tmp_path / "sub" / "run.log"
    wf._append_log_line("line one", str(log_file))
    assert log_file.exists()
    assert "line one" in log_file.read_text()
