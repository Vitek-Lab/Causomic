"""Tests for the pure/synthetic-data-testable helpers in
causomic.graph_construction.repair.

Two functions are exercised here:

* ``convert_to_y0_graph`` is a pure transformation from a posterior-DAG
  DataFrame (with 'source'/'target' columns) into a y0 ``NxMixedGraph``. The
  tests pin that nodes and directed edges are preserved (y0 stores nodes as
  ``Variable`` objects, so comparisons go through ``str(...)``).

* ``process_failed_test`` runs pearsonr-based conditional-independence tests on
  synthetic data to decide whether a failed CI test can be repaired by adding
  observed confounders or whether a latent confounder must be introduced. The
  tests build data with a genuine common cause (repair path) and with a direct
  dependency plus an unrelated candidate (latent path).

The INDRA/Neo4j-driven callers are not exercised here.
"""

import importlib

import numpy as np
import pandas as pd

repair = importlib.import_module("causomic.graph_construction.repair")


def _posterior_dag(edges):
    return pd.DataFrame(edges, columns=["source", "target"])


def test_convert_to_y0_graph_preserves_nodes_and_edges():
    dag = _posterior_dag([("X", "M"), ("M", "Y")])
    g = repair.convert_to_y0_graph(dag)

    assert {str(n) for n in g.directed.nodes} == {"X", "M", "Y"}
    assert {(str(u), str(v)) for u, v in g.directed.edges} == {("X", "M"), ("M", "Y")}


def test_convert_to_y0_graph_handles_branching_and_no_latents():
    # A branching DAG: all nodes are observed, so no bidirected (latent) edges.
    dag = _posterior_dag([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    g = repair.convert_to_y0_graph(dag)

    assert {str(n) for n in g.directed.nodes} == {"A", "B", "C", "D"}
    assert {(str(u), str(v)) for u, v in g.directed.edges} == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("C", "D"),
    }
    # Every node was marked observed, so nothing gets lifted into the ADMG's
    # undirected/bidirected component.
    assert len(list(g.undirected.edges)) == 0


def test_convert_to_y0_graph_ignores_stale_index():
    # A non-default index must not break the positional .loc access.
    dag = _posterior_dag([("X", "Y"), ("Y", "Z")])
    dag.index = [10, 20]
    g = repair.convert_to_y0_graph(dag)

    assert {(str(u), str(v)) for u, v in g.directed.edges} == {("X", "Y"), ("Y", "Z")}


def test_process_failed_test_finds_confounder_and_repairs():
    # Z is a genuine common cause of both S and T; given Z they are
    # conditionally independent, so the failed test should be repaired by
    # adding Z rather than a latent confounder.
    rng = np.random.default_rng(0)
    n = 2000
    z = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "S": 2.0 * z + rng.normal(scale=0.5, size=n),
            "T": 3.0 * z + rng.normal(scale=0.5, size=n),
            "Z": z,
        }
    )
    row = pd.Series({"left": "S", "right": "T", "given": ""})
    confounder_relations = {("S", "T"): ["Z"]}

    result = repair.process_failed_test(row, confounder_relations, data, max_conditional=2)

    assert result["source"] == "S"
    assert result["target"] == "T"
    assert result["add_latent"] is False
    assert result["Z"] == ("Z",)
    assert "error" not in result


def test_process_failed_test_no_confounder_restores_independence_adds_latent():
    # S drives T directly; the only candidate confounder W is pure noise and
    # cannot render S and T independent, so a latent confounder is proposed.
    rng = np.random.default_rng(1)
    n = 2000
    s = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "S": s,
            "T": 2.0 * s + rng.normal(scale=0.5, size=n),
            "W": rng.normal(size=n),
        }
    )
    row = pd.Series({"left": "S", "right": "T", "given": ""})
    confounder_relations = {("S", "T"): ["W"]}

    result = repair.process_failed_test(row, confounder_relations, data, max_conditional=2)

    assert result["source"] == "S"
    assert result["target"] == "T"
    assert result["add_latent"] is True
    assert result["Z"] is None
    assert "error" not in result


def test_process_failed_test_no_candidates_adds_latent():
    # When no usable confounders exist there are no combos to test, so the
    # function short-circuits to proposing a latent confounder.
    rng = np.random.default_rng(2)
    n = 500
    data = pd.DataFrame(
        {
            "S": rng.normal(size=n),
            "T": rng.normal(size=n),
        }
    )
    row = pd.Series({"left": "S", "right": "T", "given": ""})
    confounder_relations = {("S", "T"): []}

    result = repair.process_failed_test(row, confounder_relations, data)

    assert result["add_latent"] is True
    assert result["Z"] is None


def test_process_failed_test_drops_given_and_missing_candidates():
    # Candidates equal to 'given' or absent from the data are filtered out.
    # Here only the given variable and an unknown column are offered, leaving
    # no testable candidate -> latent confounder.
    rng = np.random.default_rng(3)
    n = 500
    g = rng.normal(size=n)
    data = pd.DataFrame(
        {
            "S": g + rng.normal(scale=0.5, size=n),
            "T": g + rng.normal(scale=0.5, size=n),
            "G": g,
        }
    )
    row = pd.Series({"left": "S", "right": "T", "given": "G"})
    confounder_relations = {("S", "T"): ["G", "NOT_IN_DATA"]}

    result = repair.process_failed_test(row, confounder_relations, data)

    assert result["add_latent"] is True
    assert result["Z"] is None


def test_process_failed_test_missing_relation_returns_error_latent():
    # A missing (source, target) key raises a KeyError internally; the function
    # catches it, marks add_latent and reports the error string.
    data = pd.DataFrame({"S": [0.0, 1.0], "T": [1.0, 0.0]})
    row = pd.Series({"left": "S", "right": "T", "given": ""})

    result = repair.process_failed_test(row, {}, data)

    assert result["source"] == "S"
    assert result["target"] == "T"
    assert result["add_latent"] is True
    assert result["Z"] is None
    assert "error" in result
