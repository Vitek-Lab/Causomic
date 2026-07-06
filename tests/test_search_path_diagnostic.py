"""Tests for the pure helpers in search_path_diagnostic.

random_acyclic_subgraph and compare_dag_sets are deterministic given a seeded
RNG and fixed DAG sets; the Parallel/HillClimb-driven search_path_diagnostic
itself needs an estimator and is not exercised here.
"""

import importlib

import networkx as nx
import numpy as np
from pgmpy.base import DAG

spd = importlib.import_module("causomic.graph_construction.search_path_diagnostic")


def test_random_acyclic_subgraph_is_acyclic_and_subset():
    nodes = ["A", "B", "C", "D"]
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    rng = np.random.default_rng(0)
    dag = spd.random_acyclic_subgraph(nodes, allowed, inclusion_prob=1.0, rng=rng)
    assert nx.is_directed_acyclic_graph(dag)
    assert set(dag.nodes()) == set(nodes)
    assert set(dag.edges()).issubset(set(allowed))


def test_random_acyclic_subgraph_zero_prob_adds_no_edges():
    nodes = ["A", "B", "C"]
    allowed = [("A", "B"), ("B", "C")]
    rng = np.random.default_rng(1)
    dag = spd.random_acyclic_subgraph(nodes, allowed, inclusion_prob=0.0, rng=rng)
    assert dag.number_of_edges() == 0
    assert set(dag.nodes()) == set(nodes)


def test_random_acyclic_subgraph_deterministic_with_seed():
    nodes = ["A", "B", "C", "D"]
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]
    d1 = spd.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    d2 = spd.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    assert set(d1.edges()) == set(d2.edges())


def _dag(edges):
    d = DAG()
    d.add_edges_from(edges)
    return d


def test_compare_dag_sets_frequencies_and_diff():
    # Edge A->B in all of set-a, only half of set-b.
    dags_a = [_dag([("A", "B")]), _dag([("A", "B")])]
    dags_b = [_dag([("A", "B")]), _dag([("C", "D")])]
    df = spd.compare_dag_sets(dags_a, dags_b)
    row_ab = df[(df["source"] == "A") & (df["target"] == "B")].iloc[0]
    assert row_ab["freq_random_init"] == 1.0
    assert row_ab["freq_bootstrap"] == 0.5
    assert np.isclose(row_ab["abs_diff"], 0.5)
    # Both distinct edges represented.
    assert set(zip(df["source"], df["target"])) == {("A", "B"), ("C", "D")}


def test_compare_dag_sets_sorted_by_abs_diff_desc():
    dags_a = [_dag([("A", "B"), ("C", "D")])]
    dags_b = [_dag([("C", "D")])]
    df = spd.compare_dag_sets(dags_a, dags_b)
    # A->B differs by 1.0, C->D by 0.0; the larger diff sorts first.
    assert df.iloc[0]["source"] == "A"
    assert df["abs_diff"].is_monotonic_decreasing


def test_compare_dag_sets_custom_labels():
    dags_a = [_dag([("A", "B")])]
    dags_b = [_dag([("A", "B")])]
    df = spd.compare_dag_sets(dags_a, dags_b, label_a="rand", label_b="boot")
    assert "freq_rand" in df.columns
    assert "freq_boot" in df.columns
