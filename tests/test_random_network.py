"""Tests for procedural DAG generation and INDRA-style misspecification.

Covers random and structured DAG generation, ground-truth interventional
effect computation over a known structure, generation of a misspecified
"INDRA" graph from a ground-truth DAG, and conversion of that graph into an
evidence-annotated graph.
"""

import importlib

import networkx as nx
import numpy as np
import pytest

rn = importlib.import_module("causomic.simulation.random_network")
eg = importlib.import_module("causomic.simulation.example_graphs")


# --------------------------------------------------------------------------- #
# generate_random_dag
# --------------------------------------------------------------------------- #
def test_generate_random_dag_is_acyclic():
    dag = rn.generate_random_dag(8, 0.5)
    assert isinstance(dag, nx.DiGraph)
    assert dag.number_of_nodes() == 8
    assert nx.is_directed_acyclic_graph(dag)


def test_generate_random_dag_sparsity_extremes():
    empty = rn.generate_random_dag(6, 0.0)
    full = rn.generate_random_dag(6, 1.0)
    assert empty.number_of_edges() == 0
    # sparsity 1.0 connects every valid forward pair (upper triangular).
    assert full.number_of_edges() == 6 * (6 - 1) // 2


# --------------------------------------------------------------------------- #
# generate_structured_dag
# --------------------------------------------------------------------------- #
def test_generate_structured_dag_roles_and_acyclic():
    dag, roles = rn.generate_structured_dag(seed=1)
    assert nx.is_directed_acyclic_graph(dag)
    assert {"start", "end", "mediators", "confounders"}.issubset(roles)
    # Start (ligand) and end (readout) nodes must exist.
    assert len(roles["start"]) > 0
    assert len(roles["end"]) > 0


def test_generate_structured_dag_is_reproducible():
    dag_a, _ = rn.generate_structured_dag(seed=42)
    dag_b, _ = rn.generate_structured_dag(seed=42)
    assert set(dag_a.edges()) == set(dag_b.edges())


# --------------------------------------------------------------------------- #
# ground_truth_interventional_effect
# --------------------------------------------------------------------------- #
def test_ground_truth_interventional_effect_structure():
    med = eg.mediator(n_med=1)
    gt = rn.ground_truth_interventional_effect(
        med["Networkx"], med["Coefficients"], {"X": 5.0}, ["Z"]
    )
    assert {"baseline", "interventional", "effect"}.issubset(gt)
    # Effect should be reported for the requested output node.
    assert "Z" in gt["effect"]


# --------------------------------------------------------------------------- #
# generate_indra_data / indra_dag_to_evidence_graph
# --------------------------------------------------------------------------- #
def test_generate_indra_data_adds_misspecification():
    gt_dag, _ = rn.generate_structured_dag(seed=2)
    indra_dag, edge_df, added = rn.generate_indra_data(
        gt_dag, num_incorrect_nodes=3, num_incorrect_edges=5
    )
    assert isinstance(indra_dag, nx.DiGraph)
    # Incorrect nodes are added, so the misspecified graph is at least as large.
    assert indra_dag.number_of_nodes() >= gt_dag.number_of_nodes()


def test_indra_dag_to_evidence_graph_annotates_edges():
    gt_dag, _ = rn.generate_structured_dag(seed=2)
    indra_dag, _, _ = rn.generate_indra_data(gt_dag, num_incorrect_nodes=3, num_incorrect_edges=5)
    ev_graph = rn.indra_dag_to_evidence_graph(indra_dag)
    assert isinstance(ev_graph, nx.DiGraph)
    assert ev_graph.number_of_edges() > 0
    for _, _, data in ev_graph.edges(data=True):
        assert "evidence" in data
        assert "evidence_count" in data
        assert "ground_truth" in data


# --------------------------------------------------------------------------- #
# generate_indra_data: optional misspecification branches
# --------------------------------------------------------------------------- #
def test_generate_indra_data_preferential_and_shortcut_and_missing():
    # Exercises the preferential-attachment edge branch, the mediated-shortcut
    # branch, and the p_missing_real edge-dropping branch together.
    np.random.seed(0)
    gt_dag, _ = rn.generate_structured_dag(seed=4)
    indra_dag, edge_df, missing = rn.generate_indra_data(
        gt_dag,
        num_incorrect_nodes=4,
        num_incorrect_edges=8,
        p_missing_real=0.3,
        p_mediated_shortcut=0.5,
        preferential_attachment=True,
        start_node_out_mu=2.0,
    )
    assert isinstance(indra_dag, nx.DiGraph)
    assert set(edge_df.columns) == {"source", "target", "ground_truth", "evidence_count"}
    assert (edge_df["evidence_count"] >= 1).all()
    # missing_edges is a subset of the original ground-truth edges.
    for u, v in missing:
        assert gt_dag.has_edge(u, v)


# --------------------------------------------------------------------------- #
# run_graph_sim (end-to-end smoke test, lines 594-800)
# --------------------------------------------------------------------------- #
def test_run_graph_sim_returns_metric_octuple():
    # Full recovery pipeline (causomic + PC + HC + NOTEARS). Self-seeded via
    # secrets, so we assert on structure/ranges rather than exact values.
    result = rn.run_graph_sim(verbose=False)
    assert isinstance(result, tuple)
    assert len(result) == 8
    for metric in result:
        val = float(metric)
        assert 0.0 <= val <= 1.0
