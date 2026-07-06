"""Tests for the hand-built example causal graphs.

Each constructor returns a dict with 'Networkx', 'y0', 'causomic', and
'Coefficients' representations of a canonical causal structure (mediation,
backdoor, frontdoor, signaling network). These tests pin the graph topology
and the coefficient toggle.
"""

import importlib

import networkx as nx
import pytest

eg = importlib.import_module("causomic.simulation.example_graphs")

EXPECTED_KEYS = {"Networkx", "y0", "causomic", "Coefficients"}


def test_mediator_default_topology():
    out = eg.mediator(n_med=2)
    assert EXPECTED_KEYS.issubset(out)
    # X -> M1 -> M2 -> Z
    assert set(out["Networkx"].edges()) == {("X", "M1"), ("M1", "M2"), ("M2", "Z")}
    assert nx.is_directed_acyclic_graph(out["Networkx"])


def test_mediator_single_mediator():
    out = eg.mediator(n_med=1)
    assert set(out["Networkx"].edges()) == {("X", "M1"), ("M1", "Z")}


def test_mediator_coefficients_toggle():
    with_coef = eg.mediator(n_med=1, include_coef=True)
    without_coef = eg.mediator(n_med=1, include_coef=False)
    assert with_coef["Coefficients"] is not None
    assert set(with_coef["Coefficients"]) == set(with_coef["Networkx"].nodes())
    assert without_coef["Coefficients"] is None


def test_mediator_independent_nodes_added():
    out = eg.mediator(n_med=1, add_independent_nodes=True, n_ind=5)
    ind_nodes = [n for n in out["Networkx"].nodes() if str(n).startswith("I")]
    assert len(ind_nodes) == 5


def test_mediator_rejects_zero_mediators():
    with pytest.raises(ValueError):
        eg.mediator(n_med=0)


def test_backdoor_topology():
    out = eg.backdoor()
    edges = set(out["Networkx"].edges())
    # Confounder C into both the treatment path and outcome; X -> Y core edge.
    assert ("X", "Y") in edges
    assert ("C", "Y") in edges
    assert nx.is_directed_acyclic_graph(out["Networkx"])


def test_frontdoor_topology():
    out = eg.frontdoor()
    edges = set(out["Networkx"].edges())
    # Classic frontdoor: X -> Y -> Z mediated chain with confounder C.
    assert {("X", "Y"), ("Y", "Z")}.issubset(edges)
    assert nx.is_directed_acyclic_graph(out["Networkx"])


def test_signaling_network_is_dag_with_coefficients():
    out = eg.signaling_network()
    assert EXPECTED_KEYS.issubset(out)
    assert nx.is_directed_acyclic_graph(out["Networkx"])
    assert out["Networkx"].number_of_nodes() > 0
    assert set(out["Coefficients"]) == set(out["Networkx"].nodes())
