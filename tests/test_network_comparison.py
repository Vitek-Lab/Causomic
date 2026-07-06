"""Tests for the baseline structure-learning wrappers in network_comparison.

The pure helpers (_model_to_nx, _standardize) are asserted directly; fit_pc and
fit_hc are exercised on small synthetic data to confirm they return a DiGraph
over the input variables. NOTEARS is optional and not tested here.
"""

import importlib

import numpy as np
import pandas as pd
from pgmpy.base import DAG

nc = importlib.import_module("causomic.validation.network_comparison")


def test_model_to_nx_preserves_nodes_and_edges():
    model = DAG()
    model.add_nodes_from(["A", "B", "C"])
    model.add_edges_from([("A", "B"), ("B", "C")])
    G = nc._model_to_nx(model)
    assert set(G.nodes()) == {"A", "B", "C"}
    assert set(G.edges()) == {("A", "B"), ("B", "C")}


def test_standardize_zero_mean_unit_std():
    X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    Z = nc._standardize(X)
    assert np.allclose(Z.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1.0, atol=1e-9)


def _linear_chain_data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(scale=0.1, size=n)
    c = 2.0 * b + rng.normal(scale=0.1, size=n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


def test_fit_hc_returns_digraph_over_variables():
    df = _linear_chain_data()
    G, model = nc.fit_hc(df)
    assert set(G.nodes()) == {"A", "B", "C"}
    assert G.number_of_edges() >= 1


def test_fit_pc_returns_digraph_over_variables():
    df = _linear_chain_data()
    G, model = nc.fit_pc(df)
    assert set(G.nodes()) == {"A", "B", "C"}
