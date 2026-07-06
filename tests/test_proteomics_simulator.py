"""Tests for the proteomics data simulator.

These exercise the synthetic data-generation pipeline: structural-equation
coefficient generation, node/data simulation over a causal graph, feature-level
expansion, missing-data masking, and the built-in IGF example network. Data is
constructed inline (small DiGraphs / arrays) following the suite convention.
"""

import importlib

import networkx as nx
import numpy as np
import pandas as pd

# src is placed on sys.path by tests/conftest.py
ps = importlib.import_module("causomic.simulation.proteomics_simulator")


def chain_graph():
    """Simple linear causal chain X -> Y -> Z."""
    return nx.DiGraph([("X", "Y"), ("Y", "Z")])


# --------------------------------------------------------------------------- #
# Coefficient generation
# --------------------------------------------------------------------------- #
def test_generate_coefficients_has_entry_per_node():
    G = chain_graph()
    coeffs = ps.generate_coefficients(G)

    assert set(coeffs) == set(G.nodes())
    # Root node: only intercept + error, no parent terms.
    assert set(coeffs["X"]) == {"intercept", "error"}
    # Child nodes carry a coefficient for each parent plus intercept + error.
    assert set(coeffs["Y"]) == {"X", "intercept", "error"}
    assert set(coeffs["Z"]) == {"Y", "intercept", "error"}


def test_generate_node_coefficients_root_vs_child():
    root = ps.generate_node_coefficients([])
    assert set(root) == {"intercept", "error"}
    # Root intercepts represent baseline expression in the 15-25 range.
    assert 15 <= root["intercept"] <= 25

    child = ps.generate_node_coefficients(["A", "B"])
    assert {"A", "B", "intercept", "error"}.issubset(child)


# --------------------------------------------------------------------------- #
# simulate_data
# --------------------------------------------------------------------------- #
def test_simulate_data_structure_and_shapes():
    G = chain_graph()
    out = ps.simulate_data(G, n=50, seed=1, verbose=False)

    assert set(out) == {"Protein_data", "Feature_data", "Coefficients"}
    assert set(out["Protein_data"]) == set(G.nodes())
    for node in G.nodes():
        assert np.asarray(out["Protein_data"][node]).shape == (50,)

    expected_cols = {
        "Protein",
        "Replicate",
        "Feature",
        "Intensity",
        "Obs_Intensity",
        "MNAR_threshold",
        "MAR",
        "MNAR",
    }
    assert expected_cols.issubset(out["Feature_data"].columns)


def test_simulate_data_is_reproducible_with_seed():
    G = chain_graph()
    a = ps.simulate_data(G, n=40, seed=7, verbose=False)
    b = ps.simulate_data(G, n=40, seed=7, verbose=False)

    for node in G.nodes():
        assert np.allclose(a["Protein_data"][node], b["Protein_data"][node])


def test_simulate_data_without_feature_variation():
    G = chain_graph()
    out = ps.simulate_data(G, n=20, seed=1, add_feature_var=False, verbose=False)
    assert out["Feature_data"] is None


def test_simulate_data_missingness_toggle():
    G = chain_graph()
    with_missing = ps.simulate_data(G, n=200, seed=3, verbose=False)
    without_missing = ps.simulate_data(G, n=200, seed=3, include_missing=False, verbose=False)

    # With masking enabled, the observed intensity is present and carries NaNs.
    assert "Obs_Intensity" in with_missing["Feature_data"].columns
    assert with_missing["Feature_data"]["Obs_Intensity"].isna().any()

    # With masking disabled, no masking is applied and the mask columns
    # (including Obs_Intensity) are never added.
    assert "Obs_Intensity" not in without_missing["Feature_data"].columns


def test_simulate_data_intervention_pins_node():
    G = chain_graph()
    out = ps.simulate_data(
        G, n=30, seed=2, intervention={"X": 7.0}, add_feature_var=False, verbose=False
    )
    assert np.allclose(out["Protein_data"]["X"], 7.0)


# --------------------------------------------------------------------------- #
# Lower-level helpers
# --------------------------------------------------------------------------- #
def test_simulate_node_returns_expected_length():
    np.random.seed(0)
    arr = ps.simulate_node({}, {"intercept": 10.0, "error": 1.0}, 40, False, None, "X")
    assert np.asarray(arr).shape == (40,)


def test_generate_features_columns_and_protein_label():
    np.random.seed(0)
    feats = ps.generate_features(np.random.normal(10, 1, 25), "X")
    assert isinstance(feats, pd.DataFrame)
    assert {"Protein", "Replicate", "Feature", "Intensity"}.issubset(feats.columns)
    assert set(feats["Protein"].unique()) == {"X"}


def test_add_missing_appends_mask_columns():
    df = pd.DataFrame(
        {
            "Protein": ["X"] * 20,
            "Feature": ["f"] * 20,
            "Replicate": range(20),
            "Intensity": np.linspace(5, 20, 20),
        }
    )
    out = ps.add_missing(df.copy(), 0.05, [-3, 0.4])
    for col in ("Obs_Intensity", "MNAR_threshold", "MAR", "MNAR"):
        assert col in out.columns


# --------------------------------------------------------------------------- #
# Built-in example network
# --------------------------------------------------------------------------- #
def test_build_igf_network_is_dag():
    G = ps.build_igf_network(cell_confounder=False)
    assert isinstance(G, nx.DiGraph)
    assert nx.is_directed_acyclic_graph(G)
    assert G.number_of_nodes() > 0


def test_build_igf_network_confounder_adds_nodes():
    base = ps.build_igf_network(cell_confounder=False)
    confounded = ps.build_igf_network(cell_confounder=True)
    assert confounded.number_of_nodes() >= base.number_of_nodes()


def test_simulate_data_end_to_end_on_igf_network():
    G = ps.build_igf_network(cell_confounder=False)
    out = ps.simulate_data(G, n=50, seed=11, verbose=False)
    assert set(out["Protein_data"]) == set(G.nodes())
    assert len(out["Feature_data"]) > 0
