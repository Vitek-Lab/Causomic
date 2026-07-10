"""Tests for cyclic-graph generation and data simulation.

`generate_cyclic_graph` builds a structured graph that may contain feedback
cycles; `simulate_cyclic_data` runs a fixed-point simulation over such a graph
to produce protein- and feature-level data.
"""

import importlib

import networkx as nx
import numpy as np
import pytest

cn = importlib.import_module("causomic.simulation.cyclic_network")


def test_generate_cyclic_graph_roles():
    graph, roles = cn.generate_cyclic_graph(seed=3, add_cycle_in_mediators=1)
    assert isinstance(graph, nx.DiGraph)
    assert {"start", "end", "mediators", "confounders", "cycle_nodes"}.issubset(roles)


def test_generate_cyclic_graph_requires_a_cycle_location():
    with pytest.raises(ValueError):
        cn.generate_cyclic_graph(seed=3)


def test_generate_cyclic_graph_can_introduce_cycle():
    graph, _ = cn.generate_cyclic_graph(seed=3, add_cycle_in_mediators=1, cycle_size=2)
    assert not nx.is_directed_acyclic_graph(graph)


def test_generate_cyclic_graph_is_reproducible():
    g_a, _ = cn.generate_cyclic_graph(seed=7, add_cycle_in_mediators=1)
    g_b, _ = cn.generate_cyclic_graph(seed=7, add_cycle_in_mediators=1)
    assert set(g_a.edges()) == set(g_b.edges())


def test_simulate_cyclic_data_structure():
    graph, roles = cn.generate_cyclic_graph(seed=3, add_cycle_in_mediators=1, cycle_size=2)
    out = cn.simulate_cyclic_data(graph, roles, n=40, seed=3, verbose=False)
    assert set(out) == {"Protein_data", "Feature_data", "Coefficients"}
    # Every graph node should have simulated protein-level values.
    assert set(out["Protein_data"]) == set(graph.nodes())
    for node in graph.nodes():
        assert np.asarray(out["Protein_data"][node]).shape == (40,)


def test_simulate_cyclic_data_is_reproducible():
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1)
    a = cn.simulate_cyclic_data(graph, roles, n=30, seed=9, verbose=False)
    b = cn.simulate_cyclic_data(graph, roles, n=30, seed=9, verbose=False)
    for node in graph.nodes():
        assert np.allclose(a["Protein_data"][node], b["Protein_data"][node])


def test_simulate_cyclic_data_without_feature_variation():
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1)
    out = cn.simulate_cyclic_data(graph, roles, n=20, seed=1, add_feature_var=False, verbose=False)
    assert out["Feature_data"] is None


# --------------------------------------------------------------------------- #
# generate_cyclic_graph: start-node cycles (lines 205-236)
# --------------------------------------------------------------------------- #
def test_start_cycle_through_existing_mediator_size_two():
    # mediator_cycle_prob=1.0 forces every start cycle to route through an
    # existing mediator; cycle_size=2 hits the two-node anchor<->mediator loop.
    graph, roles = cn.generate_cyclic_graph(
        seed=11,
        add_cycle_in_start=1,
        cycle_size=2,
        mediator_cycle_prob=1.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    assert "start" in roles["cycle_nodes"]
    anchor = roles["start"][0]
    # Anchor participates in a mutual edge with the routed mediator.
    assert any(
        graph.has_edge(anchor, m) and graph.has_edge(m, anchor) for m in graph.successors(anchor)
    )


def test_start_cycle_through_existing_mediator_size_three():
    # cycle_size=3 with mediator routing exercises the fresh-node chain branch.
    graph, roles = cn.generate_cyclic_graph(
        seed=13,
        add_cycle_in_start=1,
        cycle_size=3,
        mediator_cycle_prob=1.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    cy = roles["cycle_nodes"]["start"]
    assert any(name.startswith("CYS0_") for name in cy)


def test_start_cycle_all_fresh_nodes():
    # mediator_cycle_prob=0.0 forces the all-fresh-node bridge branch.
    graph, roles = cn.generate_cyclic_graph(
        seed=17,
        add_cycle_in_start=2,
        n_start=3,
        cycle_size=3,
        mediator_cycle_prob=0.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    cy = roles["cycle_nodes"]["start"]
    assert any(name.startswith("CYS") for name in cy)


# --------------------------------------------------------------------------- #
# generate_cyclic_graph: end-node cycles (lines 269-300)
# --------------------------------------------------------------------------- #
def test_end_cycle_through_existing_mediator_size_two():
    graph, roles = cn.generate_cyclic_graph(
        seed=21,
        add_cycle_in_end=1,
        cycle_size=2,
        mediator_cycle_prob=1.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    assert "end" in roles["cycle_nodes"]
    anchor = roles["end"][0]
    assert any(
        graph.has_edge(anchor, m) and graph.has_edge(m, anchor) for m in graph.successors(anchor)
    )


def test_end_cycle_through_existing_mediator_size_three():
    graph, roles = cn.generate_cyclic_graph(
        seed=23,
        add_cycle_in_end=1,
        cycle_size=3,
        mediator_cycle_prob=1.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    cy = roles["cycle_nodes"]["end"]
    assert any(name.startswith("CYE0_") for name in cy)


def test_end_cycle_all_fresh_nodes():
    graph, roles = cn.generate_cyclic_graph(
        seed=29,
        add_cycle_in_end=2,
        n_end=3,
        cycle_size=3,
        mediator_cycle_prob=0.0,
    )
    assert not nx.is_directed_acyclic_graph(graph)
    cy = roles["cycle_nodes"]["end"]
    assert any(name.startswith("CYE") for name in cy)


# --------------------------------------------------------------------------- #
# generate_cyclic_graph: validation errors
# --------------------------------------------------------------------------- #
def test_generate_cyclic_graph_rejects_small_cycle_size():
    with pytest.raises(ValueError):
        cn.generate_cyclic_graph(seed=1, add_cycle_in_mediators=1, cycle_size=1)


def test_generate_cyclic_graph_rejects_too_many_start_cycles():
    with pytest.raises(ValueError):
        cn.generate_cyclic_graph(seed=1, add_cycle_in_start=5, n_start=2)


def test_generate_cyclic_graph_rejects_too_many_end_cycles():
    with pytest.raises(ValueError):
        cn.generate_cyclic_graph(seed=1, add_cycle_in_end=5, n_end=2)


# --------------------------------------------------------------------------- #
# simulate_cyclic_data: threshold clamping / freeze-break (lines 383-399)
# --------------------------------------------------------------------------- #
def test_simulate_cyclic_data_threshold_clamps_cycle_nodes():
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1, cycle_size=3)
    # A tiny threshold forces every sample of every cycle node past saturation on
    # the first Jacobi iteration, triggering the clamp + frozen.all() break.
    out = cn.simulate_cyclic_data(
        graph, roles, n=25, seed=2, threshold=0.01, add_feature_var=False, verbose=False
    )
    cycle_nodes = roles["cycle_nodes"]["mediators"]
    for node in cycle_nodes:
        vals = np.asarray(out["Protein_data"][node])
        assert np.all(np.abs(vals) <= 0.01 + 1e-9)


# --------------------------------------------------------------------------- #
# simulate_cyclic_data: verbose + feature-level data path (lines 491-519)
# --------------------------------------------------------------------------- #
def test_simulate_cyclic_data_with_features_and_verbose(capsys):
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1)
    out = cn.simulate_cyclic_data(
        graph, roles, n=30, seed=4, add_feature_var=True, include_missing=True, verbose=True
    )
    import pandas as pd

    assert isinstance(out["Feature_data"], pd.DataFrame)
    assert not out["Feature_data"].empty
    captured = capsys.readouterr()
    assert "simulating cyclic data" in captured.out


def test_simulate_cyclic_data_features_without_missing():
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1)
    out = cn.simulate_cyclic_data(
        graph, roles, n=30, seed=4, add_feature_var=True, include_missing=False, verbose=False
    )
    import pandas as pd

    assert isinstance(out["Feature_data"], pd.DataFrame)


# --------------------------------------------------------------------------- #
# ground_truth_interventional_effect_cyclic (lines 581-653)
# --------------------------------------------------------------------------- #
def test_ground_truth_interventional_effect_cyclic_structure():
    graph, roles = cn.generate_cyclic_graph(seed=5, add_cycle_in_mediators=1, cycle_size=3)
    sim = cn.simulate_cyclic_data(graph, roles, n=20, seed=3, add_feature_var=False, verbose=False)
    coeffs = sim["Coefficients"]
    intervention = {roles["start"][0]: 30.0}
    outputs = roles["end"]
    gt = cn.ground_truth_interventional_effect_cyclic(graph, coeffs, intervention, outputs)
    assert {"baseline", "interventional", "effect"}.issubset(gt)
    # Baseline of every node equals its intercept.
    for node, val in gt["baseline"].items():
        assert val == coeffs[node]["intercept"]
    # Effect is reported for each requested output node.
    for node in outputs:
        assert node in gt["effect"]


def test_ground_truth_interventional_effect_cyclic_intervened_node_delta():
    # The intervened node's interventional expectation must equal the do-value.
    graph, roles = cn.generate_cyclic_graph(seed=8, add_cycle_in_mediators=1, cycle_size=3)
    sim = cn.simulate_cyclic_data(graph, roles, n=20, seed=6, add_feature_var=False, verbose=False)
    coeffs = sim["Coefficients"]
    target = roles["start"][0]
    gt = cn.ground_truth_interventional_effect_cyclic(graph, coeffs, {target: 42.0}, roles["end"])
    assert gt["interventional"][target] == 42.0
    # A surviving multi-node SCC (the mediator cycle) must be solved and present.
    cycle_nodes = roles["cycle_nodes"]["mediators"]
    for node in cycle_nodes:
        if node in coeffs:
            assert node in gt["interventional"]


def test_ground_truth_interventional_effect_cyclic_is_deterministic():
    graph, roles = cn.generate_cyclic_graph(seed=8, add_cycle_in_mediators=1, cycle_size=3)
    sim = cn.simulate_cyclic_data(graph, roles, n=20, seed=6, add_feature_var=False, verbose=False)
    coeffs = sim["Coefficients"]
    target = roles["start"][0]
    a = cn.ground_truth_interventional_effect_cyclic(graph, coeffs, {target: 10.0}, roles["end"])
    b = cn.ground_truth_interventional_effect_cyclic(graph, coeffs, {target: 10.0}, roles["end"])
    assert a["effect"] == b["effect"]
