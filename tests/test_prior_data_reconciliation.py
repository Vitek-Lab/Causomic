"""Tests for the pure helper functions in prior_data_reconciliation.

Covers the deterministic / data-transform helpers only:

- random_acyclic_subgraph: acyclicity, node/edge invariants, inclusion_prob
  extremes, seeded determinism, and max_indegree enforcement.
- calculate_edge_probabilities: power-law CDF mapping over evidence counts.
- prepare_indra_priors: (source, target) -> probability dictionary building
  with and without the sigmoid probability conversion.
- remove_high_corr_edges_from_blacklist: blacklist pruning and prior augmentation
  driven by a small correlation setup.

The pgmpy scoring classes and the Parallel/HillClimb bootstrap drivers
(process_bootstrap / run_bootstrap) are heavier and are not exercised here.
"""

import importlib

import networkx as nx
import numpy as np
import pandas as pd

pdr = importlib.import_module("causomic.graph_construction.prior_data_reconciliation")


# ---------------------------------------------------------------------------
# random_acyclic_subgraph
# ---------------------------------------------------------------------------
def test_random_acyclic_subgraph_is_acyclic_and_subset():
    nodes = ["A", "B", "C", "D"]
    # Contains a cycle A->B->C->D->A; result must still be acyclic.
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")]
    rng = np.random.default_rng(0)
    dag = pdr.random_acyclic_subgraph(nodes, allowed, inclusion_prob=1.0, rng=rng)
    assert nx.is_directed_acyclic_graph(dag)
    assert set(dag.nodes()) == set(nodes)
    assert set(dag.edges()).issubset(set(allowed))


def test_random_acyclic_subgraph_zero_prob_adds_no_edges():
    nodes = ["A", "B", "C"]
    allowed = [("A", "B"), ("B", "C")]
    rng = np.random.default_rng(1)
    dag = pdr.random_acyclic_subgraph(nodes, allowed, inclusion_prob=0.0, rng=rng)
    assert dag.number_of_edges() == 0
    assert set(dag.nodes()) == set(nodes)


def test_random_acyclic_subgraph_deterministic_with_seed():
    nodes = ["A", "B", "C", "D"]
    allowed = [("A", "B"), ("B", "C"), ("C", "D"), ("A", "C")]
    d1 = pdr.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    d2 = pdr.random_acyclic_subgraph(nodes, allowed, 0.6, np.random.default_rng(42))
    assert set(d1.edges()) == set(d2.edges())


def test_random_acyclic_subgraph_respects_max_indegree():
    # Every allowed edge points into node "T"; with inclusion_prob=1.0 all edges
    # would be attempted, but max_indegree caps the accepted parents.
    nodes = ["A", "B", "C", "T"]
    allowed = [("A", "T"), ("B", "T"), ("C", "T")]
    rng = np.random.default_rng(7)
    dag = pdr.random_acyclic_subgraph(
        nodes, allowed, inclusion_prob=1.0, rng=rng, max_indegree=1
    )
    assert len(dag.get_parents("T")) <= 1
    assert nx.is_directed_acyclic_graph(dag)


# ---------------------------------------------------------------------------
# calculate_edge_probabilities
# ---------------------------------------------------------------------------
def test_calculate_edge_probabilities_cdf_properties():
    df = pd.DataFrame(
        {
            "source": ["A", "B", "C", "D"],
            "target": ["W", "X", "Y", "Z"],
            "evidence_count": [1, 2, 5, 10],
        }
    )
    mapping = pdr.calculate_edge_probabilities(df)

    xmin = 1
    xmax = 10
    # Keys span the full integer support from xmin..xmax.
    assert set(mapping.keys()) == set(range(xmin, xmax + 1))

    values = [mapping[k] for k in range(xmin, xmax + 1)]
    # CDF: all in [0, 1], non-decreasing, terminating at ~1.0.
    assert all(0.0 <= v <= 1.0 for v in values)
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:]))
    assert np.isclose(values[-1], 1.0)
    # Larger evidence counts receive larger cumulative probability.
    assert mapping[10] > mapping[1]


def test_calculate_edge_probabilities_custom_count_col():
    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["X", "Y"],
            "source_count": [3, 7],
        }
    )
    mapping = pdr.calculate_edge_probabilities(df, count_col="source_count")
    assert set(mapping.keys()) == set(range(3, 8))
    assert np.isclose(mapping[7], 1.0)


# ---------------------------------------------------------------------------
# prepare_indra_priors
# ---------------------------------------------------------------------------
def test_prepare_indra_priors_no_conversion_returns_raw_counts():
    df = pd.DataFrame(
        {
            "source": ["AKT1", "TP53", "MDM2"],
            "target": ["MDM2", "MDM2", "TP53"],
            "evidence_count": [15, 25, 8],
        }
    )
    priors = pdr.prepare_indra_priors(df, convert_to_probability=False)
    assert priors == {
        ("AKT1", "MDM2"): 15,
        ("TP53", "MDM2"): 25,
        ("MDM2", "TP53"): 8,
    }


def test_prepare_indra_priors_sigmoid_conversion():
    df = pd.DataFrame(
        {
            "source": ["AKT1", "TP53"],
            "target": ["MDM2", "MDM2"],
            "evidence_count": [15, 25],
        }
    )
    priors = pdr.prepare_indra_priors(df, convert_to_probability=True)

    # Keys are the (source, target) tuples.
    assert set(priors.keys()) == {("AKT1", "MDM2"), ("TP53", "MDM2")}

    # Values match the closed-form sigmoid of log1p(count).
    def expected(count):
        log_ev = np.log1p(count)
        return 1 / (1 + np.exp(-(log_ev - 1.1) / 0.552))

    assert np.isclose(priors[("AKT1", "MDM2")], expected(15))
    assert np.isclose(priors[("TP53", "MDM2")], expected(25))
    # All resulting probabilities lie in (0, 1) and grow with evidence.
    assert 0.0 < priors[("AKT1", "MDM2")] < priors[("TP53", "MDM2")] < 1.0


def test_prepare_indra_priors_use_source_counts():
    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["X", "Y"],
            "evidence_count": [1, 2],
            "source_count": [50, 60],
        }
    )
    priors = pdr.prepare_indra_priors(
        df, convert_to_probability=False, use_source_counts=True
    )
    assert priors == {("A", "X"): 50, ("B", "Y"): 60}


# ---------------------------------------------------------------------------
# remove_high_corr_edges_from_blacklist
# ---------------------------------------------------------------------------
def test_remove_high_corr_edges_from_blacklist():
    # A and B are perfectly correlated; C is anti-correlated with both.
    data = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [2.0, 4.0, 6.0, 8.0, 10.0],
            "C": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    indra_priors = pd.DataFrame(
        {"source": ["A"], "target": ["C"], "evidence_count": [10]}
    )
    black_list = {("A", "B"), ("B", "A"), ("A", "C")}

    updated_priors, updated_blacklist = pdr.remove_high_corr_edges_from_blacklist(
        data, indra_priors, black_list, corr_threshold=0.99, verbose=False
    )

    # A<->B are highly correlated so both directions are pulled from the blacklist.
    assert ("A", "B") not in updated_blacklist
    assert ("B", "A") not in updated_blacklist
    # A->C correlation is 1.0 in magnitude (perfect anti-correlation), so it is
    # also removed given the abs-correlation threshold.
    assert ("A", "C") not in updated_blacklist
    assert updated_blacklist == set()

    # The high-correlation edges not already present are appended to the priors.
    prior_edges = set(zip(updated_priors["source"], updated_priors["target"]))
    assert ("A", "B") in prior_edges
    assert ("B", "A") in prior_edges
    # Pre-existing (A, C) prior row is retained, not duplicated.
    assert sum((updated_priors["source"] == "A") & (updated_priors["target"] == "C")) == 1


def test_remove_high_corr_edges_from_blacklist_keeps_low_corr():
    # Two independent-ish columns whose |corr| stays below the threshold.
    data = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0, 5.0],
            "B": [1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    indra_priors = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "evidence_count": [3]}
    )
    black_list = {("A", "B")}

    _, updated_blacklist = pdr.remove_high_corr_edges_from_blacklist(
        data, indra_priors, black_list, corr_threshold=0.9, verbose=False
    )
    # Correlation below threshold: blacklist edge is preserved.
    assert ("A", "B") in updated_blacklist
