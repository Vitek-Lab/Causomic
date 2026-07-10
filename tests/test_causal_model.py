"""Tests for the latent-variable structural causal model (``causomic.causal_model``).

Covers the pure helpers (``ScaleStats``, scaling round-trips), graph/data/prior
parsing, and a small end-to-end Pyro fit + interventional query on simulated
data. Fits use a tiny number of SVI steps so the suite stays fast; assertions
target structure and shapes rather than exact learned values.
"""

import numpy as np
import pandas as pd
import pyro
import pytest
from y0.graph import NxMixedGraph

from causomic.causal_model import LVM
from causomic.causal_model.LVM import ScaleStats
from causomic.simulation import generate_structured_dag, simulate_data


# --------------------------------------------------------------------------- #
# Shared fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def sim_problem():
    """A small ground-truth DAG plus a simulated wide data matrix."""
    gt, roles = generate_structured_dag(
        n_start=2, n_end=1, max_mediators=1, confounder_prob=0.0, seed=0
    )
    sim = simulate_data(gt, n=80, add_feature_var=False, add_error=True, seed=0, verbose=False)
    data = pd.DataFrame(sim["Protein_data"])
    data.columns = [str(c) for c in data.columns]
    graph = NxMixedGraph.from_edges(directed=[(str(u), str(v)) for u, v in gt.edges()])
    return {"graph": graph, "data": data, "roles": roles}


@pytest.fixture(scope="module")
def fitted_lvm(sim_problem):
    """An LVM fitted with a handful of SVI steps (Pyro backend)."""
    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=15, verbose=False)
    lvm.fit(sim_problem["data"], sim_problem["graph"])
    return lvm


# --------------------------------------------------------------------------- #
# ScaleStats
# --------------------------------------------------------------------------- #
def test_scalestats_roundtrip():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [10.0, 20.0, 30.0]})
    stats = ScaleStats(mean=df.mean(), scale=df.std())
    restored = stats.inverse(stats.transform(df))
    pd.testing.assert_frame_equal(restored, df, check_exact=False, rtol=1e-6)


def test_scalestats_zero_scale_uses_eps():
    # A constant column has scale 0; eps clipping must avoid division by zero.
    df = pd.DataFrame({"A": [5.0, 5.0, 5.0]})
    stats = ScaleStats(mean=df.mean(), scale=pd.Series({"A": 0.0}), eps=1e-6)
    z = stats.transform(df)
    assert np.isfinite(z.to_numpy()).all()


# --------------------------------------------------------------------------- #
# Construction / dunders
# --------------------------------------------------------------------------- #
def test_invalid_backend_raises():
    with pytest.raises(ValueError):
        LVM(backend="tensorflow")


def test_init_defaults_and_repr():
    lvm = LVM(backend="pyro")
    assert lvm.model is None
    assert "not fitted" in str(lvm)
    assert "fitted=False" in repr(lvm)


def test_len_before_fit_raises():
    lvm = LVM(backend="pyro")
    with pytest.raises(ValueError):
        len(lvm)


# --------------------------------------------------------------------------- #
# Scaling helpers
# --------------------------------------------------------------------------- #
def test_fit_scaler_and_z_roundtrip(sim_problem):
    lvm = LVM(backend="pyro")
    lvm.fit_scaler(sim_problem["data"])
    z = lvm._to_z(sim_problem["data"])
    back = lvm._from_z(z)
    pd.testing.assert_frame_equal(back, sim_problem["data"], check_exact=False, rtol=1e-6)


def test_to_z_without_scaler_raises():
    lvm = LVM(backend="pyro")
    lvm.scaler = None
    with pytest.raises(RuntimeError):
        lvm._to_z(pd.DataFrame({"A": [1.0]}))


# --------------------------------------------------------------------------- #
# Graph parsing
# --------------------------------------------------------------------------- #
def test_parse_graph_identifies_roots_and_leaves():
    lvm = LVM(backend="pyro")
    lvm.causal_graph = NxMixedGraph.from_edges(directed=[("A", "B"), ("B", "C")])
    lvm.parse_graph()
    assert "A" in lvm.root_nodes
    assert "C" in lvm.end_nodes
    # B has a parent, so it is a descendant mapped to its parents
    assert "A" in lvm.descendant_nodes["B"]


# --------------------------------------------------------------------------- #
# End-to-end fit
# --------------------------------------------------------------------------- #
def test_fit_populates_model_state(fitted_lvm, sim_problem):
    assert fitted_lvm.model is not None
    assert len(fitted_lvm) == len(sim_problem["data"])
    assert fitted_lvm.root_nodes is not None
    assert fitted_lvm.descendant_nodes is not None
    # a guide is built and missing values are imputed during fitting
    assert fitted_lvm.guide is not None
    assert fitted_lvm.imputed_data is not None
    assert "fitted=True" in repr(fitted_lvm)


def test_intervention_produces_samples(fitted_lvm, sim_problem):
    roles = sim_problem["roles"]
    data = sim_problem["data"]
    target = str(roles["start"][0])
    outcome = [str(roles["end"][0])]
    baseline = float(data[target].mean())

    fitted_lvm.intervention(
        {target: baseline - 2.0},
        outcome_node=outcome,
        compare_value=baseline,
        predictive_samples=25,
    )
    assert fitted_lvm.intervention_samples is not None
    assert fitted_lvm.posterior_samples is not None
    assert np.asarray(fitted_lvm.intervention_samples).shape[0] > 0


# --------------------------------------------------------------------------- #
# Stochastic-edge model variant
# --------------------------------------------------------------------------- #
def test_stochastic_edges_fit(sim_problem):
    pyro.clear_param_store()
    lvm = LVM(backend="pyro", num_steps=10, verbose=False, stochastic_edges=True)
    lvm.fit(sim_problem["data"], sim_problem["graph"])
    assert lvm.model is not None


# --------------------------------------------------------------------------- #
# NumPyro backend (fit only; the intervention path has known latent bugs)
# --------------------------------------------------------------------------- #
def test_numpyro_fit(sim_problem):
    lvm = LVM(
        backend="numpyro",
        num_samples=20,
        warmup_steps=20,
        num_chains=1,
        verbose=False,
    )
    lvm.fit(sim_problem["data"], sim_problem["graph"])
    assert lvm.model is not None
    assert len(lvm) == len(sim_problem["data"])
