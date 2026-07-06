"""Tests for pathway over-representation analysis (ORA) pure functions.

Covers membership-matrix construction, the hypergeometric ORA, Jaccard
similarity, greedy diverse-pathway selection, FDR correction, and JSON
coercion. The gseapy-backed fetchers (fetch_pathway_library,
list_pathway_libraries) hit the network and are only exercised via the
_require_gseapy guard. All other logic is pure numpy/scipy/pandas.
"""

import importlib

import networkx as nx
import numpy as np
import pandas as pd
import pytest

pa = importlib.import_module("causomic.data_analysis.pathway_analysis")


# --- build_membership_matrix --------------------------------------------------


def test_build_membership_matrix_shape_and_values():
    universe = ["A", "B", "C", "D", "E"]
    pathways = {"P1": ["A", "B", "C", "D", "E"]}
    M = pa.build_membership_matrix(universe, pathways, min_pathway_size=1)
    assert M.shape == (5, 1)
    assert M["P1"].all()
    assert M.dtypes["P1"] == bool


def test_build_membership_matrix_filters_by_size():
    universe = ["A", "B", "C", "D", "E", "F"]
    pathways = {
        "small": ["A", "B"],  # 2 universe genes
        "ok": ["A", "B", "C", "D"],  # 4 universe genes
    }
    M = pa.build_membership_matrix(universe, pathways, min_pathway_size=3, max_pathway_size=5)
    assert list(M.columns) == ["ok"]


def test_build_membership_matrix_ignores_non_universe_genes():
    universe = ["A", "B"]
    pathways = {"P1": ["A", "B", "ZZZ"]}
    M = pa.build_membership_matrix(universe, pathways, min_pathway_size=1)
    assert "ZZZ" not in M.index
    assert M["P1"].tolist() == [True, True]


# --- run_ora ------------------------------------------------------------------


def test_run_ora_flags_enriched_pathway():
    pathway_dict = {
        "P1": ["A", "B", "C", "D", "E"],
        "P2": ["F", "G", "H", "I", "J"],
    }
    network = ["A", "B", "C", "D", "E"]  # perfectly overlaps P1
    df = pa.run_ora(network, pathway_dict=pathway_dict, min_pathway_size=5)
    assert set(df["pathway"]) == {"P1", "P2"}
    p1 = df[df["pathway"] == "P1"].iloc[0]
    p2 = df[df["pathway"] == "P2"].iloc[0]
    assert p1["overlap"] == 5
    assert p2["overlap"] == 0
    assert p1["p_value"] < p2["p_value"]
    # Sorted by FDR ascending -> the enriched pathway is first.
    assert df.iloc[0]["pathway"] == "P1"


def test_run_ora_accepts_networkx_graph():
    g = nx.DiGraph()
    g.add_nodes_from(["A", "B", "C", "D", "E"])
    pathway_dict = {"P1": ["A", "B", "C", "D", "E"]}
    df = pa.run_ora(g, pathway_dict=pathway_dict, min_pathway_size=5)
    assert df.iloc[0]["overlap"] == 5
    assert set(df.columns) >= {
        "pathway",
        "overlap",
        "pathway_size",
        "network_size",
        "background_size",
        "p_value",
        "fdr",
        "overlap_genes",
    }


def test_run_ora_empty_when_no_pathways_pass_filter():
    pathway_dict = {"P1": ["A", "B"]}
    df = pa.run_ora(["A"], pathway_dict=pathway_dict, min_pathway_size=5)
    assert df.empty


# --- compute_jaccard_from_membership ------------------------------------------


def test_compute_jaccard_known_values():
    # 3 genes x 2 pathways: P1={g0,g1}, P2={g1,g2} -> intersection 1, union 3.
    M = np.array([[1, 0], [1, 1], [0, 1]], dtype=bool)
    J = pa.compute_jaccard_from_membership(M)
    assert J.shape == (2, 2)
    assert J[0, 0] == 1.0 and J[1, 1] == 1.0
    assert np.isclose(J[0, 1], 1 / 3)
    assert np.isclose(J[0, 1], J[1, 0])


def test_compute_jaccard_disjoint_is_zero():
    M = np.array([[1, 0], [1, 0], [0, 1]], dtype=bool)
    J = pa.compute_jaccard_from_membership(M)
    assert J[0, 1] == 0.0


# --- coverage_greedy_select ---------------------------------------------------


def _membership_df():
    # P1 and P3 both cover {A,B}; P2 covers {C,D}.
    return pd.DataFrame(
        {
            "P1": [True, True, False, False],
            "P2": [False, False, True, True],
            "P3": [True, True, False, False],
        },
        index=["A", "B", "C", "D"],
    )


def test_coverage_greedy_select_covers_disjoint_pathways():
    M = _membership_df()
    s = pd.Series([1.0, 1.0, 1.0], index=["P1", "P2", "P3"])
    out = pa.coverage_greedy_select(M, s, K=2, mu=0.3, max_jaccard=0.5)
    assert len(out["selected"]) == 2
    assert out["covered_genes"] == {"A", "B", "C", "D"}
    # P3 duplicates P1 (Jaccard 1.0) so it is excluded under max_jaccard=0.5.
    assert "P3" not in out["selected"]


def test_coverage_greedy_select_respects_K():
    M = _membership_df()
    s = pd.Series([1.0, 1.0, 1.0], index=["P1", "P2", "P3"])
    out = pa.coverage_greedy_select(M, s, K=1, mu=0.0)
    assert len(out["selected"]) == 1


# --- select_diverse_pathways --------------------------------------------------


def test_select_diverse_pathways_filters_by_fdr():
    ora = pd.DataFrame(
        {
            "pathway": ["P1", "P2"],
            "fdr": [0.001, 0.9],
        }
    )
    pathway_dict = {"P1": ["A", "B", "C", "D", "E"], "P2": ["F", "G"]}
    background = ["A", "B", "C", "D", "E", "F", "G"]
    out = pa.select_diverse_pathways(
        ora, pathway_dict, background, K=5, fdr_threshold=0.05, min_pathway_size=1
    )
    # Only P1 passes the FDR filter, so only P1 can be selected.
    assert out["selected"] == ["P1"]


def test_select_diverse_pathways_empty_when_none_significant():
    ora = pd.DataFrame({"pathway": ["P1"], "fdr": [0.9]})
    out = pa.select_diverse_pathways(ora, {"P1": ["A", "B"]}, ["A", "B"], fdr_threshold=0.05)
    assert out["selected"] == []
    assert out["covered_genes"] == set()


# --- _apply_fdr ---------------------------------------------------------------


def test_apply_fdr_bonferroni():
    p = np.array([0.01, 0.02])
    out = pa._apply_fdr(p, method="bonferroni")
    assert np.allclose(out, [0.02, 0.04])


def test_apply_fdr_bonferroni_clipped_at_one():
    p = np.array([0.6, 0.8])
    out = pa._apply_fdr(p, method="bonferroni")
    assert np.all(out <= 1.0)


def test_apply_fdr_bh_monotone():
    p = np.array([0.01, 0.02, 0.03])
    out = pa._apply_fdr(p, method="bh")
    # BH: each = min(1, p*3/rank) then step-up monotonized -> all 0.03 here.
    assert np.allclose(out, [0.03, 0.03, 0.03])
    # Monotone non-decreasing in the order of increasing p-value.
    assert np.all(np.diff(out[np.argsort(p)]) >= -1e-12)


# --- _jsonable ----------------------------------------------------------------


def test_jsonable_passthrough_scalars():
    assert pa._jsonable("x") == "x"
    assert pa._jsonable(3) == 3
    assert pa._jsonable(1.5) == 1.5
    assert pa._jsonable(True) is True
    assert pa._jsonable(None) is None


def test_jsonable_collections_and_fallback():
    assert pa._jsonable((1, 2)) == [1, 2]
    assert pa._jsonable({1, 2}) == [1, 2] or pa._jsonable({1, 2}) == [2, 1]
    assert pa._jsonable({"k": (1, 2)}) == {"k": [1, 2]}
    # Non-serializable object falls back to str().
    assert pa._jsonable(np.int64(5)) == 5 or pa._jsonable(np.int64(5)) == "5"


# --- _require_gseapy ----------------------------------------------------------


def test_require_gseapy_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(pa, "_GSEAPY_AVAILABLE", False)
    with pytest.raises(ImportError):
        pa._require_gseapy()
