"""Tests for pathway_correlation_analysis pure data-processing functions.

Covers protein-name parsing, MSstats wide-format prep, correlation-matrix
generation, and the gene-set querying helpers (which read a small JSON pathway
fixture). All logic here is pandas/numpy/json/re -- no external services.
"""

import importlib
import json

import numpy as np
import pandas as pd
import pytest

pca = importlib.import_module("causomic.data_analysis.pathway_correlation_analysis")


@pytest.fixture
def gene_set_file(tmp_path):
    sets = {
        "PW1": {"geneSymbols": ["TP53", "BRCA1", "EGFR"]},
        "PW2": {"geneSymbols": ["EGFR"]},
    }
    p = tmp_path / "pathways.json"
    p.write_text(json.dumps(sets))
    return str(p)


# --- parse_protein_name -------------------------------------------------------


def test_parse_protein_name_parse_gene_splits_on_underscore():
    df = pd.DataFrame({"Protein": ["P1_HUMAN", "P2_HUMAN"]})
    out = pca.parse_protein_name(df, parse_gene=True)
    assert list(out["Protein"]) == ["P1", "P2"]


def test_parse_protein_name_does_not_mutate_input():
    df = pd.DataFrame({"Protein": ["P1_HUMAN"]})
    pca.parse_protein_name(df, parse_gene=True)
    assert list(df["Protein"]) == ["P1_HUMAN"]


def test_parse_protein_name_gene_map_unmatched_becomes_nan():
    df = pd.DataFrame({"Protein": ["P1", "P2"]})
    gene_map = pd.DataFrame({"From": ["P1"], "To": ["G1"]})
    out = pca.parse_protein_name(df, gene_map=gene_map)
    assert out.loc[0, "Protein"] == "G1"
    assert pd.isna(out.loc[1, "Protein"])
    # merge helper columns are dropped
    assert "From" not in out.columns and "To" not in out.columns


# --- prep_msstats_data --------------------------------------------------------


def test_prep_msstats_data_pivots_to_wide():
    df = pd.DataFrame(
        {
            "Protein": ["A", "B"],
            "originalRUN": ["R1", "R1"],
            "LogIntensities": [1.0, 2.0],
        }
    )
    wide = pca.prep_msstats_data(df, verbose=False)
    assert list(wide.index) == ["R1"]
    assert wide.loc["R1", "A"] == 1.0
    assert wide.loc["R1", "B"] == 2.0


def test_prep_msstats_data_averages_duplicates():
    df = pd.DataFrame(
        {
            "Protein": ["A", "A"],
            "originalRUN": ["R1", "R1"],
            "LogIntensities": [2.0, 4.0],
        }
    )
    wide = pca.prep_msstats_data(df, verbose=False)
    assert wide.loc["R1", "A"] == 3.0


def test_prep_msstats_data_missing_column_raises():
    with pytest.raises(KeyError):
        pca.prep_msstats_data(pd.DataFrame({"Protein": ["A"]}), verbose=False)


# --- gen_correlation_matrix ---------------------------------------------------


def test_gen_correlation_matrix_drops_self_correlations():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})
    out = pca.gen_correlation_matrix(df, methods=["pearson"], abs_corr=False, verbose=False)
    corr = out["pearson"]
    # 2 columns => only the two off-diagonal pairs survive the value<1 filter.
    assert len(corr) == 2
    assert set(corr["index"]) == {("A", "B"), ("B", "A")}
    assert np.allclose(corr["value"].values, -1.0)


def test_gen_correlation_matrix_abs_corr():
    df = pd.DataFrame({"A": [1.0, 2.0, 3.0], "B": [3.0, 2.0, 1.0]})
    out = pca.gen_correlation_matrix(df, methods=["pearson"], abs_corr=True, verbose=False)
    assert np.allclose(out["pearson"]["value"].values, 1.0)


def test_gen_correlation_matrix_invalid_method_raises():
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]})
    with pytest.raises(ValueError):
        pca.gen_correlation_matrix(df, methods=["not_a_method"], verbose=False)


# --- test_gene_sets -----------------------------------------------------------


def _corr_frame(pairs_values):
    """Build a correlation frame with both orderings of each pair."""
    rows = []
    for (a, b), v in pairs_values.items():
        rows.append({"index": (a, b), "value": v})
        rows.append({"index": (b, a), "value": v})
    return pd.DataFrame(rows)


def test_gene_sets_counts_significant_pairs(tmp_path):
    sets = {"PW": {"geneSymbols": ["A", "B", "C"]}}
    path = tmp_path / "pw.json"
    path.write_text(json.dumps(sets))
    corr = _corr_frame({("A", "B"): 0.9, ("A", "C"): 0.1, ("B", "C"): 0.1})
    result = pca.test_gene_sets(
        {"pearson": corr}, ["A", "B", "C"], str(path), threshold=0.5, verbose=False
    )
    row = result.iloc[0]
    assert row["pathway"] == "PW"
    assert row["total_tests"] == 3
    assert row["sig_corrs"] == 1
    assert row["measured_genes"] == 3
    assert row["percent_measured"] == 1.0
    assert np.isclose(row["percent"], 1 / 3)


def test_gene_sets_insufficient_genes(tmp_path):
    sets = {"PW": {"geneSymbols": ["A", "B"]}}
    path = tmp_path / "pw.json"
    path.write_text(json.dumps(sets))
    corr = _corr_frame({("A", "B"): 0.9})
    result = pca.test_gene_sets({"pearson": corr}, ["A", "B"], str(path), verbose=False)
    assert result.iloc[0]["correlation"] == "insufficient_genes"
    assert result.iloc[0]["total_tests"] == 0


def test_gene_sets_invalid_fc_pval_raises(tmp_path):
    path = tmp_path / "pw.json"
    path.write_text(json.dumps({"PW": {"geneSymbols": ["A", "B", "C"]}}))
    with pytest.raises(ValueError):
        pca.test_gene_sets({"pearson": _corr_frame({})}, ["A"], str(path), fc_pval="bogus")


# --- extract_genes_in_path ----------------------------------------------------


def test_extract_genes_in_path_intersects_measured(gene_set_file):
    genes = pca.extract_genes_in_path(["TP53", "XYZ"], "PW1", gene_set_file)
    assert set(genes) == {"TP53"}


def test_extract_genes_in_path_return_all(gene_set_file):
    genes = pca.extract_genes_in_path(["TP53"], "PW1", gene_set_file, return_all=True)
    assert set(genes) == {"TP53", "BRCA1", "EGFR"}


def test_extract_genes_in_path_unknown_set_raises(gene_set_file):
    with pytest.raises(KeyError):
        pca.extract_genes_in_path(["TP53"], "NOPE", gene_set_file)


# --- find_sets_with_gene ------------------------------------------------------


def test_find_sets_with_gene_single(gene_set_file):
    assert pca.find_sets_with_gene("TP53", gene_set_file) == ["PW1"]


def test_find_sets_with_gene_requires_all(gene_set_file):
    # Both TP53 and BRCA1 only co-occur in PW1.
    assert pca.find_sets_with_gene(["TP53", "BRCA1"], gene_set_file) == ["PW1"]


def test_find_sets_with_gene_percent_threshold(gene_set_file):
    # EGFR is in both PW1 and PW2; threshold of 1 gene => both match.
    result = pca.find_sets_with_gene(["TP53", "EGFR"], gene_set_file, percent=1)
    assert set(result) == {"PW1", "PW2"}


def test_find_sets_with_gene_bad_type_raises(gene_set_file):
    with pytest.raises(TypeError):
        pca.find_sets_with_gene(123, gene_set_file)
