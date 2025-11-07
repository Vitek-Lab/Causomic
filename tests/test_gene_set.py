import os
import sys
import json

import pandas as pd
import numpy as np

# Ensure src/ is on sys.path so tests can import the package directly
TEST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(TEST_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import importlib

# Import the module object rather than importing names that start with `test_`
# (importing a name starting with `test_` into the test module would cause
# pytest to treat it as a test function and attempt to collect it).
gs = importlib.import_module("causomic.data_analysis.gene_set")


def get_example_gene_set_path():
    return os.path.join(os.path.dirname(__file__), "data", "example_pathways.json")


def test_parse_protein_name_parse_and_map():
    df = pd.DataFrame({"Protein": ["GENE1_001", "GENE2_002", "GENE3_003"]})

    parsed = gs.parse_protein_name(df, parse_gene=True)
    assert list(parsed["Protein"]) == ["GENE1", "GENE2", "GENE3"]

    gene_map = pd.DataFrame({"From": ["GENE1", "GENE2"], "To": ["G1", "G2"]})
    mapped = gs.parse_protein_name(df, parse_gene=True, gene_map=gene_map)
    # GENE3 has no mapping so becomes NaN -> string conversion gives 'nan'
    assert mapped.loc[0, "Protein"] == "G1"
    assert mapped.loc[1, "Protein"] == "G2"


def test_prep_msstats_data_handles_duplicates_and_pivot():
    df = pd.DataFrame(
        {
            "Protein": ["P1", "P1", "P2", "P2", "P3"],
            "originalRUN": ["R1", "R1", "R1", "R2", "R1"],
            "LogIntensities": [1.0, 3.0, 2.0, 4.0, 5.0],
        }
    )

    wide = gs.prep_msstats_data(df)
    # After grouping mean for P1@R1 -> (1+3)/2 = 2.0
    assert "P1" in wide.columns
    assert wide.loc["R1", "P1"] == 2.0
    # P2 at R2 should be 4.0
    assert wide.loc["R2", "P2"] == 4.0


def test_gen_correlation_matrix_basic():
    # Create a small wide dataframe where A and B are positively correlated but not perfect
    df = pd.DataFrame({
        "A": [1.0, 2.0, 3.0, 4.0],
        "B": [1.1, 1.9, 3.05, 4.2],
        "C": [10.0, 9.0, 7.0, 3.0],
    })

    corr = gs.gen_correlation_matrix(df, methods=["pearson"], abs_corr=False)
    assert "pearson" in corr
    pearson_df = corr["pearson"]
    # Expect at least one pair (A,B) in the results
    pairs = [t for t in pearson_df["index"].tolist()]
    assert any(("A", "B") == p or ("B", "A") == p for p in pairs)


def test_extract_genes_in_path_and_return_all(tmp_path):
    js_path = get_example_gene_set_path()
    measured = ["G1", "G3"]
    genes_measured = gs.extract_genes_in_path(measured, "Pathway1", js_path, return_all=False)
    assert set(genes_measured) <= set(measured)

    all_genes = gs.extract_genes_in_path(measured, "Pathway1", js_path, return_all=True)
    assert "G2" in all_genes


def test_find_sets_with_gene_variants():
    js_path = get_example_gene_set_path()
    # Single gene regex
    hits = gs.find_sets_with_gene("G1", js_path)
    assert "Pathway1" in hits

    # Multi-gene all-members
    gene_list = ["G1", "G2"]
    hits_all = gs.find_sets_with_gene(gene_list, js_path)
    assert "Pathway1" in hits_all

    # Percent threshold
    hits_pct = gs.find_sets_with_gene(["G1", "G4", "G5"], js_path, percent=1)
    assert isinstance(hits_pct, list)


def test_test_gene_sets_basic():
    js_path = get_example_gene_set_path()
    # Create a correlation DataFrame compatible with test_gene_sets
    # Include both orientations (A,B) and (B,A) as produced by gen_correlation_matrix
    base_pairs = [("G1", "G2"), ("G1", "G3"), ("G2", "G3")]
    pairs = []
    values = []
    for p, v in zip(base_pairs, [0.6, 0.2, 0.7]):
        pairs.append(p)
        pairs.append((p[1], p[0]))
        values.append(v)
        values.append(v)

    corr_df = pd.DataFrame({"index": pairs, "value": values})

    correlation_data = {"pearson": corr_df}
    measured = ["G1", "G2", "G3"]

    results = gs.test_gene_sets(correlation_data, measured, js_path, threshold=0.5)
    # Pathway1 has 3 genes measured -> should return a row for it
    assert "Pathway1" in results["pathway"].values
    row = results.loc[results["pathway"] == "Pathway1"].iloc[0]
    # total_tests should be 3 for 3 measured genes (3 pairs)
    assert row["total_tests"] == 3
    # sig_corrs should count pairs with value > 0.5 -> 2
    assert row["sig_corrs"] == 2
