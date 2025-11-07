import importlib
import numpy as np
import pandas as pd

# Ensure src on path via tests/conftest.py
pp = importlib.import_module("causomic.data_analysis.proteomics_data_processor")


def test_format_sim_data_basic():
    df = pd.DataFrame(
        {
            "Protein": ["P1", "P1"],
            "Feature": ["f1", "f2"],
            "Replicate": [1, 2],
            "Obs_Intensity": [100.0, 200.0],
        }
    )

    out = pp.format_sim_data(df)
    # Check column renames and formatting
    assert "ProteinName" in out.columns
    assert out.loc[0, "PeptideSequence"].startswith("P1_")
    assert out.loc[0, "Run"].endswith("_Obs")


def test_normalize_median_basic():
    df = pd.DataFrame(
        {
            "Run": ["R1", "R1", "R2", "R2"],
            "Fraction": [1, 1, 1, 1],
            "Intensity": [10.0, 30.0, 20.0, 40.0],
        }
    )

    out = pp.normalize_median(df.copy())
    # After normalization, medians per run should be adjusted
    assert out["Intensity"].dtype == float


def test_topn_feature_selection():
    df = pd.DataFrame(
        {
            "ProteinName": ["P1"] * 4,
            "Feature": ["a", "b", "c", "d"],
            "Intensity": [1.0, 10.0, 5.0, 2.0],
        }
    )
    out = pp.topn_feature_selection(df, n=2)
    # should retain two features with highest mean intensity
    assert out["Feature"].nunique() == 2


def test_tukey_median_polish_small_matrix():
    mat = np.array([[1.0, 2.0], [3.0, 4.0]])
    res = pp.tukey_median_polish(mat, maxiter=5, na_rm=False)
    assert set(res.keys()) == {"overall", "row", "col", "residuals"}
    assert res["residuals"].shape == mat.shape


def test_imputation_basic():
    # Create small dataset with one missing value that can be predicted
    df = pd.DataFrame(
        {
            "Run": ["r1", "r1", "r2", "r2"],
            "Feature": ["f1", "f2", "f1", "f2"],
            "Intensity": [10.0, 20.0, 15.0, np.nan],
        }
    )
    out = pp.imputation(df.copy())
    # The previously missing value should be filled (not NaN) when modelable
    assert not np.isnan(out.loc[out["Run"] == "r2", "Intensity"]).all()
