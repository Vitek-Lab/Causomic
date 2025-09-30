"""
Data Processing Module for Mass Spectrometry Data Analysis

This module provides functionality for processing mass spectrometry data,
including normalization, feature selection, imputation, and summarization.
It implements MSstats-like functionality in Python for proteomics data analysis.

Author: Devon Kohler
"""

import copy
from typing import Any, Dict, Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def format_sim_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Format simulated data to match MSstats input format.

    This function converts simulation data with columns [Protein, Feature, Replicate, Obs_Intensity]
    to the standard MSstats format with additional metadata columns required for processing.

    Args:
        data (pd.DataFrame): Input dataframe with columns:
            - Protein: Protein identifier
            - Feature: Feature/peptide identifier
            - Replicate: Biological replicate number
            - Obs_Intensity: Observed intensity value

    Returns:
        pd.DataFrame: Formatted dataframe with MSstats-compatible columns:
            - ProteinName: Protein identifier
            - PeptideSequence: Peptide sequence (formatted as ProteinName_Feature)
            - BioReplicate: Biological replicate
            - Intensity: Observed intensity
            - PrecursorCharge: Precursor charge (set to 2)
            - FragmentIon: Fragment ion (set to NaN)
            - ProductCharge: Product charge (set to NaN)
            - IsotopeLabelType: Isotope label type (set to 'L')
            - Condition: Experimental condition (set to 'Obs')
            - Run: Run identifier (formatted as BioReplicate_Condition)
            - Fraction: Fraction number (set to 1)
            - Feature: Feature identifier (formatted for MSstats)
    """
    # Select and rename core columns
    data = data.loc[:, ["Protein", "Feature", "Replicate", "Obs_Intensity"]].copy()
    data = data.rename(
        columns={
            "Protein": "ProteinName",
            "Feature": "PeptideSequence",
            "Replicate": "BioReplicate",
            "Obs_Intensity": "Intensity",
        }
    )

    # Add required metadata columns
    data.loc[:, "PrecursorCharge"] = 2
    data.loc[:, "FragmentIon"] = np.nan
    data.loc[:, "ProductCharge"] = np.nan
    data.loc[:, "IsotopeLabelType"] = "L"
    data.loc[:, "Condition"] = "Obs"
    data.loc[:, "Run"] = data.loc[:, "BioReplicate"]
    data.loc[:, "Fraction"] = 1

    # Format peptide sequence and run identifiers
    data.loc[:, "PeptideSequence"] = (
        data.loc[:, "ProteinName"].astype(str) + "_" + data.loc[:, "PeptideSequence"].astype(str)
    )
    data.loc[:, "Run"] = data.loc[:, "Run"].astype(str) + "_" + data.loc[:, "Condition"].astype(str)

    # Create MSstats-compatible feature identifier
    data.loc[:, "Feature"] = (
        data.loc[:, "PeptideSequence"].astype(str)
        + "_"
        + data.loc[:, "PrecursorCharge"].astype(str)
        + "_"
        + data.loc[:, "FragmentIon"].astype(str)
        + "_"
        + data.loc[:, "ProductCharge"].astype(str)
        + "_"
    )

    return data


def normalize_median(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply median-based normalization to intensity data.

    This function performs run-level normalization by adjusting intensities
    based on median values at the run and fraction levels. This helps to
    reduce systematic bias across different runs and fractions.

    Args:
        data (pd.DataFrame): Input dataframe containing columns:
            - Run: Run identifier
            - Fraction: Fraction identifier
            - Intensity: Intensity values to normalize

    Returns:
        pd.DataFrame: Normalized dataframe with adjusted intensity values

    Note:
        The normalization formula is:
        Intensity_normalized = Intensity - ABUNDANCE_RUN + ABUNDANCE_FRACTION
        where ABUNDANCE_RUN is the median intensity per run/fraction,
        and ABUNDANCE_FRACTION is the median of run medians per fraction.
    """
    # Calculate median intensity per run and fraction (ABUNDANCE_RUN)
    abundance_run = (
        data.groupby(["Run", "Fraction"])["Intensity"]
        .median()
        .reset_index()
        .rename(columns={"Intensity": "ABUNDANCE_RUN"})
    )

    # Calculate median of run abundances per fraction (ABUNDANCE_FRACTION)
    abundance_fraction = (
        abundance_run.groupby("Fraction")["ABUNDANCE_RUN"]
        .median()
        .reset_index()
        .rename(columns={"ABUNDANCE_RUN": "ABUNDANCE_FRACTION"})
    )

    # Merge abundance information with original data
    data = pd.merge(
        pd.merge(data, abundance_run, on=["Run", "Fraction"], how="left"),
        abundance_fraction,
        on="Fraction",
        how="left",
    )

    # Apply normalization
    data["Intensity"] = data["Intensity"] - data["ABUNDANCE_RUN"] + data["ABUNDANCE_FRACTION"]

    # Remove temporary columns
    data = data.drop(columns=["ABUNDANCE_RUN", "ABUNDANCE_FRACTION"])

    print("INFO: Median-based normalization completed successfully")

    return data


def topn_feature_selection(data: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Select top N features per protein based on mean intensity.

    This function performs feature selection by keeping only the N features
    with the highest mean intensity for each protein. This helps reduce
    noise and computational complexity while retaining the most informative features.

    Args:
        data (pd.DataFrame): Input dataframe containing columns:
            - ProteinName: Protein identifier
            - Feature: Feature identifier
            - Intensity: Intensity values
        n (int): Number of top features to select per protein

    Returns:
        pd.DataFrame: Filtered dataframe containing only the top N features per protein

    Note:
        Features are ranked by mean intensity across all runs for each protein.
        If a protein has fewer than N features, all available features are retained.
    """
    proteins = data["ProteinName"].unique()
    selected_data_list = []

    for protein in proteins:
        protein_data = data.loc[data["ProteinName"] == protein].copy()

        # Calculate mean intensity per feature for this protein
        feature_means = (
            protein_data.groupby("Feature")["Intensity"].mean().sort_values(ascending=False).head(n)
        )

        # Filter to keep only top N features
        protein_data_filtered = protein_data[protein_data["Feature"].isin(feature_means.index)]

        selected_data_list.append(protein_data_filtered)

    # Concatenate all protein data
    result_data = pd.concat(selected_data_list, ignore_index=True)
    return result_data


def imputation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing intensity values using linear regression.

    This function imputes missing intensity values by fitting a linear model
    with run and feature effects. It uses one-hot encoding for categorical
    variables and predicts missing values based on the trained model.

    Args:
        data (pd.DataFrame): Input dataframe containing columns:
            - Run: Run identifier
            - Feature: Feature identifier
            - Intensity: Intensity values (may contain NaN)

    Returns:
        pd.DataFrame: Dataframe with imputed intensity values

    Note:
        - Runs and features with all missing values are excluded from modeling
        - Extreme predicted values (|value| > 40) are set back to NaN
        - Uses sklearn LinearRegression with one-hot encoded run and feature effects
    """
    # Skip imputation if all values are missing
    if data["Intensity"].isna().mean() == 1:
        return data

    # Identify runs and features that have at least some non-missing values
    run_missing_rates = data["Intensity"].isna().groupby(data["Run"]).mean()
    feature_missing_rates = data["Intensity"].isna().groupby(data["Feature"]).mean()

    keep_runs = run_missing_rates[run_missing_rates != 1].index.values
    keep_features = feature_missing_rates[feature_missing_rates != 1].index.values

    # Split data into modelable and non-modelable portions
    keep_mask = data["Run"].isin(keep_runs) & data["Feature"].isin(keep_features)
    keep_data = data[keep_mask].copy()
    exclude_data = data[~keep_mask].copy()

    if len(keep_data) == 0:
        return data

    # Create one-hot encodings for runs and features
    run_dummies = pd.get_dummies(keep_data["Run"], prefix="Run")
    feature_dummies = pd.get_dummies(keep_data["Feature"], prefix="Feature")

    # Combine predictors with target variable
    model_data = pd.concat(
        [run_dummies, feature_dummies, keep_data["Intensity"].reset_index(drop=True)], axis=1
    )

    # Split into training (non-missing) and prediction (missing) sets
    train_mask = model_data["Intensity"].notna()
    train_data = model_data[train_mask]
    test_data = model_data[~train_mask]

    if len(train_data) == 0 or len(test_data) == 0:
        return data

    # Prepare features and target
    feature_columns = [col for col in model_data.columns if col != "Intensity"]
    X_train = train_data[feature_columns]
    y_train = train_data["Intensity"]
    X_test = test_data[feature_columns]

    # Fit linear regression model and predict missing values
    model = LinearRegression()
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)

    # Update missing values with predictions
    model_data.loc[~train_mask, "Intensity"] = predicted_values

    # Reconstruct the dataframe with imputed values
    keep_data.loc[:, "Intensity"] = model_data["Intensity"].values
    result_data = pd.concat([keep_data, exclude_data], ignore_index=True)

    # Set extreme values back to NaN (likely poor predictions)
    result_data.loc[:, "Intensity"] = np.where(
        abs(result_data.loc[:, "Intensity"]) > 40, np.nan, result_data.loc[:, "Intensity"]
    )

    return result_data


def tukey_median_polish(
    data: np.ndarray,
    eps: float = 0.01,
    maxiter: int = 10,
    trace_iter: bool = True,
    na_rm: bool = True,
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Perform Tukey's median polish algorithm for robust data summarization.

    This function implements Tukey's median polish, which decomposes a two-way
    table into overall effect, row effects, column effects, and residuals using
    medians instead of means for robustness to outliers.

    Args:
        data (np.ndarray): 2D array to be decomposed (rows x columns)
        eps (float, optional): Convergence tolerance. Defaults to 0.01.
        maxiter (int, optional): Maximum number of iterations. Defaults to 10.
        trace_iter (bool, optional): Whether to trace iterations. Defaults to True.
        na_rm (bool, optional): Whether to remove NAs when computing medians. Defaults to True.

    Returns:
        Dict[str, Union[float, np.ndarray]]: Dictionary containing:
            - 'overall': Overall effect (scalar)
            - 'row': Row effects (1D array)
            - 'col': Column effects (1D array)
            - 'residuals': Residual matrix (2D array)

    Note:
        The algorithm iteratively removes row and column medians until convergence.
        The decomposition satisfies: data ≈ overall + row[i] + col[j] + residuals[i,j]
    """
    # Initialize working matrix and parameters
    z = copy.deepcopy(data)
    nr, nc = data.shape
    overall = 0.0
    oldsum = 0.0

    # Initialize row and column effects
    row_effects = np.zeros(nr)
    col_effects = np.zeros(nc)

    # Iterative median polish
    for iteration in range(maxiter):
        # Remove row medians
        if na_rm:
            row_deltas = np.array([np.nanmedian(z[i, :]) for i in range(nr)])
        else:
            row_deltas = np.array([np.median(z[i, :]) for i in range(nr)])

        # Subtract row effects and update row effects
        z = z - row_deltas[:, np.newaxis]
        row_effects = row_effects + row_deltas

        # Adjust column effects for row median effect
        if na_rm:
            delta = np.nanmedian(col_effects)
        else:
            delta = np.median(col_effects)

        col_effects = col_effects - delta
        overall = overall + delta

        # Remove column medians
        if na_rm:
            col_deltas = np.array([np.nanmedian(z[:, j]) for j in range(nc)])
        else:
            col_deltas = np.array([np.median(z[:, j]) for j in range(nc)])

        # Subtract column effects and update column effects
        z = z - col_deltas[np.newaxis, :]
        col_effects = col_effects + col_deltas

        # Adjust row effects for column median effect
        if na_rm:
            delta = np.nanmedian(row_effects)
        else:
            delta = np.median(row_effects)

        row_effects = row_effects - delta
        overall = overall + delta

        # Check for convergence
        if na_rm:
            newsum = np.nansum(np.abs(z))
        else:
            newsum = np.sum(np.abs(z))

        converged = (newsum == 0) or (abs(newsum - oldsum) < eps * newsum)
        if converged:
            break

        oldsum = newsum

    # Note: Convergence warning could be added here if needed
    if not converged and trace_iter:
        print(f"WARNING: Median polish did not converge after {maxiter} iterations")

    return {"overall": overall, "row": row_effects, "col": col_effects, "residuals": z}


def summarize_data(
    data: pd.DataFrame, summarization_method: Literal["TMP", "median", "mean"], MBimpute: bool
) -> pd.DataFrame:
    """
    Summarize protein-level data across features using specified method.

    This function aggregates feature-level data to protein-level measurements
    using Tukey's median polish (TMP), median, or mean summarization methods.

    Args:
        data (pd.DataFrame): Input dataframe containing columns:
            - ProteinName: Protein identifier
            - Feature: Feature identifier
            - RUN: Run identifier (numeric)
            - Intensity: Intensity values
        summarization_method (str): Method for summarization:
            - "TMP": Tukey's median polish
            - "median": Simple median across features
            - "mean": Simple mean across features
        MBimpute (bool): Whether to apply imputation before summarization

    Returns:
        pd.DataFrame: Summarized dataframe with proteins as columns and runs as rows

    Note:
        For TMP method, the protein-level value is overall + column effects.
        For median/mean methods, values are aggregated across features for each run.
    """
    proteins = data["ProteinName"].unique()
    summarized_data = pd.DataFrame()

    for protein in proteins:
        # Extract protein-specific data
        protein_data = data[data["ProteinName"] == protein].copy()

        # Apply imputation if requested
        if MBimpute:
            protein_data = imputation(protein_data)

        # Pivot to features x runs matrix
        intensity_matrix = protein_data.pivot(index="Feature", columns="RUN", values="Intensity")

        # Apply summarization method
        if summarization_method == "TMP":
            # Convert to numpy array for median polish
            matrix_array = intensity_matrix.to_numpy()
            tmp_result = tukey_median_polish(matrix_array)
            # Protein level = overall effect + column effects
            protein_summary = tmp_result["overall"] + tmp_result["col"]

        elif summarization_method == "median":
            # Simple median across features
            protein_summary = intensity_matrix.median(axis=0, skipna=True).values

        elif summarization_method == "mean":
            # Simple mean across features
            protein_summary = intensity_matrix.mean(axis=0, skipna=True).values

        else:
            raise ValueError(f"Unknown summarization method: {summarization_method}")

        # Store results
        summarized_data.loc[:, protein] = protein_summary

    return summarized_data


def dataProcess(
    data: pd.DataFrame,
    normalization: Union[Literal["equalizeMedians"], bool] = "equalizeMedians",
    feature_selection: Literal["All", "TopN"] = "All",
    n_features: int = 3,
    summarization_method: Literal["TMP", "median", "mean"] = "TMP",
    MBimpute: bool = True,
    sim_data: bool = False,
) -> pd.DataFrame:
    """
    Process mass spectrometry data through normalization, feature selection, and summarization.

    This function implements a comprehensive data processing pipeline for mass spectrometry
    proteomics data, similar to MSstats dataProcess functionality. It handles data formatting,
    normalization, feature selection, and protein-level summarization.

    Args:
        data (pd.DataFrame): Input dataframe containing mass spectrometry data.
            For real data: should have columns like PeptideSequence, PrecursorCharge,
            FragmentIon, ProductCharge, Run, Intensity.
            For simulated data: should have columns Protein, Feature, Replicate, Obs_Intensity.

        normalization (str or bool, optional): Normalization method.
            - "equalizeMedians": Apply median-based normalization
            - False: Skip normalization
            Defaults to "equalizeMedians".

        feature_selection (str, optional): Feature selection method.
            - "All": Use all features
            - "TopN": Select top N features per protein by mean intensity
            Defaults to "All".

        n_features (int, optional): Number of features to select per protein when
            feature_selection="TopN". Defaults to 3.

        summarization_method (str, optional): Method for protein-level summarization.
            - "TMP": Tukey's median polish (robust)
            - "median": Simple median across features
            - "mean": Simple mean across features
            Defaults to "TMP".

        MBimpute (bool, optional): Whether to apply model-based imputation for
            missing values. Defaults to True.

        sim_data (bool, optional): Whether input data is from simulation (requires
            special formatting). Defaults to False.

    Returns:
        pd.DataFrame: Processed data with proteins as columns and runs as rows.
        Values represent protein-level abundances after processing.

    Raises:
        ValueError: If invalid parameter values are provided.

    Example:
        >>> # Process real MS data
        >>> processed_data = dataProcess(
        ...     data=ms_data,
        ...     normalization="equalizeMedians",
        ...     feature_selection="TopN",
        ...     n_features=5,
        ...     summarization_method="TMP"
        ... )

        >>> # Process simulated data
        >>> processed_sim = dataProcess(
        ...     data=sim_data,
        ...     sim_data=True,
        ...     normalization=False
        ... )

    Note:
        The processing pipeline follows these steps:
        1. Data formatting (simulation data only)
        2. Log2 transformation of intensities (real data only)
        3. Run factorization
        4. Normalization (if requested)
        5. Feature selection (if requested)
        6. Protein-level summarization with optional imputation
    """
    # Validate inputs
    valid_normalization = ["equalizeMedians", False]
    valid_feature_selection = ["All", "TopN"]
    valid_summarization = ["TMP", "median", "mean"]

    if normalization not in valid_normalization:
        raise ValueError(f"normalization must be one of {valid_normalization}")
    if feature_selection not in valid_feature_selection:
        raise ValueError(f"feature_selection must be one of {valid_feature_selection}")
    if summarization_method not in valid_summarization:
        raise ValueError(f"summarization_method must be one of {valid_summarization}")
    if n_features <= 0:
        raise ValueError("n_features must be a positive integer")

    # Make a copy to avoid modifying original data
    processed_data = data.copy()

    # Step 1: Format data based on source
    if sim_data:
        processed_data = format_sim_data(processed_data)
    else:
        # Log2 transform intensities for real data
        processed_data["Intensity"] = np.log2(processed_data["Intensity"])

        # Create MSstats-compatible feature identifier
        processed_data.loc[:, "Feature"] = (
            processed_data.loc[:, "PeptideSequence"].astype(str)
            + "_"
            + processed_data.loc[:, "PrecursorCharge"].astype(str)
            + "_"
            + processed_data.loc[:, "FragmentIon"].astype(str)
            + "_"
            + processed_data.loc[:, "ProductCharge"].astype(str)
            + "_"
        )

    # Step 2: Create numeric run identifiers
    processed_data.loc[:, "RUN"] = pd.factorize(processed_data.loc[:, "Run"])[0]

    # Step 3: Apply normalization
    if normalization == "equalizeMedians":
        processed_data = normalize_median(processed_data)

    # Step 4: Apply feature selection
    if feature_selection == "TopN":
        processed_data = topn_feature_selection(processed_data, n_features)

    # Step 5: Summarize to protein level
    summarized_data = summarize_data(processed_data, summarization_method, MBimpute)

    return summarized_data


def main() -> None:
    """
    Example usage and testing of the dataProcess function.

    This function demonstrates how to use the dataProcess pipeline with
    simulated data from the causomic package. It creates a signaling
    network, simulates data with missing values, processes it, and visualizes
    the results.
    """
    try:
        from causomic.simulation.example_graphs import signaling_network
        from src.causomic.simulation.proteomics_simulator import simulate_data
    except ImportError as e:
        print(f"Error importing causomic modules: {e}")
        print("Please ensure causomic is properly installed")
        return

    # Generate example signaling network
    network_data = signaling_network(add_independent_nodes=False)

    # Simulate mass spectrometry data with missing values
    simulated_data = simulate_data(
        network_data["Networkx"],
        coefficients=network_data["Coefficients"],
        mnar_missing_param=[-5, 0.4],  # Missing not at random parameters
        add_feature_var=True,
        n=25,  # Number of replicates
        seed=3,
    )

    # Process the simulated feature data
    processed_data = dataProcess(
        data=simulated_data["Feature_data"],
        normalization=False,  # Skip normalization for simulated data
        summarization_method="TMP",
        MBimpute=True,
        sim_data=True,
    )

    # Visualize results: scatter plot of two proteins
    if "SOS" in processed_data.columns and "Ras" in processed_data.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(processed_data["SOS"], processed_data["Ras"], alpha=0.7)
        ax.set_xlabel("SOS Protein Abundance")
        ax.set_ylabel("Ras Protein Abundance")
        ax.set_title("Processed Protein Abundances: SOS vs Ras")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print("SOS and/or Ras proteins not found in processed data")
        print(f"Available proteins: {list(processed_data.columns)}")


if __name__ == "__main__":
    main()
