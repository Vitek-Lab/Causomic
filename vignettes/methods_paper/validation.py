"""
Model Validation Script for Methods Paper

This script provides validation functionality for comparing MScausality results
against ground truth and other causal inference methods (Eliator). It's designed
for benchmarking and validation studies in the methods paper.

The validation compares Average Treatment Effects (ATE) across three approaches:
1. Ground truth - calculated from known causal structure and coefficients
2. Eliator - using the eliater package for causal inference
3. MScausality - using the MScausality Bayesian approach

Author: Devon Kohler
"""

import pandas as pd

from MScausality.causal_model.LVM import LVM
from MScausality.simulation.simulation import simulate_data
from MScausality.causal_model.normalization import normalize

import pyro

from eliater.regression import summary_statistics
from y0.dsl import Variable
from sklearn.impute import KNNImputer


def validate_model(data,
                   bulk_graph, 
                   y0_graph_bulk, 
                   msscausality_graph,
                   coef, 
                   int1, 
                   int2, 
                   outcome,
                   priors=None):
    """
    Validate a model by comparing ATE estimates across different methods.
    
    This function compares Average Treatment Effects (ATE) calculated using
    three different approaches: ground truth, Eliator, and MScausality.
    
    Parameters
    ----------
    data : pd.DataFrame
        Observational data for validation
    bulk_graph : networkx.DiGraph
        The true causal graph structure
    y0_graph_bulk : y0.graph.NxMixedGraph
        Graph in y0 format for Eliator
    msscausality_graph : MScausality graph format
        Graph in MScausality format
    coef : dict
        True coefficients for the causal relationships
    int1 : dict
        First intervention to compare (format: {variable: value})
    int2 : dict
        Second intervention to compare (format: {variable: value})
    outcome : str
        Target outcome variable for measuring effects
    priors : dict, optional
        Prior distributions for MScausality model
    
    Returns
    -------
    pd.DataFrame
        Comparison results with columns for each method's ATE estimate
    """

    gt_effect = gt_ate(bulk_graph, coef, int1, int2, outcome)
    eliator_effect = eliator_ate(y0_graph_bulk, data, int1, int2, outcome)
    mscausality_effect = mscausality_ate(msscausality_graph, 
                                         data, int1, int2, 
                                         outcome, priors)
    
    result_df = pd.DataFrame({
        "Ground_truth": [gt_effect],
        "Eliator": [eliator_effect],
        "MScausality": [mscausality_effect],
    })

    return result_df


def gt_ate(bulk_graph, coef, int1, int2, outcome):
    """
    Calculate the ground truth Average Treatment Effect between two interventions.

    This function simulates data under two different intervention scenarios
    using the true causal graph and coefficients, then calculates the
    difference in expected outcomes.

    Parameters
    ----------
    bulk_graph : networkx.DiGraph
        The true causal graph structure
    coef : dict
        True coefficients for the causal relationships
    int1 : dict
        First intervention (format: {variable: value})
    int2 : dict
        Second intervention (format: {variable: value})
    outcome : str
        Target outcome variable for measuring effects
        
    Returns
    -------
    float
        True Average Treatment Effect (ATE)
    """
    
    # Simulate data under first intervention
    intervention_low = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int1, 
                                    add_feature_var=False, n=10000, seed=2)

    # Simulate data under second intervention
    intervention_high = simulate_data(bulk_graph, coefficients=coef,
                                     intervention=int2, 
                                     add_feature_var=False, n=10000, seed=2)

    # Calculate difference in expected outcomes
    gt_ate = (intervention_high["Protein_data"][outcome].mean() - 
              intervention_low["Protein_data"][outcome].mean())
    
    return gt_ate


def eliator_ate(y0_graph_bulk, data, int1, int2, outcome):
    """
    Calculate the Eliator estimated Average Treatment Effect.

    This function uses the eliater package to estimate causal effects
    from observational data using the provided causal graph.

    Parameters
    ----------
    y0_graph_bulk : y0.graph.NxMixedGraph
        Causal graph in y0 format
    data : pd.DataFrame
        Observational data
    int1 : dict
        First intervention (format: {variable: value})
    int2 : dict
        Second intervention (format: {variable: value})
    outcome : str
        Target outcome variable
        
    Returns
    -------
    float
        Eliator estimated ATE
    """

    # Prepare data for Eliator (handle missing values)
    obs_data_eliator = data.copy()

    if data.isnull().values.any():
        imputer = KNNImputer(n_neighbors=3, keep_empty_features=True)
        obs_data_eliator = pd.DataFrame(
            imputer.fit_transform(obs_data_eliator), 
            columns=data.columns
        )

    # Estimate effect under first intervention
    eliator_int_low = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int1.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int1.keys())[0]): list(int1.values())[0]
        }
    )

    # Estimate effect under second intervention
    eliator_int_high = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int2.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int2.keys())[0]): list(int2.values())[0]
        }
    )
    
    # Calculate ATE
    eliator_ate = eliator_int_high.mean - eliator_int_low.mean

    return eliator_ate


def mscausality_ate(msscausality_graph, data, int1, int2, outcome, priors):
    """
    Calculate the MScausality estimated Average Treatment Effect.

    This function uses the MScausality Bayesian approach to estimate
    causal effects from observational data.

    Parameters
    ----------
    msscausality_graph : MScausality graph format
        Causal graph in MScausality format
    data : pd.DataFrame
        Observational data
    int1 : dict
        First intervention (format: {variable: value})
    int2 : dict
        Second intervention (format: {variable: value})
    outcome : str
        Target outcome variable
    priors : dict
        Prior distributions for the Bayesian model
        
    Returns
    -------
    float
        MScausality estimated ATE
    """
    
    # Clear Pyro parameter store for fresh inference
    pyro.clear_param_store()
    
    # Normalize data for Bayesian inference
    transformed_data = normalize(data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    # Fit Bayesian causal model
    lvm = LVM(backend="numpyro", informative_priors=priors)
    lvm.fit(input_data, msscausality_graph)

    # Perform intervention under first condition (normalize intervention value)
    normalized_int1_value = ((list(int1.values())[0] - scale_metrics["mean"]) / 
                            scale_metrics["std"])
    lvm.intervention({list(int1.keys())[0]: normalized_int1_value}, outcome)
    mscausality_int_low = lvm.intervention_samples
    
    # Perform intervention under second condition (normalize intervention value)
    normalized_int2_value = ((list(int2.values())[0] - scale_metrics["mean"]) / 
                            scale_metrics["std"])
    lvm.intervention({list(int2.keys())[0]: normalized_int2_value}, outcome)
    mscausality_int_high = lvm.intervention_samples

    # Transform back to original scale
    mscausality_int_low = ((mscausality_int_low * scale_metrics["std"]) + 
                          scale_metrics["mean"])
    mscausality_int_high = ((mscausality_int_high * scale_metrics["std"]) + 
                           scale_metrics["mean"])
    
    # Calculate ATE
    mscausality_ate = mscausality_int_high.mean() - mscausality_int_low.mean()

    return mscausality_ate.item()


def main():
    """
    Example validation run comparing methods on simulated signaling network data.
    
    This demonstrates the validation workflow using a simulated signaling
    network with missing data and feature-level measurements.
    """
    from MScausality.simulation.simulation import simulate_data
    from MScausality.data_analysis.proteomics_data_processor import dataProcess
    from MScausality.simulation.example_graphs import signaling_network

    print("Loading signaling network...")
    sn = signaling_network()

    print("Simulating data with missing values...")
    data = simulate_data(
        sn["Networkx"], 
        coefficients=sn["Coefficients"], 
        mnar_missing_param=[-4, 0.3],  # Missing not at random parameters
        add_feature_var=True, 
        n=30, 
        seed=200
    )
    data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    print("Processing feature-level data to protein-level...")
    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True
    )
    
    # Remove external interventions for validation
    summarized_data = summarized_data.loc[:, [
        col for col in summarized_data.columns if col not in ["IGF", "EGF"]
    ]]

    print("Running validation comparison...")
    # Compare interventions: Ras=5 vs Ras=7, measuring effect on Erk
    result = validate_model(
        summarized_data,
        sn["Networkx"], 
        sn["y0"], 
        sn["MScausality"],
        sn["Coefficients"], 
        {"Ras": 5},  # Low intervention
        {"Ras": 7},  # High intervention  
        "Erk"        # Outcome
    )
    
    print("\nValidation Results:")
    print("==================")
    print(result)
    
    # Calculate relative errors
    gt_effect = result["Ground_truth"].iloc[0]
    eliator_error = abs(result["Eliator"].iloc[0] - gt_effect) / abs(gt_effect) * 100
    mscausality_error = abs(result["MScausality"].iloc[0] - gt_effect) / abs(gt_effect) * 100
    
    print(f"\nRelative Errors (%):")
    print(f"Eliator: {eliator_error:.2f}%")
    print(f"MScausality: {mscausality_error:.2f}%")


if __name__ == "__main__":
    main()