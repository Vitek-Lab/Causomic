"""
Model Validation Script for Methods Paper

This script provides validation functionality for comparing causomic results
against ground truth and other causal inference methods (Eliator). It's designed
for benchmarking and validation studies in the methods paper.

The validation compares Average Treatment Effects (ATE) across three approaches:
1. Ground truth - calculated from known causal structure and coefficients
2. Eliator - using the eliater package for causal inference
3. causomic - using the causomic Bayesian approach

Author: Devon Kohler
"""

import pandas as pd
import numpy as np

from causomic.causal_model.LVM import LVM
from causomic.simulation.proteomics_simulator import simulate_data
from causomic.data_analysis.proteomics_data_processor import dataProcess
from causomic.simulation.example_graphs import signaling_network

import pyro

from eliater.regression import summary_statistics
from y0.dsl import Variable
from sklearn.impute import KNNImputer
import secrets
import os
import time
from concurrent.futures import as_completed


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
    three different approaches: ground truth, Eliator, and causomic.
    
    Parameters
    ----------
    data : pd.DataFrame
        Observational data for validation
    bulk_graph : networkx.DiGraph
        The true causal graph structure
    y0_graph_bulk : y0.graph.NxMixedGraph
        Graph in y0 format for Eliator
    msscausality_graph : causomic graph format
        Graph in causomic format
    coef : dict
        True coefficients for the causal relationships
    int1 : dict
        First intervention to compare (format: {variable: value})
    int2 : dict
        Second intervention to compare (format: {variable: value})
    outcome : str
        Target outcome variable for measuring effects
    priors : dict, optional
        Prior distributions for causomic model
    
    Returns
    -------
    pd.DataFrame
        Comparison results with columns for each method's ATE estimate
    """

    gt_effect = gt_ate(bulk_graph, coef, int1, int2, outcome)
    eliator_effect = eliator_ate(y0_graph_bulk, data, int1, int2, outcome)
    causomic_effect = causomic_ate(msscausality_graph, 
                                         data, int1, int2, 
                                         outcome, priors)
    
    result_df = pd.DataFrame({
        "Ground_truth": [gt_effect],
        "Eliator": [eliator_effect],
        "causomic": [causomic_effect],
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


def causomic_ate(msscausality_graph, data, int1, int2, outcome, priors):
    """
    Calculate the causomic estimated Average Treatment Effect.

    This function uses the causomic Bayesian approach to estimate
    causal effects from observational data.

    Parameters
    ----------
    msscausality_graph : causomic graph format
        Causal graph in causomic format
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
        causomic estimated ATE
    """
    
    # Clear Pyro parameter store for fresh inference
    pyro.clear_param_store()
    
    # Normalize data for Bayesian inference

    # Fit Bayesian causal model
    lvm = LVM(backend="pyro",
        num_steps=10000,
        patience=10,
        initial_lr=.01,
        verbose=True,
        informative_priors=priors)
    lvm.fit(data, msscausality_graph)

    # Perform intervention under first condition (normalize intervention value)
    lvm.intervention(int1, outcome)
    causomic_int_low = lvm.intervention_samples
    
    # Perform intervention under second condition (normalize intervention value)
    lvm.intervention(int2, outcome)
    causomic_int_high = lvm.intervention_samples
    
    # Calculate ATE
    causomic_ate = causomic_int_high.mean() - causomic_int_low.mean()

    return causomic_ate.item()

def run_validation(reps):
    
    seed = secrets.randbelow(2**31 - 1)
    np.random.seed(seed)
    print(f"Sampled random seed: {seed}")
    print("Loading signaling network...")
    sn = signaling_network()

    print("Simulating data with missing values...")
    data = simulate_data(
        sn["Networkx"], 
        coefficients=sn["Coefficients"], 
        mnar_missing_param=[-4, 0.3],  # Missing not at random parameters
        add_feature_var=True, 
        n=reps, 
        seed=seed
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
        sn["causomic"],
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
    causomic_error = abs(result["causomic"].iloc[0] - gt_effect) / abs(gt_effect) * 100
    
    print(f"\nRelative Errors (%):")
    print(f"Eliator: {eliator_error:.2f}%")
    print(f"causomic: {causomic_error:.2f}%")

    return gt_effect, result["Eliator"].iloc[0], result["causomic"].iloc[0]

if __name__ == "__main__":
    
    
    import concurrent.futures

    reps = 250  # number of samples per run

    def _worker_run(_):
        # wrapper so ProcessPoolExecutor can map easily
        return run_validation(reps)

    n_runs = 500
    
    # limit number of parallel workers sensibly; adjust if you truly want 100 simultaneous processes
    max_workers = min(n_runs, os.cpu_count() or 1)

    results = []
    start_time = time.time()
    print(f"Launching {n_runs} validation runs with up to {max_workers} workers...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_worker_run, i): i for i in range(n_runs)}
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                gt, eliator, causomic = fut.result()
                results.append({
                    "run": idx,
                    "Ground_truth": gt,
                    "Eliator": eliator,
                    "causomic": causomic,
                    "error": ""
                })
                print(f"Run {idx} completed")
            except Exception as e:
                # record failures
                results.append({
                    "run": idx,
                    "Ground_truth": None,
                    "Eliator": None,
                    "causomic": None,
                    "error": str(e)
                })
                print(f"Run {idx} failed: {e}")

    df_results = pd.DataFrame(results).sort_values("run").reset_index(drop=True)

    # write results to CSV next to this script (fallback to cwd if __file__ not defined)
    script_dir = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
    out_path = os.path.join(script_dir, f"{reps}_rep_validation_results.csv")
    df_results.to_csv(out_path, index=False)

    print(f"All runs finished in {time.time() - start_time:.1f}s. Results saved to: {out_path}")