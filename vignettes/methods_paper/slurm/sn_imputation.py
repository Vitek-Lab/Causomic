from MScausality.causal_model.LVM import LVM
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.normalization import normalize
from MScausality.data_analysis.dataProcess import dataProcess
from MScausality.simulation.example_graphs import signaling_network

import pandas as pd
import numpy as np

import pickle

import y0
from y0.dsl import Variable
from eliater.regression import summary_statistics

import pyro
from sklearn.impute import KNNImputer

def intervention(model, int1, int2, outcome, scale_metrics):
    ## MScausality results
    model.intervention({list(int1.keys())[0]: (list(int1.values())[0] \
                                            - scale_metrics["mean"]) \
                                                / scale_metrics["std"]}, outcome)
    mscausality_int_low = model.intervention_samples
    model.intervention({list(int2.keys())[0]: (list(int2.values())[0] \
                                            - scale_metrics["mean"]) \
                                                / scale_metrics["std"]}, outcome)
    mscausality_int_high = model.intervention_samples

    mscausality_int_low = ((mscausality_int_low*scale_metrics["std"]) \
                        + scale_metrics["mean"])
    mscausality_int_high = ((mscausality_int_high*scale_metrics["std"]) \
                            + scale_metrics["mean"])
    mscausality_ate = mscausality_int_high.mean() - mscausality_int_low.mean()

    return mscausality_ate

def comparison(bulk_graph, 
               y0_graph_bulk, 
               msscausality_graph,
               coef, 
               int1, 
               int2, 
               outcome,
               data):
    
    # Ground truth
    intervention_low = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int1, 
                                    mnar_missing_param=[-4, .3],
                                    add_feature_var=False, n=10000, seed=2)

    intervention_high = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int2, 
                                    add_feature_var=False, n=10000, seed=2)

    gt_ate = (intervention_high["Protein_data"][outcome].mean() \
          - intervention_low["Protein_data"][outcome].mean())
    
    # Eliator prediction
    obs_data_eliator = data.copy()

    if data.isnull().values.any():
        imputer = KNNImputer(n_neighbors=3, keep_empty_features=True)
        # Impute missing values (the result is a NumPy array, so we need to convert it back to a DataFrame)
        obs_data_eliator = pd.DataFrame(imputer.fit_transform(obs_data_eliator), columns=data.columns)

    eliator_int_low = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int1.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int1.keys())[0]): list(int1.values())[0]})

    eliator_int_high = summary_statistics(
        y0_graph_bulk, obs_data_eliator,
        treatments={Variable(list(int2.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int2.keys())[0]): list(int2.values())[0]})
    
    eliator_ate = eliator_int_high.mean - eliator_int_low.mean

    # Basic Bayesian model
    pyro.clear_param_store()
    transformed_data = normalize(data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    # Full imp Bayesian model
    lvm = LVM(backend="numpyro")
    lvm.fit(input_data, msscausality_graph)

    full_imp_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    result_df = pd.DataFrame({
        "Ground_truth": [gt_ate],
        "Eliator": [eliator_ate],
        "MScausality": [full_imp_model_ate.item()],
        "MScausality_model" : [lvm],
        "obs_data_eliator" : [obs_data_eliator]
    })

    return result_df


def generate_sn_data(replicates, temp_seed, coefs):

    sn = signaling_network()

    # Mediator loop
    data = simulate_data(
        sn["Networkx"], 
        coefficients=coefs, 
        mnar_missing_param=[-4, .3],
        add_feature_var=True, 
        n=replicates, 
        seed=temp_seed)
    # data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=True,
        sim_data=True)
    
    summarized_data = summarized_data.loc[:, [
        i for i in summarized_data.columns if i not in ["IGF", "EGF"]]]

    result = comparison(
        sn["Networkx"], sn["y0"], sn["MScausality"],
        coefs, {"Ras": 5}, {"Ras": 7}, 
        "Erk", summarized_data)

    return result

# Benchmark
igf_result = list()

uninformative_prior_coefs = {
    'EGF': {'intercept': 6., "error": 1},
    'IGF': {'intercept': 5., "error": 1},
    'SOS': {'intercept': 2, "error": 1, 'EGF': 0.6, 'IGF': 0.6},
    'Ras': {'intercept': 3, "error": 1, 'SOS': .5},
    'PI3K': {'intercept': 0, "error": 1, 'EGF': .5, 'IGF': .5, 'Ras': .5},
    'Akt': {'intercept': 1., "error": 1, 'PI3K': 0.75},
    'Raf': {'intercept': 4, "error": 1, 'Ras': 0.8, 'Akt': -.4},
    'Mek': {'intercept': 2., "error": 1, 'Raf': 0.75},
    'Erk': {'intercept': -2, "error": 1, 'Mek': 1.2}}

# sn
igf_results = generate_sn_data(50, 1, uninformative_prior_coefs)
print("snarf")

# Save results
# with open('igf_results_with_missing_subset.pkl', 'wb') as file:
#     pickle.dump(igf_result, file)