from MScausality.causal_model.LVM import LVM
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.normalization import normalize
from MScausality.data_analysis.dataProcess import dataProcess
from MScausality.simulation.example_graphs import mediator, frontdoor, backdoor


import pandas as pd
import numpy as np

import pickle

import y0
from y0.dsl import Variable
from eliater.regression import summary_statistics

import pyro

from concurrent.futures import ThreadPoolExecutor


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
                                    intervention=int1, mnar_missing_param=[-4, .3],
                                    add_feature_var=False, n=10000, seed=2)

    intervention_high = simulate_data(bulk_graph, coefficients=coef,
                                    intervention=int2, 
                                    add_feature_var=False, n=10000, seed=2)

    gt_ate = (intervention_high["Protein_data"][outcome].mean() \
          - intervention_low["Protein_data"][outcome].mean())
    
    # Eliator prediction
    obs_data_eliator = data.copy()

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
    lvm = LVM(input_data, msscausality_graph)
    lvm.prepare_graph()
    lvm.prepare_data()
    lvm.get_priors()

    # for i in lvm.priors.keys():
    #     for v in lvm.priors[i].keys():
    #         if ("coef" in v): 
    #             lvm.priors[i][v] = .75

    lvm.fit_model(num_steps=10000)

    full_imp_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    result_df = pd.DataFrame({
        "Ground_truth": [gt_ate],
        "Eliator": [eliator_ate],
        "MScausality": [full_imp_model_ate.item()],
    })

    return result_df


def generate_med_data(replicates):

    temp_seed = np.random.randint(1000000)
    med = mediator()

    # Mediator loop
    data = simulate_data(
        med["Networkx"], 
        coefficients=med["Coefficients"], 
        mnar_missing_param=[20, .3], # No missingness
        add_feature_var=True, 
        n=replicates, 
        seed=temp_seed)
    data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True)
    
    result = comparison(
        med["Networkx"], med["y0"], med["MScausality"],
        med["Coefficients"], {"X": 0}, {"X": 2}, 
        "Z", summarized_data)

    return result

def generate_bd_data(replicates):

    temp_seed = np.random.randint(1000000)
    bd = backdoor()

    # Mediator loop
    data = simulate_data(
        bd["Networkx"], 
        coefficients=bd["Coefficients"], 
        mnar_missing_param=[20, .3], # No missingness
        add_feature_var=True, 
        n=replicates, 
        seed=temp_seed)
    data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True)
    
    result = comparison(
        bd["Networkx"], bd["y0"], bd["MScausality"],
        bd["Coefficients"], {"X": 0}, {"X": 2}, 
        "Z", summarized_data)

    return result

# Initailize graphs
bd = backdoor()

# Benchmarks
N = 30
rep_range = [10, 20, 50, 100, 250, 500, 1000]

med_result = list()
bd_result = list()

# Mediator
for r in rep_range:

    temp_rep_list = list()

    for i in range(N):
        temp_rep_list.append(generate_med_data(r))

    temp_rep_list = pd.concat(temp_rep_list, ignore_index=True)
    temp_rep_list.loc[:, "Replicates"] = r

    med_result.append(temp_rep_list)

med_result = pd.concat(med_result, ignore_index=True)

# Backdoor
for r in rep_range:

    temp_rep_list = list()

    for i in range(N):
        temp_rep_list.append(generate_med_data(r))

    temp_rep_list = pd.concat(temp_rep_list, ignore_index=True)
    temp_rep_list.loc[:, "Replicates"] = r

    bd_result.append(temp_rep_list)

bd_result = pd.concat(bd_result, ignore_index=True)

# Save results
with open('results.pkl', 'wb') as file:
    pickle.dump({"Mediator": med_result,
                 "Backdoor": bd_result}, file)