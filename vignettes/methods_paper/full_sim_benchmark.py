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
from eliater.regression import summary_statistics, get_adjustment_set

import pyro

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
          - intervention_low["Protein_data"][outcome].mean() )
    
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

    for i in lvm.priors.keys():
        for v in lvm.priors[i].keys():
            if ("coef" in v): 
                lvm.priors[i][v] = .75

    pyro.clear_param_store()
    lvm.fit_model(num_steps=10000)

    full_imp_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    result_df = pd.DataFrame({
        "Ground_truth": [gt_ate],
        "Eliator": [eliator_ate],
        "MScausality": [full_imp_model_ate.item()],
    })

    return result_df


# Initailize graphs
med = mediator()
fd = frontdoor()
bd = backdoor()

# Benchmarks
N = 2

med_result = list()
fd_result = list()
bd_result = list()

for i in range(N):

    # Mediator loop
    med_data = simulate_data(med["Networkx"], 
                              coefficients=med["Coefficients"], 
                              mnar_missing_param=[20, .3], # No missingness
                              add_feature_var=True, 
                              n=250, 
                              seed=i)
    med_summarized_data = dataProcess(med_data["Feature_data"], 
                            normalization=False, 
                            feature_selection="All",
                            MBimpute=False,
                            sim_data=True)

    med_result_data = comparison(
        med["Networkx"], med["y0"], med["MScausality"],
        med["Coefficients"], {"X": 0}, {"X": 2}, 
        "Z", med_summarized_data)

    med_result.append(med_result_data)

    # Frontdoor loop
    fd_data = simulate_data(fd["Networkx"], 
                              coefficients=fd["Coefficients"], 
                              mnar_missing_param=[20, .3], # No missingness
                              add_feature_var=True, 
                              n=250, 
                              seed=i)
    fd_summarized_data = dataProcess(fd_data["Feature_data"], 
                            normalization=False, 
                            feature_selection="All",
                            MBimpute=False,
                            sim_data=True)

    fd_result_data = comparison(
        fd["Networkx"], fd["MScausality"], fd["MScausality"],
        fd["Coefficients"], {"X": 0}, {"X": 2}, 
        "Z", fd_summarized_data)

    fd_result.append(fd_result_data)

    # Backdoor loop
    bd_data = simulate_data(bd["Networkx"], 
                              coefficients=bd["Coefficients"], 
                              mnar_missing_param=[20, .3], # No missingness
                              add_feature_var=True, 
                              n=250, 
                              seed=i)
    bd_summarized_data = dataProcess(bd_data["Feature_data"], 
                            normalization=False, 
                            feature_selection="All",
                            MBimpute=False,
                            sim_data=True)

    bd_result_data = comparison(
        bd["Networkx"], bd["y0"], bd["MScausality"],
        bd["Coefficients"], {"X": 0}, {"X": 2}, 
        "Z", bd_summarized_data)

    bd_result.append(bd_result_data)

med_result = pd.concat(med_result, ignore_index=True)
fd_result = pd.concat(fd_result, ignore_index=True)
bd_result = pd.concat(bd_result, ignore_index=True)

# Save results
with open('results.pkl', 'wb') as file:
    pickle.dump({"Mediator": med_result,
                 "Frontdoor": fd_result,
                 "Backdoor": bd_result}, file)