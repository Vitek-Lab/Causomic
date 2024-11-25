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

import matplotlib.pyplot as plt

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


sn = signaling_network(add_independent_nodes=True, n_ind=20)

# Mediator loop
data = simulate_data(
    sn["Networkx"], 
    coefficients=uninformative_prior_coefs, 
    mnar_missing_param=[-4, .3],
    add_feature_var=True, 
    n=50, 
    seed=49)
# Best seed for viz is 51
# data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

summarized_data = dataProcess(
    data["Feature_data"], 
    normalization=False, 
    feature_selection="All",
    summarization_method="TMP",
    MBimpute=False,
    sim_data=True)

summarized_data = summarized_data.loc[:, [
    i for i in summarized_data.columns if i not in ["IGF", "EGF"]]]
summarized_data = summarized_data.dropna(how="all",axis=1)

# print(summarized_data.isna().mean() * 100)

result = comparison(
    sn["Networkx"], sn["y0"], sn["MScausality"],
    uninformative_prior_coefs, {"Ras": 5}, {"Ras": 7}, 
    "Erk", summarized_data)


# sn
print(result["Ground_truth"])
print(result["Eliator"])
print(result["MScausality"])


model = result["MScausality_model"][0]
imp_data = model.imputed_data
X_data = imp_data.loc[imp_data["protein"] == "Raf"]
Z_data = imp_data.loc[imp_data["protein"] == "Erk"]

X_backdoor_color = np.where(
    (X_data['imp_mean'].isna().values & Z_data['imp_mean'].isna().values), 
    "blue", 
    np.where((X_data['intensity'].isna().values & Z_data['intensity'].isna().values), 
             "red", "orange"))

X_data = np.where(
    X_data['imp_mean'].isna(),
    X_data['intensity'], 
    X_data['imp_mean'])

Z_data = np.where(
    Z_data['imp_mean'].isna(),
    Z_data['intensity'], 
    Z_data['imp_mean'])


fig, ax = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

transformed_data = normalize(summarized_data, wide_format=True)
input_data = transformed_data["df"]

eliator_data = result["obs_data_eliator"][0]

from scipy.stats import linregress

# Define a list of scatter plot configurations
temp = summarized_data.loc[:, ["Raf", "Erk"]].dropna()
plots = [
    (temp.loc[:, "Raf"], 
     temp.loc[:, "Erk"], "blue",
     "Observed data", "Raf", "Erk"),
    (((X_data*transformed_data['adj_metrics']["std"]) \
     + transformed_data['adj_metrics']["mean"]), 
    ((Z_data*transformed_data['adj_metrics']["std"]) \
     + transformed_data['adj_metrics']["mean"]), 
     X_backdoor_color, "Bayesian imputation", "Raf", "Erk"),
    (eliator_data["Raf"], eliator_data["Erk"], 
     X_backdoor_color, "KNN imputation", "Raf", "Erk"),
]

for i, (x, y, color, title, xlabel, ylabel) in enumerate(plots):

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    line_x = np.linspace(x.min()-1, x.max()+1, 100)
    line_y = slope * line_x + intercept

    ax[i].scatter(x, y, color=color,
                  edgecolor='k', s=80, alpha=0.8)
    ax[i].plot(line_x, line_y, color='red', linestyle='--', 
               linewidth=2)
 
    ax[i].set_title(title, fontsize=16, fontweight='bold')
    # ax[i].set_xlabel(xlabel, fontsize=14)
    # ax[i].set_ylabel(ylabel, fontsize=14)
    ax[i].tick_params(axis='both', which='major', labelsize=14)
    ax[i].grid(True, linestyle='--', alpha=0.3)

    ax[i].set_xlim(3, 9.5)
    ax[i].set_ylim(1, 12)

fig.suptitle("Imputation Comparison", fontsize=18, fontweight='bold')
fig.supxlabel("Ras", fontsize=16, fontweight='bold')
fig.supylabel("Erk", fontsize=16, fontweight='bold')

plt.show()
# Save results
# with open('igf_results_with_missing_subset.pkl', 'wb') as file:
#     pickle.dump(igf_result, file)