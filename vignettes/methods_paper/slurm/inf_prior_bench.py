from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.dataProcess import dataProcess
from MScausality.simulation.example_graphs import signaling_network
from MScausality.validation import validate_model

import pandas as pd
import pickle

def generate_sn_data(replicates, temp_seed, priors, coef):

    sn = signaling_network()

    # Mediator loop
    data = simulate_data(
        sn["Networkx"], 
        coefficients=coef, 
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

    result = validate_model(
        summarized_data,
        sn["Networkx"], sn["y0"], sn["MScausality"],
        sn["Coefficients"], {"Ras": 5}, {"Ras": 7}, 
        "Erk", priors)

    return result

# model coefficients
informative_prior_coefs = {
    'EGF': {'intercept': 6., "error": 1},
    'IGF': {'intercept': 5., "error": 1},
    'SOS': {'intercept': 2, "error": 1, 'EGF': 0.6, 'IGF': 0.6},
    'Ras': {'intercept': 3, "error": 1, 'SOS': .5},
    'PI3K': {'intercept': 0, "error": 1, 'EGF': .5, 'IGF': .5, 'Ras': .5},
    'Akt': {'intercept': 1., "error": 1, 'PI3K': 0.75},
    'Raf': {'intercept': 4, "error": 1, 'Ras': 0.8, 'Akt': -.4},
    'Mek': {'intercept': 2., "error": 1, 'Raf': 0.75},
    'Erk': {'intercept': -2, "error": 1, 'Mek': 1.2}}


# Benchmarks
prior_studies = 10
N = 5
rep_range = [10,20,50,100,250]

informed_results = list()

with open(f'vignettes/methods_paper/data/signaling_network/fixed_priors.pkl', 'rb') as f:
    informed_priors = pickle.load(f)

for r in rep_range:

    temp_informed_results = list()

    for i in range(250, N+250):
        temp_informed_results.append(generate_sn_data(r, i, informed_priors,
                                                      informative_prior_coefs))
        print(r, i-250)
    
    temp_informed_results = pd.concat(temp_informed_results, ignore_index=True)
    temp_informed_results.loc[:, "Replicates"] = r

    informed_results.append(temp_informed_results)

informed_results = pd.concat(informed_results, ignore_index=True)

# print("snarf")
# Save results
with open('igf_results_with_informed_prior_test.pkl', 'wb') as file:
    pickle.dump(informed_results, file)
