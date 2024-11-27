from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.dataProcess import dataProcess
from MScausality.simulation.example_graphs import signaling_network
from MScausality.validation import validate_model

import numpy as np
import pandas as pd
import pickle
import sys

def generate_sn_data(replicates, temp_seed, coef, priors):

    sn = signaling_network()

    # Mediator loop
    data = simulate_data(
        sn["Networkx"], 
        coefficients=coef, 
        mnar_missing_param=[-4, .3],
        add_feature_var=True, 
        n=replicates, 
        seed=temp_seed)
    # remove for missing features
    # data["Feature_data"]["Obs_Intensity"] = data["Feature_data"]["Intensity"]

    summarized_data = dataProcess(
        data["Feature_data"], 
        normalization=False, 
        feature_selection="All",
        summarization_method="TMP",
        MBimpute=True, # no missing, change to true for missing
        sim_data=True)
    
    summarized_data = summarized_data.loc[:, [
        i for i in summarized_data.columns if i not in ["IGF", "EGF"]]]

    result = validate_model(
        summarized_data,
        sn["Networkx"], sn["y0"], sn["MScausality"],
        coef, {"Ras": 5}, {"Ras": 7}, "Erk", priors)

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


# Get the start and end indices, and task ID from the command-line arguments
start = int(sys.argv[1])
end = int(sys.argv[2])
task_id = int(sys.argv[3])

# Benchmarks
N = 5
rep_range = [10, 20, 50, 100, 250]

idx_sims = np.repeat(rep_range, N)

igf_result = list()

with open(f'/home/kohler.d/applications_project/MScausality/vignettes/methods_paper/data/signaling_network/priors_inflated_scale.pkl', 'rb') as f:
    informed_priors = pickle.load(f)

print(f"Task {task_id}: Processing iterations {start} to {end}")
for i in range(start, end + 1):

    r = idx_sims[i]

    temp_rep_list = list()
    
    print(f"Task {task_id}, Iteration {i}")
    temp_result = generate_sn_data(r, i, informative_prior_coefs, 
                                   informed_priors)
    temp_result.loc[:, "Replicates"] = r

    igf_result.append(temp_result)

igf_result = pd.concat(igf_result, ignore_index=True)

# Save results
with open(f'compile_priors/temp_results_{task_id}.pkl', 'wb') as file:
    pickle.dump(igf_result, file)

print(f"Task {task_id} complete")