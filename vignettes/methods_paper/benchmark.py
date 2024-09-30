# Load packages
from MScausality.causal_model.LVM import LVM
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.normalization import normalize
from MScausality.data_analysis.dataProcess import dataProcess

import pyro
import pandas as pd
import numpy as np
import pickle 

import networkx as nx
import y0
from y0.algorithm.simplify_latent import simplify_latent_dag

from y0.dsl import Variable
from eliater.regression import summary_statistics, get_adjustment_set
from operator import attrgetter

from scipy.stats import linregress
from sklearn.impute import KNNImputer

import warnings
warnings.filterwarnings('ignore')

## General benchmarking functions ----------------------------------------------
def get_std_error(
        graph,
        data,
        treatments,
        outcome,
        _adjustment_set=None):

    treatments = y0.graph._ensure_set(treatments)
    if _adjustment_set:
        adjustment_set = _adjustment_set
    else:
        adjustment_set, _ = get_adjustment_set(graph=graph, 
                                               treatments=treatments, 
                                               outcome=outcome)
    variable_set = adjustment_set.union(treatments).difference({outcome})
    variables = sorted(variable_set, key=attrgetter("name"))
    model = linregress(data[[v.name for v in variables]].values.flatten(), 
                       data[outcome.name].values)
    
    return model.stderr*np.sqrt(len(data))

def comparison(full_graph, 
               y0_graph, 
               coefficients, 
               int1, int2, 
               outcome,
               obs_data=None,
               training_obs=1000,
               alt_graph=None
               ):
    """
    Compare the results of the full graph with the y0 graph
    
    """

    ## Ground truth
    if obs_data is None:
        obs_data = pd.DataFrame(simulate_data(full_graph, 
                                              coefficients=coefficients, 
                                              add_feature_var=False, 
                                              n=training_obs, seed=2)
                            ["Protein_data"])

    intervention_low = simulate_data(full_graph, coefficients=coefficients,
                                     intervention=int1, 
                                     add_feature_var=False, n=10000, seed=2)

    intervention_high = simulate_data(full_graph, coefficients=coefficients,
                                     intervention=int2, 
                                     add_feature_var=False, n=10000, seed=2)

    ## Eliator results
    if obs_data.isnull().values.any():
        imputer = KNNImputer(n_neighbors=3)

        # Impute missing values
        obs_data_eliator = obs_data.copy()
        obs_data_eliator = pd.DataFrame(imputer.fit_transform(obs_data_eliator), 
                                        columns=obs_data.columns)
        # def impute_with_normal(df):
        #     for col in df.columns:
        #         missing = df[col].isnull()
        #         if missing.any():
        #             # Sample from normal distribution based on column mean and std
        #             df.loc[missing, col] = np.random.normal(df[col].mean(), df[col].std(), missing.sum())
        #     return df
        # obs_data_eliator = impute_with_normal(obs_data_eliator)
    else:
        obs_data_eliator = obs_data.copy()

    eliator_int_low = summary_statistics(
        y0_graph, obs_data_eliator,
        treatments={Variable(list(int1.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int1.keys())[0]): list(int1.values())[0]})

    eliator_int_high = summary_statistics(
        y0_graph, obs_data_eliator,
        treatments={Variable(list(int2.keys())[0])},
        outcome=Variable(outcome),
        interventions={
            Variable(list(int2.keys())[0]): list(int2.values())[0]})
    
    eliator_std_error = get_std_error(
        y0_graph, obs_data_eliator, 
        treatments={Variable(list(int2.keys())[0])}, 
        outcome=Variable(outcome))


    ## MScausality results
    if alt_graph is None:
        alt_graph = y0_graph
    pyro.clear_param_store()

    transformed_data = normalize(obs_data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    lvm = LVM(input_data, alt_graph)
    lvm.prepare_graph()
    lvm.prepare_data()
    lvm.fit_model(num_steps=10000)

    lvm.intervention({list(int1.keys())[0]: (list(int1.values())[0] - scale_metrics["mean"]) / scale_metrics["std"]}, outcome)
    mscausality_int_low = lvm.intervention_samples
    lvm.intervention({list(int2.keys())[0]: (list(int2.values())[0] - scale_metrics["mean"]) / scale_metrics["std"]}, outcome)
    mscausality_int_high = lvm.intervention_samples
    
    
    mscausality_int_low = (mscausality_int_low*scale_metrics["std"]) + scale_metrics["mean"]
    mscausality_int_high = (mscausality_int_high*scale_metrics["std"]) + scale_metrics["mean"]

    result_df = pd.DataFrame({
        "Ground_truth": [np.mean(intervention_low["Protein_data"][outcome]), 
                         np.mean(intervention_high["Protein_data"][outcome])],
        "Ground_truth_std": [np.std(intervention_low["Protein_data"][outcome]), 
                             np.std(intervention_high["Protein_data"][outcome])],
        "Eliator": [eliator_int_low.mean, eliator_int_high.mean],
        "Eliator_std": [eliator_std_error, eliator_std_error],
        "MScausality": [np.mean(np.array(mscausality_int_low)), 
                        np.mean(np.array(mscausality_int_high))],
        "MScausality_std": [np.std(np.array(mscausality_int_low)), 
                            np.std(np.array(mscausality_int_high))]
    })

    return result_df

def benchmark_comparison(full_graph, y0_graph, coef):

    replicates = [10, 30, 100, 1000]
    result = dict()

    for r in range(len(replicates)):
        temp_r = replicates[r]

        eliate_ate = list()
        mscausality_ate = list()
        eliate_var = list()
        mscausality_var = list()

        for i in range(temp_r):

            sr_data = simulate_data(full_graph, coefficients=coef, 
                                    add_feature_var=True, n=30, seed=i)

            sr_data = dataProcess(sr_data["Feature_data"], normalization=False, sim_data=True)
            
            result_data = comparison(full_graph, y0_graph, coef, {"IL6": 5}, 
                {"IL6": 7}, "MYC", obs_data=sr_data)

            eliate_ate.append(result_data.loc[1, "Eliator"] - result_data.loc[0, "Eliator"])
            mscausality_ate.append(result_data.loc[1, "MScausality"] - result_data.loc[0, "MScausality"])
            eliate_var.append(result_data.loc[1, "Eliator_std"])
            mscausality_var.append(result_data.loc[1, "MScausality_std"])

        result[str(temp_r)] = {"eliate_ate": eliate_ate, 
                               "mscausality_ate": mscausality_ate,
                               "eliate_var": eliate_var, 
                               "mscausality_var": mscausality_var}
        
    return result

# Build graphs -----------------------------------------------------------------
# Two node
def build_sr_network():
    """
    Create TF MYC graph in networkx
    
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("IL6", "MYC")
    
    return graph

def build_sr_admg(graph):

    """

    Creates acyclic directed mixed graph (ADMG) from networkx graph with latent variables.
    
    """

    ## Define obs vs latent nodes
    all_nodes = ["IL6", "MYC"]
    obs_nodes = ["IL6", "MYC"]
            
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    mapping = dict(zip(list(graph.nodes), 
                      [Variable(i) for i in list(graph.nodes)]))
    graph = nx.relabel_nodes(graph, mapping)
    
    ## Use y0 to build ADMG
    simplified_graph = simplify_latent_dag(graph.copy(), tag="hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")
    
    return y0_graph

# 3 node
def build_tf_med_network():
    """
    Create TF MYC graph in networkx
    
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("IL6", "STAT3")
    graph.add_edge("STAT3", "MYC")
    
    return graph

def build_med_admg(graph):

    """

    Creates acyclic directed mixed graph (ADMG) from networkx graph with latent variables.
    
    """

    ## Define obs vs latent nodes
    all_nodes = ["IL6", "STAT3", "MYC"]
    obs_nodes = ["IL6", "STAT3", "MYC"]
            
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    mapping = dict(zip(list(graph.nodes), 
                      [Variable(i) for i in list(graph.nodes)]))
    graph = nx.relabel_nodes(graph, mapping)
    
    ## Use y0 to build ADMG
    simplified_graph = simplify_latent_dag(graph.copy(), tag="hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")
    
    return y0_graph


# 3 node with latent
def build_tf_med_lat_network():
    """
    Create TF MYC graph in networkx
    
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("IL6", "STAT3")
    graph.add_edge("STAT3", "MYC")
    
    graph.add_edge("C1", "STAT3")
    graph.add_edge("C1", "MYC")
    
    return graph

def build_med_lat_admg(graph):

    """

    Creates acyclic directed mixed graph (ADMG) from networkx graph with latent variables.
    
    """

    ## Define obs vs latent nodes
    all_nodes = ["IL6", "STAT3", "MYC", "C1"]
    obs_nodes = ["IL6", "STAT3", "MYC"]
            
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    mapping = dict(zip(list(graph.nodes), 
                      [Variable(i) for i in list(graph.nodes)]))
    graph = nx.relabel_nodes(graph, mapping)
    
    ## Use y0 to build ADMG
    simplified_graph = simplify_latent_dag(graph.copy(), tag="hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")
    
    return y0_graph

# 3 node 2 latents
def build_tf_med_more_lat_network():
    """
    Create TF MYC graph in networkx
    
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("IL6", "STAT3")
    graph.add_edge("STAT3", "M2")
    graph.add_edge("M2", "MYC")
    
    graph.add_edge("C1", "STAT3")
    graph.add_edge("C1", "MYC")
    graph.add_edge("C2", "M2")
    graph.add_edge("C2", "MYC")
    
    return graph

def build_med_more_lat_admg(graph):

    """

    Creates acyclic directed mixed graph (ADMG) from networkx graph with latent variables.
    
    """

    ## Define obs vs latent nodes
    all_nodes = ["IL6", "STAT3", "M2", "MYC", "C1", "C2"]
    obs_nodes = ["IL6", "STAT3", "M2", "MYC"]
            
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    mapping = dict(zip(list(graph.nodes), 
                      [Variable(i) for i in list(graph.nodes)]))
    graph = nx.relabel_nodes(graph, mapping)
    
    ## Use y0 to build ADMG
    simplified_graph = simplify_latent_dag(graph.copy(), tag="hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")
    
    return y0_graph

## Simulations -----------------------------------------------------------------
# Two node
sr_graph = build_sr_network()
y0_sr_graph = build_sr_admg(sr_graph)
sr_coef = {'IL6': {'intercept': 6, "error": 1},
            'MYC': {'intercept': 2, "error": .25, 'IL6': 1.}}

two_node_result = benchmark_comparison(sr_graph, y0_sr_graph, sr_coef)
with open('two_node_result.pickle', 'wb') as handle:
    pickle.dump(two_node_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3 node
med_graph = build_tf_med_network()
y0_med_graph = build_med_admg(med_graph)
med_coef = {'IL6': {'intercept': 6, "error": 1.},
            'STAT3': {'intercept': 1.6, "error": .25,  'IL6': 0.5},
              'MYC': {'intercept': 2, "error": .25, 'STAT3': 1.}
              }

three_node_result = benchmark_comparison(med_graph, y0_med_graph, med_coef)
with open('three_node_result.pickle', 'wb') as handle:
    pickle.dump(three_node_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3 node with latent
med_lat_graph = build_tf_med_lat_network()
y0_med_lat_graph = build_med_lat_admg(med_lat_graph)
med_lat_coef = {'IL6': {'intercept': 6, "error": 1.},
            'C1': {'intercept': 0, "error": 1.},
            'STAT3': {'intercept': 1.6, "error": .25, 'IL6': 0.5, 'C1': .35},
            'MYC': {'intercept': 2, "error": .25, 'STAT3': 1., 'C1': .35}
              }

three_node_lat_result = benchmark_comparison(med_lat_graph, 
                                             y0_med_lat_graph, 
                                             med_lat_coef)
with open('three_node_lat_result.pickle', 'wb') as handle:
    pickle.dump(three_node_lat_result, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3 node with more latent
med_more_lat_graph = build_tf_med_more_lat_network()
y0_more_med_lat_graph = build_med_more_lat_admg(med_more_lat_graph)
med_more_lat_coef = {'IL6': {'intercept': 6, "error": 1.},
            'C1': {'intercept': 0, "error": 1.},
            'C2': {'intercept': 0, "error": 1.},
            'STAT3': {'intercept': 1.6, "error": .25, 'IL6': 0.5, 'C1': .35},
            'M2': {'intercept': 1.6, "error": .25, 'STAT3': 0.5, 'C2': .6},
            'MYC': {'intercept': 2, "error": .25, 'M2': 1., 'C1': .35, 'C2':.6}
              }

three_node_more_lat_result = benchmark_comparison(med_more_lat_graph,
                                                  y0_more_med_lat_graph, 
                                                  med_more_lat_coef)
with open('three_node_more_lat_result.pickle', 'wb') as handle:
    pickle.dump(three_node_more_lat_result, handle, 
                protocol=pickle.HIGHEST_PROTOCOL)