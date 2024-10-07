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

def build_igf_network():
    """
    Create IGF graph in networkx
    
    cell_confounder : bool
        Whether to add in cell type as a confounder
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("EGF", "SOS")
    graph.add_edge("EGF", "PI3K")
    graph.add_edge("IGF", "SOS")
    graph.add_edge("IGF", "PI3K")
    graph.add_edge("SOS", "Ras")
    graph.add_edge("Ras", "PI3K")
    graph.add_edge("Ras", "Raf")
    graph.add_edge("PI3K", "Akt")
    graph.add_edge("Akt", "Raf")
    graph.add_edge("Raf", "Mek")
    graph.add_edge("Mek", "Erk")
    
    return graph

def build_admg(graph, cell_confounder=False, cell_latent=False):
    ## Define obs vs latent nodes
    all_nodes = ["SOS", "PI3K", "Ras", "Raf", "Akt", 
                 "Mek", "Erk", "EGF", "IGF"]
    obs_nodes = ["SOS", "PI3K", "Ras", "Raf", "Akt", 
                 "Mek", "Erk"]
    latent_nodes = ["EGF", "IGF"]
    
    ## Add in cell_type if included
    if cell_confounder:
        all_nodes.append("cell_type")
        if cell_latent:
            latent_nodes.append("cell_type")
        else:
            obs_nodes.append("cell_type")
        
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    # simplified_graph = simplify_latent_dag(graph.copy(), "hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph, "hidden")
    
    return y0_graph

alt_graph = y0.graph.NxMixedGraph()
alt_graph.add_directed_edge("SOS", "Ras")
alt_graph.add_directed_edge("Ras", "PI3K")
alt_graph.add_directed_edge("Ras", "Raf")
alt_graph.add_directed_edge("PI3K", "Akt")
alt_graph.add_directed_edge("Akt", "Raf")
alt_graph.add_directed_edge("Raf", "Mek")
alt_graph.add_directed_edge("Mek", "Erk")

int_graph = y0.graph.NxMixedGraph()
int_graph.add_directed_edge("SOS", "Ras")
int_graph.add_directed_edge("Ras", "Erk")

bulk_graph = build_igf_network()
y0_graph_bulk = build_admg(bulk_graph)

cell_coef = {'EGF': {'intercept': 10., "error": 1},
              'IGF': {'intercept': 8., "error": 1},
              'SOS': {'intercept': -2, "error": .5, 
                      'EGF': 0.6, 'IGF': 0.6},
              'Ras': {'intercept': 3, "error": .5, 'SOS': .5},
              'PI3K': {'intercept': 1.6, "error": .5, 
                       'EGF': .5, 'IGF': 0.5, 'Ras': .5},
              'Akt': {'intercept': 3., "error": .5, 'PI3K': 0.75},
              'Raf': {'intercept': 8, "error": .5,
                      'Ras': 0.8, 'Akt': -.4},
              'Mek': {'intercept': 3., "error": .5, 'Raf': 0.75},
              'Erk': {'intercept': 0., "error": .5, 'Mek': 1.2}
             }

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
               cell_coef, 
               int1, 
               int2, 
               outcome,
               alt_graph,
               int_graph,
               data):
    
    # Ground truth
    intervention_low = simulate_data(bulk_graph, coefficients=cell_coef,
                                    intervention=int1, mnar_missing_param=[-4, .3],
                                    add_feature_var=False, n=10000, seed=2)

    intervention_high = simulate_data(bulk_graph, coefficients=cell_coef,
                                    intervention=int2, 
                                    add_feature_var=False, n=10000, seed=2)

    gt_ate = (intervention_high["Protein_data"][outcome].mean() \
          - intervention_low["Protein_data"][outcome].mean() )
    
    # Eliator prediction
    imputer = KNNImputer(n_neighbors=5)
    obs_data_eliator = data.copy()
    obs_data_eliator = pd.DataFrame(imputer.fit_transform(obs_data_eliator), 
                                    columns=data.columns)

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

    lvm = LVM(input_data, int_graph)
    lvm.prepare_graph()
    lvm.prepare_data()
    lvm.get_priors()

    lvm.fit_model(num_steps=10000)

    pyro.clear_param_store()
    lvm.fit_model(num_steps=10000)
    
    basic_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    # Informative prior Bayesian model
    pyro.clear_param_store()
    lvm.priors['Erk']['Erk_Ras_coef'] = 1.
    lvm.fit_model(num_steps=10000)
    
    inf_prior_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    # Full imp Bayesian model
    lvm = LVM(input_data, alt_graph)
    lvm.prepare_graph()
    lvm.prepare_data()
    lvm.get_priors()

    pyro.clear_param_store()
    lvm.fit_model(num_steps=10000)

    full_imp_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)

    # Full imp Bayesian model with informative prior
    lvm = LVM(input_data, alt_graph)
    lvm.prepare_graph()
    lvm.prepare_data()
    lvm.get_priors()

    for i in lvm.priors.keys():
        for v in lvm.priors[i].keys():
            if (v != "Raf_Akt_coef") & ("coef" in v): 
                if (lvm.priors[i][v]) < .75:
                    lvm.priors[i][v] = 1
    # lvm.priors["Mek"]["Mek_Raf_coef"] = 1.

    pyro.clear_param_store()
    lvm.fit_model(num_steps=10000)

    
    full_imp_inf_post_model_ate = intervention(lvm, int1, int2, outcome, scale_metrics)
    
    result_df = pd.DataFrame({
        "Ground_truth": [gt_ate],
        "Eliator": [eliator_ate],
        "Basic_model": [basic_model_ate.item()],
        "Inf_prior": [inf_prior_model_ate.item()],
        "Full_imp": [full_imp_model_ate.item()],
        "Full_imp_inf_post": [full_imp_inf_post_model_ate.item()]
    })

    return result_df


int1 = {"Ras": 5}
int2 = {"Ras": 7}
outcome = "Erk"


result = list()

for i in range(30):

    temp_data = simulate_data(bulk_graph, coefficients=cell_coef, 
                                mnar_missing_param=[-4, .3], 
                                add_feature_var=True, n=50, seed=i)

    summarized_data = dataProcess(temp_data["Feature_data"], 
                            normalization=False, 
                            feature_selection="All",
                            MBimpute=False,
                            sim_data=True)

    
    result_data = comparison(
        bulk_graph, y0_graph_bulk, cell_coef, {"Ras": 5}, {"Ras": 7}, 
        "Erk", alt_graph, int_graph, summarized_data)

    result.append(result_data)

result = pd.concat(result, ignore_index=True)

import pickle

with open('results.pkl', 'wb') as file:
    pickle.dump(result, file)