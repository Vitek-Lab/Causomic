
import networkx as nx
import numpy as np

from y0.graph import NxMixedGraph
from y0.dsl import Variable
from y0.algorithm.simplify_latent import simplify_latent_dag
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.normalization import normalize
from MScausality.data_analysis.dataProcess import dataProcess

def mediator(include_coef=True, 
             n_med=1, 
             add_independent_nodes=False, 
             n_ind=10):

    if n_med < 1:
        raise ValueError("n_med must be at least 1")

    graph = nx.DiGraph()
    
    nodes = ["X", "Z"]
    nodes = nodes + [f"M{i}" for i in range(1, n_med+1)]
    if add_independent_nodes:
        nodes = nodes + [f"I{i}" for i in range(1, n_ind+1)]

    ## Add edges
    graph.add_edge("X", "M1")
    
    for i in range(1, n_med+1):
        if i < n_med:
            graph.add_edge(f"M{i}", f"M{i+1}")
    
    graph.add_edge(f"M{n_med}", "Z")
    
    if add_independent_nodes:
        for i in range(1, n_ind+1):
            graph.add_node(f"I{i}")

    attrs = {node: False for node in nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    mscausality_graph = NxMixedGraph()
    mscausality_graph.add_directed_edge("X", "M1")
    
    for i in range(1, n_med+1):
        if i < n_med:
            mscausality_graph.add_directed_edge(f"M{i}", f"M{i+1}")
    
    mscausality_graph.add_directed_edge(f"M{n_med}", "Z")

    if include_coef:
        coef = {
            "X": {"intercept": 3, "error": 1.},
            "M1": {"intercept": 1.6, "error": .25, "X": 0.5},
            "Z": {"intercept": -1, "error": .25, f"M{n_med}": 1.}
            }
        
        for i in range(2, n_med+1):
            coef[f"M{i}"] = {"intercept": 1.6, "error": .25, 
                             f"M{i-1}": np.random.uniform(.5, 1.5)}
        
        if add_independent_nodes:
            for i in range(1, n_ind+1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), 
                                 "error": 1.}
    else:
        coef = None

    return {
        "Networkx": graph,
        "y0": y0_graph,
        "MScausality": mscausality_graph,
        "Coefficients": coef
        }

def backdoor(include_coef=True, 
             add_independent_nodes=False, 
             n_ind=10):

    graph = nx.DiGraph()
    obs_nodes = ["B", "X", "Y", "Z"]
    all_nodes = ["B", "X", "Y", "Z", "C"]
    if add_independent_nodes:
        all_nodes = all_nodes + [f"I{i}" for i in range(1, n_ind+1)]

    ## Add edges
    graph.add_edge("B", "X")
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    graph.add_edge("C", "B")
    graph.add_edge("C", "Y")
    
    if add_independent_nodes:
        for i in range(1, n_ind+1):
            graph.add_node(f"I{i}")
    
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    mscausality_graph = NxMixedGraph()
    mscausality_graph.add_directed_edge("B", "Y")
    mscausality_graph.add_directed_edge("X", "Y")
    mscausality_graph.add_directed_edge("Y", "Z")

    if include_coef:
        coef = {
            "C": {"intercept": 6, "error": 1.}, 
            "B": {"intercept": 0, "error": 1., "C": .5},
            "X": {"intercept": 1, "error": 1., "B": 1.},
            "Y": {"intercept": 1.6, "error": .25, "X": 0.5, "C": .5},
            "Z": {"intercept": -3, "error": .25, "Y": 1.}
            }
        
        if add_independent_nodes:
            for i in range(1, n_ind+1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), 
                                 "error": 1.}
                
    else:
        coef = None

    return {
        "Networkx": graph,
        "y0": y0_graph,
        "MScausality": mscausality_graph,
        "Coefficients": coef
        }

def frontdoor(include_coef=True):

    graph = nx.DiGraph()
    obs_nodes = ["X", "Y", "Z"]
    all_nodes = ["X", "Y", "Z", "C"]

    ## Add edges
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    graph.add_edge("C", "X")
    graph.add_edge("C", "Z")
    
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    mscausality_graph = NxMixedGraph()
    mscausality_graph.add_directed_edge("X", "Z")
    mscausality_graph.add_directed_edge("X", "Y")
    mscausality_graph.add_directed_edge("Y", "Z")

    if include_coef:
        coef = {
            "C": {"intercept": 6, "error": 1.}, 
            "X": {"intercept": 1, "error": 1., "C": .5},
            "Y": {"intercept": 1.6, "error": .25, "X": 0.5},
            "Z": {"intercept": -3, "error": .25, "Y": 1., "C": .5}
            }
        

    else:
        coef = None

    return {
        "Networkx": graph,
        "y0": y0_graph,
        "MScausality": mscausality_graph,
        "Coefficients": coef
        }

def signaling_network(include_coef=True, 
             add_independent_nodes=False, 
             n_ind=10):

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

    if add_independent_nodes:
        for i in range(1, n_ind+1):
            graph.add_node(f"I{i}")
    
    ## Define obs vs latent nodes
    all_nodes = ["SOS", "PI3K", "Ras", "Raf", "Akt", 
                 "Mek", "Erk", "EGF", "IGF"]
    obs_nodes = ["SOS", "PI3K", "Ras", "Raf", "Akt", 
                 "Mek", "Erk"]
    if add_independent_nodes:
        all_nodes = all_nodes + [f"I{i}" for i in range(1, n_ind+1)]
    
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}
    
    nx.set_node_attributes(graph, attrs, name="hidden")
    # Use y0 to build ADMG
    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph, "hidden")


    mscausality_graph = NxMixedGraph()
    mscausality_graph.add_directed_edge("SOS", "PI3K")
    mscausality_graph.add_directed_edge("Ras", "PI3K")
    mscausality_graph.add_directed_edge("Ras", "Raf")
    mscausality_graph.add_directed_edge("PI3K", "Akt")
    mscausality_graph.add_directed_edge("Akt", "Raf")
    mscausality_graph.add_directed_edge("Raf", "Mek")
    mscausality_graph.add_directed_edge("Mek", "Erk")

    if include_coef:
        coef = {
            'EGF': {'intercept': 6., "error": 1},
            'IGF': {'intercept': 5., "error": 1},
            'SOS': {'intercept': 2, "error": 1, 
                      'EGF': 0.6, 'IGF': 0.6},
            'Ras': {'intercept': 3, "error": 1, 'SOS': .5},
            'PI3K': {'intercept': 0, "error": 1, 
                       'EGF': .5, 'IGF': .5, 'Ras': .5},
            'Akt': {'intercept': 1., "error": 1, 'PI3K': 0.75},
            'Raf': {'intercept': 4, "error": 1,
                      'Ras': 0.8, 'Akt': -.4},
            'Mek': {'intercept': 2., "error": 1, 'Raf': 0.75},
            'Erk': {'intercept': -2, "error": 1, 'Mek': 1.2}}
        
        if add_independent_nodes:
            for i in range(1, n_ind+1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), 
                                 "error": 1.}
                
    else:
        coef = None

    return {
        "Networkx": graph,
        "y0": y0_graph,
        "MScausality": mscausality_graph,
        "Coefficients": coef
        }

def main():

    med = mediator(n_med=3)
    bd = backdoor()
    fd = frontdoor()
    sn = signaling_network()

    simulated_fd_data = simulate_data(sn['Networkx'], 
                                coefficients=sn['Coefficients'], 
                                mnar_missing_param=[-5, .4],
                                add_feature_var=True, n=50, seed=2)
    fd_data = dataProcess(simulated_fd_data["Feature_data"], normalization=False, 
                summarization_method="TMP", MBimpute=False, sim_data=True)
    # fd_data = fd_data.dropna(how="all",axis=1)
    print(fd_data.isna().mean() * 100)

if __name__ == "__main__":
    main()
