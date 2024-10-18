
import networkx as nx
from y0.graph import NxMixedGraph
from y0.dsl import Variable
from y0.algorithm.simplify_latent import simplify_latent_dag

def mediator(include_coef=True):

    graph = nx.DiGraph()
    nodes = ["X", "Y", "Z"]

    ## Add edges
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    
    attrs = {node: True for node in nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    mscausality_graph = NxMixedGraph()
    mscausality_graph.add_directed_edge("X", "Y")
    mscausality_graph.add_directed_edge("Y", "Z")

    if include_coef:
        coef = {
            "X": {"intercept": 6, "error": 1.},
            "Y": {"intercept": 1.6, "error": .25, "X": 0.5},
            "Z": {"intercept": -3, "error": .25, "Y": 1.}
            }
    else:
        coef = None

    return {
        "Networkx": graph,
        "y0": y0_graph,
        "MScausality": mscausality_graph,
        "Coefficients": coef
        }

def backdoor(include_coef=True):

    graph = nx.DiGraph()
    obs_nodes = ["B", "X", "Y", "Z"]
    all_nodes = ["B", "X", "Y", "Z", "C"]

    ## Add edges
    graph.add_edge("B", "X")
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    graph.add_edge("C", "B")
    graph.add_edge("C", "Y")
    
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


def main():

    med = mediator()
    bd = backdoor()
    fd = frontdoor()

if __name__ == "__main__":
    main()
