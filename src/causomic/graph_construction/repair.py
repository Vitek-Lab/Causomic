
import networkx as nx
from y0.graph import NxMixedGraph
import pandas as pd

def convert_to_y0_graph(data, posterior_dag):
    """
    Convert the posterior DAG to a y0 graph format.
    """

    # Confirm index is fine
    posterior_dag = posterior_dag.reset_index(drop=True)

    # Construct NetworkX DiGraph from posterior_dag
    all_nodes = set(posterior_dag["source"]).union(set(posterior_dag["target"]))

    nx_dag = nx.DiGraph()
    for i in range(len(posterior_dag)):
        nx_dag.add_edge(posterior_dag.loc[i, "source"],
                        posterior_dag.loc[i, "target"])
        
    obs_nodes = all_nodes
    
    # Set all nodes as observed    
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}
    nx.set_node_attributes(nx_dag, attrs, name="hidden")
    
    # Use y0 to build ADMG
    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(nx_dag, "hidden")
    
    return y0_graph