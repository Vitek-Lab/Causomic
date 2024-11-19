
import networkx as nx

def mfas_greedy_min_set(graph, experimental_data=None):

    """
    Implementation of the minimum set cover problem approach to the minimum feedback arc set problem.

    Parameters
    ----------
    graph : networkx.DiGraph
        The input graph including cycles.
    
    experimental_data : pd.DataFrame
        Experimental data to be used in the calculation of relationships broken by arc set edges. Optional. Default is None.
    
    Returns
    -------
    graph
        The input graph with the minimum feedback arc set removed.
    arc_set
        The minimum feedback arc set.
    """

    # TODO: Implement version where we dont need to get all cycles everytime (takes forever)
    cycles = list(nx.simple_cycles(graph))
    arc_set = list()

    n_edges = list()
    n_weight = list()

    broken_correlations = list()

    while len(cycles) > 0:

        # Find all edges in all cycles
        edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                edges.add((cycle[i], cycle[(i+1) % len(cycle)]))
        
        # Find the edge with the combined most weight across all cycles
        weight_tracker = {e:0 for e in edges}
        edge_tracker = {e:0 for e in edges}
        for edge in edges:
            for cycle in cycles:
                for cyc_node in range(len(cycle)):
                    if edge == (cycle[cyc_node], cycle[(cyc_node+1) % len(cycle)]):
                        weight_tracker[edge] += graph[edge[0]][edge[1]]['weight']
                        edge_tracker[edge] += 1

        remove_edge = max(weight_tracker, key=weight_tracker.get)
        n_edges.append(edge_tracker[remove_edge])
        n_weight.append(weight_tracker[remove_edge])

        if experimental_data is not None:
            broken_correlations.append(check_broken_correlalations(graph, remove_edge, experimental_data))

        # Remove the edge with the most weight
        graph.remove_edge(remove_edge[0], remove_edge[1])

        arc_set.append(remove_edge)

        cycles = list(nx.simple_cycles(graph))

    return graph, arc_set, broken_correlations

def check_broken_correlalations(graph, edge, experimental_data):

    """
    Check if the removal of an edge has broken major mediated correlations.

    Parameters
    ----------
    graph : networkx.DiGraph
        The input graph including cycles.
    edge : tuple
        Tuple of two nodes making up edge.
    experimental_data : pd.DataFrame
        Experimental data to be used in the calculation of relationships broken by arc set edges.
    
    Returns
    -------
    Dictionary
        Dictionary of broken correlations.
    """

    broken_correlations = dict()
    if graph.nodes[edge[0]]["observed"]:
        out_edges = graph.out_edges(edge[1])
        for e in out_edges:
            if graph.nodes[e[1]]["observed"]:
                edge_data = experimental_data.loc[:, [edge[0], e[1]]]
                edge_corr = edge_data.corr().iloc[0, 1]

                # there might be a correlation, but if a path still exists we dont really care
                temp = graph.copy()
                temp.remove_edge(edge[0], edge[1])
                try:
                    path_exists = nx.shortest_path(temp, source=edge[0], target=e[1])
                except:
                    path_exists = []

                if len(path_exists) == 0:
                    broken_correlations[(edge[0], e[1])] = abs(edge_corr)
    
    elif graph.nodes[edge[1]]["observed"]:
        in_edges = graph.in_edges(edge[0])
        for e in in_edges:
            if graph.nodes[e[0]]["observed"]:
                edge_data = experimental_data.loc[:, [e[0], edge[1]]]
                edge_corr = edge_data.corr().iloc[0, 1]

                # there might be a correlation, but if a path still exists we dont really care
                temp = graph.copy()
                temp.remove_edge(edge[0], edge[1])
                try:
                    path_exists = nx.shortest_path(temp, source=e[0], target=edge[1])
                except:
                    path_exists = []

                if len(path_exists) == 0:
                    broken_correlations[(e[0], edge[1])] = abs(edge_corr)

    # TODO: Add something in to check on if both are latent. Need to check how often this even happens and WHY it happens first.
    # else:
    #     return None   

    return broken_correlations

# Example usage:
if __name__ == "__main__":

    graph = Graph()
    graph.add_edge('A', 'B', 1)
    graph.add_edge('A', 'C', 5)
    graph.add_edge('B', 'C', 2)
    graph.add_edge('B', 'D', 4)
    graph.add_edge('C', 'D', 3)

