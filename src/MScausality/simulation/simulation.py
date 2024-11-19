
import numpy as np
import pandas as pd
import networkx as nx

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

def simulate_data(graph,
                  coefficients=None,
                  include_missing=True,
                  cell_type=False,
                  n_cells=3,
                  mar_missing_param=.05,
                  mnar_missing_param=[-3, .4],
                  add_error=False,
                  error_node=None,
                  intervention=dict(),
                  add_feature_var=True,
                  n=1000,
                  seed=None):
    """Simulate data from a given graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to simulate data from.
    coefficients : dict
        A dictionary of coefficients to use for the simulation. If None, random
        coefficients are generated.
    include_missing : bool
        Whether to include missing data in the simulation.
    cell_type : bool
        Whether cell type is included in the simulation (must pass custom coefficients if True).
    n_cells : int
        The number of cell types to simulate.
    mar_missing_param : float
        The probability of missing data for missing at random (MAR) missingness.
    mnar_missing_param : list
        The parameters for missing not at random (MNAR) missingness.
    add_error : bool
        Whether to add extra measurement error to a node in the simulation.
    error_node : str
        The node to add extra measurement error to.
    intervention : dict
        A dictionary of interventions to apply to the data. Default is None.
    n : int
        The number of samples to simulate.
    seed : int
        The seed to use for the random number generator.

    Returns
    -------
    data : pandas.DataFrame
        The simulated data.
    """
    if seed is not None:
        np.random.seed(seed)

    if coefficients is None:
        coefficients = generate_coefficients(graph)

    data = dict()#pd.DataFrame(columns=graph.nodes())

    sorted_nodes = [i for i in nx.topological_sort(graph) if i != "cell_type"]

    print("simulating data...")
    for node in sorted_nodes:
        node_coefficients = coefficients[node]
        if node in intervention.keys():
            temp_int = intervention[node]
        else:
            temp_int = None
        data[node] = simulate_node(data, node_coefficients, n, cell_type,
                                   temp_int)

    if cell_type:
        data["cell_type"] = np.repeat([i for i in range(n_cells)], n//n_cells)
        if len(data["cell_type"]) < n:
            data["cell_type"] = np.append(data["cell_type"], n_cells-1)

    if add_error:
        data[error_node] += np.random.normal(0, 5, n)

    # break data into features
    if add_feature_var:
        print("adding feature level data...")
        feature_level_data_list = list()
        for node in sorted_nodes:
            feature_level_data_list.append(generate_features(data[node], node))

        feature_level_data = pd.concat(feature_level_data_list, 
                                       ignore_index=True)

        print("masking data...")
        if include_missing:
            feature_level_data = add_missing(
                feature_level_data, 
                mar_missing_param, 
                mnar_missing_param)
    else:
        feature_level_data = None

    return {"Protein_data" : data, 
            "Feature_data" : feature_level_data, 
            "Coefficients" : coefficients}

def generate_coefficients(graph):
    """Generate random coefficients for a graph.

    Parameters
    ----------
    graph : networkx.DiGraph
        The graph to generate coefficients for.

    Returns
    -------
    coefficients : dict
        A dictionary of coefficients for each node.
    """
    coefficients = {}

    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        coefficients[node] = generate_node_coefficients(parents)

    return coefficients

def generate_node_coefficients(parents):
    """Generate random coefficients for a node.

    Parameters
    ----------
    parents : list
        The parents of the node.

    Returns
    -------
    coefficients : dict
        A dictionary of coefficients for the node.
    """
    coefficients = {}

    for parent in parents:
        coefficients[parent] = np.random.choice([np.random.uniform(-1., -.5),
                                                 np.random.uniform(.5, 1.5)])


    if len(coefficients.keys()) == 0:
        coefficients["intercept"] = np.random.uniform(15, 25)
    else:
        coefficients["intercept"] = np.random.uniform(-5, 5)

    coefficients["error"] = np.random.uniform(0,1)

    return coefficients

def simulate_node(data, coefficients, n, cell_type, intervention):
    """Simulate a node.

    Parameters
    ----------
    data : dict
        The data to use for the simulation.
    coefficients : dict
        A dictionary of coefficients for the node.
    n : int
        The number of samples to simulate.
    cell_type : bool
        Whether cell type is included in the simulation.
    intervention : dict
        A dictionary of interventions to apply to the data.
    Returns
    -------
    node_data : numpy.ndarray
        The simulated node data.
    """

    remove = ["intercept", "error", "cell_type"]
    parents = list(coefficients.keys())
    parents = [i for i in parents if i not in remove]

    node_data = coefficients["intercept"]

    for parent in parents:
        node_data += coefficients[parent] * data[parent]

    if cell_type & ("cell_type" in coefficients.keys()):
        ## add in cell_effect
        cells = coefficients["cell_type"]
        n_obs_per_cell = round(n / len(cells))

        for i in range(len(cells)):
            node_data[(i*n_obs_per_cell):(i*n_obs_per_cell+n_obs_per_cell)] += cells[i]

    node_data += np.random.normal(0, coefficients["error"], n)

    if intervention is not None:
        node_data = np.repeat(intervention, n)

    return node_data

def generate_features(data, node):
    """Generate features from a list of data.

    Parameters
    ----------
    data : dict
        The data to generate features from.

    node : str
        The node to generate features for.

    Returns
    -------
    feature_level_data : list
        The data at the feature level.
    """
    feature_level_data = pd.DataFrame(columns=["Protein", "Replicate", 
                                               "Feature", "Intensity"])

    number_features = np.random.randint(15, 30)
    feature_effects = [np.random.uniform(-.75, .75) for _ in range(number_features)]

    for i in range(len(data)):
        for j in range(number_features):
            
            # Measurement error
            error = np.random.normal(0, .1)
            feature_level_data = pd.concat(
                [feature_level_data, 
                 pd.DataFrame(
                     {"Protein": [node],
                      "Replicate": [i],
                      "Feature": [j],
                      "Intensity": [data[i] + feature_effects[j] + error]}
                      )], ignore_index=True)

    return feature_level_data

def add_missing(feature_level_data, mar_missing_param, mnar_missing_param):
    """Add missing data to a feature level dataset.

    Parameters
    ----------
    feature_level_data : pandas.DataFrame
        The feature level data to add missing data to.
    mar_missing_param : float
        The probability of missing data for missing at random (MAR) missingness.
    mnar_missing_param : list
        The parameters for missing not at random (MNAR) missingness.
    Returns
    -------
    feature_level_data : pandas.DataFrame
        The feature level data with missing data added.
    """

    feature_level_data.loc[:, "Obs_Intensity"] = feature_level_data.loc[:, "Intensity"]

    for i in range(len(feature_level_data)):
        mar_prob = np.random.uniform(0, 1)
        mnar_prob = np.random.uniform(0, 1)

        mnar_thresh = 1 / (1 + np.exp(mnar_missing_param[0] +
                                      (mnar_missing_param[1] * feature_level_data.loc[i, "Intensity"])))
        feature_level_data.loc[i, "MNAR_threshold"] = mnar_thresh

        if mar_prob < mar_missing_param:
            feature_level_data.loc[i, "Obs_Intensity"] = np.nan
            feature_level_data.loc[i, "MAR"] = True
        else:
            feature_level_data.loc[i, "MAR"] = False

        if mnar_prob < mnar_thresh:
            feature_level_data.loc[i, "Obs_Intensity"] = np.nan
            feature_level_data.loc[i, "MNAR"] = True
        else:
            feature_level_data.loc[i, "MNAR"] = False

    return feature_level_data

def simple_profile_plot(data, protein, intensity_col="Obs_Intensity"):
    fig, ax = plt.subplots()

    plot_data = data[data["Protein"] == protein]

    sns.scatterplot(x=plot_data.loc[:, "Replicate"], 
                y=plot_data.loc[:, intensity_col], 
                hue=plot_data.loc[:, "Feature"].astype(str))
    sns.lineplot(x=plot_data.loc[:, "Replicate"], 
                y=plot_data.loc[:, intensity_col], 
                hue=plot_data.loc[:, "Feature"].astype(str))
    ax.get_legend().remove()
    ax.set_title(protein)


def build_igf_network(cell_confounder):
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

    if cell_confounder:
        graph.add_edge("cell_type", "Ras")
        graph.add_edge("cell_type", "Raf")
        graph.add_edge("cell_type", "Mek")
        graph.add_edge("cell_type", "Erk")

    return graph



def main():

    from MScausality.simulation.simulation import simple_profile_plot
    from MScausality.simulation.example_graphs import signaling_network

    fd = signaling_network(add_independent_nodes=False)
    simulated_fd_data = simulate_data(fd['Networkx'], 
                                    coefficients=fd['Coefficients'], 
                                    mnar_missing_param=[-3, .3],
                                    add_feature_var=True, n=25, seed=3)

    simple_profile_plot(simulated_fd_data["Feature_data"], "Mek")
    plt.show()

if __name__ == "__main__":
    main()