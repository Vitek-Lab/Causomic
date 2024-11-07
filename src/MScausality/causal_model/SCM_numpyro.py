
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
from MScausality.graph_construction.graph import GraphBuilder
import pickle

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import constraints
from numpyro import handlers

from operator import attrgetter
from y0.dsl import Variable

import networkx as nx
import y0
from y0.algorithm.simplify_latent import simplify_latent_dag

import jax
from jax import numpy as jnp
from jax import random

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

def scm_model(data, root_nodes, downstream_nodes, missing, learned_params=None): #TODO: add priors and missing

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:

        root_coef_dict_mean[node_name] = numpyro.sample(f"{node_name}_mean", dist.Normal(10., 5.))
        root_coef_dict_scale[node_name] = numpyro.sample(f"{node_name}_scale", dist.Exponential(.2))

    for node_name, items in downstream_nodes.items():

        downstream_coef_dict_mean[f"{node_name}_intercept"] = numpyro.sample(f"{node_name}_intercept",
                                                                          dist.Normal(0., 10.))

        for item in items:

            downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = numpyro.sample(f"{node_name}_{item}_coef",
                                                                                dist.Normal(0., 2.))

        downstream_coef_dict_scale[f"{node_name}_scale"] = numpyro.sample(f"{node_name}_scale",
                                                                       dist.Exponential(1.))


    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()

    # Create Pyro Normal distributions for each node
    for node_name in root_nodes:

        # Create a Normal distribution object
        if "latent" in node_name:
            root_sample = numpyro.sample(f"{node_name}",
                         dist.Normal(root_coef_dict_mean[node_name],
                                     root_coef_dict_scale[node_name]
                                     ).expand(
                             [data[list(data.keys())[0]].shape[0]])
            )
        else:

            imp = numpyro.sample(
                f"imp_{node_name}", dist.Normal(
                    root_coef_dict_mean[node_name],
                    root_coef_dict_scale[node_name]).expand(
                    [sum(missing[node_name] == 1)]
                ).mask(False)
            )

            observed = jnp.asarray(data[node_name]).at[missing[node_name] == 1].set(imp)

            root_sample = numpyro.sample(f"{node_name}",
                                      dist.Normal(root_coef_dict_mean[node_name],
                                                  root_coef_dict_scale[node_name]
                                                  ).expand([data[list(data.keys())[0]].shape[0]]),
                                      obs=observed)

        # Store the distribution in the dictionary
        downstream_distributions[node_name] = root_sample

    # Create pyro linear regression obj for each downstream node
    for node_name, items in downstream_nodes.items():

        # calculate mean as sum of upstream items
        mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
        for item in items:
            coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
            mean = mean + coef*downstream_distributions[item]

        # Define scale
        scale = downstream_coef_dict_scale[f"{node_name}_scale"]

        imp = numpyro.sample(
            f"imp_{node_name}", dist.Normal(
                mean[missing[node_name] == 1],
                scale).mask(False)
        )

        observed = jnp.asarray(data[node_name]).at[missing[node_name] == 1].set(imp)

        # Create a Normal distribution object
        downstream_sample = numpyro.sample(f"{node_name}",
                                        dist.Normal(mean, scale),
                                        obs=observed)

        # Store the distribution in the dictionary
        downstream_distributions[node_name] = downstream_sample

    return downstream_distributions

def int_model(data, root_nodes, downstream_nodes, missing, learned_params=None): #TODO: add priors and missing

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    # with handlers.seed(rng_seed=np.random.randint(-100000, 100000)):
    for node_name in root_nodes:
        root_coef_dict_mean[node_name] = numpyro.sample(f"{node_name}_mean",
                                                        dist.Normal(learned_params[f"{node_name}_mean_mean_param"],
                                                                    learned_params[f"{node_name}_mean_scale_param"]))

        root_coef_dict_scale[node_name] = numpyro.sample(f"{node_name}_scale",
                                                         dist.Exponential(learned_params[f"{node_name}_scale_param"]))

    for node_name, items in downstream_nodes.items():

        downstream_coef_dict_mean[f"{node_name}_intercept"] = numpyro.sample(f"{node_name}_intercept",
                                                                             dist.Normal(
                                                              learned_params[f"{node_name}_intercept_mean_param"],
                                                              learned_params[f"{node_name}_intercept_scale_param"]))

        for item in items:

            downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = numpyro.sample(f"{node_name}_{item}_coef",
                                                                                   dist.Normal(
                                                               learned_params[f"{node_name}_{item}_coef_mean_param"],
                                                               learned_params[f"{node_name}_{item}_coef_scale_param"]))


        downstream_coef_dict_scale[f"{node_name}_scale"] = numpyro.sample(f"{node_name}_scale",
                                                       dist.Exponential(learned_params[f"{node_name}_scale_param"]))

    downstream_distributions = dict()
    for node_name in root_nodes:

        root_sample = numpyro.sample(f"{node_name}",
                                  dist.Normal(root_coef_dict_mean[node_name],
                                              root_coef_dict_scale[node_name]))

        # Store the distribution in the dictionary
        downstream_distributions[node_name] = root_sample

    # Create pyro linear regression obj for each downstream node
    for node_name, items in downstream_nodes.items():

        # calculate mean as sum of upstream items
        mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
        for item in items:
            coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
            mean = mean + coef*downstream_distributions[item]

        # Define scale
        scale = downstream_coef_dict_scale[f"{node_name}_scale"]

        # Create a Normal distribution object
        downstream_sample = numpyro.sample(f"{node_name}",
                                        dist.Normal(mean, scale))

        # Store the distribution in the dictionary
        downstream_distributions[node_name] = downstream_sample
    return downstream_distributions

class SCM: ## TODO: rename to LVM? LVSCM?
    """
    Class for building and fitting a latent variable structural causal model.

    Parameters:
    - observational_data (pd.DataFrame): Dataframe of observational data. Must in long 
    format with proteins as columns as observations as rows.
    - causal_graph (networkx.DiGraph): Causal graph of the data.

    Attributes:
    - obs_data (pd.DataFrame): Dataframe of observational data.
    - causal_graph (networkx.DiGraph): Causal graph of the data.

    Methods:
    - prepare_scm_input: Prepares the input for the SCM model.
    - fit_scm: Fits the SCM model.
    - intervention: Performs an intervention on the SCM model.
    """

    def __init__(self, observational_data, causal_graph):
        self.obs_data = observational_data
        self.causal_graph = causal_graph

    def prepare_scm_input(self):

        # Sort graph
        sorted_nodes = [i for i in self.causal_graph.topological_sort()]

        # Get ancestors of each node
        ancestors = {i: list(self.causal_graph.ancestors_inclusive(i)) for i in sorted_nodes}

        # Find starting nodes
        print("Finding root nodes...")
        root_nodes = [i for i in sorted_nodes if len(ancestors[i]) == 1]
        descendent_nodes = [i for i in sorted_nodes if len(ancestors[i]) != 1]

        temp_descendent_nodes = dict()
        for i in descendent_nodes:
            in_edges = self.causal_graph.directed.in_edges(i)
            temp_descendent_nodes[i] = [i[0] for i in list(self.causal_graph.directed.in_edges(i))]
        descendent_nodes = temp_descendent_nodes

        # Find latent confounder nodes
        latent_edges = list(self.causal_graph.undirected.edges())
        latent_nodes = ["latent_{}".format(i) for i in range(len(latent_edges))]

        # Add in latent confounders to descendent nodes
        print("Adding latent confounders to graph...")
        for i in range(len(latent_nodes)):
            root_nodes.append(latent_nodes[i])
            for node in latent_edges[i]:
                if node in root_nodes:
                    temp = dict()
                    root_nodes.remove(node)
                    temp[node] = [latent_nodes[i]]
                    temp.update(descendent_nodes)
                    descendent_nodes = temp.copy()
                else:
                    descendent_nodes[node].append(latent_nodes[i])

        # Finalize output
        descendent_nodes = {str(name): [str(item) for item in nodes if item != name] for name, nodes in
                            descendent_nodes.items() if name in descendent_nodes}
        root_nodes = [str(i) for i in root_nodes]

        self.root_nodes = root_nodes
        self.descendent_nodes = descendent_nodes

    def compile_model_stats(self, model, prob=.9):

        """
        By default, numpyro only provides model summary statistics in the `print_summary()` function. This is a pain
        because it cannot be saved to a variable and cannot be read with thousands of parameters learned. This function
        simply runs the underlying code for `print_summary()` and returns the summary statistics.

        :param model: trained mcmc model
        :param prob: the probability mass of samples within the HPDI interval.

        :return: dictionary of summary statistics.
        """

        sites = model._states[model._sample_field]

        if isinstance(sites, dict):
            state_sample_field = attrgetter(model._sample_field)(model._last_state)
            if isinstance(state_sample_field, dict):
                sites = {
                    k: v
                    for k, v in model._states[model._sample_field].items()
                    if k in state_sample_field
                }

        for site_name in list(sites):
            if len(sites[site_name].shape) == 3:
                if sites[site_name].shape[2] == 0:
                    # remove key from dictionary
                    sites.pop(site_name)

        summary_stats = numpyro.diagnostics.summary(sites, prob, group_by_chain=True)

        self.summary_stats = summary_stats


    def fit_scm(self, num_samples=1000, warmup_steps=1000, num_chains=4):

        print("Determining missing obs...")
        filled_data = self.obs_data
        data = dict()
        missing = dict()
        for i in filled_data.columns:
            data[i] = np.array(filled_data[i].values)
            missing[i] = np.array(filled_data[i].isna().values)

        # if algorithm == "MCMC":
        print("Running MCMC...")
        mcmc = MCMC(NUTS(scm_model), num_warmup=warmup_steps, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(random.PRNGKey(0), data, self.root_nodes, self.descendent_nodes, missing)
        self.compile_model_stats(mcmc)

            # nuts_kernel = NUTS(model=scm_model)
            # mcmc = MCMC(
            #     nuts_kernel,
            #     num_samples=num_samples,
            #     warmup_steps=warmup_steps,
            #     num_chains=num_chains)
            #
            # mcmc.run(data, self.root_nodes, self.descendent_nodes, missing)
            #
        self.model = scm_model
        self.int_model = int_model
        self.mcmc = mcmc
            #
        sample_keys = list(mcmc.get_samples().keys())
        learned_params = dict()

        for i in range(len(sample_keys)):
            if "scale" not in sample_keys[i]:
                learned_params[f"{sample_keys[i]}_mean_param"] = mcmc.get_samples()[sample_keys[i]].mean()
                learned_params[f"{sample_keys[i]}_scale_param"] = mcmc.get_samples()[sample_keys[i]].std()
            else:
                learned_params[f"{sample_keys[i]}_param"] = mcmc.get_samples()[sample_keys[i]].mean()

        self.learned_params = learned_params

        # else:
        #     raise ValueError("Please choose a valid algorithm: MCMC or SVI")

    def intervention(self, intervention_node, outcome_node, intervention_value, n=1000,
                     return_all = False):

        # Posterior samples
        if return_all:
            post_samples = [numpyro.handlers.seed(self.int_model, np.random.randint(-1*10**10, 1*10**10)
                                                  )(None, self.root_nodes, self.descendent_nodes,
                                       [], self.learned_params) for i in range(n)]

            intervened_model = numpyro.handlers.do(self.int_model, data={intervention_node: intervention_value})
            int_samples = [numpyro.handlers.seed(intervened_model, np.random.randint(-1*10**10, 1*10**10)
                                                 )(None, self.root_nodes, self.descendent_nodes,
                                            [], self.learned_params) for i in range(n)]
        else:
            post_samples = [numpyro.handlers.seed(self.int_model, np.random.randint(-1*10**10, 1*10**10)
                                                  )(None, self.root_nodes, self.descendent_nodes,
                                       [], self.learned_params)[outcome_node] for i in range(n)]

            intervened_model = numpyro.handlers.do(self.int_model, data={intervention_node: intervention_value})
            int_samples = [numpyro.handlers.seed(intervened_model, np.random.randint(-1*10**10, 1*10**10)
                                                 )(None, self.root_nodes, self.descendent_nodes,
                                            [], self.learned_params)[outcome_node] for i in range(n)]

        self.posterior_samples = post_samples
        self.intervention_samples = int_samples

def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def build_igf_network(cell_confounder):
    """
    Create IGF graph in networkx
    
    cell_confounder : bool
        Whether to add in cell type as a confounder
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("E", "B")
    graph.add_edge("E", "A")
    
    return graph

def build_admg(graph, cell_confounder=False, cell_latent=False):
    ## Define obs vs latent nodes
    all_nodes = ["A", "B", "C", "E"]
    obs_nodes = ["A", "B", "C"]
    latent_nodes = ["E"]
        
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph, "hidden")
    
    return y0_graph


def main():

    # # Data
    # data = pd.read_csv("data/Talus/processed_data/ProteinLevelData.csv")

    # # Build graph
    # graph = pd.read_csv("data/INDRA_networks/Talus_networks/GCM_TPR_obs.tsv", sep="\t")
    # graph = GraphBuilder(graph, data)
    # graph.build(data_type="LF",
    #             protein_format="Gene_Name_Organism",
    #             evidence_filter=2,
    #             source_name="source_hgnc_symbol",
    #             target_name="target_hgnc_symbol")
    # graph.create_latent_graph()

    data = pd.read_csv("data/sim_data/four_node_protein_data.csv")

    graph = build_igf_network(cell_confounder=False)
    y0_graph = build_admg(graph, cell_confounder=False)

    scm = SCM(data.iloc[:, 1:-1], y0_graph)
    scm.prepare_scm_input()
    scm.fit_scm()
    scm.intervention("MEN1", "TPR", 10.)

    print(scm.learned_params)

if __name__ == "__main__":
    main()