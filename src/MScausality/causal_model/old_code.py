
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
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive

from operator import attrgetter
from y0.dsl import Variable

import networkx as nx
import y0
from y0.algorithm.simplify_latent import simplify_latent_dag

import jax
from jax import numpy as jnp
from jax import random


def scm_model(data, root_nodes, downstream_nodes, missing, sample=False): #TODO: add priors and missing

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:

        root_coef_dict_mean[node_name] = numpyro.sample(f"{node_name}_mean", dist.Normal(0., 5.))
        root_coef_dict_scale[node_name] = numpyro.sample(f"{node_name}_scale", dist.Exponential(.1))

    for node_name, items in downstream_nodes.items():

        downstream_coef_dict_mean[f"{node_name}_intercept"] = numpyro.sample(f"{node_name}_intercept",
                                                                          dist.Normal(0., 5.))

        for item in items:

            downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = numpyro.sample(f"{node_name}_{item}_coef",
                                                                                dist.Normal(0., 5.))

        downstream_coef_dict_scale[f"{node_name}_scale"] = numpyro.sample(f"{node_name}_scale",
                                                                       dist.Exponential(1.))


    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()

    # Create Pyro Normal distributions for each node
    for node_name in root_nodes:
        if not sample:
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
        else:
            root_sample = numpyro.sample(f"{node_name}",
                                    dist.Normal(root_coef_dict_mean[node_name],
                                                root_coef_dict_scale[node_name]
                                                ))
            
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

        if not sample:

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
        else:
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
        self.missing = missing
        self.data = data

        # if algorithm == "MCMC":
        print("Running MCMC...")
        mcmc = MCMC(NUTS(scm_model), num_warmup=warmup_steps, num_samples=num_samples, num_chains=num_chains)
        mcmc.run(random.PRNGKey(0), data, self.root_nodes, self.descendent_nodes, missing)
        self.compile_model_stats(mcmc)

        self.model = scm_model
        self.mcmc = mcmc

        sample_keys = list(mcmc.get_samples().keys())
        learned_params = dict()

        for i in range(len(sample_keys)):
            if "scale" not in sample_keys[i] and "imp" not in sample_keys[i]:
                learned_params[f"{sample_keys[i]}_mean_param"] = mcmc.get_samples()[sample_keys[i]].mean()
                learned_params[f"{sample_keys[i]}_scale_param"] = mcmc.get_samples()[sample_keys[i]].std()
            elif "scale" in sample_keys[i]:
                learned_params[f"{sample_keys[i]}_param"] = mcmc.get_samples()[sample_keys[i]].mean()
            else:
                learned_params[f"{sample_keys[i]}"] = mcmc.get_samples()[sample_keys[i]].mean(axis=0)
                learned_params[f"{sample_keys[i]}_scale"] = mcmc.get_samples()[sample_keys[i]].std(axis=0)

        self.learned_params = learned_params
        self.add_imputed_values()

        # else:
        #     raise ValueError("Please choose a valid algorithm: MCMC or SVI")

    def intervention(self, intervention_node, outcome_node, intervention_value, n=1000,
                     return_all = False):

        rng_key, rng_key_ = random.split(random.PRNGKey(2))

        zero_model = numpyro.handlers.do(self.model, 
                                data={intervention_node: 0.})
        zero_predictive = Predictive(zero_model, self.mcmc.get_samples())
        zero_predictions = zero_predictive(rng_key_, None, 
                                           self.root_nodes,
                                           self.descendent_nodes, 
                                           [], sample=True)

        int_model = numpyro.handlers.do(self.model, 
                                data={intervention_node: intervention_value})
        int_predictive = Predictive(int_model, self.mcmc.get_samples())
        int_predictions = int_predictive(rng_key_, None, 
                                           self.root_nodes,
                                           self.descendent_nodes, 
                                           [], sample=True)
        
        if not return_all:
            zero_predictions = zero_predictions[outcome_node]
            int_predictions = int_predictions[outcome_node]
        
        self.posterior_samples = zero_predictions
        self.intervention_samples = int_predictions

    def add_imputed_values(self):
        """
        Adds imputed values back into data with mean and scale.
        """
        
        # Put data into long format
        long_data = pd.melt(self.obs_data, var_name="protein", 
                            value_name="intensity")
        long_data.loc[long_data["intensity"] == 0, "intensity"] = np.nan
        long_data['was_missing'] = long_data['intensity'].isna()
        long_data.loc[:, "imp_mean"] = np.nan

        # Extract imputation info from model parameters
        # TODO: (?) Put this into a function to stop code repeat
        loc_params = [i for i in self.learned_params.keys() if ("imp" in i)]
        loc_params = {key.replace("imp_", ""): self.learned_params[key] \
                   for key in loc_params}

        for variable, values in loc_params.items():
            mask = (long_data['protein'] == variable) & long_data['was_missing']
            if sum(mask) > 0:
                na_indices = long_data[mask].index  # Get the indices for missing values in each variable
                fill_len = min(len(values), len(na_indices))  # Determine how many values to use
                long_data.loc[na_indices[:fill_len], 'imp_mean'] = values[:fill_len]

        self.imputed_data = long_data

def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

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

    # data = pd.read_csv("data/sim_data/four_node_protein_data.csv")

    # graph = build_igf_network(cell_confounder=False)
    # y0_graph = build_admg(graph, cell_confounder=False)

    from MScausality.simulation.example_graphs import mediator
    from MScausality.simulation.simulation import simulate_data
    from MScausality.data_analysis.dataProcess import dataProcess
    from MScausality.data_analysis.normalization import normalize


    med = mediator(add_independent_nodes=False, n_ind=50)
    simulated_med_data = simulate_data(med['Networkx'], 
                                    coefficients=med['Coefficients'], 
                                    mnar_missing_param=[20, .4],
                                    add_feature_var=True, n=50, seed=2)
    med_data = dataProcess(simulated_med_data["Feature_data"], normalization=False, 
                summarization_method="TMP", MBimpute=False, sim_data=True)
    
    transformed_data = normalize(med_data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    scm = SCM(input_data, med['MScausality'])
    scm.prepare_scm_input()
    scm.fit_scm()
    scm.intervention("X", "Z", 2.)

    print(np.array(scm.intervention_samples).mean()-np.array(scm.posterior_samples).mean())

    imp_data = scm.imputed_data
    X_data = imp_data.loc[imp_data["protein"] == "X"]
    Y_data = imp_data.loc[imp_data["protein"] == "M1"]
    Z_data = imp_data.loc[imp_data["protein"] == "Z"]

    X_backdoor_color = np.where(
        (X_data['imp_mean'].isna().values & Y_data['imp_mean'].isna().values), 
        "blue", 
        np.where((X_data['intensity'].isna().values & Y_data['intensity'].isna().values), 
                "red", "orange"))

    Y_backdoor_color = np.where(
        (Y_data['imp_mean'].isna().values & Z_data['imp_mean'].isna().values), 
        "blue", 
        np.where((Y_data['intensity'].isna().values & Z_data['intensity'].isna().values), 
                "red", "orange"))

    import matplotlib.pyplot as plt

    X_data = np.where(
        X_data['imp_mean'].isna(),
        X_data['intensity'], 
        X_data['imp_mean'])

    Y_data = np.where(
        Y_data['imp_mean'].isna(),
        Y_data['intensity'], 
        Y_data['imp_mean'])

    Z_data = np.where(
        Z_data['imp_mean'].isna(),
        Z_data['intensity'], 
        Z_data['imp_mean'])


    fig, ax = plt.subplots(2,2, figsize=(10,5))

    ax[0,0].scatter(input_data.loc[:, "X"], input_data.loc[:, "M1"])
    ax[0,1].scatter(X_data, Y_data, color=X_backdoor_color)

    ax[1,0].scatter(input_data.loc[:, "M1"], input_data.loc[:, "Z"])
    ax[1,1].scatter(Y_data, Z_data, color=Y_backdoor_color)
    plt.show()
    # ax[0,0].set_xlim(0,2.5)
    # ax[0,1].set_xlim(0,2.5)
    # ax[0,0].set_ylim(-.5,.8)
    # ax[0,1].set_ylim(-.5,.8)

    # ax[1,0].set_xlim(-1,1)
    # ax[1,1].set_xlim(-1,1)
    # ax[1,0].set_ylim(-2.5,-.5)
    # ax[1,1].set_ylim(-2.5,-.5)

if __name__ == "__main__":
    main()