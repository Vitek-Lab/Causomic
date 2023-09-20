
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
import numpy as np
from MScausality.graph import GraphBuilder
import pickle

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.distributions import constraints
numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)
from operator import attrgetter
from y0.dsl import Variable

import jax
from jax import numpy as jnp
from jax import random

def scm_model(data, root_nodes, downstream_nodes, missing, learned_params=None): #TODO: add priors and missing

    key = np.random.randint(-1000000, 1000000)

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:

        if learned_params is None:
            root_coef_dict_mean[node_name] = numpyro.sample(f"{node_name}_mean", dist.Normal(10., 5.))
            root_coef_dict_scale[node_name] = numpyro.sample(f"{node_name}_scale", dist.Exponential(.2))
        else:
            root_coef_dict_mean[node_name] = numpyro.sample(f"{node_name}_mean",
                                                         dist.Normal(learned_params[f"{node_name}_mean_mean_param"],
                                                                     learned_params[f"{node_name}_mean_scale_param"]),
                                                            rng_key=random.PRNGKey(key))

            root_coef_dict_scale[node_name] = numpyro.sample(f"{node_name}_scale",
                                                          dist.Exponential(learned_params[f"{node_name}_scale_param"]),
                                                             rng_key=random.PRNGKey(key))

    for node_name, items in downstream_nodes.items():

        if learned_params is None:
            downstream_coef_dict_mean[f"{node_name}_intercept"] = numpyro.sample(f"{node_name}_intercept",
                                                                              dist.Normal(0., 10.))
        else:
            downstream_coef_dict_mean[f"{node_name}_intercept"] = numpyro.sample(f"{node_name}_intercept",
                                                              dist.Normal(learned_params[f"{node_name}_intercept_mean_param"],
                                                              learned_params[f"{node_name}_intercept_scale_param"]),
                                                                                 rng_key=random.PRNGKey(key))

        for item in items:

            if learned_params is None:
                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = numpyro.sample(f"{node_name}_{item}_coef",
                                                                                    dist.Normal(0., 2.))
            else:
                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = numpyro.sample(f"{node_name}_{item}_coef",
                                                        dist.Normal(learned_params[f"{node_name}_{item}_coef_mean_param"],
                                                                    learned_params[f"{node_name}_{item}_coef_scale_param"]),
                                                            rng_key=random.PRNGKey(key))

        if learned_params is None:
            downstream_coef_dict_scale[f"{node_name}_scale"] = numpyro.sample(f"{node_name}_scale",
                                                                           dist.Exponential(1.))
        else:
            downstream_coef_dict_scale[f"{node_name}_scale"] = numpyro.sample(f"{node_name}_scale",
                                                           dist.Exponential(learned_params[f"{node_name}_scale_param"]),
                                                            rng_key=random.PRNGKey(key))

    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()
    if data is not None:
        # Create Pyro Normal distributions for each node
        for node_name in root_nodes:

            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = numpyro.sample(f"{node_name}", dist.Normal(root_coef_dict_mean[node_name],
                                                                      root_coef_dict_scale[node_name]
                                                                         ).expand([data[list(data.keys())[0]].shape[0]]))
            else:

                imp = numpyro.sample(
                    f"imp_{node_name}", dist.Normal(
                        root_coef_dict_mean[node_name] - (2*root_coef_dict_scale[node_name]),
                        root_coef_dict_scale[node_name]).expand([sum(missing[node_name] == 1)]).mask(False)
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
                    mean[missing[node_name] == 1] - (2*scale),
                    scale).mask(False)
            )

            observed = jnp.asarray(data[node_name]).at[missing[node_name] == 1].set(imp)
            # data[node_name][missing[node_name]] = imp[missing[node_name]].to(torch.double)

            # Create a Normal distribution object
            downstream_sample = numpyro.sample(f"{node_name}",
                                            dist.Normal(mean, scale),
                                            obs=observed)

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = downstream_sample
    else:
        for node_name in root_nodes:

            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = numpyro.sample(f"{node_name}",
                                          dist.Normal(root_coef_dict_mean[node_name],
                                                      root_coef_dict_scale[node_name]),
                                             rng_key=random.PRNGKey(key))
            else:
                root_sample = numpyro.sample(f"{node_name}",
                                          dist.Normal(root_coef_dict_mean[node_name],
                                                      root_coef_dict_scale[node_name]),
                                             rng_key=random.PRNGKey(key))

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
                                            dist.Normal(mean, scale),
                                               rng_key=random.PRNGKey(key))

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = downstream_sample
    return downstream_distributions

class SCM: ## TODO: rename to LVM? LVSCM?
    def __init__(self, observational_data, causal_graph):
        self.obs_data = observational_data
        self.causal_graph = causal_graph

    def prepare_scm_input(self):

        # Sort graph
        sorted_nodes = [i for i in self.causal_graph.topological_sort()]

        # Get ancestors of each node
        ancestors = {i: list(self.causal_graph.ancestors_inclusive(i)) for i in sorted_nodes}

        # Find starting nodes
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

        # filled_data = self.obs_data.fillna(15.)
        filled_data = self.obs_data.iloc[:,:] #367)
        data = dict()
        missing = dict()
        for i in filled_data.columns:
            data[i] = np.array(filled_data[i].values)
            missing[i] = np.array(filled_data[i].isna().values)

        # if algorithm == "MCMC":

        mcmc = MCMC(NUTS(scm_model), num_warmup =warmup_steps, num_samples =num_samples, num_chains =num_chains)
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

    def intervention(self, intervention_node, outcome_node, intervention_value, n=1000):

        # Posterior samples
        post_samples = [self.model(None, self.root_nodes, self.descendent_nodes,
                                   [], self.learned_params) for _ in range(n)]

        intervened_model = numpyro.handlers.do(self.model, data={intervention_node: intervention_value})
        int_samples = [intervened_model(None, self.root_nodes, self.descendent_nodes,
                                        [], self.learned_params)[outcome_node] for _ in range(n)]

        self.posterior_samples = post_samples
        self.intervention_samples = int_samples

def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def main():

    # experimental_data = pd.read_csv("/mnt/d/OneDrive-NortheasternUniversity/Northeastern/Research/MS_data/Single_cell/Leduc/MSstats/MSstats_summarized.csv")
    # indra_statements = pd.read_csv("../../data/sox_pathway.csv")

    # graph = GraphBuilder(indra_statements, experimental_data)
    # graph.build(data_type="TMT",
    #             protein_format="UniProtKB_AC/ID",
    #             evidence_filter=1,
    #             source_name="source_hgnc_symbol",
    #             target_name="target_hgnc_symbol")
    # graph.create_latent_graph()
    # graph.find_all_identifiable_pairs()

    pickle_filename = '../../data/IGF_pathway/igf_graph.pkl'



    # Save the object to a pickle file
    # with open(pickle_filename, 'wb') as pickle_file:
    #     pickle.dump(graph, pickle_file)

    # Load the object from the pickle file
    with open(pickle_filename, 'rb') as pickle_file:
        graph = pickle.load(pickle_file)

    data = pd.read_csv("../../data/IGF_pathway/protein_data.csv")

    # graph = example[0]
    # data = example[1]

    scm = SCM(data, graph)
    scm.prepare_scm_input()
    scm.fit_scm()
    scm.intervention("Ras", "Erk", 10.)

    print(scm.learned_params)

if __name__ == "__main__":
    main()