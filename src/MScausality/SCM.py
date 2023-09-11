
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from graph import GraphBuilder
import networkx as nx
import pickle

# import numpyro
# from numpyro.infer import MCMC, NUTS
# from numpyro.distributions import constraints
# numpyro.set_platform('cpu')
# numpyro.set_host_device_count(4)

# from jax import random
# from jax import numpy as jnp

import pyro
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer import MCMC, NUTS, Predictive

import torch

def scm_model(data, root_nodes, downstream_nodes, missing): #TODO: add priors and missing

    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:
        loc = pyro.param(f'{node_name}_mean_param', torch.tensor(10.))
        scale = pyro.param(f'{node_name}_scale_param', torch.tensor(10.))

        root_coef_dict_mean[node_name] = pyro.sample(f"{node_name}_mean", dist.Normal(loc, 2))
        root_coef_dict_scale[node_name] = pyro.sample(f"{node_name}_scale", dist.Exponential(scale))

    for node_name, items in downstream_nodes.items():

        intercept_loc = pyro.param(f'{node_name}_int_loc', torch.tensor(0.))
        intercept_scale = pyro.param(f'{node_name}_int_scale', torch.tensor(2.))

        downstream_coef_dict_mean[f"{node_name}_intercept"] = pyro.sample(f"{node_name}_intercept",
                                                                          dist.Normal(intercept_loc,
                                                                                      intercept_scale))

        for item in items:
            coef_loc = pyro.param(f'{node_name}_coef_loc', torch.tensor(0.))
            coef_scale = pyro.param(f'{node_name}_coef_scale', torch.tensor(2.))

            downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(f"{node_name}_{item}_coef",
                                                                                dist.Normal(coef_loc, coef_scale))

        scale = pyro.param(f'{node_name}_scale_param', torch.tensor(2.))
        downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(f"{node_name}_scale",
                                                                       dist.Exponential(scale))

    with pyro.plate("obs", len(data[list(data.keys())[0]])):
        # Create Pyro Normal distributions for each node
        for node_name in root_nodes:

            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = pyro.sample(f"{node_name}", dist.Normal(root_coef_dict_mean[node_name],
                                                                      root_coef_dict_scale[node_name]))
            else:

                imp = pyro.sample(
                    f"imp_{node_name}", dist.Normal(
                        root_coef_dict_mean[node_name],
                        root_coef_dict_scale[node_name]).mask(False)
                ).detach()

                data[node_name][missing[node_name]] = imp[missing[node_name]].to(torch.double)

                root_sample = pyro.sample(f"{node_name}",
                                          dist.Normal(root_coef_dict_mean[node_name],
                                                      root_coef_dict_scale[node_name]),
                                          obs=data[node_name])

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = root_sample

        # Create pyro linear regression obj for each downstream node
        for node_name, items in downstream_nodes.items():

            # calculate mean as sum of upstream items
            mean = torch.tensor(0.)
            intercept = downstream_coef_dict_mean[f"{node_name}_intercept"]
            mean += intercept
            for item in items:
                coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                mean = mean + coef*downstream_distributions[item]

            # Define scale
            scale = downstream_coef_dict_scale[f"{node_name}_scale"]

            imp = pyro.sample(
                f"imp_{node_name}", dist.Normal(
                    mean, scale).mask(False)
            ).detach()

            data[node_name][missing[node_name]] = imp[missing[node_name]].to(torch.double)

            # Create a Normal distribution object
            downstream_sample = pyro.sample(f"{node_name}",
                                            dist.Normal(mean, scale),
                                            obs=data[node_name])

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = downstream_sample

class SCM:
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

        # Find latent confounder nodes
        latent_edges = list(self.causal_graph.undirected.edges())
        latent_nodes = ["latent_{}".format(i) for i in range(len(latent_edges))]

        # Add in latent confounders to descendent nodes
        for i in range(len(latent_nodes)):
            root_nodes.append(latent_nodes[i])
            for node in latent_edges[i]:
                if node in root_nodes:
                    root_nodes.remove(node)
                    descendent_nodes.append(node)
                    ancestors[node] = [latent_nodes[i]]
                else:
                    ancestors[node].append(latent_nodes[i])

        # Finalize output
        descendent_nodes = {str(name): [str(item) for item in nodes if item != name] for name, nodes in
                            ancestors.items() if name in descendent_nodes}
        root_nodes = [str(i) for i in root_nodes]

        self.root_nodes = root_nodes
        self.descendent_nodes = descendent_nodes

    def fit_scm(self):

        # filled_data = self.obs_data.fillna(15.)
        filled_data = self.obs_data.iloc[:20,:] #367)
        data = dict()
        missing = dict()
        for i in filled_data.columns:
            data[i] = torch.tensor(filled_data[i].values)
            missing[i] = torch.tensor(filled_data[i].isna().values)

        # nuts_kernel = NUTS(model=scm_model)
        # mcmc = MCMC(
        #     nuts_kernel,
        #     num_samples=100,
        #     warmup_steps=100,
        #     num_chains=1)
        # mcmc.run(data, self.root_nodes, self.descendent_nodes)
        # samples_fully_pooled = mcmc.get_samples()

        # set up the optimizer
        adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
        optimizer = Adam(adam_params)

        guide = AutoDiagonalNormal(scm_model)

        # setup the inference algorithm
        svi = SVI(scm_model, guide, optimizer, loss=Trace_ELBO())

        n_steps = 5000

        # do gradient steps
        print("starting training")
        for step in range(n_steps):
            loss = svi.step(data, self.root_nodes, self.descendent_nodes, missing)
            if step % 10 == 0:
                print(loss)

        print("tracker")
        params = [i for i in pyro.get_param_store().items()]
        # num_samples = 100
        # predictive = Predictive(scm_model, guide=guide, num_samples=num_samples)
        # svi_samples = {k: v.reshape(num_samples).detach().cpu().numpy()
        #                for k, v in predictive(None, self.root_nodes, self.descendent_nodes).items()
        #                if k != "obs"}

        # Numpyro
        # warmup_steps = 1000
        # sample_steps = 250
        # mcmc = MCMC(NUTS(scm_model), num_warmup=warmup_steps, num_samples=sample_steps,
        #             num_chains=4)
        # mcmc.run(random.PRNGKey(69), data, self.root_nodes, self.descendent_nodes)

        # hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

        # mcmc.print_summary()

def summary(samples):
    site_stats = {}
    for site_name, values in samples.items():
        marginal_site = pd.DataFrame(values)
        describe = marginal_site.describe(percentiles=[.05, 0.25, 0.5, 0.75, 0.95]).transpose()
        site_stats[site_name] = describe[["mean", "std", "5%", "25%", "50%", "75%", "95%"]]
    return site_stats

def main():

    experimental_data = pd.read_csv("/Users/kohler.d/Library/CloudStorage/OneDrive-NortheasternUniversity/Northeastern/Research/MS_data/Single_cell/Leduc/MSstats/MSstats_summarized.csv")
    indra_statements = pd.read_csv("../../data/sox_pathway.csv")

    # graph = GraphBuilder(indra_statements, experimental_data)
    # graph.build(data_type="TMT",
    #             protein_format="UniProtKB_AC/ID",
    #             evidence_filter=1,
    #             source_name="source_hgnc_symbol",
    #             target_name="target_hgnc_symbol")
    # graph.create_latent_graph()
    # graph.find_all_identifiable_pairs()

    pickle_filename = '../../data/graph_obj.pickle'

    # Save the object to a pickle file
    # with open(pickle_filename, 'wb') as pickle_file:
    #     pickle.dump(graph, pickle_file)

    # Load the object from the pickle file
    with open(pickle_filename, 'rb') as pickle_file:
        graph = pickle.load(pickle_file)

    scm = SCM(graph.experimental_data, graph.causal_graph)
    scm.prepare_scm_input()
    scm.fit_scm()

if __name__ == "__main__":
    main()