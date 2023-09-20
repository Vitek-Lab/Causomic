
import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pandas as pd
from MScausality.graph import GraphBuilder
import pickle

import pyro
import pyro.distributions as dist
from pyro.distributions import constraints

from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO
from pyro.infer import MCMC, NUTS

import torch

def scm_model(data, root_nodes, downstream_nodes, missing, learned_params=None): #TODO: add priors and missing

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:

        if learned_params is None:
            root_coef_dict_mean[node_name] = pyro.sample(f"{node_name}_mean", dist.Normal(10., 1.))
            root_coef_dict_scale[node_name] = pyro.sample(f"{node_name}_scale", dist.Exponential(1.))
        else:
            root_coef_dict_mean[node_name] = pyro.sample(f"{node_name}_mean",
                                                         dist.Normal(learned_params[f"{node_name}_mean_mean_param"],
                                                                     learned_params[f"{node_name}_mean_scale_param"]))
            if "latent" in node_name:
                root_coef_dict_scale[node_name] = pyro.sample(f"{node_name}_scale",
                                                              dist.Exponential(1.))
            else:
                root_coef_dict_scale[node_name] = pyro.sample(f"{node_name}_scale",
                                                              dist.Exponential(learned_params[f"{node_name}_scale_param"]))

    for node_name, items in downstream_nodes.items():

        if learned_params is None:
            downstream_coef_dict_mean[f"{node_name}_intercept"] = pyro.sample(f"{node_name}_intercept",
                                                                              dist.Normal(0., 10.))
        else:
            downstream_coef_dict_mean[f"{node_name}_intercept"] = pyro.sample(f"{node_name}_intercept",
                                                              dist.Normal(learned_params[f"{node_name}_intercept_mean_param"],
                                                              learned_params[f"{node_name}_intercept_scale_param"]))

        for item in items:

            if learned_params is None:
                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(f"{node_name}_{item}_coef",
                                                                                    dist.Normal(0., 1.))
            else:
                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(f"{node_name}_{item}_coef",
                                                        dist.Normal(learned_params[f"{node_name}_{item}_coef_mean_param"],
                                                                    learned_params[f"{node_name}_{item}_coef_scale_param"]))

        if learned_params is None:
            downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(f"{node_name}_scale",
                                                                           dist.LogNormal(1., 1.))
        else:
            downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(f"{node_name}_scale",
                                                           dist.Exponential(learned_params[f"{node_name}_scale_param"], 1.))

    # Dictionary to store the Pyro root distribution objects
    downstream_distributions = dict()
    if data is not None:
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
                            root_coef_dict_mean[node_name],#-2*root_coef_dict_scale[node_name],
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
                mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
                for item in items:
                    coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                    mean = mean + coef*downstream_distributions[item]

                # Define scale
                scale = downstream_coef_dict_scale[f"{node_name}_scale"]

                imp = pyro.sample(
                    f"imp_{node_name}", dist.Normal(
                        mean,#-2*scale,
                        scale).mask(False)
                ).detach()

                data[node_name][missing[node_name]] = imp[missing[node_name]].to(torch.double)

                # Create a Normal distribution object
                downstream_sample = pyro.sample(f"{node_name}",
                                                dist.Normal(mean, scale),
                                                obs=data[node_name])

                # Store the distribution in the dictionary
                downstream_distributions[node_name] = downstream_sample
    else:
        for node_name in root_nodes:

            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = pyro.sample(f"{node_name}",
                                          dist.Normal(root_coef_dict_mean[node_name],
                                                      root_coef_dict_scale[node_name]))
            else:
                root_sample = pyro.sample(f"{node_name}",
                                          dist.Normal(root_coef_dict_mean[node_name],#-2*root_coef_dict_mean[node_name],
                                                      root_coef_dict_scale[node_name]))

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = root_sample

        # Create pyro linear regression obj for each downstream node
        for node_name, items in downstream_nodes.items():

            # calculate mean as sum of upstream items
            mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
            for item in items:
                coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                mean = mean + coef * downstream_distributions[item]

            # Define scale
            scale = downstream_coef_dict_scale[f"{node_name}_scale"]

            # Create a Normal distribution object
            downstream_sample = pyro.sample(f"{node_name}",
                                            dist.Normal(mean,#-2*mean,
                                                        scale))

            # Store the distribution in the dictionary
            downstream_distributions[node_name] = downstream_sample
    return downstream_distributions

def guide(data, root_nodes, downstream_nodes, missing): #TODO: add priors and missing

    root_coef_dict_mean = dict()
    root_coef_dict_scale = dict()

    downstream_coef_dict_mean = dict()
    downstream_coef_dict_scale = dict()

    for node_name in root_nodes:
        loc = pyro.param(f'{node_name}_mean_mean_param', torch.tensor(10.))
        loc_scale = pyro.param(f'{node_name}_mean_scale_param', torch.tensor(1.),
                        constraint=constraints.positive)
        scale = pyro.param(f'{node_name}_scale_param', torch.tensor(2.),
                        constraint=constraints.positive)

        root_coef_dict_mean[node_name] = pyro.sample(f"{node_name}_mean", dist.Normal(loc, loc_scale))
        root_coef_dict_scale[node_name] = pyro.sample(f"{node_name}_scale", dist.Exponential(scale))

    for node_name, items in downstream_nodes.items():

        intercept_loc = pyro.param(f'{node_name}_intercept_mean_param', torch.tensor(5.))
        intercept_scale = pyro.param(f'{node_name}_intercept_scale_param', torch.tensor(2.),
                                     constraint=constraints.positive)

        downstream_coef_dict_mean[f"{node_name}_intercept"] = pyro.sample(f"{node_name}_intercept",
                                                                          dist.Normal(intercept_loc,
                                                                                      intercept_scale))

        for item in items:
            coef_loc = pyro.param(f'{node_name}_{item}_coef_mean_param', torch.tensor(1.))
            coef_scale = pyro.param(f'{node_name}_{item}_coef_scale_param', torch.tensor(2.),
                                    constraint=constraints.positive)

            downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(f"{node_name}_{item}_coef",
                                                                                dist.Normal(coef_loc, coef_scale))

        scale = pyro.param(f'{node_name}_scale_param', torch.tensor(2.),
                        constraint=constraints.positive)
        downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(f"{node_name}_scale",
                                                                       dist.Exponential(scale))

    downstream_distributions = dict()
    with pyro.plate("obs", len(data[list(data.keys())[0]])):
        # Create Pyro Normal distributions for each node
        for node_name in root_nodes:

            # Create a Normal distribution object
            if "latent" in node_name:
                root_sample = pyro.sample(f"{node_name}", dist.Normal(root_coef_dict_mean[node_name],
                                                                      root_coef_dict_scale[node_name]))


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

    def fit_scm(self, algorithm="MCMC",
                num_samples=100, warmup_steps=100, num_chains=1,
                num_steps=1000, initial_lr=.001, gamma=.01):

        # filled_data = self.obs_data.fillna(15.)
        filled_data = self.obs_data.iloc[:,:] #367)
        data = dict()
        missing = dict()
        for i in filled_data.columns:
            data[i] = torch.tensor(filled_data[i].values)
            missing[i] = torch.tensor(filled_data[i].isna().values)

        if algorithm == "MCMC":
            nuts_kernel = NUTS(model=scm_model)
            mcmc = MCMC(
                nuts_kernel,
                num_samples=num_samples,
                warmup_steps=warmup_steps,
                num_chains=num_chains)

            mcmc.run(data, self.root_nodes, self.descendent_nodes, missing)

            self.model = scm_model
            self.mcmc = mcmc

            sample_keys = list(mcmc.get_samples().keys())
            learned_params = dict()

            for i in range(len(sample_keys)):
                if "scale" not in sample_keys[i]:
                    learned_params[f"{sample_keys[i]}_mean_param"] = mcmc.get_samples()[sample_keys[i]].mean()
                    learned_params[f"{sample_keys[i]}_scale_param"] = mcmc.get_samples()[sample_keys[i]].std()
                else:
                    learned_params[f"{sample_keys[i]}_param"] = mcmc.get_samples()[sample_keys[i]].mean()

            self.learned_params = learned_params

        elif algorithm == "SVI":
            # set up the optimizer
            lrd = gamma ** (1 / num_steps)
            optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

            guide = AutoNormal(scm_model)

            # setup the inference algorithm
            svi = SVI(scm_model, guide, optim, loss=Trace_ELBO())

            # do gradient steps
            print("starting training")
            for step in range(num_steps):
                loss = svi.step(data, self.root_nodes, self.descendent_nodes, missing)
                if step % 100 == 0:
                    print(loss)
            params = [i for i in pyro.get_param_store().items()]

            self.model = scm_model

            learned_params = dict(params)
            learned_params = {key : value.detach() for key, value in learned_params.items()}
            # print(learned_params)
            rename_params = dict()
            sample_keys = list(learned_params.keys())

            for i in range(len(sample_keys)):
                if ("coef" not in sample_keys[i]) & ("intercept" not in sample_keys[i]) & \
                        ("mean" not in sample_keys[i]) & ("_scale" not in sample_keys[i]):
                    rename_params[f"{sample_keys[i]}"] = learned_params[f"{sample_keys[i]}"]
                elif "_scale" not in sample_keys[i]:
                    if "loc"  in sample_keys[i]:
                        rename_params[f"{sample_keys[i]}_mean_param".replace("AutoNormal.locs.", "")] = learned_params[f"{sample_keys[i]}"]
                    elif "scales" in sample_keys[i]:
                        rename_params[f"{sample_keys[i]}_scale_param".replace("AutoNormal.scales.", "")] = learned_params[f"{sample_keys[i]}"]
                else:
                    rename_params[f"{sample_keys[i]}_param".replace("AutoNormal.locs.", "")] = learned_params[f"{sample_keys[i]}"]

            self.original_params = learned_params
            self.learned_params = rename_params
            # self.learned_params = learned_params

        else:
            raise ValueError("Please choose a valid algorithm: MCMC or SVI")

    def intervention(self, intervention_node, outcome_node, intervention_value):

        # Posterior samples
        post_samples = [self.model(None, self.root_nodes, self.descendent_nodes,
                                   [], self.learned_params)[outcome_node] for _ in range(1000)]

        intervened_model = pyro.do(self.model, data={intervention_node: torch.tensor(intervention_value)})
        int_samples = [intervened_model(None, self.root_nodes, self.descendent_nodes,
                                        [], self.learned_params)[outcome_node] for _ in range(1000)]

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

    data = pd.read_csv("../../data/IGF_pathway/protein_data_10_reps.csv")
    data = data.drop(columns="originalRUN")

    # graph = example[0]
    # data = example[1]

    scm = SCM(data, graph)
    scm.prepare_scm_input()
    scm.fit_scm("SVI", num_steps=2000, initial_lr=.01, gamma=.01)
    scm.intervention("Ras", "Erk", 2.)

    print(scm.learned_params)

if __name__ == "__main__":
    main()