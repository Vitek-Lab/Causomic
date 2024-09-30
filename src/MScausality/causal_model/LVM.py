

from MScausality.simulation.simulation import simulate_data

import os

import pandas as pd
import numpy as np

import pyro
import pyro.distributions as dist

from pyro.infer.autoguide import AutoNormal
from pyro.infer import SVI, Trace_ELBO

from pyro.nn import PyroModule
import pyro.poutine as poutine

import torch

import networkx as nx
import y0
from y0.dsl import Variable
from y0.algorithm.simplify_latent import simplify_latent_dag

from chirho.interventional.handlers import do
from chirho.counterfactual.handlers import MultiWorldCounterfactual

# TODO: give user the option to reset parameters or not (new models vs more training)
# pyro.clear_param_store()
# pyro.settings.set(module_local_params=True)

class ProteomicPerturbationModel(PyroModule):

    def __init__(self, n_obs, root_nodes, downstream_nodes):
        
        super().__init__()
        self.n_obs = n_obs
        self.root_nodes = root_nodes
        self.downstream_nodes = downstream_nodes

    def forward(self, data):

        downstream_coef_dict_mean = dict()
        downstream_coef_dict_scale = dict()

        root_coef_dict_mean = dict()
        root_coef_dict_scale = dict()

        for node_name, items in self.downstream_nodes.items():

            downstream_coef_dict_mean[f"{node_name}_intercept"] = pyro.sample(
                    f"{node_name}_intercept", dist.Normal(0., 1.))
            
            for item in items:

                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(
                    f"{node_name}_{item}_coef", dist.Normal(0., 1.))

            downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                f"{node_name}_scale", dist.Exponential(2.))

        for node_name in self.root_nodes:
            root_coef_dict_mean[f"{node_name}_int"] = pyro.sample(
                f"{node_name}_int", dist.Normal(0,1))
            root_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                f"{node_name}_scale", dist.Exponential(.25))

        # Dictionary to store the Pyro root distribution objects
        downstream_distributions = dict()
        with pyro.plate("observations", self.n_obs):
            
            # Create Pyro Normal distributions for each node
            for node_name in self.root_nodes:
            
                # Create a Normal distribution object
                if "latent" in node_name:
                    root_sample = pyro.sample(f"{node_name}", dist.Normal(
                        root_coef_dict_scale[f"{node_name}_scale"],
                        root_coef_dict_scale[f"{node_name}_scale"])
                        )
                else:

                    missing = pyro.sample(f"missing_{node_name}", dist.Bernoulli(0.25),
                                      obs=data[f"missing_{node_name}"])

                    # Impute missing values where needed
                    with poutine.mask(mask=missing.bool()):
                        missing_values = pyro.sample(f"imp_{node_name}", dist.Normal(0, 1))  # no obs

                    if f"obs_{node_name}" in data:
                        observed = data[f"obs_{node_name}"].detach_()
                        observed[missing.bool()] = missing_values[missing.bool()]
                        
                        root_sample = pyro.sample(
                            f"{node_name}",
                            dist.Normal(root_coef_dict_mean[f"{node_name}_int"], 
                                        root_coef_dict_scale[f"{node_name}_scale"]),
                            obs=observed)
                    else:
                        root_sample = pyro.sample(
                            f"{node_name}",
                            dist.Normal(root_coef_dict_mean[f"{node_name}_int"], 
                                        root_coef_dict_scale[f"{node_name}_scale"]))

                # Store the distribution in the dictionary
                downstream_distributions[node_name] = root_sample

            # Create pyro linear regression obj for each downstream node
            for node_name, items in self.downstream_nodes.items():

                # calculate mean as sum of upstream items
                mean = downstream_coef_dict_mean[f"{node_name}_intercept"]
                for item in items:
                    coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                    mean = mean + coef*downstream_distributions[item]

                # Define scale
                scale = downstream_coef_dict_scale[f"{node_name}_scale"]
                
                # Mask this a boolean
                missing = pyro.sample(f"missing_{node_name}", dist.Bernoulli(0.25),
                                      obs=data[f"missing_{node_name}"])

                # Impute missing values where needed
                with poutine.mask(mask=missing.bool()):
                    missing_values = pyro.sample(f"imp_{node_name}", dist.Normal(mean-scale, scale))  # no obs

                if f"obs_{node_name}" in data:
                    observed = data[f"obs_{node_name}"].detach_()
                    observed[missing.bool()] = missing_values[missing.bool()]

                    downstream_sample = pyro.sample(
                            f"{node_name}",
                            dist.Normal(mean, scale),
                            obs=observed)
                else:
                    downstream_sample = pyro.sample(
                        f"{node_name}",
                        dist.Normal(mean, scale))

                # Store the distribution in the dictionary
                downstream_distributions[node_name] = downstream_sample

        return downstream_distributions

# class ConditionedProteomicModel(PyroModule):
#     def __init__(self, model: ProteomicPerturbationModel):
#         super().__init__()
#         self.model = model

#     def forward(self, condition_data):#**kwargs
#         # with condition(data=condition_data):
#         return self.model(data=condition_data)

class ProteomicPerturbationCATE(pyro.nn.PyroModule):
    def __init__(self, model: ProteomicPerturbationModel):
        super().__init__()
        self.model = model

    def forward(self, intervention, condition_data):#, obs_data, missing, root_nodes, downstream_nodes, intervention, intervention_node

        # with do(actions={intervention_node: (torch.tensor(intervention).float())}):#, \
            # condition(data=condition_data):#, MultiWorldCounterfactual(), 
        with do(actions=intervention):
            return self.model(data=condition_data)
        # with MultiWorldCounterfactual(), do(actions=intervention):
        #     return self.model(data=condition_data)
        
class LVM: ## TODO: rename to LVM? LVSCM?
    def __init__(self, observational_data, causal_graph):
        self.obs_data = observational_data
        self.causal_graph = causal_graph

    def prepare_graph(self):

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

    def prepare_data(self):

        data = dict()
        missing = dict()
        for i in self.obs_data.columns:
            data[i] = torch.tensor(self.obs_data[i].values).float()
            missing[i] = torch.tensor(self.obs_data[i].isna().values).float()
            data[i][self.obs_data[i].isna()] = 0.
        
        self.input_data = pd.DataFrame.from_dict(data)
        self.input_missing = pd.DataFrame.from_dict(missing)
    
    def prep_condition_data(self):
        condition_data = dict()
        for node in self.root_nodes:
            if "latent" not in node:
                condition_data[f"obs_{node}"] = torch.tensor(
                    self.input_data.loc[:, node].values)
                condition_data[f"missing_{node}"] = torch.tensor(
                    self.input_missing.loc[:, node].values)

        for node in self.descendent_nodes:
            condition_data[f"obs_{node}"] = torch.tensor(
                self.input_data.loc[:, node].values)
            condition_data[f"missing_{node}"] = torch.tensor(
                self.input_missing.loc[:, node].values)

        return condition_data
    
    def compile_parameters(self):
        
        """
        Takes the learned pyro parameters and compiles them into readable format.
        """

        params = [i for i in pyro.get_param_store().items()]
        params = dict(params)
        params = {key : value.detach() for key, value in params.items()}
        self.original_params = params

        # Extract coefficients locs
        loc_params = [i for i in params.keys() if ("imp" not in i) & \
                             ("latent" not in i) & ("locs" in i)]
        loc_coefs = pd.DataFrame().from_dict(
            {key.replace("AutoNormal.locs.", ""): params[key] \
             for key in loc_params}, 
             orient='index', columns=["mean"]).reset_index(names="parameter")
        
        scale_params = [i for i in params.keys() if ("imp" not in i) & \
                             ("latent" not in i) & ("scales" in i)]
        scale_coefs = pd.DataFrame().from_dict(
            {key.replace("AutoNormal.scales.", ""): params[key] \
             for key in scale_params}, 
             orient='index', columns=["scale"]).reset_index(names="parameter")
        
        coef_data = pd.merge(loc_coefs, scale_coefs, on="parameter")
        coef_data["mean"] = [i.numpy() for  i in coef_data["mean"]]
        coef_data["scale"] = [i.numpy() for  i in coef_data["scale"]]

        self.parameters = coef_data

    def add_imputed(self):
        """
        Adds imputed values back into data with mean and scale.
        """
        
        # Put data into long format
        long_data = pd.melt(self.input_data, var_name="protein", 
                            value_name="intensity")
        long_data.loc[long_data["intensity"] == 0, "intensity"] = np.nan
        long_data.loc[:, "imp_mean"] = np.nan
        long_data.loc[:, "imp_scale"] = np.nan

        # Extract imputation info from model parameters
        # TODO: (?) Put this into a function to stop code repeat
        loc_params = [i for i in self.original_params.keys() if ("imp" in i) & \
                      ("locs" in i)]
        loc_imp = pd.DataFrame().from_dict(
            {key.replace("AutoNormal.locs.imp_", ""): self.original_params[key] \
             for key in loc_params})
        loc_imp = loc_imp[self.input_missing.astype(bool)]
        loc_imp = pd.melt(loc_imp, var_name="protein", 
                            value_name="imp_loc")
        loc_imp = loc_imp.dropna(ignore_index=True)
                
        scale_params = [i for i in self.original_params.keys() if ("imp" in i) & \
                      ("scales" in i)]
        scale_imp = pd.DataFrame().from_dict(
            {key.replace("AutoNormal.scales.imp_", ""): self.original_params[key] \
             for key in scale_params})
        scale_imp = scale_imp[self.input_missing.astype(bool)]
        scale_imp = pd.melt(scale_imp, var_name="protein", 
                            value_name="imp_scale")
        scale_imp = scale_imp.dropna(ignore_index=True)

        for col in loc_imp["protein"].unique():

            long_data.loc[(long_data["protein"] == col) & \
                        (long_data["intensity"].isna()), "imp_mean"] = \
                            loc_imp.loc[loc_imp["protein"] == col, 
                                        "imp_loc"].values
            
            long_data.loc[(long_data["protein"] == col) & \
                        (long_data["intensity"].isna()), "imp_scale"] = \
                            scale_imp.loc[scale_imp["protein"] == col, 
                                        "imp_scale"].values
        
        self.imputed_data = long_data
        

    def fit_model(self, num_steps=2000, 
                  initial_lr=.03, gamma=.01,
                  patience=100, min_delta=5):
        pyro.set_rng_seed(1234)

        #ConditionedProteomicModel(
        model = ProteomicPerturbationModel(n_obs = len(self.input_data), 
                                       root_nodes = self.root_nodes, 
                                       downstream_nodes = self.descendent_nodes)
                                    #    )
        condition_data = self.prep_condition_data()
        
        # set up the optimizer
        lrd = gamma ** (1 / num_steps)
        optim = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})

        guide = AutoNormal(model)

        # setup the inference algorithm
        svi = SVI(model, guide, optim, loss=Trace_ELBO())

        # do gradient steps
        best_loss = float('inf')
        steps_since_improvement = 0

        for step in range(num_steps):
            loss = svi.step(condition_data)
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss}")

            # Check for improvement
            if loss < best_loss - min_delta:
                best_loss = loss
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            # Early stopping condition
            if steps_since_improvement >= patience:
                print(f"Stopping early at step {step} with loss {loss}")
                break

        self.model = model
        self.guide = guide

        self.compile_parameters()
        self.add_imputed()

    def intervention(self, intervention, outcome_node, compare_value=0.):
        
        # Prep interventional conditioning data
        condition_data_test = dict()
        for node in self.root_nodes:
            if "latent" not in node:
                condition_data_test[f"missing_{node}"] = torch.tensor(
                    [0 for _ in range(len(self.input_data))])
                # condition_data_test[f"obs_{node}"] = torch.tensor(
                #     [0. for _ in range(len(self.input_data))])

        for node in self.descendent_nodes:
            condition_data_test[f"missing_{node}"] = torch.tensor(
                [0 for _ in range(len(self.input_data))])
            # condition_data_test[f"obs_{node}"] = torch.tensor(
            #     [0. for _ in range(len(self.input_data))])

        # Posterior samples
        ate_model = ProteomicPerturbationCATE(
            ProteomicPerturbationModel(n_obs = len(self.input_data),
                                       root_nodes = self.root_nodes, 
                                       downstream_nodes = self.descendent_nodes
                                       ))
        
        ate_predictive = pyro.infer.Predictive(
            ate_model, guide=self.guide, num_samples=50)
        zero_int = dict()
        for key, value in intervention.items():
            zero_int[key] = torch.tensor(compare_value).float()
            intervention[key] = torch.tensor(value).float()

        zero_int = ate_predictive(zero_int, 
                                  condition_data_test)
        intervention = ate_predictive(intervention, 
                                      condition_data_test)

        self.posterior_samples = zero_int[outcome_node].flatten()
        self.intervention_samples = intervention[outcome_node].flatten()

# Some functions for testing
def build_igf_network():
    """
    Create IGF graph in networkx
    
    cell_confounder : bool
        Whether to add in cell type as a confounder
    """
    graph = nx.DiGraph()

    ## Add edges
    graph.add_edge("IL6", "STAT3")
    graph.add_edge("STAT3", "MYC")
    
    return graph

def build_admg(graph):
    ## Define obs vs latent nodes
    all_nodes = ["IL6", "STAT3", "MYC"]
    obs_nodes = ["IL6", "STAT3", "MYC"]
            
    attrs = {node: (True if node not in obs_nodes and 
                    node != "\\n" else False) for node in all_nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")
    
    ## Use y0 to build ADMG
    mapping = dict(zip(list(graph.nodes), 
                      [Variable(i) for i in list(graph.nodes)]))
    graph = nx.relabel_nodes(graph, mapping)
    
    ## Use y0 to build ADMG
    simplified_graph = simplify_latent_dag(graph.copy(), tag="hidden")
    y0_graph = y0.graph.NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")
    
    return y0_graph

def main():

    graph = build_igf_network()
    y0_graph = build_admg(graph)

    # data = pd.read_csv("data/IGF_pathway/high_missing_protein_data.csv")
    # data = data.drop("originalRUN", axis=1)
    # data = (data - data.mean()) / data.std()

    ## Coefficients for relations
    med_coef = {'IL6': {'intercept': 15, "error": 1.},
                'STAT3': {'intercept': 1.6, "error": .25,  'IL6': 0.5},
                'MYC': {'intercept': 2, "error": .25, 'STAT3': 1.}
                }
    data = pd.DataFrame(simulate_data(graph,
                    coefficients=med_coef,
                    include_missing=True,
                    mar_missing_param=.05,
                    mnar_missing_param=[-3, .4],
                    add_feature_var=False,
                    n=10000,
                  seed=2)["Protein_data"])

    lvm = LVM(data, y0_graph)
    lvm.prepare_graph()
    lvm.prepare_data()

    lvm.fit_model(num_steps=100)
    # lvm.intervention({"Ras": (torch.tensor(3.))}, "Erk")

    # int1 = lvm.posterior_samples
    # int2 = lvm.intervention_samples
    # plot_data = lvm.imputed_data[lvm.imputed_data["protein"].isin(["Raf", "Mek"])]
    # pivot wide
    # raf_data = plot_data[plot_data["protein"] == "Raf"]
    # mek_data = plot_data[plot_data["protein"] == "Mek"]

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()

    # ax.scatter(raf_data["intensity"], mek_data["intensity"], alpha=.5)
    # ax.scatter(raf_data["imp_mean"], mek_data["intensity"], alpha=.5, color="orange")
    # ax.scatter(raf_data["intensity"], mek_data["imp_mean"], alpha=.5, color="orange")
    # ax.scatter(raf_data["imp_mean"], mek_data["imp_mean"], alpha=.5, color="red")

    # ax.hist(int1, bins=20, alpha=.5, label="Pre-training", density=True)
    # ax.hist(int2, bins=20, alpha=.5, label="pred", density=True)

    # plt.show()

if __name__ == "__main__":
    main()