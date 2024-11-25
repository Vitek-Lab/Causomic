
from MScausality.simulation.simulation import simulate_data
from MScausality.data_analysis.dataProcess import dataProcess
from MScausality.data_analysis.normalization import normalize
from MScausality.causal_model.models import ProteomicPerturbationModel, ProteomicPerturbationCATE
from MScausality.causal_model.models import NumpyroProteomicPerturbationModel
from MScausality.causal_model.utils import prep_data_for_model

import pandas as pd
import numpy as np
from operator import attrgetter

import torch
import pyro
from pyro import poutine
from pyro.infer.autoguide import AutoDelta, AutoMultivariateNormal, AutoGuideList, AutoNormal
from pyro.infer import SVI, Trace_ELBO

import numpyro
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive
from jax import random

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

import networkx as nx
import y0
from y0.dsl import Variable
from y0.algorithm.simplify_latent import simplify_latent_dag

numpyro.set_platform('cpu')
numpyro.set_host_device_count(4)

# TODO: give user the option to reset parameters or not (new models vs more training)
# pyro.clear_param_store()
# pyro.settings.set(module_local_params=True)

class LVM:
    def __init__(self, backend="numpyro", 
                 num_samples=1000, 
                 warmup_steps=1000, 
                 num_chains=4,
                 num_steps=2000, 
                 initial_lr=.01, gamma=.01,
                 patience=300, min_delta=5,
                 informative_priors=None):
        
        self.backend = backend
        self.num_samples = num_samples
        self.warmup_steps = warmup_steps
        self.num_chains = num_chains
        self.num_steps = num_steps
        self.initial_lr = initial_lr
        self.gamma = gamma
        self.patience = patience
        self.min_delta = min_delta
        self.informative_priors = informative_priors

    def __repr__(self):
        return f"Latent Variable Structural Causal Model"
    
    def __str__(self):
        return f"Latent Variable Structural Causal Model"
    
    def __len__(self):
        return len(self.obs_data)

    def parse_graph(self):

        """
        Parse graph into root nodes and descendent nodes.

        Returns
        -------
        root_nodes : list
            A list of root nodes in the causal graph.
        descendent_nodes : dict
            A dictionary containing the descendent nodes for each root node.
        """

        # Sort graph
        sorted_nodes = [i for i in self.causal_graph.topological_sort()]

        # Get ancestors of each node
        ancestors = {i: list(self.causal_graph.ancestors_inclusive(i)) \
                     for i in sorted_nodes}

        # Find starting nodes
        root_nodes = [i for i in sorted_nodes if len(ancestors[i]) == 1]
        descendent_nodes = [i for i in sorted_nodes if len(ancestors[i]) != 1]

        temp_descendent_nodes = dict()
        for i in descendent_nodes:
            temp_descendent_nodes[i] = [
                i[0] for i in list(self.causal_graph.directed.in_edges(i))
                ]
            
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
        descendent_nodes = {
            str(name): 
            [str(item) for item in nodes if item != name] for name, nodes in 
            descendent_nodes.items() if name in descendent_nodes
            }
        root_nodes = [str(i) for i in root_nodes]

        self.root_nodes = root_nodes
        self.descendent_nodes = descendent_nodes

    def parse_data(self):

        data = dict()
        missing = dict()
        for i in self.obs_data.columns:
            if self.backend == "numpyro":
                data[i] = np.array(self.obs_data[i].values)
                missing[i] = np.array(self.obs_data[i].isna().values)
            elif self.backend == "pyro":
                data[i] = torch.tensor(self.obs_data[i].values).float()
                missing[i] = torch.tensor(self.obs_data[i].isna().values).float()
        
        self.input_data = pd.DataFrame.from_dict(data)
        self.input_missing = pd.DataFrame.from_dict(missing)

    def parse_priors(self):
        
        priors = {}

        if self.informative_priors is None:
            inf_priors = list()
        else:
            inf_priors = list(self.informative_priors.keys())

        # TODO: determine correct scale for uninformative priors
        for col in self.root_nodes:
            if col in inf_priors:
                priors[col] = {
                    f"{col}_int": self.informative_priors[col]["int"],
                    f"{col}_int_scale": self.informative_priors[col]["scale"]}
            else:
                priors[col] = {f"{col}_int": 0,
                               f"{col}_int_scale": 5}

        for col, value in self.descendent_nodes.items():
            
            temp = {}
            if col in inf_priors:
                for v in value: 
                    temp[f"{col}_{v}_coef"
                         ] = self.informative_priors[col][f"{v}_coef"]
                    temp[f"{col}_{v}_scale"
                         ] = self.informative_priors[col][f"{v}_coef_scale"]
            else:
                for v in value:
                    temp[f"{col}_{v}_coef"] = 0
                    temp[f"{col}_{v}_coef_scale"] = 5
            
            temp[f"{col}_int"] = 0
            temp[f"{col}_int_scale"] = 5
            priors[col] = temp

        self.priors = priors

    # TODO: Fix this for AutoDelta
    def compile_pyro_parameters(self):
        
        """
        Takes the learned pyro parameters and compiles them into readable format.
        """

        params = [i for i in pyro.get_param_store().items()]
        params = dict(params)
        params = {key : value.detach() for key, value in params.items()}
        self.original_params = params

        loc_params = [i for i in params.keys() if ("imp" not in i)] #& \
                             #("locs" in i)]#("latent" not in i) & 
        coef_data = pd.DataFrame().from_dict(
            {key.replace("AutoDelta.", ""): params[key] \
             for key in loc_params}, 
             orient='index', columns=["mean"]).reset_index(names="parameter")

        coef_data["mean"] = [i.numpy() for  i in coef_data["mean"]]

    def compile_numpyro_parameters(self, prob=.9):

        """
        Custom function to compile summary statistics from numpyro model.

        :param model: trained mcmc model
        :param prob: the probability mass of samples within the HPDI interval.

        :return: dictionary of summary statistics.
        """

        sites = self.model._states[self.model._sample_field]

        if isinstance(sites, dict):
            state_sample_field = attrgetter(
                self.model._sample_field)(self.model._last_state)
            if isinstance(state_sample_field, dict):
                sites = {
                    k: v
                    for k, v in self.model._states[
                        self.model._sample_field].items()
                    if k in state_sample_field
                }

        for site_name in list(sites):
            if len(sites[site_name].shape) == 3:
                if sites[site_name].shape[2] == 0:
                    # remove key from dictionary
                    sites.pop(site_name)

        summary_stats = numpyro.diagnostics.summary(sites, 
                                                    prob, 
                                                    group_by_chain=True)

        sample_keys = list(self.model.get_samples().keys())
        samples = self.model.get_samples()
        learned_params = dict()

        for i in range(len(sample_keys)):
            if "scale" not in sample_keys[i] and "imp" not in sample_keys[i]:
                learned_params[
                    f"{sample_keys[i]}"] = samples[
                        sample_keys[i]].mean().item()
                learned_params[
                    f"{sample_keys[i]}_scale"] = self.model.get_samples()[
                        sample_keys[i]].std().item()
            elif "scale" in sample_keys[i]:
                learned_params[
                    f"{sample_keys[i]}"] = samples[sample_keys[i]].mean().item()
            else:
                learned_params[
                    f"{sample_keys[i]}"] = samples[sample_keys[i]].mean(axis=0)
                learned_params[
                    f"{sample_keys[i]}_scale"] = samples[
                        sample_keys[i]].std(axis=0)

        self.learned_params = learned_params
        self.summary_stats = summary_stats

    def train_numpyro(self, verbose=True):
        
        condition_data = dict()
        condition_missing = dict()
        
        for node in self.root_nodes:
            condition_data[f"{node}"] = np.array(
                np.nan_to_num(self.input_data.loc[:, node].values))
            condition_missing[f"{node}"] = np.array(
                self.input_missing.loc[:, node].values)
            
        for node in self.descendent_nodes:
            condition_data[f"{node}"] = np.array(
                np.nan_to_num(self.input_data.loc[:, node].values))
            condition_missing[f"{node}"] = np.array(
                self.input_missing.loc[:, node].values)
    
        model = MCMC(NUTS(NumpyroProteomicPerturbationModel), 
                    num_warmup=self.warmup_steps, 
                    num_samples=self.num_samples, 
                    num_chains=self.num_chains)
        model.run(random.PRNGKey(0), 
                 condition_data, 
                 condition_missing,
                 self.priors,
                 self.root_nodes, 
                 self.descendent_nodes)
        self.model = model


    def train_pyro(self, verbose=True):
        
        pyro.set_rng_seed(1234)

        model = ProteomicPerturbationModel(n_obs = len(self.input_data), 
                                       root_nodes = self.root_nodes, 
                                       downstream_nodes = self.descendent_nodes)
        # dpc_slope = calc_dpc(self.input_data)
        # self.dpc_slope = dpc_slope
        condition_data = prep_data_for_model(self.root_nodes, 
                                             self.descendent_nodes,
                                             self.input_data,
                                             self.input_missing)
        

        # set up the optimizer
        lrd = self.gamma ** (1 / self.num_steps)
        optim = pyro.optim.ClippedAdam({'lr': self.initial_lr, 
                                        'lrd': lrd})

        guide = AutoDelta(model)

        # setup the inference algorithm
        svi = SVI(model, guide, optim, loss=Trace_ELBO())

        # do gradient steps
        best_loss = float('inf')
        steps_since_improvement = 0

        for step in range(self.num_steps):
            loss = svi.step(condition_data, self.priors)
            if step % 100 == 0 & verbose:
                print(f"Step {step}: Loss = {loss}")

            # Check for improvement
            if loss < best_loss - self.min_delta:
                best_loss = loss
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            # Early stopping condition
            if steps_since_improvement >= self.patience:
                print(f"Stopping early at step {step} with loss {loss}")
                break

        self.model = model
        self.guide = guide

    def add_imputed_values(self):
        """
        Adds imputed values back into data with mean and scale.
        """
        
        # Put data into long format
        long_data = pd.melt(self.input_data, var_name="protein", 
                            value_name="intensity")
        long_data.loc[long_data["intensity"] == 0, "intensity"] = np.nan
        long_data['was_missing'] = long_data['intensity'].isna()
        long_data.loc[:, "imp_mean"] = np.nan
        
        if self.backend == "pyro":

            # TODO: (?) Put this into a function to stop code repeat
            loc_params = [i for i in self.original_params.keys() if ("imp" in i)]
            loc_imp = pd.DataFrame().from_dict(
                {key.replace("AutoDelta.imp_", ""): self.original_params[key] \
                for key in loc_params})
            loc_imp = loc_imp[self.input_missing.astype(bool)]
            loc_imp = pd.melt(loc_imp, var_name="protein", 
                                value_name="imp_loc")
            loc_imp = loc_imp.dropna(ignore_index=True)

            for col in loc_imp["protein"].unique():

                long_data.loc[(long_data["protein"] == col) & \
                            (long_data["intensity"].isna()), "imp_mean"] = \
                                loc_imp.loc[loc_imp["protein"] == col, 
                                            "imp_loc"].values
                
            self.imputed_data = long_data
        
        elif self.backend == "numpyro":

            # Extract imputation info from model parameters
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

    def fit(self, 
            observational_data, 
            causal_graph, 
            verbose=True):
        
        self.obs_data = observational_data
        self.causal_graph = causal_graph
        
        # Prepare information for model
        self.parse_graph()
        self.parse_data()
        self.parse_priors()

        # Train model and extract results
        if self.backend == "numpyro":
            self.train_numpyro(verbose=verbose)
        elif self.backend == "pyro":
            self.train_pyro(verbose=verbose)

        if self.backend == "numpyro":
            self.compile_numpyro_parameters()
        elif self.backend == "pyro":
            self.compile_pyro_parameters()
        self.add_imputed_values()

    def intervention(self, intervention, outcome_node, compare_value=0.):
        
        # Prep interventional conditioning data
        if self.backend == "pyro":
            condition_data_test = dict()
            for node in self.root_nodes:
                if "latent" not in node:
                    condition_data_test[f"missing_{node}"] = torch.tensor(
                        [0 for _ in range(len(self.input_data))])

            for node in self.descendent_nodes:
                condition_data_test[f"missing_{node}"] = torch.tensor(
                    [0 for _ in range(len(self.input_data))])

            # Posterior samples
            ate_model = ProteomicPerturbationCATE(
                ProteomicPerturbationModel(n_obs = len(self.input_data),
                                        root_nodes = self.root_nodes, 
                                        downstream_nodes = self.descendent_nodes
                                        ))
            
            ate_predictive = pyro.infer.Predictive(
                ate_model, guide=self.guide, num_samples=500)
            zero_int = dict()
            for key, value in intervention.items():
                zero_int[key] = torch.tensor(compare_value).float()
                intervention[key] = torch.tensor(value).float()

            zero_int = ate_predictive(zero_int, 
                                    condition_data_test,
                                    self.priors)
            intervention = ate_predictive(intervention, 
                                        condition_data_test,
                                        self.priors)

            self.posterior_samples = zero_int[outcome_node].flatten()
            self.intervention_samples = intervention[outcome_node].flatten()
        
        elif self.backend == "numpyro":
            rng_key, rng_key_ = random.split(random.PRNGKey(2))

            zero_model = numpyro.handlers.do(
                NumpyroProteomicPerturbationModel, 
                data={next(iter(intervention)): compare_value})
            zero_predictive = Predictive(zero_model, self.model.get_samples())
            zero_predictions = zero_predictive(rng_key_, None, [],
                                               self.priors,
                                               self.root_nodes,
                                               self.descendent_nodes)

            int_model = numpyro.handlers.do(
                NumpyroProteomicPerturbationModel, 
                data=intervention)
            int_predictive = Predictive(int_model, self.model.get_samples())
            int_predictions = int_predictive(rng_key_, None, [],
                                             self.priors,
                                             self.root_nodes,
                                             self.descendent_nodes)
            
            zero_predictions = zero_predictions[outcome_node]
            int_predictions = int_predictions[outcome_node]
            
            self.posterior_samples = zero_predictions
            self.intervention_samples = int_predictions

def main():

    from MScausality.simulation.example_graphs import mediator

    med = mediator(add_independent_nodes=False, n_ind=50)
    simulated_med_data = simulate_data(med['Networkx'], 
                                    coefficients=med['Coefficients'], 
                                    mnar_missing_param=[-3, .4],
                                    add_feature_var=True, n=50, seed=2)
    med_data = dataProcess(simulated_med_data["Feature_data"], normalization=False, 
                summarization_method="TMP", MBimpute=False, sim_data=True)
    # med_data = med_data.dropna(how="any",axis=0).reset_index(drop=True)


    transformed_data = normalize(med_data, wide_format=True)
    input_data = transformed_data["df"]
    scale_metrics = transformed_data["adj_metrics"]

    lvm = LVM(backend="numpyro")
    lvm.fit(input_data, med["MScausality"])
    lvm.intervention({"X": (0 - scale_metrics["mean"]) / scale_metrics["std"]}, "Z")
    print("finished")

    # imp_data = lvm.imputed_data
    # X_data = imp_data.loc[imp_data["protein"] == "X"]
    # Y_data = imp_data.loc[imp_data["protein"] == "M1"]
    # Z_data = imp_data.loc[imp_data["protein"] == "Z"]

    # X_backdoor_color = np.where(
    #     (X_data['imp_mean'].isna().values & Y_data['imp_mean'].isna().values), 
    #     "blue", 
    #     np.where((X_data['intensity'].isna().values & Y_data['intensity'].isna().values), 
    #             "red", "orange"))

    # Y_backdoor_color = np.where(
    #     (Y_data['imp_mean'].isna().values & Z_data['imp_mean'].isna().values), 
    #     "blue", 
    #     np.where((Y_data['intensity'].isna().values & Z_data['intensity'].isna().values), 
    #             "red", "orange"))

    # import matplotlib.pyplot as plt

    # X_data = np.where(
    #     X_data['imp_mean'].isna(),
    #     X_data['intensity'], 
    #     X_data['imp_mean'])

    # Y_data = np.where(
    #     Y_data['imp_mean'].isna(),
    #     Y_data['intensity'], 
    #     Y_data['imp_mean'])

    # Z_data = np.where(
    #     Z_data['imp_mean'].isna(),
    #     Z_data['intensity'], 
    #     Z_data['imp_mean'])


    # fig, ax = plt.subplots(2,2, figsize=(10,5))

    # ax[0,0].scatter(input_data.loc[:, "X"], input_data.loc[:, "M1"])
    # ax[0,1].scatter(X_data, Y_data, color=X_backdoor_color)

    # ax[1,0].scatter(input_data.loc[:, "M1"], input_data.loc[:, "Z"])
    # ax[1,1].scatter(Y_data, Z_data, color=Y_backdoor_color)
    # plt.show()

if __name__ == "__main__":
    main()