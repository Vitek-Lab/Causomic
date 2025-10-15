"""
Bayesian causal inference models for proteomics data analysis.

This module contains PyroModule and NumPyro model implementations for
causal inference in mass spectrometry proteomics data. The models support
interventional analysis, missing data imputation, and counterfactual reasoning.

Author: Devon Kohler
"""

from typing import Any, Dict, List, Optional

import numpyro
import numpyro.distributions as numpyro_dist
import pyro
import pyro.distributions as pyro_dist
import pyro.poutine as poutine
import torch
from jax import numpy as jnp
from pyro.nn import PyroModule

# Configure Pyro
# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
device = torch.device("cpu")

class ProteomicPerturbationModel(PyroModule):
    """
    Pyro model for causal inference in proteomics perturbation experiments.

    This model represents a causal graph structure with root nodes
    and downstream nodes connected by linear relationships.
    Supports missing data imputation and interventional analysis.

    Parameters
    ----------
    n_obs : int
        Number of observations in the dataset
    root_nodes : List[str]
        List of root node names (variables with no parents)
    downstream_nodes : Dict[str, List[str]]
        Dictionary mapping downstream node names to their parent node names

    Attributes
    ----------
    n_obs : int
        Number of observations
    root_nodes : List[str]
        Root nodes in the causal graph
    downstream_nodes : Dict[str, List[str]]
        Downstream nodes and their parents
    """

    def __init__(self, n_obs: int, root_nodes: List[str], downstream_nodes: Dict[str, List[str]]):
        super().__init__()
        self.n_obs = n_obs
        self.root_nodes = root_nodes
        self.downstream_nodes = downstream_nodes

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        priors: Dict[str, Dict[str, float]],
        dpc_slope: float = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the proteomic perturbation model.

        Generates samples from the causal model, handling missing data imputation
        and interventional queries. Uses linear relationships between variables
        with Gaussian distributions for continuous variables and Bernoulli for binary outcomes.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary containing observed data tensors and missing data masks
        priors : Dict[str, Dict[str, float]]
            Nested dictionary specifying prior distributions for model parameters
        dpc_slope : float, default=1
            Slope parameter for dose-response curves (currently unused)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping variable names to their sampled values
        """

        # Define objects that will store the coefficients
        downstream_coef_dict_mean: Dict[str, torch.Tensor] = dict()
        downstream_coef_dict_scale: Dict[str, torch.Tensor] = dict()
        root_coef_dict_mean: Dict[str, torch.Tensor] = dict()
        root_coef_dict_scale: Dict[str, torch.Tensor] = dict()

        # Initial priors for coefficients
        for node_name, items in self.downstream_nodes.items():

            downstream_coef_dict_mean[f"{node_name}_int"] = pyro.sample(
                f"{node_name}_int",
                pyro_dist.Normal(
                    priors[node_name][f"{node_name}_int"],
                    priors[node_name][f"{node_name}_int_scale"],
                ),
            )

            for item in items:

                downstream_coef_dict_mean[f"{node_name}_{item}_coef"] = pyro.sample(
                    f"{node_name}_{item}_coef",
                    pyro_dist.Normal(
                        priors[node_name][f"{node_name}_{item}_coef"],
                        priors[node_name][f"{node_name}_{item}_coef_scale"],
                    ),
                )

            if node_name not in "Output":
                downstream_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                    f"{node_name}_scale", pyro_dist.Exponential(torch.tensor(1.0, device=device))
                )

        for node_name in self.root_nodes:
            root_coef_dict_mean[f"{node_name}_int"] = pyro.sample(
                f"{node_name}_int",
                pyro_dist.Normal(
                    priors[node_name][f"{node_name}_int"],
                    priors[node_name][f"{node_name}_int_scale"],
                ),
            )

            root_coef_dict_scale[f"{node_name}_scale"] = pyro.sample(
                f"{node_name}_scale", pyro_dist.Exponential(torch.tensor(1.0, device=device))
            )

        # Loop through the data
        downstream_distributions: Dict[str, torch.Tensor] = dict()

        with pyro.plate("observations", self.n_obs):

            # Start with root nodes (sampled from normal)
            for node_name in self.root_nodes:

                mean = root_coef_dict_mean[f"{node_name}_int"]
                scale = root_coef_dict_scale[f"{node_name}_scale"]

                x = pyro.sample(node_name, pyro_dist.Normal(mean, scale))
                obs_eps = pyro.sample(f"{node_name}_obs_eps", 
                                      pyro_dist.HalfCauchy(torch.tensor(0.1, device=device)))

                # If data passed in, condition on observed data
                if f"obs_{node_name}" in data:

                    mask = ~data[f"missing_{node_name}"].bool()

                    with poutine.mask(mask=mask):
                        pyro.sample(
                            f"obs_{node_name}",
                            pyro_dist.Normal(x, obs_eps),
                            obs=data[f"obs_{node_name}"],
                        )

                downstream_distributions[node_name] = x

            # Linear regression for each downstream node
            for node_name, items in self.downstream_nodes.items():

                # calculate mean as sum of upstream nodes and coefficients
                mean = downstream_coef_dict_mean[f"{node_name}_int"]
                for item in items:
                    coef = downstream_coef_dict_mean[f"{node_name}_{item}_coef"]
                    mean = mean + coef * downstream_distributions[item]

                # Define scale
                if "Output" not in node_name:
                    scale = 1  # downstream_coef_dict_scale[f"{node_name}_scale"]

                if f"obs_{node_name}" in data:

                    if "Output" in node_name:
                        # Binary outcome variable (e.g., toxicity)
                        pyro.sample(
                            f"obs_{node_name}",
                            pyro_dist.Bernoulli(logits=mean),
                            obs=data[f"obs_{node_name}"],
                        )
                    else:
                        if "Z" in node_name:
                            scale = torch.tensor(1.0)

                        # Impute missing values where needed
                        y = pyro.sample(f"{node_name}_real", pyro_dist.Normal(mean, scale))

                        mask = ~data[f"missing_{node_name}"].bool()
                        obs_eps = pyro.sample(f"{node_name}_obs_eps", 
                                              pyro_dist.HalfCauchy(torch.tensor(0.1, device=device)))

                        # Clamp only observed entries
                        with poutine.mask(mask=mask):
                            pyro.sample(
                                f"obs_{node_name}",
                                pyro_dist.Normal(y, obs_eps),
                                obs=data[f"obs_{node_name}"],
                            )

                else:
                    if "Output" in node_name:
                        y = pyro.sample(node_name, pyro_dist.Bernoulli(logits=mean))
                    else:
                        y = pyro.sample(node_name, pyro_dist.Normal(mean, scale))

                downstream_distributions[node_name] = y

        return downstream_distributions


def NumpyroProteomicPerturbationModel(
    data: Dict[str, jnp.ndarray],
    missing: Dict[str, jnp.ndarray],
    priors: Dict[str, Dict[str, float]],
    root_nodes: List[str],
    downstream_nodes: Dict[str, List[str]],
) -> Dict[str, jnp.ndarray]:
    """
    NumPyro version of the proteomic perturbation model.

    Implements the same causal model structure as ProteomicPerturbationModel
    but using NumPyro for MCMC sampling instead of Pyro's SVI. Handles missing
    data imputation and supports interventional queries.

    Parameters
    ----------
    data : Dict[str, jnp.ndarray]
        Dictionary containing observed data arrays
    missing : Dict[str, jnp.ndarray]
        Dictionary containing boolean masks for missing data
    priors : Dict[str, Dict[str, float]]
        Nested dictionary specifying prior distributions
    root_nodes : List[str]
        List of root node names in the causal graph
    downstream_nodes : Dict[str, List[str]]
        Dictionary mapping downstream nodes to their parents

    Returns
    -------
    Dict[str, jnp.ndarray]
        Dictionary mapping variable names to their latent values
    """

    # Sample parameters
    params: Dict[str, jnp.ndarray] = {}

    # Root node parameters
    for root_node in root_nodes:
        int_key = f"{root_node}_int"
        int_scale_key = f"{root_node}_int_scale"

        if root_node in priors:
            int_mean = priors[root_node].get(int_key, 0.0)
            int_scale = priors[root_node].get(int_scale_key, 1.0)
        else:
            int_mean = 0.0
            int_scale = 1.0

        params[int_key] = numpyro.sample(int_key, numpyro_dist.Normal(int_mean, int_scale))
        params[f"{root_node}_scale"] = numpyro.sample(
            f"{root_node}_scale", numpyro_dist.Exponential(1)
        )

    # Downstream node parameters
    for node, parents in downstream_nodes.items():
        int_key = f"{node}_int"
        int_scale_key = f"{node}_int_scale"

        if node in priors:
            int_mean = priors[node].get(int_key, 0.0)
            int_scale = priors[node].get(int_scale_key, 1.0)
        else:
            int_mean = 0.0
            int_scale = 1.0

        params[int_key] = numpyro.sample(int_key, numpyro_dist.Normal(int_mean, int_scale))
        params[f"{node}_scale"] = numpyro.sample(f"{node}_scale", numpyro_dist.Exponential(1))

        for parent in parents:
            coef_key = f"{node}_{parent}_coef"
            coef_scale_key = f"{node}_{parent}_coef_scale"

            if node in priors:
                coef_mean = priors[node].get(coef_key, 0.0)
                coef_scale = priors[node].get(coef_scale_key, 1.0)
            else:
                coef_mean = 0.0
                coef_scale = 1.0

            params[coef_key] = numpyro.sample(coef_key, numpyro_dist.Normal(coef_mean, coef_scale))

    # Sample latent variables and handle observations
    n_obs = len(data[list(data.keys())[0]])
    latent_vars: Dict[str, jnp.ndarray] = {}

    with numpyro.plate("observations", n_obs):
        # Root nodes
        for root_node in root_nodes:
            latent_sample = numpyro.sample(
                root_node,
                numpyro_dist.Normal(params[f"{root_node}_int"], params[f"{root_node}_scale"]),
            )
            latent_vars[root_node] = latent_sample

            # Handle missing data imputation
            obs_data = data.get(root_node, jnp.zeros(n_obs))
            missing_mask = missing.get(root_node, jnp.zeros(n_obs, dtype=bool))

            # Impute missing values
            imputed_values = numpyro.sample(
                f"imp_{root_node}",
                numpyro_dist.Normal(latent_sample, 0.1),
                obs=jnp.where(missing_mask, jnp.nan, obs_data),
            )

            # Observe non-missing values
            numpyro.sample(
                f"{root_node}_obs",
                numpyro_dist.Normal(latent_sample, 0.1),
                obs=jnp.where(missing_mask, imputed_values, obs_data),
            )

        # Downstream nodes
        for node, parents in downstream_nodes.items():
            mean = params[f"{node}_int"]
            for parent in parents:
                mean = mean + params[f"{node}_{parent}_coef"] * latent_vars[parent]

            latent_sample = numpyro.sample(node, numpyro_dist.Normal(mean, params[f"{node}_scale"]))
            latent_vars[node] = latent_sample

            # Handle missing data imputation
            obs_data = data.get(node, jnp.zeros(n_obs))
            missing_mask = missing.get(node, jnp.zeros(n_obs, dtype=bool))

            # Impute missing values
            imputed_values = numpyro.sample(
                f"imp_{node}",
                numpyro_dist.Normal(latent_sample, 0.1),
                obs=jnp.where(missing_mask, jnp.nan, obs_data),
            )

            # Observe non-missing values
            numpyro.sample(
                f"{node}_obs",
                numpyro_dist.Normal(latent_sample, 0.1),
                obs=jnp.where(missing_mask, imputed_values, obs_data),
            )

    return latent_vars
