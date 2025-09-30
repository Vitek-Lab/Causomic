"""
Latent Variable Model (LVM) for Causal Inference

This module implements a Latent Variable Structural Causal Model for causal inference
in proteomics data. It supports both Pyro and NumPyro probabilistic programming
backends and provides functionality for handling observational data, causal graphs,
missing data imputation, and interventional analysis.

The LVM class is designed to:
- Handle mixed observational and interventional data
- Support informative priors for domain knowledge integration
- Perform missing data imputation using model-based approaches
- Enable counterfactual and interventional queries
- Provide uncertainty quantification through Bayesian inference

Author: Devon Kohler
"""

from dataclasses import dataclass

# Standard library imports
from operator import attrgetter
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

# Third-party imports
import networkx as nx
import numpy as np

# Probabilistic programming - NumPyro
import numpyro
import pandas as pd

# Probabilistic programming - Pyro
import pyro
import torch
from jax import random
from numpyro.infer import MCMC, NUTS
from numpyro.infer import Predictive as NumpyroPredicitve
from pyro import poutine
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta

# CausOmic package imports
from causomic.causal_model.models import (
    NumpyroProteomicPerturbationModel,
    ProteomicPerturbationModel,
)
from causomic.data_analysis.proteomics_data_processor import dataProcess
from causomic.simulation.proteomics_simulator import simulate_data

# Configure NumPyro
numpyro.set_platform("cpu")
numpyro.set_host_device_count(4)


@dataclass
class ScaleStats:
    mean: pd.Series
    scale: pd.Series
    eps: float = 1e-6

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return (df - self.mean) / self.scale.clip(lower=self.eps)

    def inverse(self, df: pd.DataFrame) -> pd.DataFrame:
        return df * self.scale.clip(lower=self.eps) + self.mean


class LVM:
    """
    Latent Variable Structural Causal Model for Proteomics Data.

    This class implements a Bayesian latent variable model for causal inference
    in proteomics data. It supports both Pyro and NumPyro backends for
    probabilistic programming and provides methods for model fitting, missing
    data imputation, and interventional analysis.

    The model handles:
    - Mixed observational and interventional data
    - Missing data through model-based imputation
    - Latent confounders in causal graphs
    - Informative priors for domain knowledge integration
    - Uncertainty quantification through Bayesian inference

    Parameters
    ----------
    backend : {"pyro", "numpyro"}, default="numpyro"
        Probabilistic programming backend to use for inference
    num_samples : int, default=1000
        Number of posterior samples to draw during inference
    warmup_steps : int, default=1000
        Number of warmup steps for MCMC sampling (NumPyro only)
    num_chains : int, default=4
        Number of MCMC chains to run in parallel (NumPyro only)
    num_steps : int, default=10000
        Number of optimization steps for SVI (Pyro only)
    initial_lr : float, default=0.01
        Initial learning rate for optimizer (Pyro only)
    gamma : float, default=0.01
        Learning rate decay factor (Pyro only)
    patience : int, default=200
        Early stopping patience for SVI convergence (Pyro only)
    min_delta : float, default=5
        Minimum loss improvement to reset patience counter (Pyro only)
    informative_priors : dict, optional
        Dictionary specifying informative priors for model parameters
    verbose : bool, default=False
        Whether to print detailed progress information during model fitting

    Attributes
    ----------
    obs_data : pd.DataFrame
        Observational data used for model fitting
    causal_graph : networkx.DiGraph or y0.graph.NxMixedGraph
        Causal graph representing relationships between variables
    root_nodes : List[str]
        List of root nodes (no parents) in the causal graph
    descendant_nodes : Dict[str, List[str]]
        Dictionary mapping each node to its parent nodes
    end_nodes : List[str]
        List of leaf nodes (no children) in the causal graph
    input_data : pd.DataFrame
        Processed input data for the model (backend-specific format)
    input_missing : pd.DataFrame
        Boolean mask indicating missing values in input data
    priors : dict
        Dictionary of prior distributions for model parameters
    model : object
        Trained model object (backend-specific)
    guide : object
        Variational guide object (Pyro only)
    learned_params : dict
        Dictionary of learned model parameters (NumPyro)
    summary_stats : dict
        Summary statistics of posterior samples (NumPyro)
    original_params : dict
        Dictionary of learned model parameters (Pyro)
    imputed_data : pd.DataFrame
        Data with missing values imputed using model predictions
    posterior_samples : np.ndarray or torch.Tensor
        Posterior samples for outcome under baseline condition
    intervention_samples : np.ndarray or torch.Tensor
        Posterior samples for outcome under specified intervention

    Examples
    --------
    Basic usage with NumPyro backend:

    >>> # Initialize model
    >>> lvm = LVM(backend="numpyro", num_samples=500)
    >>>
    >>> # Fit model to data and causal graph
    >>> lvm.fit(observational_data, causal_graph)
    >>>
    >>> # Perform intervention analysis
    >>> lvm.intervention({"Treatment": 1.0}, outcome_node="Outcome")
    >>>
    >>> # Access results
    >>> baseline_effect = lvm.posterior_samples.mean()
    >>> intervention_effect = lvm.intervention_samples.mean()
    >>> causal_effect = intervention_effect - baseline_effect

    With informative priors:

    >>> # Define informative priors
    >>> priors = {
    ...     "Outcome": {
    ...         "Outcome_Treatment_coef": 0.5,
    ...         "Outcome_Treatment_coef_scale": 0.1,
    ...         "Outcome_int": 0.0,
    ...         "Outcome_int_scale": 0.5
    ...     }
    ... }
    >>>
    >>> # Initialize with priors
    >>> lvm = LVM(backend="numpyro", informative_priors=priors)
    >>> lvm.fit(data, graph)

    Using Pyro backend with early stopping:

    >>> lvm = LVM(
    ...     backend="pyro",
    ...     num_steps=5000,
    ...     patience=100,
    ...     min_delta=1.0
    ... )
    >>> lvm.fit(data, graph)
    """

    def __init__(
        self,
        backend: Literal["pyro", "numpyro"] = "numpyro",
        num_samples: int = 1000,
        warmup_steps: int = 1000,
        num_chains: int = 4,
        num_steps: int = 10000,
        initial_lr: float = 0.01,
        gamma: float = 0.01,
        patience: int = 200,
        min_delta: float = 5,
        informative_priors: Optional[Dict[str, Dict[str, float]]] = None,
        verbose: bool = False,
    ):

        # Validate backend
        if backend not in ["pyro", "numpyro"]:
            raise ValueError("Backend must be either 'pyro' or 'numpyro'")

        # Store configuration parameters
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
        self.verbose = verbose

        # Initialize attributes that will be set during fitting
        self.obs_data: Optional[pd.DataFrame] = None
        self.causal_graph: Optional[Any] = None
        self.root_nodes: Optional[List[str]] = None
        self.descendant_nodes: Optional[Dict[str, List[str]]] = None
        self.end_nodes: Optional[List[str]] = None
        self.input_data: Optional[pd.DataFrame] = None
        self.input_missing: Optional[pd.DataFrame] = None
        self.priors: Optional[Dict] = None
        self.model: Optional[Any] = None
        self.guide: Optional[Any] = None
        self.learned_params: Optional[Dict] = None
        self.summary_stats: Optional[Dict] = None
        self.original_params: Optional[Dict] = None
        self.imputed_data: Optional[pd.DataFrame] = None
        self.posterior_samples: Optional[Union[np.ndarray, torch.Tensor]] = None
        self.intervention_samples: Optional[Union[np.ndarray, torch.Tensor]] = None

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

    def __repr__(self) -> str:
        """Return string representation of the LVM model."""
        return f"LVM(backend='{self.backend}', fitted={self.model is not None})"

    def __str__(self) -> str:
        """Return detailed string representation of the LVM model."""
        status = "fitted" if self.model is not None else "not fitted"
        n_obs = len(self.obs_data) if self.obs_data is not None else "unknown"
        return (
            f"Latent Variable Structural Causal Model\n"
            f"  Backend: {self.backend}\n"
            f"  Status: {status}\n"
            f"  Observations: {n_obs}"
        )

    def __len__(self) -> int:
        """Return the number of observations in the dataset."""
        if self.obs_data is None:
            raise ValueError("Model has not been fitted yet")
        return len(self.obs_data)

    def fit_scaler(self, df_obs: pd.DataFrame) -> None:

        mean = df_obs.mean(skipna=True)
        scale = df_obs.std(skipna=True)

        # don’t scale the Output column
        if "Output" in df_obs.columns:
            mean.loc["Output"] = 0.0
            scale.loc["Output"] = 1.0

        self.scaler = ScaleStats(mean=mean, scale=scale)

    def _to_z(self, df_nat: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler first.")
        return self.scaler.transform(df_nat)

    def _from_z(self, df_z: pd.DataFrame) -> pd.DataFrame:
        if self.scaler is None:
            raise RuntimeError("Call fit_scaler first.")
        return self.scaler.inverse(df_z)

    def parse_graph(self) -> None:
        """
        Parse causal graph into root nodes, descendant nodes, and latent confounders.

        This method processes the causal graph to identify:
        - Root nodes: variables with no parents
        - Descendant nodes: variables with parents (mapped to their parents)
        - Latent confounders: unobserved variables affecting multiple nodes

        The method handles both directed edges (causal relationships) and
        undirected edges (latent confounders) in mixed graphs.

        Sets
        ----
        self.root_nodes : List[str]
            List of root nodes in the causal graph
        self.descendant_nodes : Dict[str, List[str]]
            Dictionary mapping each descendant node to its parent nodes
        self.end_nodes : List[str]
            List of leaf nodes (no children) in the causal graph

        Raises
        ------
        AttributeError
            If causal_graph has not been set
        """
        if self.causal_graph is None:
            raise AttributeError("Causal graph must be set before parsing")

        # Get topologically sorted nodes
        sorted_nodes = [node for node in self.causal_graph.topological_sort()]

        # Get ancestors and descendants for each node
        ancestors = {
            node: [anc for anc in self.causal_graph.ancestors_inclusive(node)]
            for node in sorted_nodes
        }

        descendants = {
            node: [desc for desc in self.causal_graph.descendants_inclusive(node)]
            for node in sorted_nodes
        }

        # Identify root nodes (only ancestor is themselves)
        root_nodes = [node for node in sorted_nodes if len(ancestors[node]) == 1]

        # Identify descendant nodes (have parents)
        descendant_nodes = [node for node in sorted_nodes if len(ancestors[node]) != 1]

        # Identify leaf nodes (no descendants other than themselves)
        leaf_nodes = [node for node in sorted_nodes if len(descendants[node]) == 1]

        # Map descendant nodes to their immediate parents
        descendant_node_mapping = {}
        for node in descendant_nodes:
            parents = [str(parent) for parent, _ in self.causal_graph.directed.in_edges(node)]
            descendant_node_mapping[str(node)] = parents

        # Handle latent confounders (undirected edges)
        latent_edges = list(self.causal_graph.undirected.edges())
        latent_nodes = [f"latent_{i}" for i in range(len(latent_edges))]

        # Add latent confounders to the graph structure
        for i, latent_node in enumerate(latent_nodes):
            root_nodes.append(latent_node)

            # Each latent confounder affects the nodes in its edge
            for affected_node in latent_edges[i]:
                affected_node_str = str(affected_node)

                if affected_node_str in root_nodes:
                    # Convert root node to descendant node
                    root_nodes.remove(affected_node_str)
                    descendant_node_mapping[affected_node_str] = [latent_node]
                else:
                    # Add latent confounder as additional parent
                    if affected_node_str not in descendant_node_mapping:
                        descendant_node_mapping[affected_node_str] = []
                    descendant_node_mapping[affected_node_str].append(latent_node)

        # Finalize output
        descendant_node_mapping = {
            str(name): [str(item) for item in nodes if item != name]
            for name, nodes in descendant_node_mapping.items()
            if name in descendant_node_mapping
        }
        root_nodes = [str(i) for i in root_nodes]
        leaf_nodes = [str(i) for i in leaf_nodes]

        # Store results
        self.root_nodes = root_nodes
        self.descendant_nodes = descendant_node_mapping
        self.end_nodes = leaf_nodes

    def parse_data(self) -> None:
        """
        Process observational data for the selected backend.

        Converts pandas DataFrame to backend-specific tensor formats and creates
        missing value masks. NumPyro uses numpy arrays while Pyro uses torch tensors.

        Sets
        ----
        self.input_data : pd.DataFrame
            Data converted to backend-specific format
        self.input_missing : pd.DataFrame
            Boolean mask indicating missing values

        Raises
        ------
        ValueError
            If backend is not supported
        AttributeError
            If obs_data has not been set
        """
        if self.obs_data is None:
            raise AttributeError("Observational data must be set before parsing")

        data_dict = {}
        missing_dict = {}

        for column in self.obs_data.columns:
            if self.backend == "numpyro":
                # NumPyro uses numpy arrays
                data_dict[column] = np.array(self.obs_data[column].values, dtype=np.float32)
                missing_dict[column] = np.array(self.obs_data[column].isna().values, dtype=bool)

            elif self.backend == "pyro":
                # Pyro uses torch tensors
                data_dict[column] = torch.tensor(self.obs_data[column].values, dtype=torch.float32)
                missing_dict[column] = torch.tensor(
                    self.obs_data[column].isna().values, dtype=torch.float32
                )
            else:
                raise ValueError(f"Unsupported backend: {self.backend}")

        self.input_data = pd.DataFrame.from_dict(data_dict)
        self.input_missing = pd.DataFrame.from_dict(missing_dict)

    def parse_priors(self) -> None:
        """
        Construct prior distributions for model parameters.

        Creates prior specifications for:
        - Root node intercepts and scales
        - Causal coefficients between nodes
        - Intercepts for descendant nodes

        Uses informative priors if provided, otherwise defaults to
        weakly informative priors (mean=0, scale=1).

        Sets
        ----
        self.priors : Dict[str, Dict[str, float]]
            Dictionary of prior parameters organized by node

        Raises
        ------
        AttributeError
            If root_nodes or descendant_nodes have not been set
        """
        if self.root_nodes is None or self.descendant_nodes is None:
            raise AttributeError("Graph must be parsed before setting priors")

        priors = {}

        # Get list of nodes with informative priors
        informative_nodes = (
            list(self.informative_priors.keys()) if self.informative_priors is not None else []
        )

        # Set priors for root nodes (intercept and scale only)
        for node in self.root_nodes:
            if node in informative_nodes:
                priors[node] = {
                    f"{node}_int": self.informative_priors[node][f"{node}_int"],
                    f"{node}_int_scale": self.informative_priors[node][f"{node}_int_scale"],
                }
            else:
                # Default weakly informative priors
                priors[node] = {f"{node}_int": 0.0, f"{node}_int_scale": 1.0}

        # Set priors for descendant nodes (coefficients and intercepts)
        for node, parent_nodes in self.descendant_nodes.items():
            node_priors = {}

            if node in informative_nodes:
                # Use informative priors if available
                for parent in parent_nodes:
                    coef_key = f"{node}_{parent}_coef"
                    scale_key = f"{node}_{parent}_coef_scale"

                    node_priors[coef_key] = self.informative_priors[node][coef_key]
                    node_priors[scale_key] = self.informative_priors[node][scale_key]

                # Intercept priors
                node_priors[f"{node}_int"] = self.informative_priors[node][f"{node}_int"]
                node_priors[f"{node}_int_scale"] = self.informative_priors[node][
                    f"{node}_int_scale"
                ]

            else:
                # Use default weakly informative priors
                for parent in parent_nodes:
                    node_priors[f"{node}_{parent}_coef"] = 0.0
                    node_priors[f"{node}_{parent}_coef_scale"] = 1.0

                node_priors[f"{node}_int"] = 0.0
                node_priors[f"{node}_int_scale"] = 1.0

            priors[node] = node_priors

        self.priors = priors

    def compile_pyro_parameters(self) -> None:
        """
        Extract and compile learned parameters from Pyro's parameter store.

        Processes the learned parameters from Pyro's automatic differentiation
        parameter store and formats them for analysis. Filters out imputation
        parameters and focuses on structural model parameters.

        Sets
        ----
        self.original_params : Dict[str, torch.Tensor]
            Dictionary of learned parameters with detached gradients

        Note
        ----
        This method currently has a TODO to complete the parameter compilation
        for coefficient analysis. The parameter extraction logic may need
        refinement based on the specific AutoGuide used.
        """
        # Extract parameters from Pyro's parameter store
        param_items = list(pyro.get_param_store().items())
        params = dict(param_items)

        # Detach gradients for analysis
        params = {key: value.detach() for key, value in params.items()}
        self.original_params = params

        # Filter location parameters (excluding imputation parameters)
        loc_params = [
            key for key in params.keys() if "imp" not in key  # Exclude imputation parameters
        ]

        # Create coefficient dataframe
        coef_data = pd.DataFrame.from_dict(
            {key.replace("AutoDelta.", ""): params[key] for key in loc_params},
            orient="index",
            columns=["mean"],
        ).reset_index(names="parameter")

        # Convert tensors to numpy arrays for analysis
        coef_data["mean"] = [tensor.numpy() for tensor in coef_data["mean"]]

        # TODO: Complete parameter compilation and analysis
        # Additional processing could include:
        # - Uncertainty quantification from guide parameters
        # - Coefficient significance testing
        # - Parameter interpretation and reporting

    def compile_numpyro_parameters(self, prob: float = 0.9) -> None:
        """
        Extract summary statistics and learned parameters from NumPyro MCMC model.

        Compiles posterior samples into summary statistics and point estimates
        for model parameters. Calculates mean and standard deviation for each
        parameter, excluding empty parameter arrays.

        Parameters
        ----------
        prob : float, default=0.9
            Probability mass for highest posterior density intervals (HPDI)

        Sets
        ----
        self.learned_params : Dict[str, float]
            Dictionary of parameter means and standard deviations
        self.summary_stats : Dict
            NumPyro summary statistics including HPDI intervals

        Raises
        ------
        AttributeError
            If model has not been fitted yet
        """
        if self.model is None:
            raise AttributeError("Model must be fitted before compiling parameters")

        # Extract posterior samples from MCMC chains
        sites = self.model._states[self.model._sample_field]

        # Handle different sample field formats
        if isinstance(sites, dict):
            state_sample_field = attrgetter(self.model._sample_field)(self.model._last_state)
            if isinstance(state_sample_field, dict):
                sites = {
                    k: v
                    for k, v in self.model._states[self.model._sample_field].items()
                    if k in state_sample_field
                }

        # Remove empty parameter arrays (shape with 0 dimension)
        sites_to_remove = []
        for site_name, site_values in sites.items():
            if len(site_values.shape) == 3 and site_values.shape[2] == 0:
                sites_to_remove.append(site_name)

        for site_name in sites_to_remove:
            sites.pop(site_name)

        # Generate summary statistics with HPDI
        summary_stats = numpyro.diagnostics.summary(sites, prob=prob, group_by_chain=True)

        # Extract posterior samples and compute parameter estimates
        sample_keys = list(self.model.get_samples().keys())
        samples = self.model.get_samples()
        learned_params = {}

        for param_name in sample_keys:
            param_samples = samples[param_name]

            if "scale" not in param_name and "imp" not in param_name:
                # Regular parameters: store mean and compute scale
                learned_params[param_name] = param_samples.mean().item()
                learned_params[f"{param_name}_scale"] = param_samples.std().item()

            elif "scale" in param_name:
                # Scale parameters: store mean only
                learned_params[param_name] = param_samples.mean().item()

            else:
                # Imputation parameters: store mean and scale across observations
                learned_params[param_name] = param_samples.mean(axis=0)
                learned_params[f"{param_name}_scale"] = param_samples.std(axis=0)

        self.learned_params = learned_params
        self.summary_stats = summary_stats

    def train_numpyro(self, verbose: Optional[bool] = None) -> None:
        """
        Train the model using NumPyro's MCMC inference.

        Sets up MCMC sampling using the No-U-Turn Sampler (NUTS) algorithm
        to draw posterior samples from the latent variable model. Handles
        missing data by converting NaN values to zeros and tracking missingness.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print training progress information.
            If None, uses the instance's verbose setting.

        Sets
        ----
        self.model : numpyro.infer.MCMC
            Fitted MCMC model with posterior samples

        Raises
        ------
        AttributeError
            If required attributes (input_data, priors, etc.) are not set
        """
        if any(
            attr is None
            for attr in [
                self.input_data,
                self.input_missing,
                self.priors,
                self.root_nodes,
                self.descendant_nodes,
            ]
        ):
            raise AttributeError("Data and graph must be parsed before training")

        # Use instance verbose setting if not provided
        if verbose is None:
            verbose = self.verbose

        # Prepare conditioning data for NumPyro model
        condition_data = {}
        condition_missing = {}

        # Process root nodes
        for node in self.root_nodes:
            condition_data[node] = np.array(
                np.nan_to_num(self.input_data.loc[:, node].values, nan=0.0), dtype=np.float32
            )
            condition_missing[node] = np.array(self.input_missing.loc[:, node].values, dtype=bool)

        # Process descendant nodes
        for node in self.descendant_nodes:
            condition_data[node] = np.array(
                np.nan_to_num(self.input_data.loc[:, node].values, nan=0.0), dtype=np.float32
            )
            condition_missing[node] = np.array(self.input_missing.loc[:, node].values, dtype=bool)

        if verbose:
            print(
                f"Starting MCMC inference with {self.num_chains} chains, "
                f"{self.num_samples} samples, {self.warmup_steps} warmup steps"
            )

        # Set up MCMC with NUTS sampler
        mcmc = MCMC(
            NUTS(NumpyroProteomicPerturbationModel),
            num_warmup=self.warmup_steps,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
        )

        # Run MCMC inference
        mcmc.run(
            random.PRNGKey(0),
            condition_data,
            condition_missing,
            self.priors,
            self.root_nodes,
            self.descendant_nodes,
        )

        self.model = mcmc

        if verbose:
            print("MCMC inference completed successfully")

    def train_pyro(self, verbose: Optional[bool] = None) -> None:
        """
        Train the model using Pyro's Stochastic Variational Inference (SVI).

        Uses automatic differentiation variational inference with early stopping
        to fit the latent variable model. Employs AutoDelta guide for point
        estimates and ClippedAdam optimizer with learning rate decay.

        Parameters
        ----------
        verbose : bool, optional
            Whether to print training progress every 100 steps.
            If None, uses the instance's verbose setting.

        Sets
        ----
        self.model : ProteomicPerturbationModel
            The fitted Pyro model
        self.guide : AutoDelta
            The fitted variational guide

        Raises
        ------
        AttributeError
            If required attributes are not set before training
        """
        if any(
            attr is None
            for attr in [
                self.input_data,
                self.input_missing,
                self.priors,
                self.root_nodes,
                self.descendant_nodes,
            ]
        ):
            raise AttributeError("Data and graph must be parsed before training")

        # Use instance verbose setting if not provided
        if verbose is None:
            verbose = self.verbose

        # Set random seed for reproducibility
        pyro.set_rng_seed(1234)

        # Initialize the model
        model = ProteomicPerturbationModel(
            n_obs=len(self.input_data),
            root_nodes=self.root_nodes,
            downstream_nodes=self.descendant_nodes,
        )

        # Prepare conditioning data for Pyro model
        condition_data = {}

        # Process root nodes (excluding latent nodes)
        for node in self.root_nodes:
            if "latent" not in node:
                condition_data[f"obs_{node}"] = torch.tensor(
                    np.nan_to_num(self.input_data.loc[:, node].values)
                )
                condition_data[f"missing_{node}"] = torch.tensor(
                    self.input_missing.loc[:, node].values
                )

        # Process descendant nodes
        for node in self.descendant_nodes:
            condition_data[f"obs_{node}"] = torch.tensor(
                np.nan_to_num(self.input_data.loc[:, node].values)
            )
            condition_data[f"missing_{node}"] = torch.tensor(self.input_missing.loc[:, node].values)

        # Set up optimizer with learning rate decay
        learning_rate_decay = self.gamma ** (1 / self.num_steps)
        optimizer = pyro.optim.ClippedAdam({"lr": self.initial_lr, "lrd": learning_rate_decay})

        # Use AutoDelta guide for point estimates
        guide = AutoDelta(model)

        # Set up SVI inference
        svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

        if verbose:
            print(f"Starting SVI training for {self.num_steps} steps with early stopping")
            print(f"Patience: {self.patience}, Min delta: {self.min_delta}")

        # Training loop with early stopping
        best_loss = float("inf")
        steps_since_improvement = 0

        for step in range(self.num_steps):
            loss = svi.step(condition_data, self.priors)

            if verbose and step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}")

            # Check for improvement
            if loss < best_loss - self.min_delta:
                best_loss = loss
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            # Early stopping condition
            if steps_since_improvement >= self.patience:
                if verbose:
                    print(f"Early stopping at step {step} with loss {loss:.4f}")
                break

        if verbose and steps_since_improvement < self.patience:
            print(f"Training completed at step {self.num_steps} with loss {loss:.4f}")

        self.model = model
        self.guide = guide

    def add_imputed_values(self) -> None:
        """
        Add model-based imputed values back into the dataset.

        Extracts imputed values from the fitted model and merges them with
        the original data, creating a long-format dataset that indicates
        which values were originally missing and their imputed replacements.

        Sets
        ----
        self.imputed_data : pd.DataFrame
            Long-format dataframe with columns:
            - protein: Variable name
            - intensity: Original or imputed value
            - was_missing: Boolean indicating if value was originally missing
            - imp_mean: Imputed mean value (NaN for non-missing)

        Raises
        ------
        AttributeError
            If model has not been fitted yet
        ValueError
            If backend is not supported
        """
        if self.input_data is None or self.input_missing is None:
            raise AttributeError("Model must be fitted before adding imputed values")

        # Convert data to long format for easier manipulation
        long_data = pd.melt(self.input_data, var_name="protein", value_name="intensity")

        # Replace zeros (used for missing values in model) with NaN
        long_data.loc[long_data["intensity"] == 0, "intensity"] = np.nan
        long_data["was_missing"] = long_data["intensity"].isna()
        long_data.loc[:, "imp_mean"] = np.nan

        if self.backend == "pyro":
            if self.original_params is None:
                raise AttributeError("Pyro parameters not compiled yet")

            # Extract imputation parameters from Pyro model
            imputation_param_keys = [key for key in self.original_params.keys() if "imp" in key]

            if imputation_param_keys:
                # Create dataframe of imputation values
                imputation_params = pd.DataFrame.from_dict(
                    {
                        key.replace("AutoDelta.imp_", ""): self.original_params[key]
                        for key in imputation_param_keys
                    }
                )

                # Apply missing value mask and melt to long format
                imputation_long = imputation_params[self.input_missing.astype(bool)]
                imputation_long = pd.melt(imputation_long, var_name="protein", value_name="imp_loc")
                imputation_long = imputation_long.dropna()

                # Merge imputed values with original data
                for protein in imputation_long["protein"].unique():
                    mask = (long_data["protein"] == protein) & long_data["intensity"].isna()

                    imputed_values = imputation_long.loc[
                        imputation_long["protein"] == protein, "imp_loc"
                    ].values

                    long_data.loc[mask, "imp_mean"] = imputed_values

        elif self.backend == "numpyro":
            if self.learned_params is None:
                raise AttributeError("NumPyro parameters not compiled yet")

            # Extract imputation parameters
            imputation_params = {
                key.replace("imp_", ""): self.learned_params[key]
                for key in self.learned_params.keys()
                if "imp" in key
            }

            # Apply imputed values for each variable
            for variable, imputed_values in imputation_params.items():
                missing_mask = (long_data["protein"] == variable) & long_data["was_missing"]

                if missing_mask.sum() > 0:
                    missing_indices = long_data[missing_mask].index

                    # Handle both scalar and array imputed values
                    if np.isscalar(imputed_values):
                        long_data.loc[missing_indices, "imp_mean"] = imputed_values
                    else:
                        # Use only as many values as needed
                        n_to_fill = min(len(imputed_values), len(missing_indices))
                        long_data.loc[missing_indices[:n_to_fill], "imp_mean"] = imputed_values[
                            :n_to_fill
                        ]
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.imputed_data = long_data

    def fit(
        self, observational_data: pd.DataFrame, causal_graph: Any, verbose: Optional[bool] = None
    ) -> None:
        """
        Fit the latent variable model to observational data and causal graph.

        This is the main method that orchestrates the entire model fitting process:
        1. Stores the input data and causal graph
        2. Parses the graph structure to identify root and descendant nodes
        3. Converts data to backend-specific format
        4. Sets up prior distributions
        5. Trains the model using the specified backend
        6. Compiles learned parameters
        7. Adds imputed values for missing data

        Parameters
        ----------
        observational_data : pd.DataFrame
            Input data with variables as columns and observations as rows.
            Can contain missing values (NaN) which will be handled through
            model-based imputation.
        causal_graph : networkx.DiGraph or y0.graph.NxMixedGraph
            Causal graph representing the relationships between variables.
            Should have directed edges for causal relationships and optional
            undirected edges for latent confounders.
        verbose : bool, optional
            Whether to print progress information during fitting.
            If None, uses the instance's verbose setting.

        Raises
        ------
        ValueError
            If observational_data is empty or causal_graph is invalid
        RuntimeError
            If model training fails

        Examples
        --------
        Basic model fitting:

        >>> lvm = LVM(backend="numpyro")
        >>> lvm.fit(data, graph)
        >>> print(f"Model fitted with {len(lvm)} observations")

        Fitting with custom parameters:

        >>> lvm = LVM(
        ...     backend="pyro",
        ...     num_steps=5000,
        ...     informative_priors=my_priors
        ... )
        >>> lvm.fit(data, graph, verbose=False)
        """
        # Use instance verbose setting if not provided
        if verbose is None:
            verbose = self.verbose

        # Validate inputs
        if observational_data.empty:
            raise ValueError("Observational data cannot be empty")

        if causal_graph is None:
            raise ValueError("Causal graph must be provided")

        if verbose:
            print(f"Fitting LVM with {self.backend} backend")
            print(f"Data shape: {observational_data.shape}")

        # Store input data and graph
        self.obs_data = observational_data.copy()
        self.causal_graph = causal_graph

        try:
            # Prepare information for model
            if verbose:
                print("Parsing causal graph structure...")
            self.parse_graph()

            if verbose:
                print("Scaling observational data...")
            self.fit_scaler(self.obs_data)
            self.obs_data = self._to_z(self.obs_data)

            if verbose:
                print(
                    f"Found {len(self.root_nodes)} root nodes and "
                    f"{len(self.descendant_nodes)} descendant nodes"
                )

            if verbose:
                print("Processing observational data...")
            self.parse_data()

            if verbose:
                print("Setting up prior distributions...")
            self.parse_priors()

            # Train model and extract results
            if self.backend == "numpyro":
                self.train_numpyro(verbose=verbose)
                self.compile_numpyro_parameters()
            elif self.backend == "pyro":
                self.train_pyro(verbose=verbose)
                self.compile_pyro_parameters()

            if verbose:
                print("Adding imputed values for missing data...")
            self.add_imputed_values()

            if verbose:
                print("Model fitting completed successfully!")

        except Exception as e:
            raise RuntimeError(f"Model fitting failed: {str(e)}") from e

    def intervention(
        self, intervention: Dict[str, float], outcome_node: str, compare_value: float = 0.0
    ) -> None:
        """
        Perform interventional analysis to estimate causal effects.

        Estimates the causal effect of an intervention by comparing outcomes
        under the intervention versus a baseline (control) condition. Uses
        the do-calculus to perform counterfactual inference.

        Parameters
        ----------
        intervention : Dict[str, float]
            Dictionary specifying the intervention. Keys are variable names
            and values are the intervention levels.
            Example: {"Treatment": 1.0} or {"Drug_A": 2.0, "Drug_B": 1.5}
        outcome_node : str
            Name of the outcome variable to measure the intervention effect on
        compare_value : float, default=0.0
            Baseline value for comparison (control condition)

        Sets
        ----
        self.posterior_samples : np.ndarray or torch.Tensor
            Posterior samples for outcome under baseline condition
        self.intervention_samples : np.ndarray or torch.Tensor
            Posterior samples for outcome under intervention condition

        Raises
        ------
        AttributeError
            If model has not been fitted yet
        ValueError
            If intervention variables or outcome_node are not in the graph
        KeyError
            If specified nodes are not found in the causal graph

        Examples
        --------
        Single intervention:

        >>> lvm.intervention({"Treatment": 1.0}, "Outcome")
        >>> baseline_mean = lvm.posterior_samples.mean()
        >>> intervention_mean = lvm.intervention_samples.mean()
        >>> causal_effect = intervention_mean - baseline_mean

        Multiple interventions:

        >>> lvm.intervention({"Drug_A": 2.0, "Drug_B": 1.5}, "Survival")
        >>> effect_size = (lvm.intervention_samples - lvm.posterior_samples).mean()

        Note
        ----
        The method performs two separate model evaluations:
        1. Baseline: all intervention variables set to compare_value
        2. Intervention: variables set to specified intervention values

        The causal effect is the difference between these two conditions.
        """
        if self.model is None:
            raise AttributeError("Model must be fitted before performing interventions")

        # Validate intervention and outcome variables
        all_nodes = set(self.root_nodes + list(self.descendant_nodes.keys()))

        for var in intervention.keys():
            if var not in all_nodes:
                raise ValueError(f"Intervention variable '{var}' not found in causal graph")

        if outcome_node not in all_nodes:
            raise ValueError(f"Outcome node '{outcome_node}' not found in causal graph")

        intervention_df = self._to_z(pd.DataFrame(intervention, index=[0])).dropna(
            axis=1, how="all"
        )
        intervention = intervention_df.iloc[0].dropna().to_dict()

        # if compare_value != 0.0:
        #     compare_value = self._to_z(torch.tensor([compare_value]))[0].item()

        # Handle different backends
        if self.backend == "pyro":
            self._intervention_pyro(intervention, outcome_node, compare_value)
        elif self.backend == "numpyro":
            self._intervention_numpyro(intervention, outcome_node, compare_value)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        # Convert results from z-score scale back to natural/original scale
        if outcome_node != "Output":
            if self.posterior_samples is not None:
                self.posterior_samples = self._from_z(torch.tensor(self.posterior_samples)).numpy()
            if self.intervention_samples is not None:
                self.intervention_samples = self._from_z(
                    torch.tensor(self.intervention_samples)
                ).numpy()

    def _intervention_pyro(
        self, intervention: Dict[str, float], outcome_node: str, compare_value: float
    ) -> None:
        """Perform intervention analysis using Pyro backend."""
        # Prepare conditioning data (no missing values for intervention)
        condition_data = {}

        for node in self.root_nodes:
            if "latent" not in node:
                condition_data[f"missing_{node}"] = torch.tensor(
                    [0.0] * len(self.input_data), dtype=torch.float32
                )

        for node in self.descendant_nodes:
            condition_data[f"missing_{node}"] = torch.tensor(
                [0.0] * len(self.input_data), dtype=torch.float32
            )

        # Create baseline intervention (control condition)
        if len(intervention) > 1:
            intervention_vars = list(intervention.keys())
            baseline_intervention = {
                intervention_vars[0]: torch.tensor(compare_value, dtype=torch.float32),
                intervention_vars[1]: torch.tensor(compare_value, dtype=torch.float32),
            }
        else:
            baseline_intervention = {
                list(intervention.keys())[0]: torch.tensor(compare_value, dtype=torch.float32)
            }

        # Convert intervention values to tensors
        intervention_tensors = {
            var: torch.tensor(value, dtype=torch.float32) for var, value in intervention.items()
        }

        # Baseline model (control)
        baseline_model = poutine.do(self.model, data=baseline_intervention)
        baseline_predictive = pyro.infer.Predictive(
            baseline_model, guide=self.guide, num_samples=self.num_samples
        )
        baseline_predictions = baseline_predictive(condition_data, self.priors)

        # Intervention model
        intervention_model = poutine.do(self.model, data=intervention_tensors)
        intervention_predictive = pyro.infer.Predictive(
            intervention_model, guide=self.guide, num_samples=self.num_samples
        )
        intervention_predictions = intervention_predictive(condition_data, self.priors)

        # Extract predictions for outcome node
        self.posterior_samples = baseline_predictions[outcome_node].flatten()
        self.intervention_samples = intervention_predictions[outcome_node].flatten()

    def _intervention_numpyro(
        self, intervention: Dict[str, float], outcome_node: str, compare_value: float
    ) -> None:
        """Perform intervention analysis using NumPyro backend."""
        rng_key, rng_key_intervention = random.split(random.PRNGKey(2))

        # Create baseline intervention (control condition)
        if len(intervention) > 1:
            intervention_vars = list(intervention.keys())
            baseline_intervention = {
                intervention_vars[0]: compare_value,
                intervention_vars[1]: compare_value,
            }
        else:
            baseline_intervention = {list(intervention.keys())[0]: compare_value}

        # Baseline model evaluation
        baseline_model = numpyro.handlers.do(
            NumpyroProteomicPerturbationModel, data=baseline_intervention
        )
        baseline_predictive = NumpyroPredicitve(baseline_model, self.model.get_samples())
        baseline_predictions = baseline_predictive(
            rng_key,
            None,
            [],  # No observed data for intervention
            self.priors,
            self.root_nodes,
            self.descendant_nodes,
        )

        # Intervention model evaluation
        intervention_model = numpyro.handlers.do(
            NumpyroProteomicPerturbationModel, data=intervention
        )
        intervention_predictive = NumpyroPredicitve(intervention_model, self.model.get_samples())
        intervention_predictions = intervention_predictive(
            rng_key_intervention,
            None,
            [],  # No observed data for intervention
            self.priors,
            self.root_nodes,
            self.descendant_nodes,
        )

        # Extract outcome predictions
        self.posterior_samples = baseline_predictions[outcome_node]
        self.intervention_samples = intervention_predictions[outcome_node]


def main() -> None:
    """
    Example usage and testing of the LVM class.

    This function demonstrates a complete workflow using the LVM for causal
    inference on a simulated mediator model. It shows:
    1. Data simulation with missing values
    2. Data processing and normalization
    3. Model fitting with Pyro backend
    4. Intervention analysis
    5. Visualization of results
    """
    import matplotlib.pyplot as plt

    from causomic.simulation.example_graphs import mediator, signaling_network

    print("=== LVM Example Workflow ===")

    # Generate mediator graph and simulate data
    print("1. Generating mediator model and simulating data...")
    med_graph = mediator(add_independent_nodes=False, output_node=True)

    simulated_data = simulate_data(
        med_graph["Networkx"],
        coefficients=med_graph["Coefficients"],
        add_error=False,
        mnar_missing_param=[-3, 0.4],  # Missing not at random
        add_feature_var=True,
        n=100,
        seed=2,
    )

    # Process feature-level data to protein-level
    print("2. Processing feature-level data...")
    input_data = dataProcess(
        simulated_data["Feature_data"],
        normalization=False,
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True,
    )

    input_data.loc[:, "Output"] = simulated_data["Protein_data"]["Output"]

    # Fit LVM model
    print("4. Fitting LVM model with Pyro backend...")
    lvm = LVM(backend="pyro", num_steps=2000, verbose=True)
    lvm.fit(input_data, med_graph["causomic"])

    # Perform intervention analysis
    print("5. Performing intervention analysis...")
    intervention_value = 3.0
    lvm.intervention({"X": intervention_value}, "Output")

    # Process imputed data for visualization
    print("6. Processing imputed data...")
    imputed_data = lvm.imputed_data.copy()
    imputed_data["intensity"] = np.where(
        imputed_data["was_missing"], imputed_data["imp_mean"], imputed_data["intensity"]
    )

    # Create wide format for plotting
    imputed_data["index"] = np.tile(
        np.arange(1, len(input_data) + 1), len(imputed_data["protein"].unique())
    )
    lvm_imputed = imputed_data.pivot(index="index", columns="protein", values="intensity")

    # Visualization 1: Imputed vs Original Data
    print("7. Creating visualizations...")
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    missing_indices = imputed_data[imputed_data["was_missing"]]["index"].unique()
    colors = ["red" if idx in missing_indices else "blue" for idx in lvm_imputed.index]

    plt.scatter(lvm_imputed["M1"], lvm_imputed["Z"], alpha=0.6, c=colors, s=20)
    plt.xlabel("M1 (Mediator)")
    plt.ylabel("Z (Outcome)")
    plt.title("M1 vs Z\n(Red: Imputed, Blue: Observed)")
    plt.grid(True, alpha=0.3)

    # Visualization 2: Distribution by Output
    plt.subplot(1, 2, 2)
    for output_val in sorted(input_data["Output"].unique()):
        subset = input_data[input_data["Output"] == output_val]
        plt.hist(subset["X"], bins=20, alpha=0.6, label=f"Output={output_val}")

    plt.xlabel("X (Treatment)")
    plt.ylabel("Count")
    plt.title("Distribution of X by Output")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Intervention analysis across multiple values
    print("8. Analyzing intervention effects across values...")
    intervention_values = [0, 1, 2, 3, 4, 5, 6]
    intervention_results = []

    for value in intervention_values:
        lvm.intervention({"X": value}, "Output")
        mean_effect = lvm.intervention_samples.mean()
        std_effect = lvm.intervention_samples.std()
        intervention_results.append((value, mean_effect, std_effect))

    # Plot intervention effects
    plt.figure(figsize=(10, 6))
    values, means, stds = zip(*intervention_results)

    plt.errorbar(
        values,
        means,
        yerr=stds,
        fmt="o-",
        capsize=5,
        color="royalblue",
        ecolor="darkorange",
        linewidth=2,
        markersize=8,
        markeredgewidth=2,
    )

    plt.xlabel("Intervention Value (X)", fontsize=12)
    plt.ylabel("Expected Outcome (Output)", fontsize=12)
    plt.title("Causal Effect of X on Output", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n=== Results Summary ===")
    print(f"Model fitted with {len(lvm)} observations")
    print(f"Missing data imputed: {lvm.imputed_data['was_missing'].sum():.0f} values")
    print(f"Intervention effects range: {min(means):.3f} to {max(means):.3f}")
    print("Analysis completed successfully!")


if __name__ == "__main__":
    main()
