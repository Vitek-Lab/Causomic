"""
Causal Data Simulation for Proteomics Research

This module implements comprehensive data simulation capabilities for causal
inference studies in proteomics and systems biology. The simulation framework
generates realistic datasets that capture the complex characteristics of mass
spectrometry proteomics data, including hierarchical feature structure,
realistic missing data patterns, and measurement noise.

The core innovation is the multi-level simulation approach that models both
protein-level causal relationships and feature-level technical variation,
enabling realistic evaluation of causal discovery algorithms under conditions
that closely mirror real experimental data.

Key Features
------------
- Structural Equation Model (SEM) simulation from causal graphs
- Hierarchical protein-feature data structure matching proteomics workflows
- Realistic missing data patterns (MAR and MNAR)
- Technical measurement error and biological variation
- Cell type effects and batch confounding
- Intervention simulation for causal effect evaluation
- Integration with proteomics data processing pipelines

Simulation Workflow
------------------
1. Generate structural equation coefficients from causal graph
2. Simulate protein-level data following causal relationships
3. Generate feature-level technical replicates with measurement error
4. Apply realistic missing data patterns based on intensity values
5. Output multi-level data structure for downstream analysis

Examples
--------
>>> # Simulate proteomics data from causal graph
>>> import networkx as nx
>>> from causomic.simulation.example_graphs import signaling_network
>>>
>>> # Get example signaling network
>>> network = signaling_network(include_coef=True)
>>>
>>> # Simulate realistic proteomics data
>>> sim_data = simulate_data(
...     network['Networkx'],
...     coefficients=network['Coefficients'],
...     n=100,  # Typical proteomics sample size
...     mnar_missing_param=[-3, 0.4],  # Realistic missing pattern
...     add_feature_var=True  # Multi-level structure
... )
>>>
>>> protein_data = sim_data['Protein_data']  # Causal relationships
>>> feature_data = sim_data['Feature_data']  # Technical measurements

Author: Devon Kohler
Date: 2024
"""

import pickle
from typing import Any, Dict, List, Optional, Union

# Visualization imports
import matplotlib.pyplot as plt
import networkx as nx

# Scientific computing imports
import numpy as np
import pandas as pd
import seaborn as sns


def simulate_data(
    graph: nx.DiGraph,
    coefficients: Optional[Dict] = None,
    include_missing: bool = True,
    cell_type: bool = False,
    n_cells: int = 3,
    mar_missing_param: float = 0.05,
    mnar_missing_param: List[float] = [-3, 0.4],
    add_error: bool = False,
    error_node: Optional[str] = None,
    intervention: Dict = {},
    add_feature_var: bool = True,
    n: int = 1000,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Simulate realistic proteomics data from causal graph structure.

    Generates multi-level proteomics data that captures both causal relationships
    between proteins and technical characteristics of mass spectrometry measurements.
    The simulation produces protein-level data following structural equation models
    and feature-level data reflecting technical measurement variation and realistic
    missing data patterns.

    This function is designed to create realistic test datasets for evaluating
    causal discovery algorithms under conditions that closely mirror real
    proteomics experiments, including small sample sizes, hierarchical data
    structure, and complex missing data mechanisms.

    Parameters
    ----------
    graph : nx.DiGraph
        Causal graph structure defining relationships between proteins.
        Nodes represent proteins/variables, edges represent causal relationships.
    coefficients : Optional[Dict], default=None
        Structural equation coefficients for each node. If None, generates
        random coefficients. Format: {node: {parent: coef, 'intercept': val, 'error': var}}
    include_missing : bool, default=True
        Whether to include realistic missing data patterns in feature-level data
    cell_type : bool, default=False
        Whether to include cell type effects as systematic variation.
        Requires custom coefficients with 'cell_type' specifications.
    n_cells : int, default=3
        Number of different cell types to simulate if cell_type=True
    mar_missing_param : float, default=0.05
        Probability of Missing At Random (MAR) missingness.
        Represents random technical failures in measurement.
    mnar_missing_param : List[float], default=[-3, 0.4]
        Parameters [intercept, slope] for Missing Not At Random (MNAR) pattern.
        Models intensity-dependent detection limits common in mass spectrometry.
    add_error : bool, default=False
        Whether to add additional measurement error to specific nodes
    error_node : Optional[str], default=None
        Specific node to add extra measurement error. If None and add_error=True,
        adds error to all non-output nodes.
    intervention : Dict, default={}
        Dictionary specifying interventional values for specific nodes.
        Format: {node: intervention_value}. Useful for causal effect simulation.
    add_feature_var : bool, default=True
        Whether to generate feature-level technical replicates with measurement
        variation. Essential for realistic proteomics simulation.
    n : int, default=1000
        Number of samples (biological replicates) to simulate
    seed : Optional[int], default=None
        Random seed for reproducible simulations

    Returns
    -------
    Dict[str, Any]
        Dictionary containing simulated data at multiple levels:
        - 'Protein_data': Dict with protein-level causal data
        - 'Feature_data': pd.DataFrame with feature-level technical measurements
        - 'Coefficients': Dict of structural equation coefficients used

    Examples
    --------
    >>> # Basic simulation with random coefficients
    >>> import networkx as nx
    >>> G = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])
    >>> data = simulate_data(G, n=100, seed=42)
    >>>
    >>> # Simulation with custom coefficients and realistic missing data
    >>> coeffs = {
    ...     'X': {'intercept': 5, 'error': 1},
    ...     'Y': {'intercept': 2, 'error': 0.5, 'X': 0.8},
    ...     'Z': {'intercept': -1, 'error': 0.3, 'Y': 1.2}
    ... }
    >>> data = simulate_data(
    ...     G,
    ...     coefficients=coeffs,
    ...     n=50,  # Small sample like real proteomics
    ...     mnar_missing_param=[-2, 0.5],  # Strong intensity dependence
    ...     add_feature_var=True
    ... )
    >>>
    >>> # Access different data levels
    >>> protein_data = data['Protein_data']  # Causal structure
    >>> feature_data = data['Feature_data']  # Technical measurements
    >>> print(f"Missing data: {feature_data['Obs_Intensity'].isna().mean():.2%}")

    Notes
    -----
    Simulation Process:
    1. Generate/validate structural equation coefficients
    2. Simulate protein-level data following topological ordering
    3. Add cell type effects if specified
    4. Generate feature-level technical replicates with measurement error
    5. Apply realistic missing data patterns (MAR and MNAR)
    6. Return multi-level data structure

    Missing Data Mechanisms:
    - MAR: Random technical failures (constant probability)
    - MNAR: Intensity-dependent detection (logistic model)
    - Combined pattern reflects real mass spectrometry limitations

    The feature-level simulation captures key proteomics characteristics:
    - Multiple features (peptides) per protein
    - Technical measurement variation
    - Intensity-dependent missing data
    - Realistic effect sizes and noise levels

    This simulation framework enables rigorous evaluation of causal discovery
    algorithms under realistic proteomics constraints including small samples,
    high-dimensional feature space, and complex missing data patterns.
    """
    if seed is not None:
        np.random.seed(seed)

    if coefficients is None:
        coefficients = generate_coefficients(graph)

    data = dict()

    sorted_nodes = [i for i in nx.topological_sort(graph) if i != "cell_type"]

    print("simulating data...")
    for node in sorted_nodes:
        node_coefficients = coefficients[node]
        if node in intervention.keys():
            temp_int = intervention[node]
        else:
            temp_int = None

        data[node] = simulate_node(data, node_coefficients, n, cell_type, temp_int, node)

    if cell_type:
        data["cell_type"] = np.repeat([i for i in range(n_cells)], n // n_cells)
        if len(data["cell_type"]) < n:
            data["cell_type"] = np.append(data["cell_type"], n_cells - 1)

    if add_error:
        if error_node is None:
            for node, _ in data.items():
                if node != "Output":
                    data[node] += np.random.normal(0, 0.25, n)
        else:
            data[error_node] += np.random.normal(0, 5, n)

    # break data into features
    if add_feature_var:
        print("adding feature level data...")
        feature_level_data_list = list()
        for node in sorted_nodes:
            if node != "Output":
                feature_level_data_list.append(generate_features(data[node], node))

        feature_level_data = pd.concat(feature_level_data_list, ignore_index=True)

        print("masking data...")
        if include_missing:
            feature_level_data = add_missing(
                feature_level_data, mar_missing_param, mnar_missing_param
            )
    else:
        feature_level_data = None

    return {"Protein_data": data, "Feature_data": feature_level_data, "Coefficients": coefficients}


def generate_coefficients(graph: nx.DiGraph) -> Dict[str, Dict[str, float]]:
    """
    Generate random structural equation coefficients for causal graph.

    Creates realistic structural equation model coefficients for each node
    in the causal graph, including parent relationships, intercepts, and
    error variances. The coefficient ranges are chosen to produce realistic
    effect sizes typical of biological systems.

    Parameters
    ----------
    graph : nx.DiGraph
        Causal graph structure with nodes as variables and edges as relationships

    Returns
    -------
    Dict[str, Dict[str, float]]
        Nested dictionary of coefficients for each node:
        {node: {parent: coefficient, 'intercept': value, 'error': variance}}

    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.DiGraph([('X', 'Y'), ('Y', 'Z')])
    >>> coeffs = generate_coefficients(G)
    >>> print(coeffs)
    # {'X': {'intercept': 18.2, 'error': 0.7},
    #  'Y': {'X': 0.8, 'intercept': -2.1, 'error': 0.3},
    #  'Z': {'Y': -0.9, 'intercept': 1.5, 'error': 0.6}}

    Notes
    -----
    Coefficient Generation Rules:
    - Root nodes (no parents): Intercept 15-25, representing baseline expression
    - Child nodes: Intercept -5 to 5, representing regulated expression
    - Parent effects: [-1, -0.5] or [0.5, 1.5], avoiding weak effects near zero
    - Error variance: [0, 1], representing measurement and biological noise

    The bimodal parent effect distribution ensures meaningful causal relationships
    while allowing both positive and negative regulation typical in biological
    networks. Effect sizes are calibrated for realistic biological variation.
    """
    coefficients = {}

    for node in graph.nodes():
        parents = list(graph.predecessors(node))
        coefficients[node] = generate_node_coefficients(parents)

    return coefficients


def generate_node_coefficients(parents: List[str]) -> Dict[str, float]:
    """
    Generate random structural equation coefficients for a single node.

    Creates coefficients for a node's structural equation including parent
    effects, intercept, and error variance. Uses biologically realistic
    ranges to ensure meaningful causal relationships and appropriate
    signal-to-noise ratios.

    Parameters
    ----------
    parents : List[str]
        List of parent node names that causally influence this node

    Returns
    -------
    Dict[str, float]
        Dictionary of coefficients including:
        - Parent effects: Coefficient for each parent variable
        - 'intercept': Baseline value when parents are zero
        - 'error': Error variance (standard deviation)

    Examples
    --------
    >>> # Root node (no parents)
    >>> coeffs = generate_node_coefficients([])
    >>> print(coeffs)
    # {'intercept': 19.3, 'error': 0.4}

    >>> # Child node with parents
    >>> coeffs = generate_node_coefficients(['X', 'Y'])
    >>> print(coeffs)
    # {'X': 0.7, 'Y': -0.8, 'intercept': 2.1, 'error': 0.6}

    Notes
    -----
    Coefficient Ranges:
    - Parent effects: Bimodal distribution [-1, -0.5] ∪ [0.5, 1.5]
      * Avoids weak effects near zero that are difficult to detect
      * Allows both activation and inhibition relationships
      * Realistic for biological regulatory strength

    - Intercept (root nodes): Uniform [15, 25]
      * Represents baseline protein expression levels
      * High enough to allow meaningful regulation

    - Intercept (child nodes): Uniform [-5, 5]
      * Represents regulated expression around baseline
      * Can be positive or negative depending on regulation

    - Error variance: Uniform [0, 1]
      * Represents measurement noise and biological variation
      * Scaled appropriately for typical protein expression ranges

    The bimodal parent effect distribution is crucial for ensuring that
    simulated causal relationships are detectable and biologically meaningful,
    avoiding the common problem of weak effects that disappear in noise.
    """
    coefficients = {}

    for parent in parents:
        coefficients[parent] = np.random.choice(
            [np.random.uniform(-1.0, -0.5), np.random.uniform(0.5, 1.5)]
        )

    if len(coefficients.keys()) == 0:
        coefficients["intercept"] = np.random.uniform(15, 25)
    else:
        coefficients["intercept"] = np.random.uniform(-5, 5)

    coefficients["error"] = np.random.uniform(0, 1)

    return coefficients


def simulate_node(
    data: Dict[str, np.ndarray],
    coefficients: Dict[str, float],
    n: int,
    cell_type: bool,
    intervention: Optional[float],
    node_name: str,
) -> np.ndarray:
    """
    Simulate data for a single node using its structural equation.

    Generates data for one variable following its structural equation model,
    incorporating parent effects, cell type variation, and special handling
    for binary output variables. Supports interventional data generation
    for causal effect simulation.

    Parameters
    ----------
    data : Dict[str, np.ndarray]
        Dictionary containing simulated data for parent nodes
    coefficients : Dict[str, float]
        Structural equation coefficients for this node including
        parent effects, intercept, and error variance
    n : int
        Number of samples to generate
    cell_type : bool
        Whether to include cell type effects in the simulation
    intervention : Optional[float]
        Fixed interventional value to set for all samples.
        If provided, overrides structural equation simulation.
    node_name : str
        Name of the node being simulated (used for special output handling)

    Returns
    -------
    np.ndarray
        Simulated data values for this node across all samples

    Examples
    --------
    >>> # Simulate root node
    >>> coeffs = {'intercept': 20, 'error': 1}
    >>> data = {}
    >>> values = simulate_node(data, coeffs, 100, False, None, 'X')
    >>> print(f"Mean: {values.mean():.1f}, Std: {values.std():.1f}")
    # Mean: 20.1, Std: 1.0

    >>> # Simulate child node with parent effects
    >>> data = {'X': np.random.normal(20, 1, 100)}
    >>> coeffs = {'X': 0.5, 'intercept': 5, 'error': 0.5}
    >>> values = simulate_node(data, coeffs, 100, False, None, 'Y')
    >>> # Y = 5 + 0.5*X + noise

    >>> # Simulate binary output variable
    >>> data = {'Y': np.random.normal(10, 2, 100)}
    >>> coeffs = {'Y': 0.3, 'intercept': -2, 'error': 0.1}
    >>> binary_out = simulate_node(data, coeffs, 100, False, None, 'Output')
    >>> print(f"Probability of 1: {binary_out.mean():.2f}")

    Notes
    -----
    Structural Equation Form:
    node_value = intercept + Σ(parent_coeff × parent_value) + ε

    Where:
    - intercept: Baseline value when all parents are zero
    - parent_coeff: Linear effect of each parent
    - ε ~ N(0, error_variance): Gaussian noise term

    Special Cases:
    - Output nodes: Apply logistic transformation + Bernoulli sampling
      * Useful for binary outcomes (success/failure, high/low expression)
      * node_value → P(success) → Bernoulli(P) → {0,1}

    - Cell type effects: Add systematic variation across cell types
      * Requires 'cell_type' key in coefficients
      * Models batch effects or cell-specific baseline differences

    - Interventions: Override structural equation with fixed values
      * Useful for simulating experimental perturbations
      * Enables causal effect estimation validation

    The simulation follows standard structural equation modeling principles
    while incorporating biological realism through appropriate noise models
    and special handling for common biological measurement types.
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
            node_data[(i * n_obs_per_cell) : (i * n_obs_per_cell + n_obs_per_cell)] += cells[i]

    if node_name == "Output":
        p = 1 / (1 + np.exp(-node_data))  # logistic transform
        node_data = np.random.binomial(1, p)  # Bernoulli draw
    else:
        node_data += np.random.normal(0, coefficients["error"], n)

    if intervention is not None:
        node_data = np.repeat(intervention, n)

    return node_data


def generate_features(data: np.ndarray, node: str) -> pd.DataFrame:
    """
    Generate feature-level technical measurements from protein-level data.

    Simulates the hierarchical structure of proteomics data where each protein
    is measured through multiple features (peptides/transitions) with technical
    variation. This captures the key characteristic of mass spectrometry data
    where protein quantification relies on aggregating multiple lower-level
    measurements.

    Each feature represents a peptide or transition measurement with:
    - Systematic bias relative to protein level (feature effect)
    - Random measurement error
    - Realistic number of features per protein (15-30)

    Parameters
    ----------
    data : np.ndarray
        Protein-level true values across samples (from structural equation model)
    node : str
        Protein/node name for labeling features

    Returns
    -------
    pd.DataFrame
        Feature-level data with columns:
        - 'Protein': Protein name (node)
        - 'Replicate': Sample index (biological replicate)
        - 'Feature': Feature index (peptide/transition ID)
        - 'Intensity': Measured intensity value

    Examples
    --------
    >>> # Generate features for protein with 5 samples
    >>> protein_data = np.array([10, 12, 8, 15, 11])
    >>> features = generate_features(protein_data, 'AKT1')
    >>> print(features.head())
    #   Protein  Replicate  Feature   Intensity
    # 0    AKT1          0        0   10.234
    # 1    AKT1          0        1    9.876
    # 2    AKT1          0        2   10.512
    # ...

    >>> # Check feature count and structure
    >>> n_features = features['Feature'].nunique()
    >>> n_samples = features['Replicate'].nunique()
    >>> print(f"Features per protein: {n_features}")
    >>> print(f"Samples: {n_samples}")
    # Features per protein: 23
    # Samples: 5

    Notes
    -----
    Feature Generation Model:
    intensity_ij = protein_level_i + feature_effect_j + ε_ij

    Where:
    - i indexes biological samples (replicates)
    - j indexes features (peptides/transitions)
    - feature_effect_j ~ Uniform[-0.75, 0.75]: Systematic peptide bias
    - ε_ij ~ N(0, 0.1): Random measurement error

    Realistic Characteristics:
    - Features per protein: 15-30 (typical for targeted proteomics)
    - Feature effects: ±0.75 log-intensity units (realistic bias range)
    - Measurement error: σ=0.1 (typical technical precision)
    - Each sample gets independent measurements for all features

    This structure enables testing of:
    - Protein summarization algorithms (feature → protein aggregation)
    - Missing data imputation at feature level
    - Technical vs biological variation separation
    - Realistic proteomics data processing pipelines

    The hierarchical data structure is essential for realistic evaluation
    of causal discovery algorithms under conditions that mirror real
    mass spectrometry proteomics experiments.
    """
    feature_level_data = pd.DataFrame(columns=["Protein", "Replicate", "Feature", "Intensity"])

    number_features = np.random.randint(15, 30)
    feature_effects = [np.random.uniform(-0.75, 0.75) for _ in range(number_features)]

    for i in range(len(data)):
        for j in range(number_features):

            # Measurement error
            error = np.random.normal(0, 0.1)
            feature_level_data = pd.concat(
                [
                    feature_level_data,
                    pd.DataFrame(
                        {
                            "Protein": [node],
                            "Replicate": [i],
                            "Feature": [j],
                            "Intensity": [data[i] + feature_effects[j] + error],
                        }
                    ),
                ],
                ignore_index=True,
            )

    return feature_level_data


def add_missing(
    feature_level_data: pd.DataFrame, mar_missing_param: float, mnar_missing_param: List[float]
) -> pd.DataFrame:
    """
    Add realistic missing data patterns to feature-level measurements.

    Implements both Missing At Random (MAR) and Missing Not At Random (MNAR)
    mechanisms that reflect real missing data patterns in mass spectrometry
    proteomics. The combined pattern captures both random technical failures
    and systematic detection limits based on signal intensity.

    Missing data is a critical challenge in proteomics where low-abundance
    proteins often fall below detection thresholds, creating systematic
    bias that can confound causal inference if not properly modeled.

    Parameters
    ----------
    feature_level_data : pd.DataFrame
        Feature-level data with 'Intensity' column containing true values
    mar_missing_param : float
        Probability of MAR missingness (constant across all intensities).
        Represents random technical failures, instrument downtime, etc.
    mnar_missing_param : List[float]
        Parameters [intercept, slope] for MNAR logistic model.
        Controls intensity-dependent detection probability.

    Returns
    -------
    pd.DataFrame
        Enhanced dataframe with additional columns:
        - 'Obs_Intensity': Observed intensity (NaN for missing values)
        - 'MAR': Boolean indicator for MAR missingness
        - 'MNAR': Boolean indicator for MNAR missingness
        - 'MNAR_threshold': Detection probability for each measurement

    Examples
    --------
    >>> # Create sample feature data
    >>> data = pd.DataFrame({
    ...     'Protein': ['A', 'A', 'B', 'B'],
    ...     'Replicate': [0, 1, 0, 1],
    ...     'Feature': [0, 0, 0, 0],
    ...     'Intensity': [15, 8, 12, 5]  # High to low intensities
    ... })
    >>>
    >>> # Add realistic missing pattern
    >>> missing_data = add_missing(
    ...     data,
    ...     mar_missing_param=0.1,      # 10% random missing
    ...     mnar_missing_param=[-2, 0.5]  # Strong intensity dependence
    ... )
    >>>
    >>> # Analyze missing patterns
    >>> print(f"Total missing: {missing_data['Obs_Intensity'].isna().mean():.1%}")
    >>> print(f"MAR missing: {missing_data['MAR'].mean():.1%}")
    >>> print(f"MNAR missing: {missing_data['MNAR'].mean():.1%}")

    Notes
    -----
    Missing Data Mechanisms:

    1. MAR (Missing At Random):
       P(missing) = mar_missing_param (constant)
       - Models random technical failures
       - Independent of intensity values
       - Easier to handle statistically

    2. MNAR (Missing Not At Random):
       P(missing) = 1 / (1 + exp(α + β × intensity))
       - Models detection limits and signal-dependent dropout
       - Higher probability for low-intensity measurements
       - Biologically realistic for mass spectrometry
       - More challenging for statistical analysis

    Parameter Interpretation:
    - mnar_missing_param[0] (α): Baseline log-odds of detection
    - mnar_missing_param[1] (β): Intensity dependence strength
    - Negative α → higher baseline missing rate
    - Positive β → less missing at higher intensities

    Realistic Parameter Ranges:
    - MAR: 0.01-0.1 (1-10% random technical failures)
    - MNAR α: -3 to -1 (moderate to high baseline dropout)
    - MNAR β: 0.2-0.6 (moderate to strong intensity dependence)

    Missing Pattern Effects:
    - MAR: Reduces power but preserves validity with proper handling
    - MNAR: Can bias causal estimates if not properly modeled
    - Combined: Reflects real proteomics missing data complexity

    This realistic missing data simulation enables proper evaluation of:
    - Missing data imputation methods
    - Causal discovery robustness to systematic dropout
    - Statistical methods for MNAR data handling
    - Realistic proteomics analysis pipelines
    """
    feature_level_data.loc[:, "Obs_Intensity"] = feature_level_data.loc[:, "Intensity"]

    for i in range(len(feature_level_data)):
        mar_prob = np.random.uniform(0, 1)
        mnar_prob = np.random.uniform(0, 1)

        mnar_thresh = 1 / (
            1
            + np.exp(
                mnar_missing_param[0]
                + (mnar_missing_param[1] * feature_level_data.loc[i, "Intensity"])
            )
        )
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


def simple_profile_plot(
    data: pd.DataFrame, protein: str, intensity_col: str = "Obs_Intensity"
) -> None:
    """
    Create feature profile plot for protein across samples.

    Visualizes the intensity patterns of individual features (peptides/transitions)
    for a specific protein across biological replicates. This diagnostic plot
    helps assess data quality, missing patterns, and feature behavior in
    simulated proteomics data.

    The plot shows both scatter points and connecting lines for each feature,
    enabling easy identification of problematic features, systematic trends,
    and missing data patterns that could affect downstream causal analysis.

    Parameters
    ----------
    data : pd.DataFrame
        Feature-level data containing protein measurements
    protein : str
        Name of protein to plot (must exist in 'Protein' column)
    intensity_col : str, default='Obs_Intensity'
        Column name containing intensity values to plot.
        Use 'Intensity' for true values, 'Obs_Intensity' for values with missing data.

    Examples
    --------
    >>> # Plot simulated protein data
    >>> from causomic.simulation.simulation import simulate_data, simple_profile_plot
    >>> from causomic.simulation.example_graphs import signaling_network
    >>>
    >>> # Generate data
    >>> network = signaling_network(include_coef=True)
    >>> sim_data = simulate_data(network['Networkx'], n=20, seed=42)
    >>>
    >>> # Plot specific protein
    >>> simple_profile_plot(sim_data['Feature_data'], 'AKT')
    >>> plt.show()
    >>>
    >>> # Compare true vs observed intensities
    >>> simple_profile_plot(sim_data['Feature_data'], 'AKT', 'Intensity')
    >>> plt.title('True Intensities')
    >>> plt.show()

    Notes
    -----
    Plot Elements:
    - X-axis: Biological replicate number (sample index)
    - Y-axis: Intensity values (log-scale typical for proteomics)
    - Colors: Different features (peptides/transitions) for the protein
    - Points: Individual measurements
    - Lines: Connecting measurements across samples for each feature

    Diagnostic Value:
    - Feature consistency: Similar trends across features indicate good data
    - Missing patterns: Gaps in lines reveal systematic dropout
    - Outliers: Unusual points may indicate measurement artifacts
    - Signal-to-noise: Spread around trend lines shows measurement precision

    This visualization is particularly useful for:
    - Validating simulation parameters
    - Diagnosing missing data patterns
    - Quality control of simulated datasets
    - Understanding feature-level variation
    - Preparing data for causal analysis

    The plot removes legend for clarity when many features are present,
    focusing attention on overall patterns rather than individual feature tracking.
    """
    fig, ax = plt.subplots()

    plot_data = data[data["Protein"] == protein]

    sns.scatterplot(
        x=plot_data.loc[:, "Replicate"],
        y=plot_data.loc[:, intensity_col],
        hue=plot_data.loc[:, "Feature"].astype(str),
    )
    sns.lineplot(
        x=plot_data.loc[:, "Replicate"],
        y=plot_data.loc[:, intensity_col],
        hue=plot_data.loc[:, "Feature"].astype(str),
    )
    ax.get_legend().remove()
    ax.set_title(protein)


def build_igf_network(cell_confounder: bool) -> nx.DiGraph:
    """
    Construct IGF signaling network as NetworkX directed graph.

    Creates a canonical representation of the Insulin-like Growth Factor (IGF)
    and Epidermal Growth Factor (EGF) signaling pathways. This network captures
    key regulatory relationships in growth factor signaling cascades commonly
    studied in cancer biology and metabolic research.

    The network includes major signaling components from receptor activation
    through downstream kinase cascades, representing well-characterized
    protein-protein interactions and phosphorylation events measurable
    by mass spectrometry proteomics.

    Parameters
    ----------
    cell_confounder : bool
        Whether to include cell type as a confounding variable.
        When True, adds 'cell_type' node with edges to downstream
        signaling components (Ras, Raf, Mek, Erk), representing
        cell-type-specific expression differences that could confound
        causal discovery.

    Returns
    -------
    nx.DiGraph
        Directed graph representing IGF/EGF signaling network:

        Core Pathway Nodes:
        - Growth factors: 'EGF', 'IGF' (upstream stimuli)
        - Adapters: 'SOS' (guanine nucleotide exchange factor)
        - Central switches: 'Ras' (small GTPase), 'PI3K' (lipid kinase)
        - Kinase cascade: 'Akt', 'Raf', 'Mek', 'Erk' (MAPK pathway)

        Optional Confounding:
        - 'cell_type': Added when cell_confounder=True

    Examples
    --------
    >>> # Build basic IGF network
    >>> from causomic.simulation.simulation import build_igf_network
    >>>
    >>> # Network without cell type confounder
    >>> network = build_igf_network(cell_confounder=False)
    >>> print(f"Network has {network.number_of_nodes()} proteins")
    >>> print(f"Network has {network.number_of_edges()} interactions")
    >>>
    >>> # Network with cell type effects
    >>> network_conf = build_igf_network(cell_confounder=True)
    >>> print(f"With confounder: {network_conf.number_of_nodes()} nodes")
    >>>
    >>> # Visualize network structure
    >>> import matplotlib.pyplot as plt
    >>> import networkx as nx
    >>> plt.figure(figsize=(10, 6))
    >>> pos = nx.spring_layout(network, seed=42)
    >>> nx.draw(network, pos, with_labels=True,
    ...         node_color='lightblue', arrows=True)
    >>> plt.title('IGF/EGF Signaling Network')
    >>> plt.show()
    >>>
    >>> # Use for simulation studies
    >>> from causomic.simulation.simulation import simulate_data
    >>> sim_data = simulate_data(network, n=30, seed=123)

    Notes
    -----
    Network Architecture:
    - Parallel Input Pathways: Both EGF and IGF converge on shared signaling
    - Cross-talk: Ras activates PI3K; Akt inhibits Raf (negative feedback)
    - Linear Cascade: Raf → Mek → Erk (classical MAPK pathway)
    - Branching: PI3K → Akt (survival/growth pathway)

    Biological Significance:
    - Growth Factor Receptors: EGF and IGF represent major oncogenic drivers
    - MAPK Pathway: Central to proliferation and differentiation
    - PI3K/Akt Axis: Critical for cell survival and metabolism
    - Ras Hub: Integration point for multiple growth signals

    Confounder Modeling:
    - Cell Type Effects: Represents systematic differences in protein expression
    - Confounding Edges: Added to kinase cascade components most likely
      to show cell-type-specific variation
    - Causal Challenge: Tests algorithm ability to distinguish true causal
      relationships from spurious correlations due to hidden confounders

    This network structure is particularly valuable for:
    - Testing causal discovery algorithms on realistic signaling pathways
    - Benchmarking intervention prediction in growth factor networks
    - Evaluating robustness to confounding variables
    - Simulating proteomics data with known ground truth structure

    The network captures essential features of receptor tyrosine kinase
    signaling while maintaining computational tractability for simulation studies.
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


def main() -> None:
    """
    Demonstration script for causomic simulation capabilities.

    Provides example usage of the simulation framework, including data generation,
    visualization, and basic analysis workflows. This function serves as both
    a usage tutorial and validation test for the simulation components.

    The demonstration includes:
    1. Building realistic signaling networks
    2. Generating simulated proteomics data with hierarchical structure
    3. Creating diagnostic visualizations
    4. Showing data preparation for causal analysis

    Examples
    --------
    >>> # Run demonstration
    >>> from causomic.simulation.simulation import main
    >>> main()  # Will generate plots and print data summaries

    Notes
    -----
    This function demonstrates several key workflows:

    1. Network Construction:
       - Uses predefined signaling network templates
       - Shows how to specify realistic coefficient structures
       - Demonstrates network properties and validation

    2. Data Simulation:
       - Multi-level data generation (protein + feature levels)
       - Realistic missing data patterns
       - Biologically motivated noise models

    3. Quality Assessment:
       - Profile plots for data visualization
       - Statistical summaries of simulated data
       - Diagnostic checks for simulation parameters

    4. Workflow Integration:
       - Preparation for causal discovery algorithms
       - Data formatting for downstream analysis
       - Integration with causomic modeling framework

    The function uses predefined coefficient structures that reflect
    realistic biological relationships in growth factor signaling,
    making the simulated data suitable for algorithm benchmarking
    and method development.

    Expected Outputs:
    - Console output with data summaries and network statistics
    - Matplotlib plots showing protein expression profiles
    - Demonstration of data structures and formats

    This main function is particularly useful for:
    - Learning causomic simulation workflow
    - Validating installation and dependencies
    - Understanding data formats and structures
    - Testing custom network configurations
    - Benchmarking algorithm performance
    """

    from causomic.simulation.example_graphs import signaling_network
    from src.causomic.simulation.proteomics_simulator import simple_profile_plot

    informative_prior_coefs = {
        "EGF": {"intercept": 6.0, "error": 1},
        "IGF": {"intercept": 5.0, "error": 1},
        "SOS": {"intercept": 2, "error": 0.25, "EGF": 0.6, "IGF": 0.6},
        "Ras": {"intercept": 3, "error": 0.25, "SOS": 0.5},
        "PI3K": {"intercept": 0, "error": 0.25, "EGF": 0.5, "IGF": 0.5, "Ras": 0.5},
        "Akt": {"intercept": 1.0, "error": 0.25, "PI3K": 0.75},
        "Raf": {"intercept": 4, "error": 0.25, "Ras": 1.2, "Akt": -0.4},
        "Mek": {"intercept": 2.0, "error": 0.25, "Raf": 0.75},
        "Erk": {"intercept": -2, "error": 0.25, "Mek": 1.0},
    }

    fd = signaling_network(add_independent_nodes=False)
    simulated_fd_data = simulate_data(
        fd["Networkx"],
        coefficients=informative_prior_coefs,
        mnar_missing_param=[-3, 0.3],
        add_feature_var=True,
        n=25,
        seed=3,
    )

    simple_profile_plot(simulated_fd_data["Feature_data"], "Mek")
    plt.show()


if __name__ == "__main__":
    main()
