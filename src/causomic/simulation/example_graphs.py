"""
Example Causal Graphs for Simulation Studies

This module provides pre-defined causal graph structures commonly used in
causal inference research and proteomics simulation studies. The graphs
represent various challenging causal scenarios including mediation, confounding,
and complex biological signaling networks.

Each graph function returns multiple representations (NetworkX, y0, CausOmic)
and optional structural equation model coefficients for data simulation. This
enables systematic evaluation of causal discovery algorithms across diverse
graph structures and confounding patterns.

Key Graph Types
---------------
- Mediation: Sequential causal chains with intermediate mediator variables
- Backdoor: Confounding structures requiring adjustment for valid inference
- Frontdoor: Alternative identification strategies when backdoor blocked
- Signaling Networks: Realistic biological pathway structures

The module supports adding independent noise variables to test algorithm
robustness and includes realistic coefficient structures based on biological
knowledge for meaningful simulation studies.

Examples
--------
>>> # Generate a mediation graph with 2 mediators
>>> med_graph = mediator(n_med=2, include_coef=True)
>>> networkx_graph = med_graph['Networkx']
>>> coefficients = med_graph['Coefficients']
>>>
>>> # Create signaling network for proteomics simulation
>>> signal_graph = signaling_network(add_independent_nodes=True, n_ind=5)
>>> y0_graph = signal_graph['y0']  # For causal inference algorithms

Author: Devon Kohler
Date: 2024
"""

from typing import Any, Dict, Optional, Union

# Scientific computing imports
import networkx as nx
import numpy as np
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.dsl import Variable

# Causal inference libraries
from y0.graph import NxMixedGraph

from causomic.data_analysis.proteomics_data_processor import dataProcess

# CausOmic imports
from causomic.simulation.proteomics_simulator import simulate_data


def mediator(
    include_coef: bool = True,
    n_med: int = 1,
    add_independent_nodes: bool = False,
    n_ind: int = 10,
    output_node: bool = False,
) -> Dict[str, Any]:
    """
    Generate mediation causal graph with sequential mediator chain.

    Creates a linear mediation structure X → M₁ → M₂ → ... → Mₙ → Z, where
    causal effects flow through a sequence of intermediate mediator variables.
    This structure is fundamental for testing mediation analysis algorithms
    and understanding indirect causal pathways.

    The mediation pattern is common in biological systems where upstream
    signals propagate through cascading protein interactions to produce
    downstream effects, making this graph essential for proteomics simulation.

    Parameters
    ----------
    include_coef : bool, default=True
        Whether to include structural equation coefficients for simulation.
        If True, returns realistic coefficient values for data generation.
    n_med : int, default=1
        Number of mediator variables in the chain (minimum 1).
        Higher values create longer causal pathways for complex mediation.
    add_independent_nodes : bool, default=False
        Whether to add independent noise variables unconnected to main pathway.
        Useful for testing algorithm robustness to irrelevant variables.
    n_ind : int, default=10
        Number of independent nodes to add if add_independent_nodes=True.
    output_node : bool, default=False
        Whether to add final output node Z → Output for endpoint analysis.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing multiple graph representations:
        - 'Networkx': NetworkX DiGraph with full causal structure
        - 'y0': y0.NxMixedGraph for causal inference algorithms
        - 'causomic': CausOmic-compatible graph representation
        - 'Coefficients': Structural equation coefficients (if include_coef=True)

    Raises
    ------
    ValueError
        If n_med < 1 (at least one mediator required)

    Examples
    --------
    >>> # Simple mediation: X → M1 → Z
    >>> simple_med = mediator(n_med=1)
    >>> print(list(simple_med['Networkx'].edges()))
    [('X', 'M1'), ('M1', 'Z')]

    >>> # Complex mediation chain with noise variables
    >>> complex_med = mediator(
    ...     n_med=3,
    ...     add_independent_nodes=True,
    ...     n_ind=5,
    ...     include_coef=True
    ... )
    >>> # Structure: X → M1 → M2 → M3 → Z + 5 independent nodes

    >>> # Use for data simulation
    >>> from causomic.simulation.simulation import simulate_data
    >>> data = simulate_data(
    ...     complex_med['Networkx'],
    ...     coefficients=complex_med['Coefficients'],
    ...     n=1000
    ... )

    Notes
    -----
    Graph Structure:
    - Source: X (treatment/exposure variable)
    - Mediators: M1, M2, ..., Mn (sequential intermediate variables)
    - Target: Z (outcome variable)
    - Optional: Independent noise variables I1, I2, ..., In
    - Optional: Final output node

    Coefficient Structure:
    - X: Exogenous with intercept and error variance
    - Mediators: Linear dependence on previous mediator with random coefficients
    - Z: Depends on final mediator Mn
    - Coefficients chosen for realistic effect sizes in biological contexts

    This structure is ideal for testing:
    - Mediation analysis algorithms
    - Indirect effect estimation
    - Multi-step causal pathway discovery
    - Robustness to irrelevant variables
    """
    if n_med < 1:
        raise ValueError("n_med must be at least 1")

    graph = nx.DiGraph()

    nodes = ["X", "Z"]
    nodes = nodes + [f"M{i}" for i in range(1, n_med + 1)]
    if add_independent_nodes:
        nodes = nodes + [f"I{i}" for i in range(1, n_ind + 1)]

    ## Add edges
    graph.add_edge("X", "M1")

    for i in range(1, n_med + 1):
        if i < n_med:
            graph.add_edge(f"M{i}", f"M{i+1}")

    graph.add_edge(f"M{n_med}", "Z")

    if add_independent_nodes:
        for i in range(1, n_ind + 1):
            graph.add_node(f"I{i}")

    if output_node:
        graph.add_edge("Z", "Output")
        nodes = nodes + ["Output"]

    attrs = {node: False for node in nodes}

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    causomic_graph = NxMixedGraph()
    causomic_graph.add_directed_edge("X", "M1")

    for i in range(1, n_med + 1):
        if i < n_med:
            causomic_graph.add_directed_edge(f"M{i}", f"M{i+1}")

    causomic_graph.add_directed_edge(f"M{n_med}", "Z")

    if output_node:
        causomic_graph.add_directed_edge("Z", "Output")

    if include_coef:
        coef = {
            "X": {"intercept": 3, "error": 1.0},
            "M1": {"intercept": 1.6, "error": 0.25, "X": 0.5},
            "Z": {"intercept": -1, "error": 0.25, f"M{n_med}": 1.0},
        }

        for i in range(2, n_med + 1):
            coef[f"M{i}"] = {
                "intercept": 1.6,
                "error": 0.25,
                f"M{i-1}": np.random.uniform(0.5, 1.5),
            }

        if add_independent_nodes:
            for i in range(1, n_ind + 1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), "error": 1.0}
        if output_node:
            coef["Output"] = {"intercept": -2, "error": 1.0, "Z": 1}
    else:
        coef = None

    return {"Networkx": graph, "y0": y0_graph, "causomic": causomic_graph, "Coefficients": coef}


def backdoor(
    include_coef: bool = True, add_independent_nodes: bool = False, n_ind: int = 10
) -> Dict[str, Any]:
    """
    Generate backdoor confounding causal graph.

    Creates a confounding structure where a latent confounder C affects both
    treatment and outcome, creating spurious correlation that must be controlled
    for valid causal inference. This represents the classic confounding scenario
    in observational studies.

    Structure: C → {B, Y}, B → X, X → Y, Y → Z
    Where C is latent (unobserved) and creates backdoor confounding between
    X and Y. The observable B serves as a proxy for adjusting confounding.

    This graph is essential for testing confounding adjustment algorithms
    and understanding when causal identification is possible through
    observable proxies of latent confounders.

    Parameters
    ----------
    include_coef : bool, default=True
        Whether to include structural equation coefficients for simulation
    add_independent_nodes : bool, default=False
        Whether to add independent noise variables for robustness testing
    n_ind : int, default=10
        Number of independent nodes to add if add_independent_nodes=True

    Returns
    -------
    Dict[str, Any]
        Dictionary containing multiple graph representations:
        - 'Networkx': NetworkX DiGraph with latent confounders marked
        - 'y0': y0.NxMixedGraph with proper latent variable handling
        - 'causomic': Observable graph after marginalizing latent variables
        - 'Coefficients': Structural equation coefficients (if include_coef=True)

    Examples
    --------
    >>> # Standard backdoor confounding scenario
    >>> backdoor_graph = backdoor(include_coef=True)
    >>>
    >>> # Check latent variable marking
    >>> nx_graph = backdoor_graph['Networkx']
    >>> latent_nodes = [n for n, attr in nx_graph.nodes(data=True)
    ...                if attr.get('hidden', False)]
    >>> print(latent_nodes)  # ['C']
    >>>
    >>> # Use for confounding robustness testing
    >>> confounded_data = simulate_data(
    ...     backdoor_graph['Networkx'],
    ...     coefficients=backdoor_graph['Coefficients'],
    ...     n=1000
    ... )

    Notes
    -----
    Graph Structure:
    - C: Latent confounder (hidden=True)
    - B: Observable proxy affected by confounder
    - X: Treatment variable affected by B
    - Y: Outcome affected by both X and C (confounded)
    - Z: Downstream outcome affected by Y

    Confounding Pattern:
    - Direct confounding: C → Y creates bias in X → Y effect
    - Backdoor path: X ← B ← C → Y biases X-Y association
    - Adjustment strategy: Control for B to block backdoor path

    This structure tests:
    - Backdoor criterion implementation
    - Confounding adjustment methods
    - Proxy variable strategies
    - Causal identification algorithms

    The y0 representation automatically handles latent variable marginalization
    while causomic format shows the observable adjustment structure.
    """

    graph = nx.DiGraph()
    obs_nodes = ["B", "X", "Y", "Z"]
    all_nodes = ["B", "X", "Y", "Z", "C"]
    if add_independent_nodes:
        all_nodes = all_nodes + [f"I{i}" for i in range(1, n_ind + 1)]

    ## Add edges
    graph.add_edge("B", "X")
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    graph.add_edge("C", "B")
    graph.add_edge("C", "Y")

    if add_independent_nodes:
        for i in range(1, n_ind + 1):
            graph.add_node(f"I{i}")

    attrs = {
        node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes
    }

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    causomic_graph = NxMixedGraph()
    causomic_graph.add_directed_edge("B", "Y")
    causomic_graph.add_directed_edge("X", "Y")
    causomic_graph.add_directed_edge("Y", "Z")

    if include_coef:
        coef = {
            "C": {"intercept": 6, "error": 1.0},
            "B": {"intercept": 0, "error": 1.0, "C": 0.5},
            "X": {"intercept": 1, "error": 1.0, "B": 1.0},
            "Y": {"intercept": 1.6, "error": 0.25, "X": 0.5, "C": 0.5},
            "Z": {"intercept": -3, "error": 0.25, "Y": 1.0},
        }

        if add_independent_nodes:
            for i in range(1, n_ind + 1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), "error": 1.0}

    else:
        coef = None

    return {"Networkx": graph, "y0": y0_graph, "causomic": causomic_graph, "Coefficients": coef}


def frontdoor(include_coef: bool = True) -> Dict[str, Any]:
    """
    Generate frontdoor identification causal graph.

    Creates a frontdoor criterion structure where causal identification is
    possible through a mediator variable even when direct backdoor paths
    are blocked by latent confounding. This represents advanced causal
    identification scenarios requiring sophisticated inference strategies.

    Structure: C → {X, Z}, X → Y → Z
    Where C is a latent confounder that affects both X and Z directly,
    blocking standard backdoor adjustment. However, the mediator Y enables
    frontdoor identification of the X → Z causal effect.

    This graph is crucial for testing advanced causal identification
    algorithms that go beyond simple backdoor adjustment, particularly
    relevant in biological systems with complex confounding patterns.

    Parameters
    ----------
    include_coef : bool, default=True
        Whether to include structural equation coefficients for simulation.
        Coefficients are calibrated for realistic frontdoor effect sizes.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing multiple graph representations:
        - 'Networkx': NetworkX DiGraph with latent confounder marked
        - 'y0': y0.NxMixedGraph for advanced causal inference algorithms
        - 'causomic': Observable graph showing mediation structure
        - 'Coefficients': Structural equation coefficients (if include_coef=True)

    Examples
    --------
    >>> # Generate frontdoor identification scenario
    >>> frontdoor_graph = frontdoor(include_coef=True)
    >>>
    >>> # Analyze identification structure
    >>> from y0.algorithm.identify import identify
    >>> effect = identify(
    ...     frontdoor_graph['y0'],
    ...     treatments=['X'],
    ...     outcomes=['Z']
    ... )
    >>>
    >>> # Simulate data for frontdoor testing
    >>> fd_data = simulate_data(
    ...     frontdoor_graph['Networkx'],
    ...     coefficients=frontdoor_graph['Coefficients'],
    ...     n=2000  # Larger sample for frontdoor precision
    ... )

    Notes
    -----
    Graph Structure:
    - C: Latent confounder affecting X and Z (hidden=True)
    - X: Treatment variable confounded with Z
    - Y: Mediator variable (key for frontdoor identification)
    - Z: Outcome variable with direct confounding from C

    Identification Strategy:
    - Backdoor blocked: X ← C → Z prevents direct adjustment
    - Frontdoor available: X → Y → Z provides alternative pathway
    - Requires: Y fully mediates X → Z and Y-Z unconfounded given X

    Frontdoor Formula:
    P(Z|do(X)) = Σy P(Y=y|X) × Σx' P(Z|Y=y,X=x') × P(X=x')

    This structure tests:
    - Frontdoor criterion algorithms
    - Advanced causal identification methods
    - Mediation-based inference strategies
    - Robustness when backdoor adjustment fails

    The coefficient structure ensures identifiability conditions are met
    while maintaining realistic effect sizes for biological applications.
    """

    graph = nx.DiGraph()
    obs_nodes = ["X", "Y", "Z"]
    all_nodes = ["X", "Y", "Z", "C"]

    ## Add edges
    graph.add_edge("X", "Y")
    graph.add_edge("Y", "Z")
    graph.add_edge("C", "X")
    graph.add_edge("C", "Z")

    attrs = {
        node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes
    }

    nx.set_node_attributes(graph, attrs, name="hidden")

    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph)

    # TODO: replace with automated code
    causomic_graph = NxMixedGraph()
    causomic_graph.add_directed_edge("X", "Z")
    causomic_graph.add_directed_edge("X", "Y")
    causomic_graph.add_directed_edge("Y", "Z")

    if include_coef:
        coef = {
            "C": {"intercept": 6, "error": 1.0},
            "X": {"intercept": 1, "error": 1.0, "C": 0.5},
            "Y": {"intercept": 1.6, "error": 0.25, "X": 0.5},
            "Z": {"intercept": -3, "error": 0.25, "Y": 1.0, "C": 0.5},
        }

    else:
        coef = None

    return {"Networkx": graph, "y0": y0_graph, "causomic": causomic_graph, "Coefficients": coef}


def signaling_network(
    include_coef: bool = True, add_independent_nodes: bool = False, n_ind: int = 10
) -> Dict[str, Any]:
    """
    Generate realistic biological signaling network graph.

    Creates a complex signaling network based on well-characterized protein
    interactions in growth factor and stress response pathways. The network
    includes multiple input signals (EGF, IGF, TNF, Insulin), core signaling
    modules (RAS/RAF/MEK/ERK, PI3K/AKT), and downstream transcriptional
    outputs (NFKB, JNK).

    This network represents realistic biological complexity with:
    - Multiple convergent pathways
    - Feedback inhibition (AKT → RAF)
    - Cross-pathway interactions
    - Latent growth factor inputs
    - Observable protein measurements

    The structure is ideal for testing causal discovery algorithms on
    realistic biological networks with known ground truth relationships.

    Parameters
    ----------
    include_coef : bool, default=True
        Whether to include structural equation coefficients based on
        biological knowledge of pathway strengths and interactions
    add_independent_nodes : bool, default=False
        Whether to add independent noise variables to test robustness
        against irrelevant measurements common in proteomics data
    n_ind : int, default=10
        Number of independent nodes if add_independent_nodes=True

    Returns
    -------
    Dict[str, Any]
        Dictionary containing multiple graph representations:
        - 'Networkx': Full NetworkX DiGraph including latent growth factors
        - 'y0': y0.NxMixedGraph for causal inference with latent variables
        - 'causomic': Observable signaling network for proteomics analysis
        - 'Coefficients': Biologically-informed structural equation coefficients

    Examples
    --------
    >>> # Generate realistic signaling network
    >>> signal_net = signaling_network(
    ...     include_coef=True,
    ...     add_independent_nodes=True,
    ...     n_ind=20
    ... )
    >>>
    >>> # Simulate proteomics-like data
    >>> proteomics_data = simulate_data(
    ...     signal_net['Networkx'],
    ...     coefficients=signal_net['Coefficients'],
    ...     n=100,  # Typical proteomics sample size
    ...     mnar_missing_param=[-2, 0.3]  # Realistic missing data
    ... )
    >>>
    >>> # Test causal discovery on observable network
    >>> observable_graph = signal_net['causomic']
    >>> # Run causal discovery algorithms...

    Notes
    -----
    Network Components:

    Growth Factor Inputs (Latent):
    - EGF: Epidermal Growth Factor
    - IGF: Insulin-like Growth Factor
    - TNF: Tumor Necrosis Factor

    Core Signaling Proteins (Observable):
    - SOS: Son of Sevenless (adaptor protein)
    - RAS: RAS GTPase (central hub)
    - PI3K: Phosphoinositide 3-kinase
    - AKT: Protein kinase B
    - RAF: RAF kinase
    - MEK: MAP kinase kinase
    - ERK: Extracellular signal-regulated kinase

    Additional Components:
    - Insulin: Observable hormone input
    - NFKB: Nuclear factor kappa B (transcription factor)
    - JNK: c-Jun N-terminal kinase

    Key Biological Relationships:
    - Growth factor convergence: EGF, IGF → SOS, PI3K
    - Central integration: RAS coordinates multiple pathways
    - Negative feedback: AKT inhibits RAF (realistic regulation)
    - Cross-pathway communication: Multiple interaction points

    Coefficient Design:
    - Positive regulation: Typical range 0.5-1.5
    - Negative regulation: AKT → RAF = -0.4 (feedback inhibition)
    - Growth factors: Strong upstream effects (intercepts 5-6)
    - Noise levels: Realistic for biological measurements

    This network tests:
    - Complex pathway discovery
    - Feedback loop detection
    - Multi-input integration
    - Latent variable handling
    - Realistic biological constraints

    The network structure is based on canonical MAPK and PI3K pathway
    organization with additional complexity reflecting real biological systems.
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

    # Additional stuff
    graph.add_edge("Erk", "Nfkb")
    graph.add_edge("Erk", "Jnk")
    graph.add_edge("Ins", "Erk")
    graph.add_edge("Tnf", "Erk")
    graph.add_edge("Tnf", "Ins")

    if add_independent_nodes:
        for i in range(1, n_ind + 1):
            graph.add_node(f"I{i}")

    ## Define obs vs latent nodes
    all_nodes = [
        "SOS",
        "PI3K",
        "Ras",
        "Raf",
        "Akt",
        "Mek",
        "Erk",
        "EGF",
        "IGF",
        "Ins",
        "Tnf",
        "Nfkb",
        "Jnk",
    ]
    obs_nodes = ["SOS", "PI3K", "Ras", "Raf", "Akt", "Mek", "Erk", "Ins", "Nfkb", "Jnk"]
    if add_independent_nodes:
        all_nodes = all_nodes + [f"I{i}" for i in range(1, n_ind + 1)]

    attrs = {
        node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes
    }

    nx.set_node_attributes(graph, attrs, name="hidden")
    # Use y0 to build ADMG
    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(graph, "hidden")

    causomic_graph = NxMixedGraph()
    causomic_graph.add_directed_edge("SOS", "PI3K")
    causomic_graph.add_directed_edge("Ras", "PI3K")
    causomic_graph.add_directed_edge("Ras", "Raf")
    causomic_graph.add_directed_edge("PI3K", "Akt")
    causomic_graph.add_directed_edge("Akt", "Raf")
    causomic_graph.add_directed_edge("Raf", "Mek")
    causomic_graph.add_directed_edge("Mek", "Erk")

    if include_coef:
        coef = {
            "EGF": {"intercept": 6.0, "error": 1},
            "IGF": {"intercept": 5.0, "error": 1},
            "Tnf": {"intercept": 5.0, "error": 1},
            "SOS": {"intercept": 2, "error": 1, "EGF": 0.6, "IGF": 0.6},
            "Ras": {"intercept": 3, "error": 1, "SOS": 0.5},
            "PI3K": {"intercept": 0, "error": 1, "EGF": 0.5, "IGF": 0.5, "Ras": 0.5},
            "Akt": {"intercept": 1.0, "error": 1, "PI3K": 0.75},
            "Raf": {"intercept": 4, "error": 1, "Ras": 0.8, "Akt": -0.4},
            "Mek": {"intercept": 2.0, "error": 1, "Raf": 0.75},
            "Erk": {"intercept": -2, "error": 1, "Mek": 1.2, "Ins": -0.4, "Tnf": 1.5},
            "Ins": {"intercept": 1.0, "Tnf": 1.5, "error": 1},
            "Nfkb": {"intercept": 1.0, "Erk": 1.5, "error": 1},
            "Jnk": {"intercept": 1.0, "Erk": 1.75, "error": 1},
        }

        if add_independent_nodes:
            for i in range(1, n_ind + 1):
                coef[f"I{i}"] = {"intercept": np.random.uniform(-5, 5), "error": 1.0}

    else:
        coef = None

    return {"Networkx": graph, "y0": y0_graph, "causomic": causomic_graph, "Coefficients": coef}


def main() -> None:
    med = mediator(n_med=3)
    bd = backdoor()
    fd = frontdoor()
    sn = signaling_network()

    simulated_fd_data = simulate_data(
        sn["Networkx"],
        coefficients=sn["Coefficients"],
        mnar_missing_param=[-5, 0.4],
        add_feature_var=True,
        n=50,
        seed=2,
    )
    fd_data = dataProcess(
        simulated_fd_data["Feature_data"],
        normalization=False,
        summarization_method="TMP",
        MBimpute=False,
        sim_data=True,
    )
    # fd_data = fd_data.dropna(how="all",axis=1)
    print(fd_data.isna().mean() * 100)


if __name__ == "__main__":
    main()
