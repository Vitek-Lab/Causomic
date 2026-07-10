"""
Cyclic Causal Graph Simulation

This module extends the Causomic simulation framework to support causal graphs with
directed feedback cycles. Standard Causomic methods assume DAGs; this module enables
benchmarking of how much cycles degrade structure-learning and causal-inference methods.

Three cycle locations are supported, each controlled by a boolean flag:

- **Start-node cycle** (``add_cycle_in_start``): one start node (L0) anchors a loop
  through ``cycle_size - 1`` fresh mediator nodes, e.g. L0 → CYS0 → CYS1 → L0.
  The anchor retains all its existing downstream connections to the base DAG.

- **Mediator cycle** (``add_cycle_in_mediators``): ``cycle_size`` fresh mediator nodes
  form a self-contained loop (CYM0 → CYM1 → … → CYM0), connected to the main
  causal path via one in-edge from an upstream node and one out-edge to a downstream
  node.

- **End-node cycle** (``add_cycle_in_end``): one end node (R0) anchors a loop through
  ``cycle_size - 1`` fresh mediator nodes, e.g. R0 → CYE0 → CYE1 → R0.  The anchor
  retains all its existing upstream connections from the base DAG.

Data Simulation with Threshold Clamping
----------------------------------------
Cyclic graphs have no global topological order, but ``nx.condensation`` decomposes
any directed graph into a DAG over its strongly-connected components (SCCs).  Singleton
SCCs are handled by the standard ``simulate_node`` from ``proteomics_simulator``.
Multi-node SCCs (cycles) use iterative Jacobi updates: all cycle nodes are updated
simultaneously from the previous iteration's values.  When any sample's absolute value
for any cycle node exceeds ``threshold``, ALL cycle nodes for that sample are clamped
to ±threshold and frozen for the remainder of the iterations.  This models biological
saturation and prevents numerical divergence.

INDRA Priors
------------
The existing :func:`~causomic.simulation.random_network.generate_indra_data` works
directly with cyclic graphs — it iterates over edges without requiring acyclicity and
marks cycle edges ``ground_truth=True`` because they are part of the input graph.  No
wrapper is needed::

    indra_dag, indra_df, missing = generate_indra_data(cyclic_graph)
    # Cycle edges appear in indra_df with ground_truth=True.

Example
-------
>>> import networkx as nx
>>> from causomic.simulation.cyclic_network import generate_cyclic_graph, simulate_cyclic_data
>>> from causomic.simulation.random_network import generate_indra_data
>>>
>>> graph, roles = generate_cyclic_graph(
...     n_start=5, n_end=3,
...     add_cycle_in_start=0,
...     add_cycle_in_mediators=2,
...     add_cycle_in_end=2,
...     cycle_size=3,
...     seed=42,
... )
>>> assert not nx.is_directed_acyclic_graph(graph)
>>> print(roles['cycle_nodes'])
>>>
>>> sim = simulate_cyclic_data(graph, roles, n=200, threshold=20.0, seed=42,
...                            add_feature_var=False)
>>>
>>> # INDRA priors — reuse existing function directly
>>> indra_dag, indra_df, missing = generate_indra_data(graph)
"""

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import numpy.linalg as _npl
import pandas as pd

from causomic.simulation.proteomics_simulator import (
    add_missing,
    generate_coefficients,
    generate_features,
    simulate_node,
)
from causomic.simulation.random_network import generate_structured_dag


def generate_cyclic_graph(
    n_start: int = 3,
    n_end: int = 3,
    max_mediators: int = 3,
    add_cycle_in_start: int = 0,
    add_cycle_in_mediators: int = 0,
    add_cycle_in_end: int = 0,
    cycle_size: int = 2,
    mediator_cycle_prob: float = 0.5,
    confounder_prob: float = 0.0,
    shared_mediator_prob: float = 0.3,
    seed: Optional[int] = None,
) -> Tuple[nx.DiGraph, dict]:
    """
    Generate a causal graph with directed cycles injected at specified node roles.

    Builds a structured base DAG via :func:`generate_structured_dag`, then injects
    directed cycles at the requested locations.  At least one of the three
    ``add_cycle_in_*`` parameters must be non-zero.

    Parameters
    ----------
    n_start : int
        Number of start (ligand) nodes in the base DAG.  Named L0, L1, …
    n_end : int
        Number of end (readout) nodes in the base DAG.  Named R0, R1, …
    max_mediators : int
        Maximum mediator nodes per start-to-end path in the base DAG.
    add_cycle_in_start : int
        Number of cycles to inject at start nodes.  Cycle *k* is anchored at
        ``L{k}``.  A fraction ``mediator_cycle_prob`` of cycles route through a
        randomly chosen existing mediator node as the first bridge; the
        remainder use all-fresh nodes named ``CYS{k}_{i}``.  The anchor keeps
        all its existing outgoing edges.  Requires
        ``n_start >= add_cycle_in_start``.  ``True`` is treated as 1.
    add_cycle_in_mediators : int
        Number of independent pure-mediator cycles to inject.  Cycle *k*
        consists of ``cycle_size`` fresh nodes ``CYM{k}_{i}`` forming a loop,
        attached to the main graph via one in-edge from a random upstream node
        and one out-edge to a random downstream node.  ``True`` is treated as 1.
    add_cycle_in_end : int
        Number of cycles to inject at end nodes.  Cycle *k* is anchored at
        ``R{k}``.  A fraction ``mediator_cycle_prob`` of cycles route through a
        randomly chosen existing mediator node as the last bridge (so
        ``M → R{k}`` closes the loop); the remainder use all-fresh nodes named
        ``CYE{k}_{i}``.  The anchor keeps all its existing incoming edges.
        Requires ``n_end >= add_cycle_in_end``.  ``True`` is treated as 1.
    cycle_size : int
        Total number of nodes in each individual cycle (must be ≥ 2).  For
        start/end cycles, one of those nodes is the existing anchor; the rest
        are fresh mediators.
    mediator_cycle_prob : float
        Fraction of start/end cycles that route through an existing mediator
        node instead of using all-fresh bridge nodes.  Must be in [0, 1].
        Default 0.5 means half of the cycles involve existing mediators.
        Has no effect on pure mediator cycles (``add_cycle_in_mediators``).
    confounder_prob : float
        Fraction of observable nodes in the base DAG to add as confounders.
    shared_mediator_prob : float
        Probability of reusing an existing mediator at each slot in the base DAG.
    seed : int or None
        Random seed.  Passed to both ``generate_structured_dag`` and the local
        RNG used for attachment decisions.

    Returns
    -------
    graph : nx.DiGraph
        Causal graph with injected cycles.  Guaranteed to be non-acyclic when
        at least one cycle is enabled.
    node_roles : dict
        Keys ``'start'``, ``'end'``, ``'mediators'``, ``'confounders'`` carry
        the same node lists as ``generate_structured_dag``.  An additional key
        ``'cycle_nodes'`` holds a sub-dict keyed by enabled location
        (``'start'``, ``'mediators'``, ``'end'``), each mapping to a **flat**
        list of all nodes that participate in cycles at that location (anchors
        and fresh mediators from every injected cycle, in injection order).

    Raises
    ------
    ValueError
        If no cycle location is enabled, ``cycle_size < 2``, or ``n_start`` /
        ``n_end`` is too small to anchor the requested number of cycles.
    """
    n_start_cycles = int(add_cycle_in_start)
    n_med_cycles = int(add_cycle_in_mediators)
    n_end_cycles = int(add_cycle_in_end)

    if not (n_start_cycles or n_med_cycles or n_end_cycles):
        raise ValueError(
            "At least one of add_cycle_in_start, add_cycle_in_mediators, "
            "or add_cycle_in_end must be non-zero."
        )
    if cycle_size < 2:
        raise ValueError(f"cycle_size must be >= 2, got {cycle_size}.")
    if n_start_cycles > n_start:
        raise ValueError(
            f"add_cycle_in_start={n_start_cycles} requires n_start >= {n_start_cycles}."
        )
    if n_end_cycles > n_end:
        raise ValueError(f"add_cycle_in_end={n_end_cycles} requires n_end >= {n_end_cycles}.")

    if seed is not None:
        np.random.seed(seed)

    graph, node_roles = generate_structured_dag(
        n_start=n_start,
        n_end=n_end,
        max_mediators=max_mediators,
        confounder_prob=confounder_prob,
        shared_mediator_prob=shared_mediator_prob,
        seed=seed,
    )

    node_roles["cycle_nodes"] = {}
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Start-node cycles: L{k} → … → L{k}
    # First ceil(n/2) cycles route through an existing mediator node;
    # remaining cycles use all-fresh bridge nodes.
    # ------------------------------------------------------------------
    if n_start_cycles:
        all_cy_nodes: List[str] = []
        med_pool = node_roles["mediators"]
        n_med_routed = round(n_start_cycles * mediator_cycle_prob)

        for k in range(n_start_cycles):
            anchor = node_roles["start"][k]
            use_existing_med = k < n_med_routed and len(med_pool) > 0

            if use_existing_med:
                med = med_pool[int(rng.integers(len(med_pool)))]
                if cycle_size == 2:
                    graph.add_edge(anchor, med)
                    graph.add_edge(med, anchor)
                    cycle_node_list = [anchor, med]
                else:
                    fresh = [f"CYS{k}_{i}" for i in range(cycle_size - 2)]
                    graph.add_nodes_from(fresh)
                    chain = [anchor, med] + fresh + [anchor]
                    for u, v in zip(chain[:-1], chain[1:]):
                        graph.add_edge(u, v)
                    cycle_node_list = [anchor, med] + fresh
            else:
                cy_mediators = [f"CYS{k}_{i}" for i in range(cycle_size - 1)]
                chain = [anchor] + cy_mediators + [anchor]
                graph.add_nodes_from(cy_mediators)
                for u, v in zip(chain[:-1], chain[1:]):
                    graph.add_edge(u, v)
                cycle_node_list = [anchor] + cy_mediators

            all_cy_nodes.extend(cycle_node_list)
        node_roles["cycle_nodes"]["start"] = all_cy_nodes

    # ------------------------------------------------------------------
    # Mediator cycles: CYM{k}_0 → … → CYM{k}_0, attached to main path
    # ------------------------------------------------------------------
    if n_med_cycles:
        all_cy_nodes = []
        for k in range(n_med_cycles):
            cy_nodes = [f"CYM{k}_{i}" for i in range(cycle_size)]
            chain = cy_nodes + [cy_nodes[0]]
            graph.add_nodes_from(cy_nodes)
            for u, v in zip(chain[:-1], chain[1:]):
                graph.add_edge(u, v)

            upstream = node_roles["start"] + node_roles["mediators"]
            if upstream:
                src = upstream[int(rng.integers(len(upstream)))]
                graph.add_edge(src, cy_nodes[0])

            downstream = node_roles["mediators"] + node_roles["end"]
            if downstream:
                dst = downstream[int(rng.integers(len(downstream)))]
                graph.add_edge(cy_nodes[-1], dst)

            all_cy_nodes.extend(cy_nodes)
        node_roles["cycle_nodes"]["mediators"] = all_cy_nodes

    # ------------------------------------------------------------------
    # End-node cycles: R{k} → … → R{k}
    # First ceil(n/2) cycles route through an existing mediator node
    # (placed last in the chain so M → R{k} closes the loop);
    # remaining cycles use all-fresh bridge nodes.
    # ------------------------------------------------------------------
    if n_end_cycles:
        all_cy_nodes = []
        med_pool = node_roles["mediators"]
        n_med_routed = round(n_end_cycles * mediator_cycle_prob)

        for k in range(n_end_cycles):
            anchor = node_roles["end"][k]
            use_existing_med = k < n_med_routed and len(med_pool) > 0

            if use_existing_med:
                med = med_pool[int(rng.integers(len(med_pool)))]
                if cycle_size == 2:
                    graph.add_edge(anchor, med)
                    graph.add_edge(med, anchor)
                    cycle_node_list = [anchor, med]
                else:
                    fresh = [f"CYE{k}_{i}" for i in range(cycle_size - 2)]
                    graph.add_nodes_from(fresh)
                    chain = [anchor] + fresh + [med, anchor]
                    for u, v in zip(chain[:-1], chain[1:]):
                        graph.add_edge(u, v)
                    cycle_node_list = [anchor] + fresh + [med]
            else:
                cy_mediators = [f"CYE{k}_{i}" for i in range(cycle_size - 1)]
                chain = [anchor] + cy_mediators + [anchor]
                graph.add_nodes_from(cy_mediators)
                for u, v in zip(chain[:-1], chain[1:]):
                    graph.add_edge(u, v)
                cycle_node_list = [anchor] + cy_mediators

            all_cy_nodes.extend(cycle_node_list)
        node_roles["cycle_nodes"]["end"] = all_cy_nodes

    assert not nx.is_directed_acyclic_graph(graph), (
        "generate_cyclic_graph produced a DAG — cycle injection failed. " "This is a bug."
    )

    return graph, node_roles


def _simulate_cycle_scc(
    graph: nx.DiGraph,
    members: set,
    data: dict,
    coefficients: dict,
    n: int,
    threshold: float,
    max_iterations: int,
) -> None:
    """
    Simulate one strongly-connected component (cycle) via iterative Jacobi updates.

    Upstream (non-cycle) parents must already be present in ``data``.
    Results are written into ``data`` in-place.

    The update rule mirrors ``simulate_node``: each cycle node is computed as::

        val = intercept + Σ coef[parent] * (parent_vals - parent_vals.mean()) + ε

    where cycle parents use values from the *previous* iteration (Jacobi scheme)
    and non-cycle parents use their already-settled values from ``data``.

    Per-sample threshold clamping: at each iteration, any sample whose absolute
    value for any cycle node reaches ``threshold`` has ALL its cycle-node values
    clamped to ±threshold and is frozen for all subsequent iterations.

    Parameters
    ----------
    graph : nx.DiGraph
        Full causal graph.
    members : set of str
        Node names belonging to this SCC.
    data : dict
        Simulation state; maps node name → np.ndarray of shape (n,).
        Modified in-place.
    coefficients : dict
        SEM coefficient dict for all graph nodes.
    n : int
        Number of samples.
    threshold : float
        Absolute-value saturation threshold.
    max_iterations : int
        Maximum Jacobi update steps before halting.
    """
    _non_coef = {"intercept", "error", "cell_type"}

    # Initialise every cycle node at its intercept.  Because all n samples share
    # the same starting value, the mean-centred cycle contribution is exactly 0
    # in the first iteration — a clean, unbiased initialisation.
    current = {node: np.full(n, coefficients[node]["intercept"], dtype=float) for node in members}
    frozen = np.zeros(n, dtype=bool)

    for _ in range(max_iterations):
        new_vals: Dict[str, np.ndarray] = {}

        for node in members:
            coefs = coefficients[node]
            parents = [k for k in coefs if k not in _non_coef]

            val = np.full(n, coefs["intercept"], dtype=float)
            for parent in parents:
                # Cycle parents: use previous-iteration values (Jacobi update).
                # Upstream parents: already settled values from data[].
                parent_vals = current[parent] if parent in members else data[parent]
                val += coefs[parent] * (parent_vals - parent_vals.mean())

            val += np.random.normal(0, coefs["error"], n)
            new_vals[node] = val

        # Per-sample: if any cycle node exceeds threshold, freeze that sample.
        exceeded = np.zeros(n, dtype=bool)
        for node in members:
            exceeded |= np.abs(new_vals[node]) >= threshold

        frozen |= exceeded

        # Clamp ALL cycle nodes for every frozen sample.
        for node in members:
            new_vals[node] = np.where(
                frozen,
                np.clip(new_vals[node], -threshold, threshold),
                new_vals[node],
            )

        current = new_vals

        if frozen.all():
            break

    for node in members:
        data[node] = current[node]


def simulate_cyclic_data(
    graph: nx.DiGraph,
    node_roles: dict,
    coefficients: Optional[Dict] = None,
    threshold: float = 20.0,
    max_iterations: int = 100,
    n: int = 1000,
    seed: Optional[int] = None,
    add_feature_var: bool = True,
    include_missing: bool = True,
    mar_missing_param: float = 0.05,
    mnar_missing_param: List[float] = [-3, 0.4],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Simulate data from a causal graph that may contain directed cycles.

    Nodes are processed via :func:`nx.condensation`, which collapses each
    strongly-connected component (SCC) to a single node and returns a DAG over
    SCCs.  Singleton SCCs (ordinary DAG nodes) are handled by the standard
    ``simulate_node``.  Multi-node SCCs (cycles) are handled by
    :func:`_simulate_cycle_scc`.

    Parameters
    ----------
    graph : nx.DiGraph
        Causal graph, typically from :func:`generate_cyclic_graph`.
        May contain directed cycles.
    node_roles : dict
        Node-role dictionary as returned by :func:`generate_cyclic_graph`.
    coefficients : dict or None
        SEM coefficients.  If ``None``, generated automatically via
        :func:`generate_coefficients`.
        Format: ``{node: {parent: coef, 'intercept': val, 'error': var}}``.
    threshold : float
        Absolute-value saturation threshold for cycle nodes.  When any cycle
        node's value for sample *i* reaches this level, all cycle nodes for
        sample *i* are clamped to ±threshold and frozen.
    max_iterations : int
        Maximum Jacobi update steps for each cycle SCC.
    n : int
        Number of samples to simulate.
    seed : int or None
        Random seed for reproducibility.
    add_feature_var : bool
        Whether to generate feature-level (peptide) measurements with
        technical noise.
    include_missing : bool
        Whether to apply MAR/MNAR missing-data patterns to feature-level data.
        Ignored when ``add_feature_var=False``.
    mar_missing_param : float
        Probability of Missing At Random (random technical failure).
    mnar_missing_param : list of float
        ``[intercept, slope]`` for the MNAR logistic detection model.

    Returns
    -------
    dict
        - ``'Protein_data'``: ``dict`` mapping each node name to a
          ``np.ndarray`` of shape ``(n,)``.
        - ``'Feature_data'``: ``pd.DataFrame`` (or ``None`` if
          ``add_feature_var=False``), formatted identically to
          :func:`~causomic.simulation.proteomics_simulator.simulate_data`.
        - ``'Coefficients'``: SEM coefficient dict used for the simulation.

    Notes
    -----
    Because cycle nodes are processed as a unit (via SCC condensation), all
    upstream non-cycle parents are settled before any cycle node is first
    updated.  This ensures that mean-centering within the iterative loop uses
    the correct observational baselines.

    Cycle nodes whose values are clamped at ``threshold`` will produce
    distributions with a spike at ±threshold.  Downstream nodes that receive
    clamped cycle-node values as parents will also be affected.  This is
    intentional — it faithfully represents the saturation behaviour of a
    thresholded feedback loop.
    """
    if seed is not None:
        np.random.seed(seed)

    if coefficients is None:
        coefficients = generate_coefficients(graph)

    data: dict = {}

    # Condense graph into a DAG over SCCs for correct processing order.
    condensation = nx.condensation(graph)

    if verbose:
        print("simulating cyclic data...")
    for scc_idx in nx.topological_sort(condensation):
        members: set = condensation.nodes[scc_idx]["members"]

        if len(members) == 1:
            node = next(iter(members))
            if node not in coefficients:
                # Node has no SEM entry (e.g. added by generate_indra_data);
                # skip so it doesn't block downstream nodes.
                continue
            data[node] = simulate_node(data, coefficients[node], n, False, None, node)
        else:
            _simulate_cycle_scc(graph, members, data, coefficients, n, threshold, max_iterations)

    feature_level_data: Optional[pd.DataFrame] = None
    if add_feature_var:
        if verbose:
            print("adding feature level data...")
        all_nodes = [node for node in graph.nodes() if node in data and node != "Output"]
        feature_list = [generate_features(data[node], node) for node in all_nodes]
        feature_level_data = pd.concat(feature_list, ignore_index=True)

        if verbose:
            print("masking data...")
        if include_missing:
            feature_level_data = add_missing(
                feature_level_data, mar_missing_param, mnar_missing_param
            )

    return {
        "Protein_data": data,
        "Feature_data": feature_level_data,
        "Coefficients": coefficients,
    }


def ground_truth_interventional_effect_cyclic(
    graph: nx.DiGraph,
    coefficients: dict,
    intervention_nodes: dict,
    output_nodes: list,
) -> dict:
    """
    Compute the ground-truth expected interventional effect for a graph that
    may contain directed cycles.

    Mirrors :func:`~causomic.simulation.random_network.ground_truth_interventional_effect`
    but handles cycles via two mechanisms:

    1. **Do-operator edge cutting**: a copy of the graph has every incoming edge
       to each intervened node removed.  This often breaks cycles that include
       intervened nodes, reducing those SCCs to singletons in the modified graph.

    2. **Linear system for remaining cycles**: if a multi-node SCC survives in
       the modified graph (a cycle whose members are all non-intervened but that
       receives a changed signal from upstream), the steady-state deviations
       satisfy ``Δ = A Δ + b``, i.e. ``(I - A) Δ = b``, which is solved exactly.
       The system has a unique solution when the spectral radius of ``A`` is < 1
       (guaranteed when all cycle-edge coefficients are in the [-0.75, 0.75]
       range used by :func:`~causomic.simulation.proteomics_simulator.generate_coefficients`).

    Parameters
    ----------
    graph : nx.DiGraph
        Causal graph, typically from :func:`generate_cyclic_graph`.
        May contain directed cycles.
    coefficients : dict
        SEM coefficients as returned in the ``'Coefficients'`` key of
        :func:`simulate_cyclic_data`.
        Format: ``{node: {parent: coef, 'intercept': val, 'error': var}}``.
    intervention_nodes : dict
        Mapping of node name → interventional value, e.g. ``{'L0': 30.0}``.
    output_nodes : list
        Names of the nodes whose post-intervention expected values are of
        interest, e.g. ``['R0', 'R1']``.

    Returns
    -------
    dict
        Keys:
        - ``'baseline'``: ``{node: E[node]}`` under observation (= intercept).
        - ``'interventional'``: ``{node: E[node | do(...)]}`` for every node.
        - ``'effect'``: ``{node: E[node | do(...)] - E[node]}`` for each output node.

    Raises
    ------
    numpy.linalg.LinAlgError
        If the linear system for a cycle SCC is singular (spectral radius ≥ 1).
    """
    _non_coef = {"intercept", "error", "cell_type"}

    # Observational baseline: every node's expected value equals its intercept
    # because simulate_node mean-centers parent values.
    baseline = {
        node: coefficients[node]["intercept"] for node in graph.nodes() if node in coefficients
    }

    # Apply do-operator: cut all incoming edges to intervened nodes.
    modified = graph.copy()
    for node in intervention_nodes:
        for pred in list(modified.predecessors(node)):
            modified.remove_edge(pred, node)

    # Re-condense the modified graph. Cutting edges may break cycles that
    # contained intervened nodes, so their SCCs collapse to singletons.
    condensation = nx.condensation(modified)

    delta: dict = {}  # deviation from baseline under the intervention

    for scc_idx in nx.topological_sort(condensation):
        members: set = condensation.nodes[scc_idx]["members"]

        if len(members) == 1:
            node = next(iter(members))
            if node not in coefficients:
                continue
            if node in intervention_nodes:
                delta[node] = intervention_nodes[node] - baseline[node]
            else:
                coefs = coefficients[node]
                parents = [k for k in coefs if k not in _non_coef]
                node_delta = 0.0
                for parent in parents:
                    if modified.has_edge(parent, node) and parent in delta:
                        node_delta += coefs[parent] * delta[parent]
                delta[node] = node_delta

        else:
            # Multi-node SCC: none of these nodes are intervened (intervention
            # edges were cut above, so intervened nodes can't form cycles here).
            # Solve (I - A) * delta_vec = b.
            member_list = list(members)
            idx = {n: i for i, n in enumerate(member_list)}
            m = len(member_list)

            A = np.zeros((m, m))
            b = np.zeros(m)

            for node in member_list:
                if node not in coefficients:
                    continue
                i = idx[node]
                coefs = coefficients[node]
                parents = [k for k in coefs if k not in _non_coef]
                for parent in parents:
                    if not modified.has_edge(parent, node):
                        continue
                    if parent in members:
                        A[i, idx[parent]] = coefs[parent]
                    elif parent in delta:
                        b[i] += coefs[parent] * delta[parent]

            delta_vec = _npl.solve(np.eye(m) - A, b)
            for node in member_list:
                delta[node] = delta_vec[idx[node]]

    interventional = {
        node: baseline[node] + delta.get(node, 0.0) for node in graph.nodes() if node in baseline
    }
    effect = {node: interventional[node] - baseline[node] for node in output_nodes}

    return {"baseline": baseline, "interventional": interventional, "effect": effect}
