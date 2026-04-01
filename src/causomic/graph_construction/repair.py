from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators.CITests import pearsonr
from y0.graph import NxMixedGraph


def convert_to_y0_graph(posterior_dag):
    """
    Convert the posterior DAG to a y0 graph format.
    """

    # Confirm index is fine
    posterior_dag = posterior_dag.reset_index(drop=True)

    # Construct NetworkX DiGraph from posterior_dag
    all_nodes = set(posterior_dag["source"]).union(set(posterior_dag["target"]))

    nx_dag = nx.DiGraph()
    for i in range(len(posterior_dag)):
        nx_dag.add_edge(posterior_dag.loc[i, "source"], posterior_dag.loc[i, "target"])

    obs_nodes = all_nodes

    # Set all nodes as observed
    attrs = {
        node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes
    }
    nx.set_node_attributes(nx_dag, attrs, name="hidden")

    # Use y0 to build ADMG
    y0_graph = NxMixedGraph()
    y0_graph = y0_graph.from_latent_variable_dag(nx_dag, "hidden")

    return y0_graph


def process_failed_test(
    row: pd.Series, confounder_relations: dict, data: pd.DataFrame, max_conditional: int = 2
):
    """
    Process a single failed conditional independence test to identify potential confounders.

    This function attempts to repair confounding relationships by testing whether
    adding observed confounding variables can restore conditional independence
    between two variables. If successful, it returns the confounding variables
    that should be added to the causal graph. If unsuccessful, it indicates
    that a latent (unobserved) confounder should be considered.

    The function systematically tests combinations of potential confounders up to
    a maximum size, looking for a set that renders the source and target variables
    conditionally independent given the existing conditioning set plus the new
    confounders.

    Parameters
    ----------
    row : pd.Series or dict
        Row from failed conditional independence tests containing:
        - 'left': Source variable name (str)
        - 'right': Target variable name (str)
        - 'given': Existing conditioning variables (str, list, or empty)

    confounder_relations : dict
        Dictionary mapping (source, target) tuples to lists of potential
        confounder variable names extracted from biological databases.
        Format: {(source, target): [confounder1, confounder2, ...]}

    data : pd.DataFrame
        Observational data matrix where rows are samples and columns are variables.
        Must contain all variables referenced in row and confounder_relations.

    max_conditional : int
        Maximum number of confounding variables to test in combination.
        Higher values allow more complex confounding patterns but increase
        computational cost. Typical range: 1-3.

    Returns
    -------
    dict
        Dictionary containing repair results with keys:
        - 'source': Source variable name (str)
        - 'target': Target variable name (str)
        - 'add_latent': Whether to add latent confounder (bool)
        - 'Z': Confounding variables that restore independence (tuple or None)
        - 'error': Error message if exception occurred (str, optional)

        If add_latent=False and Z is not None, the variables in Z should be
        added as confounders in the causal graph with edges to both source
        and target. If add_latent=True, a bidirectional edge or latent
        confounder should be considered.

    Examples
    --------
    >>> # Example failed test row
    >>> failed_test = {'left': 'EGFR', 'right': 'ERK', 'given': 'MEK'}
    >>>
    >>> # Potential confounders from INDRA
    >>> confounders = {('EGFR', 'ERK'): ['AKT', 'PI3K', 'RAS']}
    >>>
    >>> # Process the failed test
    >>> result = process_failed_test(
    ...     failed_test, confounders, data, max_conditional=2
    ... )
    >>>
    >>> if not result['add_latent']:
    ...     print(f"Add confounders: {result['Z']}")
    ... else:
    ...     print("Add latent confounder")

    """

    try:
        # Create or reuse a client in this process
        source = row["left"]
        target = row["right"]
        given = row["given"]

        add_latent = False
        found_adjustment = False
        found_Z = None

        # build all non-empty confounder combos (kept same range as original: r in [1])
        confounders = confounder_relations[(source, target)]
        confounders = [i for i in confounders if i != given and i in data.columns]

        # sort by combined absolute correlation with source and target so the
        # most promising candidates are tested first, improving early termination
        if confounders:
            corr_scores = (
                data[confounders].corrwith(data[source]).abs()
                + data[confounders].corrwith(data[target]).abs()
            )
            confounders = corr_scores.sort_values(ascending=False).index.tolist()

        conf_list = list(confounders)
        all_combos = [
            combo for r in range(1, max_conditional + 1) for combo in combinations(conf_list, r)
        ]

        # normalize 'given' once
        if isinstance(given, (list, tuple, np.ndarray)):
            given_list = list(given)
        elif given is None or (isinstance(given, str) and given == "") or pd.isna(given):
            given_list = []
        else:
            given_list = [given]

        # no confounders → plan to add latent
        if not all_combos:
            return {"source": source, "target": target, "add_latent": True, "Z": None}

        # test combos; stop at first success
        for combo in all_combos:
            Z = given_list + list(combo)
            try:
                independent = pearsonr(source, target, Z, data, significance_level=0.05)
            except Exception:
                independent = False
            if independent:
                found_adjustment = True
                found_Z = combo
                break

        if found_adjustment:
            return {"source": source, "target": target, "add_latent": False, "Z": found_Z}
        else:
            return {"source": source, "target": target, "add_latent": True, "Z": None}

    except Exception as e:
        # On error be conservative: mark as latent confounding
        return {
            "source": row.get("left"),
            "target": row.get("right"),
            "add_latent": True,
            "Z": None,
            "error": str(e),
        }
