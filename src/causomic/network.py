"""
Network estimation module for causal inference using Bayesian network learning.

This module provides functionality for estimating posterior directed acyclic graphs (DAGs)
from observational data combined with prior knowledge from biological networks. It uses
bootstrap sampling to quantify uncertainty in the learned network structure and returns
edges that meet a specified probability threshold.

The main workflow involves:
1. Extracting prior knowledge from INDRA biological databases using multi-step queries
2. Running bootstrap sampling on the data with prior knowledge constraints
3. Aggregating edge counts across bootstrap samples
4. Computing edge probabilities
5. Filtering edges based on probability threshold

Key Functions:
    - extract_indra_prior: Query INDRA databases for biological pathway information
    - estimate_posterior_dag: Learn causal network structure using bootstrap sampling

Dependencies:
    - pandas: For data manipulation and DataFrame operations
    - numpy: For array operations and data processing
    - pgmpy: For Bayesian network structures and expert knowledge
    - collections.Counter: For efficient edge counting
    - indra_cogex.client: For querying INDRA biological knowledge graphs
    - causomic.graph_construction: Custom modules for prior data reconciliation and utilities
"""

import os
import copy
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from indra_cogex.client import Neo4jClient

# Parallel processing and progress tracking
from joblib import Parallel, delayed
from pgmpy.estimators import ExpertKnowledge
from sklearn.impute import KNNImputer
from tqdm import tqdm
from y0.algorithm.falsification import get_graph_falsifications
from y0.dsl import Variable
from y0.graph import NxMixedGraph

from causomic.graph_construction.neo4j_indra_queries import format_query_results, get_ids
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    SparseHillClimb,
    calculate_edge_probabilities,
    run_bootstrap,
)
from causomic.graph_construction.repair import convert_to_y0_graph, process_failed_test
from causomic.graph_construction.utils_neo4j import (
    get_one_step_root_down,
    get_three_step_root,
    get_two_step_root_known_med,
    query_confounder_relationships,
)
from causomic.graph_construction.utils_nx import query_confounders


def extract_indra_prior(
    source: list,
    target: list,
    measured_proteins: list,
    client: Neo4jClient,
    one_step_evidence: int = 1,
    two_step_evidence: int = 1,
    three_step_evidence: int = 3,
    confounder_evidence: int = 10,
) -> pd.DataFrame:
    """
    Extract prior biological knowledge from INDRA databases using multi-step pathway queries.

    This function queries the INDRA knowledge graph to extract causal relationships
    between source proteins, target proteins, and measured proteins. It performs
    queries at different path lengths (1-3 steps) and different evidence thresholds
    to build a comprehensive prior network for causal inference.

    Parameters
    ----------
    source : list of str
        List of source protein names (e.g., ['EGFR', 'IGF1']). These represent
        the upstream regulators or treatment conditions in the causal model.

    target : list of str
        List of target protein names (e.g., ['MEK', 'ERK', 'MAPK']). These represent
        the downstream outcomes or endpoints of interest.

    measured_proteins : list of str
        List of all measured protein names in the dataset. Used to constrain
        queries to only include relationships between measured variables.

    client : Neo4jClient
        Authenticated INDRA Neo4j client for querying the biological knowledge graph.
        Should be initialized with proper credentials and database URL.

    one_step_evidence : int, optional
        Minimum evidence count threshold for direct (1-step) relationships.
        Lower values include more relationships but with less evidence support.
        Default is 1.

    two_step_evidence : int, optional
        Minimum evidence count threshold for 2-step relationships (source -> mediator -> target).
        Default is 1.

    three_step_evidence : int, optional
        Minimum evidence count threshold for 3-step relationships.
        Higher threshold due to increased uncertainty in longer paths.
        Default is 3.

    confounder_evidence : int, optional
        Minimum evidence count threshold for relationships between confounding variables.
        Higher threshold to focus on well-supported confounding relationships.
        Default is 10.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the extracted prior network with columns:
        - 'source': Source protein name (gene symbol)
        - 'target': Target protein name (gene symbol)
        - 'evidence_count': Total evidence count supporting this relationship

        Protein names have hyphens removed for consistency with data formatting.

    Notes
    -----
    - Queries are restricted to "IncreaseAmount" and "DecreaseAmount" relationships
    - Evidence counts are summed across multiple query results for the same edge
    - Confounder relationships are identified among all proteins in the network
    - The function prints summary statistics of extracted relationships

    Examples
    --------
    >>> from indra_cogex.client import Neo4jClient
    >>> import os
    >>>
    >>> # Initialize INDRA client
    >>> client = Neo4jClient(
    ...     url=os.getenv("API_URL"),
    ...     auth=("neo4j", os.getenv("PASSWORD"))
    ... )
    >>>
    >>> # Extract prior network
    >>> priors = extract_indra_prior(
    ...     source=['EGFR', 'IGF1'],
    ...     target=['MEK', 'ERK'],
    ...     measured_proteins=['EGFR', 'IGF1', 'MEK', 'ERK', 'AKT'],
    ...     client=client,
    ...     one_step_evidence=2,
    ...     two_step_evidence=2,
    ...     three_step_evidence=5
    ... )
    """

    # Query one-step relationships: direct connections between source and target
    one_step_relations = format_query_results(
        get_one_step_root_down(
            root_nodes=get_ids(source, "gene"),
            downstream_nodes=get_ids(target, "gene"),
            client=client,
            relation=["IncreaseAmount", "DecreaseAmount"],
            minimum_evidence_count=one_step_evidence,
        )
    )

    # Query two-step relationships: source -> mediator -> target
    two_step_relations = format_query_results(
        get_two_step_root_known_med(
            root_nodes=get_ids(source, "gene"),
            downstream_nodes=get_ids(target, "gene"),
            client=client,
            relation=["IncreaseAmount", "DecreaseAmount"],
            minimum_evidence_count=two_step_evidence,
            mediators=get_ids(measured_proteins, "gene"),
        )
    )

    # Query three-step relationships: source -> med1 -> med2 -> target
    three_step_relations = format_query_results(
        get_three_step_root(
            root_nodes=get_ids(source, "gene"),
            downstream_nodes=get_ids(target, "gene"),
            client=client,
            relation=["IncreaseAmount", "DecreaseAmount"],
            minimum_evidence_count=three_step_evidence,
            mediators=get_ids(measured_proteins, "gene"),
        )
    )

    # Combine initial relationship queries
    all_relations = pd.concat(
        [one_step_relations, two_step_relations, three_step_relations], ignore_index=True
    )
    all_network_nodes = pd.unique(all_relations[["source", "target"]].values.ravel())

    # Query confounder relationships among all discovered network nodes
    # confounder_relations = format_query_results(
    #     query_confounder_relationships(
    #         get_ids(all_network_nodes, "gene"),
    #         client, minimum_evidence_count=confounder_evidence,
    #         mediators=get_ids(measured_proteins, "gene"))
    # )
    # confounder_relations = confounder_relations[
    #     confounder_relations["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]

    # # Remove duplicate confounder relationships
    # confounder_relations = confounder_relations.drop_duplicates(
    #     subset=["source", "target", "relation", "evidence_count"])

    # Combine all relationship types into final network
    all_relations = pd.concat(
        [one_step_relations, two_step_relations, three_step_relations], ignore_index=True
    )
    all_network_nodes = pd.unique(all_relations[["source", "target"]].values.ravel())

    # Extract relevant columns and aggregate evidence counts
    prior_network = all_relations.loc[:, ["source", "target", "evidence_count"]]

    # Sum evidence counts for duplicate edges (same source-target pair)
    prior_network = prior_network.groupby(["source", "target"], as_index=False)[
        "evidence_count"
    ].sum()

    # Clean protein names by removing hyphens for consistency
    prior_network["source"] = prior_network["source"].str.replace("-", "")
    prior_network["target"] = prior_network["target"].str.replace("-", "")

    # Print summary statistics
    print(f"Number of proteins pulled: {len(all_network_nodes)}")
    print(f"Number of reconciled edges pulled: {len(prior_network)}")

    return prior_network


def consensus_dag(bootstrap_dags, indra_priors, lam=0.25, min_freq=0.5):

    # build edge priors dict from indra_priors DataFrame
    df = indra_priors.copy()
    df["source"] = df["source"].astype(str).str.replace("-", "")
    df["target"] = df["target"].astype(str).str.replace("-", "")

    edge_priors = {(row["source"], row["target"]): row["edge_p"] for _, row in df.iterrows()}

    counts = Counter()
    total = 0

    for dag in bootstrap_dags:
        if dag is None:
            continue
        counts.update(list(dag.edges()))
        total += 1

    G = nx.DiGraph()
    for dag in bootstrap_dags:
        if dag:
            G.add_nodes_from(dag.nodes())

    # def weight(edge):
    #     f = counts[edge] / max(total, 1)
    #     return f
    def weight(edge):
        f = counts[edge] / max(total, 1)
        p = np.clip(edge_priors.get(edge, 0.5), 1e-6, 1 - 1e-6)
        return f + lam * np.log(p / (1 - p))

    candidates = [e for e, c in counts.items() if c / max(total, 1) >= min_freq]

    candidates.sort(key=weight, reverse=True)
    for u, v in candidates:
        G.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(u, v)
    return G


def estimate_posterior_dag(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float = 5.0,
    scoring_function: type = BICGaussIndraPriors,
    search_algorithm: type = SparseHillClimb,
    n_bootstrap: int = 100,
    add_high_corr_edges_to_priors: bool = False,
    corr_threshold: float = 0.95,
    edge_probability: float = 0.5,
    convert_to_probability: bool = True,
) -> NxMixedGraph:
    """
    Estimate a posterior directed acyclic graph (DAG) using bootstrap sampling.

    This function combines observational data with prior biological knowledge to learn
    a causal network structure. It uses bootstrap resampling to quantify uncertainty
    in the learned edges and returns only those edges that appear with sufficient
    frequency across bootstrap samples. The function automatically creates expert
    knowledge constraints by forbidding edges not present in the prior network.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data matrix where rows are samples and columns are variables.
        Should contain numeric values for all variables in the network.
        Column names should match protein names in indra_priors.

    indra_priors : pd.DataFrame
        Prior knowledge about causal relationships extracted from INDRA databases.
        Should contain columns: 'source', 'target', 'evidence_count'.
        Typically generated using the extract_indra_prior function.

    prior_strength : float, optional
        Weight given to prior knowledge relative to data. Higher values give more
        importance to the priors, while lower values rely more heavily on the data.
        Default is 5.0. Typical range is 0.1 to 10.0.

    scoring_function : type, optional
        Class implementing the scoring function for evaluating DAG quality.
        Default is BICGaussIndraPriors which incorporates INDRA prior information.
        Other options include standard BIC or BDeu scores.

    search_algorithm : type, optional
        Class implementing the structure learning algorithm for DAG search.
        Default is SparseHillClimb which is optimized for sparse biological networks.
        Other options include standard hill climbing or genetic algorithms.

    n_bootstrap : int, optional
        Number of bootstrap samples to generate. Higher values provide more
        stable estimates but increase computational cost. Default is 100.
        Typical range: 50-1000.

    edge_probability : float, optional
        Minimum probability threshold for including edges in the final network.
        Edges appearing in fewer than this fraction of bootstrap samples are
        excluded. Default is 0.5 (50% threshold).
    
    convert_to_probability : bool, optional
        Whether to convert edge counts to probabilities before thresholding. Default is True.

    Returns
    -------
    NxMixedGraph
        y0 graph object representing the posterior DAG edges.

    Examples
    --------
    >>> import pandas as pd
    >>> from indra_cogex.client import Neo4jClient
    >>>
    >>> # Load your data
    >>> data = pd.read_csv('expression_data.csv')
    >>>
    >>> # Extract priors from INDRA
    >>> client = Neo4jClient(url=api_url, auth=("neo4j", password))
    >>> priors = extract_indra_prior(
    ...     source=['EGFR'], target=['ERK'],
    ...     measured_proteins=data.columns.tolist(), client=client
    ... )
    >>>
    >>> # Estimate network
    >>> posterior_dag = estimate_posterior_dag(
    ...     data=data,
    ...     indra_priors=priors,
    ...     prior_strength=5.0,
    ...     n_bootstrap=100,
    ...     edge_probability=0.8
    ... )

    Notes
    -----
    - The function automatically creates expert knowledge constraints by forbidding
      all edges not present in the INDRA prior network
    - Protein names are cleaned by removing hyphens for consistency
    - Higher edge_probability thresholds result in sparser but more confident networks
    - Computational complexity scales with n_bootstrap and the size of the search space
    - Failed bootstrap runs (returning None) are excluded from probability calculations
    """

    # Extract unique nodes from prior network and clean names
    nodes = pd.unique(indra_priors[["source", "target"]].values.ravel())
    nodes = np.array([node.replace("-", "") for node in nodes])

    # Generate all possible edges between nodes
    all_possible_edges = [
        (u.replace("-", ""), v.replace("-", "")) for u in nodes for v in nodes if u != v
    ]

    # Extract observed edges from prior network
    obs_edges = [
        (
            indra_priors.loc[i, "source"].replace("-", ""),
            indra_priors.loc[i, "target"].replace("-", ""),
        )
        for i in range(len(indra_priors))
    ]

    # Define forbidden edges as all edges not in the prior network
    forbidden_edges = [edge for edge in all_possible_edges if edge not in obs_edges]

    # Create expert knowledge object with forbidden edges constraint
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

    # Remove hyphens from data column names
    data.columns = [str(col).replace("-", "") for col in data.columns]

    # Verify that every node from the priors appears in the data columns
    missing_nodes = [str(n) for n in nodes if str(n) not in data.columns]
    if missing_nodes:
        raise ValueError(
            "The following nodes from indra_priors are not present in data.columns: "
            + ", ".join(sorted(missing_nodes))
        )

    # Prepare input arguments for bootstrap sampling
    model_input = (
        data,
        indra_priors,
        prior_strength,
        scoring_function,
        search_algorithm,
        expert_knowledge,
        add_high_corr_edges_to_priors,
        corr_threshold,
        n_bootstrap,
        convert_to_probability,
    )

    # Run bootstrap sampling to generate multiple DAG hypotheses
    bootstrap_dags = run_bootstrap(*model_input)

    # Integrate bootstrap results into one final DAG using consensus approach
    posterior_dag = consensus_dag(bootstrap_dags, indra_priors, lam=0.25, min_freq=edge_probability)
    posterior_dag = pd.DataFrame(posterior_dag.edges, columns=["source", "target"])

    # Convert posterior DAG to y0 graph format
    y0_graph = convert_to_y0_graph(posterior_dag)

    return y0_graph


def repair_confounding(
    data: pd.DataFrame,
    posterior_dag: NxMixedGraph,
    indra_graph: nx.DiGraph,
    max_conditional: int = 2,
    n_jobs: int = -2,
    confounder_evidence: int = 1,
) -> NxMixedGraph:
    """
    Check for potential confounders in the estimated posterior DAG and repair if possible.

    This function identifies unobserved confounders in the posterior DAG that
    may act as confounders. It then attempts to repair the DAG by looking in
    INDRA for potential nodes that can explain the confounding. If the
    confounding is resolved, the function returns the repaired DAG. If not,
    it will add a bidirectional edge to indicate unresolved confounding.
    """

    repaired_dag = copy.deepcopy(posterior_dag)

    # Identify relations with latent confounders
    knn_imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(knn_imputer.fit_transform(data), index=data.index, columns=data.columns)

    falsification_results = get_graph_falsifications(
        repaired_dag,
        data,
        max_given=max_conditional,
        method="pearson",
        verbose=True,
        significance_level=0.05,
    ).evidence

    failed_tests = falsification_results.loc[
        (falsification_results["p_adj_significant"] == True)
        & (falsification_results["given"] != "")
    ].reset_index(drop=True)

    # combine unique nodes from both 'left' and 'right' columns
    query_relations = failed_tests[["left", "right"]].drop_duplicates()

    confounder_relations = dict()
    for i in tqdm(range(len(query_relations)), desc="Pulling confounder relations"):
        nodes = [query_relations.loc[i, "left"], query_relations.loc[i, "right"]]
        # indra_relations = format_query_results(
        #         query_confounder_relationships(
        #             get_ids(nodes, "gene"),
        #             client, minimum_evidence_count=confounder_evidence,
        #             mediators=get_ids(data.columns, "gene"),
        #             relation=["IncreaseAmount", "DecreaseAmount"])
        # )

        indra_relations = query_confounders(indra_graph, nodes)

        indra_relations = (
            indra_relations.groupby(["source"], as_index=False)["evidence_count"]
            .sum()
            .sort_values(by="evidence_count", ascending=False)["source"]
            .values
        )

        confounder_relations[tuple(nodes)] = indra_relations

    # Parallel processing of failed tests
    n = len(failed_tests)
    print(f"Processing {n} failed tests for confounding repair...")

    # Pre-convert rows to dicts to avoid serializing the full DataFrame per worker
    failed_test_rows = [failed_tests.loc[i].to_dict() for i in range(n)]

    results = list(
        tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(process_failed_test)(row, confounder_relations, data, max_conditional)
                for row in failed_test_rows
            ),
            total=n,
            desc="Processing failed tests",
        )
    )

    # Process results and collect statistics
    total_results = len(results)
    none_results = sum(1 for res in results if not res)
    valid_results = total_results - none_results
    repaired_count = 0
    unrepaired_count = 0
    new_nodes_added = set()
    new_edges_added = 0

    for res in results:
        if not res:
            continue
        src = Variable(res.get("source"))
        tgt = Variable(res.get("target"))
        Z = res.get("Z")
        if res.get("add_latent") or Z is None:
            repaired_dag.add_undirected_edge(src, tgt)
            unrepaired_count += 1
            pass
        else:
            repaired_count += 1
            # add nodes and directed edges from Z -> source and Z -> target
            for node in Z:
                node = Variable(node)
                if node not in repaired_dag.directed.nodes:
                    repaired_dag.add_node(node)
                    new_nodes_added.add(str(node))
                if ((node, src) not in repaired_dag.directed.edges) and (
                    not nx.has_path(repaired_dag.directed, src, node)
                ):
                    repaired_dag.add_directed_edge(node, src, directed=True)
                    new_edges_added += 1
                if ((node, tgt) not in repaired_dag.directed.edges) and (
                    not nx.has_path(repaired_dag.directed, tgt, node)
                ):
                    repaired_dag.add_directed_edge(node, tgt, directed=True)
                    new_edges_added += 1

    # Print summary of confounding repair results
    print("\n" + "=" * 60)
    print("CONFOUNDING REPAIR SUMMARY")
    print("=" * 60)
    print(f"Total failed tests processed: {total_results}")
    print(f"Valid results obtained: {valid_results}")
    print(f"Failed/invalid results: {none_results}")
    print(f"Successfully repaired confounders: {repaired_count}")
    print(f"Unrepaired confounders: {unrepaired_count}")
    if new_nodes_added:
        print(f"New confounder nodes added: {len(new_nodes_added)}")
        print(f"Added nodes: {', '.join(sorted(new_nodes_added))}")
    else:
        print("No new confounder nodes were added")
    if new_edges_added > 0:
        print(f"New edges added to repair confounding: {new_edges_added}")
    else:
        print("No new edges were added during confounding repair")
    repair_rate = (repaired_count / valid_results * 100) if valid_results > 0 else 0
    print(f"Repair success rate: {repair_rate:.1f}%")
    print("=" * 60 + "\n")

    return repaired_dag


def main():
    """
    Example usage of the network estimation pipeline.

    Demonstrates how to:
    1. Set up source and target proteins for pathway analysis
    2. Connect to INDRA database using authentication
    3. Extract prior biological knowledge
    4. Use the extracted priors for causal network learning

    This function serves as a template for typical causomic workflows
    involving IGF and EGFR signaling pathways.
    """

    from indra_cogex.client import Neo4jClient
    from sklearn.model_selection import train_test_split

    client = Neo4jClient(url=os.getenv("API_URL"), 
                            auth=(os.getenv("USER"), 
                                os.getenv("PASSWORD"))
                        )
    
    input_data = pd.read_csv("../../AstraZeneca_project/case_studies/compounds/model_input.csv")

    input_data_graph, input_data_scm = train_test_split(
        input_data, test_size=0.5, random_state=42)
        
    trog_targets = ['SERPINE1', 'CYP3A4', 'CTNNB1', 'MAPK1']

    dili_targets = ['ALB']

    indra_prior = extract_indra_prior(
        trog_targets, dili_targets, input_data_graph.columns, client,
        one_step_evidence=1, two_step_evidence=1,
        three_step_evidence=3, confounder_evidence=5000000)
    
if __name__ == "__main__":
    main()
