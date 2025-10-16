
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

from collections import Counter
import os
import pandas as pd
import numpy as np
from pgmpy.estimators import ExpertKnowledge
from indra_cogex.client import Neo4jClient
from y0.graph import NxMixedGraph
from y0.algorithm.falsification import get_graph_falsifications
from y0.dsl import Variable

from pgmpy.estimators.CITests import pearsonr
import networkx as nx

from causomic.graph_construction.prior_data_reconciliation import run_bootstrap, \
    SparseHillClimb, BICGaussIndraPriors
from causomic.graph_construction.utils import get_one_step_root_down, \
    query_confounder_relationships, get_three_step_root, \
        get_two_step_root_known_med
from causomic.graph_construction.indra_queries import format_query_results, get_ids
from causomic.graph_construction.repair import convert_to_y0_graph
from sklearn.impute import KNNImputer
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_indra_prior(source: list, 
                        target: list, 
                        measured_proteins: list, 
                        client: Neo4jClient, 
                        one_step_evidence: int = 1, 
                        two_step_evidence: int = 1,
                        three_step_evidence: int = 3, 
                        confounder_evidence: int = 10) -> pd.DataFrame:
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
        - 'source_symbol': Source protein name (gene symbol)
        - 'target_symbol': Target protein name (gene symbol)
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
            minimum_evidence_count=one_step_evidence)
    )

    # Query two-step relationships: source -> mediator -> target
    two_step_relations = format_query_results(
        get_two_step_root_known_med(
            root_nodes=get_ids(source, "gene"), 
            downstream_nodes=get_ids(target, "gene"), 
            client=client,
            relation=["IncreaseAmount", "DecreaseAmount"],
            minimum_evidence_count=two_step_evidence,
            mediators=get_ids(measured_proteins, "gene"))
    )

    # Query three-step relationships: source -> med1 -> med2 -> target
    three_step_relations = format_query_results(
        get_three_step_root(
            root_nodes=get_ids(source, "gene"),
            downstream_nodes=get_ids(target, "gene"), 
            client=client, relation=["IncreaseAmount", "DecreaseAmount"],
            minimum_evidence_count=three_step_evidence,
            mediators=get_ids(measured_proteins, "gene"))
    )

    # Combine initial relationship queries
    all_relations = pd.concat(
        [one_step_relations, two_step_relations, three_step_relations],
        ignore_index=True)
    all_network_nodes = pd.unique(
        all_relations[['source_symbol', 'target_symbol']].values.ravel())
    
    # Query confounder relationships among all discovered network nodes
    confounder_relations = format_query_results(
        query_confounder_relationships(
            get_ids(all_network_nodes, "gene"), 
            client, minimum_evidence_count=confounder_evidence,
            mediators=get_ids(measured_proteins, "gene"))
    )
    confounder_relations = confounder_relations[
        confounder_relations["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]

    # Remove duplicate confounder relationships
    confounder_relations = confounder_relations.drop_duplicates(
        subset=["source_symbol", "target_symbol", "relation", "evidence_count"])

    # Combine all relationship types into final network
    all_relations = pd.concat(
        [one_step_relations, two_step_relations, three_step_relations, 
         confounder_relations], ignore_index=True)
    all_network_nodes = pd.unique(
        all_relations[['source_symbol', 'target_symbol']].values.ravel())

    # Extract relevant columns and aggregate evidence counts
    prior_network = all_relations.loc[
        :, ["source_symbol", "target_symbol", "evidence_count"]]

    # Sum evidence counts for duplicate edges (same source-target pair)
    prior_network = prior_network.groupby(
        ["source_symbol", "target_symbol"], as_index=False)["evidence_count"].sum()

    # Clean protein names by removing hyphens for consistency
    prior_network["source_symbol"] = prior_network["source_symbol"].str.replace("-", "")
    prior_network["target_symbol"] = prior_network["target_symbol"].str.replace("-", "")

    # Print summary statistics
    print(f"Number of proteins pulled: {len(all_network_nodes)}")
    print(f"Number of reconciled edges pulled: {len(prior_network)}")

    return prior_network

def estimate_posterior_dag(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float = 5.0,
    scoring_function: type = BICGaussIndraPriors,
    search_algorithm: type = SparseHillClimb,
    n_bootstrap: int = 100,
    add_high_corr_edges_to_priors: bool = True,
    corr_threshold: float = 0.95,
    edge_probability: float = 0.9) -> pd.DataFrame:
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
        Should contain columns: 'source_symbol', 'target_symbol', 'evidence_count'.
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
        excluded. Default is 0.9 (90% threshold).
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the posterior DAG edges with columns:
        - 'source': Source node of the edge
        - 'target': Target node of the edge  
        - 'Probability': Fraction of bootstrap samples containing this edge
        
        Only edges with probability > edge_probability are included.
        
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
    nodes = pd.unique(
        indra_priors[['source_symbol', 'target_symbol']].values.ravel())
    nodes = np.array([node.replace("-", "") for node in nodes])

    # Generate all possible edges between nodes
    all_possible_edges = [(u.replace("-", ""), v.replace("-", "")) for u in nodes for v in nodes if u != v]
    
    # Extract observed edges from prior network
    obs_edges = [
        (indra_priors.loc[i, "source_symbol"].replace("-", ""), 
        indra_priors.loc[i, "target_symbol"].replace("-", "")) for i in range(len(indra_priors))]
    
    # Define forbidden edges as all edges not in the prior network
    forbidden_edges = [edge for edge in all_possible_edges if edge not in obs_edges]

    # Create expert knowledge object with forbidden edges constraint
    expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
    
    # Remove hyphens from data column names
    data.columns = [str(col).replace("-", "") for col in data.columns]
    
    # Prepare input arguments for bootstrap sampling
    model_input = (
        data, indra_priors, prior_strength, 
        scoring_function, search_algorithm, expert_knowledge, 
        add_high_corr_edges_to_priors, corr_threshold
    )

    # Run bootstrap sampling to generate multiple DAG hypotheses
    bootstrap_dags = run_bootstrap(*model_input, n_bootstrap)

    # Count occurrences of each edge across all bootstrap samples
    edge_counts = Counter()
    for dag in bootstrap_dags:
        if dag is not None:  # Skip failed bootstrap runs
            # Update the edge counts with edges from this DAG
            edge_counts.update(list(dag.edges()))
    
    # Calculate edge probabilities based on bootstrap frequency
    n_dags = len(bootstrap_dags)  # Total number of bootstrap samples

    # Create DataFrame with edge probabilities
    edge_probabilities = pd.DataFrame(
        [(edge, count / n_dags) for edge, count in edge_counts.items()],
        columns=["Edge", "Probability"])

    # Filter edges based on probability threshold
    posterior_dag = edge_probabilities.loc[
        edge_probabilities["Probability"] > edge_probability, :]
    
    # Split edge tuples into separate source and target columns
    posterior_dag[['source', 'target']] = pd.DataFrame(posterior_dag['Edge'].tolist(), 
                                                       index=posterior_dag.index)
    # Remove the original Edge column containing tuples
    posterior_dag = posterior_dag.drop('Edge', axis=1)
    
    # Reset index for clean output
    posterior_dag = posterior_dag.reset_index(drop=True)
    
    return posterior_dag

def repair_confounding(data: pd.DataFrame, 
                       posterior_dag: pd.DataFrame,
                       client: Neo4jClient,
                       max_conditional: int = 2,
                       ) -> NxMixedGraph:
    """
    Check for potential confounders in the estimated posterior DAG and repair if possible.
    
    This function identifies unobserved confounders in the posterior DAG that 
    may act as confounders. It then attempts to repair the DAG by looking in 
    INDRA for potential nodes that can explain the confounding. If the 
    confounding is resolved, the function returns the repaired DAG. If not,
    it will add a bidirectional edge to indicate unresolved confounding.
    """

    # Convert posterior DAG to y0 graph format
    y0_graph = convert_to_y0_graph(data, posterior_dag)

    # Identify relations with latent confounders
    knn_imputer = KNNImputer(n_neighbors=5)
    data = pd.DataFrame(knn_imputer.fit_transform(data), 
                        index=data.index, columns=data.columns)

    falsification_results = get_graph_falsifications(
        y0_graph,
        data,
        max_given=max_conditional,
        method="pearson",
        verbose=True,
        significance_level=0.05,
    ).evidence
    
    failed_tests = falsification_results.loc[
        (falsification_results["p_adj_significant"] == True) & \
            (falsification_results["given"] != "")].reset_index(drop=True)
    
    def _process_failed_test(row):
        try:
            source = row["left"]
            target = row["right"]
            given = row["given"]

            confounder_relations = format_query_results(
                query_confounder_relationships(
                    get_ids([source, target], "gene"),
                    client, minimum_evidence_count=1,
                    mediators=get_ids(data.columns, "gene"))
            )
            confounder_relations = confounder_relations[
                confounder_relations["relation"].isin(["IncreaseAmount", "DecreaseAmount"])
            ]
            confounder_relations = confounder_relations.groupby(
                ["source_symbol"], as_index=False)["evidence_count"].sum().sort_values(
                by="evidence_count", ascending=False)["source_symbol"].values

            add_latent = False
            found_adjustment = False
            found_Z = None

            # build all non-empty confounder combos (kept same range as original: r in [1])
            conf_list = list(confounder_relations)
            all_combos = [
                combo
                for r in range(1, 2)
                for combo in combinations(conf_list, r)
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
                add_latent = True
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
            return {"source": row.get("left"), "target": row.get("right"), "add_latent": True, "Z": None, "error": str(e)}


    # Parallel processing of failed tests
    results = []
    print(f"Processing {len(failed_tests)} failed tests for confounding repair...")
    if len(failed_tests) > 0:
        max_workers = min(32, (os.cpu_count() or 4), len(failed_tests))
        with ThreadPoolExecutor(max_workers=max_workers) as exc:
            futures = {
                exc.submit(_process_failed_test, failed_tests.loc[i]): i
                for i in range(len(failed_tests))
            }
            for fut in as_completed(futures):
                results.append(fut.result())

    # Apply results to y0_graph (sequential to avoid concurrency issues in graph updates)
    for res in results:
        src = Variable(res.get("source"))
        tgt = Variable(res.get("target"))
        Z = res.get("Z")
        if res.get("add_latent") or Z is None:
            # y0_graph.add_undirected_edge(src, tgt)
            pass
        else:
            # add nodes and directed edges from Z -> source and Z -> target
            for node in Z:
                node = Variable(node)
                if node not in y0_graph.directed.nodes:
                    y0_graph.add_node(node)
                if ((node, src) not in y0_graph.directed.edges) and (not nx.has_path(y0_graph.directed, src, node)):
                    y0_graph.add_directed_edge(node, src, directed=True)
                if ((node, tgt) not in y0_graph.directed.edges) and (not nx.has_path(y0_graph.directed, tgt, node)):
                    y0_graph.add_directed_edge(node, tgt, directed=True)

    
    # Final check on confounding
    # falsification_results = get_graph_falsifications(
    #     y0_graph,
    #     data,
    #     max_given=max_conditional,
    #     method="pearson",
    #     verbose=True,
    #     significance_level=0.05,
    # ).evidence
        
    
    
    return y0_graph

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
    
    # Define proteins of interest for pathway analysis
    source = ['SERPINE1', 'CYP3A4', 'CTNNB1', 'MAPK1'] # Upstream regulators/treatments
    target = ['ABCC2', 'ALB', 'CAT', 'CYP2C19', 'CYP2C9']  # Downstream targets/outcomes
    measured_proteins = ['SERPINE1', 'CYP3A4', 'CTNNB1', 'MAPK1',
                         'ABCC2', 'ALB', 'CAT', 'CYP2C19', 'CYP2C9', 
                         'CYP2E1', 'ENO1', 'GPT', 'GSR', 'GSTM1', 
                         'GSTT1', 'HLA-A', 'HMOX1', 'HPD', 'KNG1', 
                         'MTHFR', 'NAT2', 'SOD1']

    # Initialize INDRA client with authentication from environment variables
    client = Neo4jClient(
        url=os.getenv("API_URL"), 
        auth=("neo4j", os.getenv("PASSWORD"))
    )

    # Extract biological prior knowledge from INDRA databases
    network_priors = extract_indra_prior(
        source, target, measured_proteins, client,
        one_step_evidence=1, two_step_evidence=1, three_step_evidence=1, 
        confounder_evidence=1)
    
    # Note: Additional steps would typically include:
    # - Loading observational data
    # - Running estimate_posterior_dag with the extracted priors
    # - Analyzing and visualizing the resulting network
    
    
if __name__ == "__main__":
    main()