import networkx as nx
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score

from causomic.simulation.proteomics_simulator import simulate_data
from causomic.graph_construction.prior_data_reconciliation import (
    BICGaussIndraPriors,
    AICGaussIndraPriors,
    SparseHillClimb
)
from causomic.network import estimate_posterior_dag
import seaborn as sns

from causomic.validation.network_comparison import fit_pc, fit_hc, fit_notears
import multiprocessing as mp
import time
import pandas as pd
import os
import traceback
import secrets
import random as _random

def generate_random_dag(num_nodes, sparsity_factor):
    """
    Generate a random Directed Acyclic Graph (DAG) using networkx with letter-based node names.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    sparsity_factor : float
        Probability of creating an edge between two nodes.

    Returns
    -------
    nx.DiGraph
        A random DAG.
    """
    dag = nx.DiGraph()
    node_names = [f"{chr(65 + i)}{i}" for i in range(num_nodes)]  # Generate node names as letters (A, B, C, ...)
    dag.add_nodes_from(node_names)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Ensure directionality i -> j
            if np.random.random() < sparsity_factor:
                dag.add_edge(node_names[i], node_names[j])

    return dag


def generate_structured_dag(
    n_start: int = 3,
    n_end: int = 3,
    max_mediators: int = 3,
    confounder_prob: float = 0.1,
    shared_mediator_prob: float = 0.3,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate a structured DAG with semantic node roles: ligands (start), readouts (end),
    mediators, and confounders.

    Parameters
    ----------
    n_start : int
        Number of start (ligand) nodes. Named L0, L1, ...
    n_end : int
        Number of end (readout) nodes. Named R0, R1, ...
    max_mediators : int
        Maximum number of mediator nodes in any single start-to-end chain.
        Each path independently samples 0..max_mediators mediators.
    confounder_prob : float
        Fraction of observable nodes to add as confounders.
        n_confounders = max(0, round(confounder_prob * n_observable_nodes)).
        Each confounder has edges into 2 randomly selected observable nodes.
    shared_mediator_prob : float
        Probability of reusing an existing mediator node at each slot in a
        new chain (cross-talk). Falls back to a fresh node if reuse would
        create a cycle.
    seed : int or None
        Seed for numpy.random.default_rng. None gives non-deterministic output.

    Returns
    -------
    dag : nx.DiGraph
        A valid DAG with all structural edges.
    node_roles : dict
        Keys: 'start', 'end', 'mediators', 'confounders'.
        Each value is a list of node name strings.
    """
    rng = np.random.default_rng(seed)
    dag = nx.DiGraph()

    start_nodes = [f"L{i}" for i in range(n_start)]
    end_nodes = [f"R{i}" for i in range(n_end)]
    dag.add_nodes_from(start_nodes + end_nodes)

    mediator_pool: list = []
    med_count = 0

    # Truncated geometric weights: biases toward 1-3 connections, not all-to-all
    p_geom = 0.5
    ks = np.arange(1, n_end + 1)
    weights = p_geom * (1 - p_geom) ** (ks - 1)
    weights /= weights.sum()

    for li in start_nodes:
        k = int(rng.choice(ks, p=weights))
        target_ends = rng.choice(end_nodes, size=k, replace=False).tolist()

        for rj in target_ends:
            n_med = int(rng.integers(0, max_mediators + 1))

            chain = [li]
            new_meds_for_chain: list = []

            for _ in range(n_med):
                placed = False
                if mediator_pool and rng.random() < shared_mediator_prob:
                    shuffled_pool = list(mediator_pool)
                    rng.shuffle(shuffled_pool)
                    for cand in shuffled_pool:
                        if cand not in chain:
                            chain.append(cand)
                            placed = True
                            break
                if not placed:
                    new_name = f"M{med_count}"
                    med_count += 1
                    chain.append(new_name)
                    new_meds_for_chain.append(new_name)

            chain.append(rj)
            proposed_edges = [(chain[i], chain[i + 1]) for i in range(len(chain) - 1)]

            g_test = dag.copy()
            g_test.add_nodes_from(new_meds_for_chain)
            g_test.add_edges_from(proposed_edges)

            if nx.is_directed_acyclic_graph(g_test):
                dag.add_nodes_from(new_meds_for_chain)
                dag.add_edges_from(proposed_edges)
                mediator_pool.extend(new_meds_for_chain)
            else:
                # Fallback: all-new mediators — guaranteed acyclic
                chain_fb = [li]
                fb_meds: list = []
                for _ in range(n_med):
                    nm = f"M{med_count}"
                    med_count += 1
                    chain_fb.append(nm)
                    fb_meds.append(nm)
                chain_fb.append(rj)
                dag.add_nodes_from(fb_meds)
                dag.add_edges_from(
                    [(chain_fb[i], chain_fb[i + 1]) for i in range(len(chain_fb) - 1)]
                )
                mediator_pool.extend(fb_meds)

    observable_nodes = start_nodes + mediator_pool + end_nodes
    n_conf = max(0, round(confounder_prob * len(observable_nodes)))
    conf_nodes: list = []

    for ci in range(n_conf):
        cname = f"C{ci}"
        dag.add_node(cname)
        conf_nodes.append(cname)
        n_targets = min(2, len(observable_nodes))
        targets = rng.choice(observable_nodes, size=n_targets, replace=False).tolist()
        for t in targets:
            dag.add_edge(cname, t)

    assert nx.is_directed_acyclic_graph(dag), (
        "generate_structured_dag produced a cyclic graph — this is a bug."
    )

    node_roles = {
        "start": start_nodes,
        "end": end_nodes,
        "mediators": mediator_pool,
        "confounders": conf_nodes,
    }

    return dag, node_roles


def ground_truth_interventional_effect(
    dag: nx.DiGraph,
    coefficients: dict,
    intervention_nodes: dict,
    output_nodes: list,
) -> dict:
    """
    Compute the ground-truth expected interventional effect using the SEM coefficients.

    Uses the do-calculus: sets each intervened node to its fixed value (cutting
    incoming edges) and propagates expected values analytically through the DAG.
    Because ``simulate_node`` mean-centers parent values before multiplying by
    coefficients, the observational expected value of every node equals its
    intercept.  Under do(X = v) the deviation from that baseline propagates
    linearly downstream.

    Parameters
    ----------
    dag : nx.DiGraph
        The causal graph returned by ``generate_structured_dag`` (or any DAG
        whose nodes match the keys in ``coefficients``).
    coefficients : dict
        Structural equation coefficients as returned by ``simulate_data`` in
        the ``'Coefficients'`` key.  Format::

            {node: {parent: coef, 'intercept': val, 'error': var}}

    intervention_nodes : dict
        Mapping of node name → interventional value, e.g. ``{'L0': 30.0}``.
    output_nodes : list
        Names of the nodes whose post-intervention expected values are of
        interest, e.g. ``['R0', 'R1']``.

    Returns
    -------
    dict
        Keys:
        - ``'baseline'``: {node: E[node]} under observation (= intercept for each node)
        - ``'interventional'``: {node: E[node | do(...)]} for every node in the DAG
        - ``'effect'``: {node: E[node | do(...)] - E[node]} for each output node
          (positive = increase, negative = decrease)

    Examples
    --------
    >>> dag, roles = generate_structured_dag(n_start=1, n_end=1, seed=0)
    >>> sim = simulate_data(dag, n=500, add_feature_var=False, seed=0)
    >>> result = ground_truth_interventional_effect(
    ...     dag,
    ...     sim['Coefficients'],
    ...     intervention_nodes={'L0': 30.0},
    ...     output_nodes=['R0'],
    ... )
    >>> print(result['effect'])
    """
    _non_coef = {"intercept", "error", "cell_type"}

    # Observational expected value = intercept for every node (mean-centering
    # cancels all parent contributions in expectation).
    baseline = {node: coefficients[node]["intercept"] for node in dag.nodes()}

    # Propagate interventional expectations in topological order.
    interventional = dict(baseline)
    for node in nx.topological_sort(dag):
        if node in intervention_nodes:
            interventional[node] = intervention_nodes[node]
        else:
            node_coefs = coefficients[node]
            parents = [k for k in node_coefs if k not in _non_coef]
            expected = node_coefs["intercept"]
            for parent in parents:
                # Mirror simulate_node: coefficient * (parent_value - parent_obs_mean)
                expected += node_coefs[parent] * (interventional[parent] - baseline[parent])
            interventional[node] = expected

    effect = {node: interventional[node] - baseline[node] for node in output_nodes}

    return {"baseline": baseline, "interventional": interventional, "effect": effect}


def generate_indra_data(
    ground_truth_dag,
    num_incorrect_nodes=20,
    num_incorrect_edges=40,
    p_missing_real=0.0,
    p_on_path=0.7,
):
    """
    Generate INDRA-compatible data from a ground truth DAG by introducing incorrect
    nodes/edges and simulating realistic integer evidence counts.

    Parameters
    ----------
    ground_truth_dag : nx.DiGraph
        The original DAG.
    num_incorrect_nodes : int
        Number of spurious nodes (prefixed X) to add with one outgoing edge each.
    num_incorrect_edges : int
        Number of additional spurious edges to add between existing nodes.
    p_missing_real : float
        Probability that each ground-truth edge is excluded from the INDRA output,
        simulating edges with no literature support. Excluded edges are absent from
        both the returned graph and DataFrame.
    p_on_path : float
        Fraction of spurious nodes that are inserted *on* a source-to-sink path,
        i.e. between two nodes u and v where (u, v) lies on such a path.
        The remaining fraction get a single outgoing edge to a random existing node
        (the previous behaviour). Default 0.7.

    Returns
    -------
    modified_dag : nx.DiGraph
        DAG containing ground-truth edges that survived the missingness draw,
        plus all spurious nodes and edges.
    edges_df : pd.DataFrame
        Columns: source, target, ground_truth, evidence_count.
        evidence_count is a discrete integer in [1, 300] drawn from a log-normal
        distribution (true edges have higher median than false edges).
    missing_edges : list of (str, str)
        Ground-truth edges that were excluded from the INDRA output.
    """
    modified_dag = nx.DiGraph()

    # Add ground-truth edges, randomly dropping some to simulate INDRA gaps
    missing_edges = []
    for u, v in ground_truth_dag.edges():
        if np.random.random() < p_missing_real:
            missing_edges.append((u, v))
        else:
            modified_dag.add_edge(u, v)
            modified_dag.edges[u, v]["ground_truth"] = True

    # Carry over any isolated ground-truth nodes that had all edges dropped
    for node in ground_truth_dag.nodes():
        if node not in modified_dag:
            modified_dag.add_node(node)

    existing_nodes = list(ground_truth_dag.nodes)

    # Identify edges that lie on at least one source-to-sink path so that
    # on-path spurious nodes can be inserted between real path nodes.
    sources = [n for n in ground_truth_dag.nodes() if ground_truth_dag.in_degree(n) == 0]
    sinks   = [n for n in ground_truth_dag.nodes() if ground_truth_dag.out_degree(n) == 0]

    forward_reachable: set = set()
    for s in sources:
        forward_reachable.update(nx.descendants(ground_truth_dag, s) | {s})

    reversed_dag = ground_truth_dag.reverse(copy=False)
    backward_reachable: set = set()
    for t in sinks:
        backward_reachable.update(nx.descendants(reversed_dag, t) | {t})

    path_edges = [
        (u, v) for u, v in ground_truth_dag.edges()
        if u in forward_reachable and v in backward_reachable
    ]

    # Add incorrect nodes.
    # Majority (p_on_path) are inserted *between* two nodes on a real path,
    # creating a spurious u → X → v shortcut that mimics an intermediate mediator.
    # The rest receive a single outgoing edge to a random existing node.
    for i in range(num_incorrect_nodes):
        new_node = f"X{i}"
        modified_dag.add_node(new_node)
        if path_edges and np.random.random() < p_on_path:
            # Pick a random path edge and insert the fake node between its endpoints
            u, v = path_edges[np.random.randint(len(path_edges))]
            modified_dag.add_edge(u, new_node)
            modified_dag.edges[u, new_node]["ground_truth"] = False
            modified_dag.add_edge(new_node, v)
            modified_dag.edges[new_node, v]["ground_truth"] = False
        elif existing_nodes:
            target_node = np.random.choice(existing_nodes)
            modified_dag.add_edge(new_node, target_node)
            modified_dag.edges[new_node, target_node]["ground_truth"] = False

    # Add incorrect edges between existing (ground-truth) nodes only
    for _ in range(num_incorrect_edges):
        src = np.random.choice(existing_nodes)
        dst = np.random.choice(existing_nodes)
        if src != dst and not modified_dag.has_edge(src, dst):
            modified_dag.add_edge(src, dst)
            modified_dag.edges[src, dst]["ground_truth"] = False

    # Assign integer evidence counts drawn from log-normal distributions.
    # True edges: lognormal(1.0, 1.5) → median ≈ 3, heavy tail up to 300.
    # False edges: lognormal(0.0, 0.8) → median ≈ 1, mostly 1–5.
    for u, v in modified_dag.edges():
        is_true = modified_dag.edges[u, v].get("ground_truth", False)
        if is_true:
            raw = np.random.lognormal(mean=1.0, sigma=1.5)
        else:
            raw = np.random.lognormal(mean=0.0, sigma=0.8)
        count = int(np.clip(np.round(raw), 1, 300))
        modified_dag.edges[u, v]["evidence_count"] = count

    edges_data = {
        'source': [],
        'target': [],
        'ground_truth': [],
        'evidence_count': [],
    }
    for u, v in modified_dag.edges():
        edges_data['source'].append(u)
        edges_data['target'].append(v)
        edges_data['ground_truth'].append(modified_dag.edges[u, v]["ground_truth"])
        edges_data['evidence_count'].append(modified_dag.edges[u, v]["evidence_count"])

    edges_df = pd.DataFrame(edges_data)

    return modified_dag, edges_df, missing_edges

def run_graph_sim():

    # grab a CSPRNG seed (32-bit signed range) and apply it to numpy and python's random
    seed = secrets.randbelow(2**31 - 1)
    np.random.seed(seed)
    _random.seed(seed)

    print(f"Using random seed: {seed}")

    print("Generating random DAG...")
    num_nodes = 10
    gt_dag = generate_random_dag(num_nodes, .2)
    
    # pos = nx.spring_layout(gt_dag)
    # nx.draw(gt_dag, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    # plt.title("Random DAG Visualization")
    # plt.show()
    # print("Generated DAG edges:")
    # print(gt_dag.edges())

    print("Generating fake INDRA data...")
    indra_nx, indra_data, _ = generate_indra_data(gt_dag, num_incorrect_nodes=10, num_incorrect_edges=100)
    # pos = nx.spring_layout(indra_nx)
    # nx.draw(indra_nx, pos, with_labels=True, node_color='lightblue', node_size=500, arrowsize=20)
    # plt.title("Random DAG Visualization")
    # plt.show()
    # print("\nINDRa-compatible data:")
    # print(indra_data)

    # plt.figure(figsize=(8, 6))
    # sns.kdeplot(
    #     data=indra_data.dropna(subset=["evidence_count", "ground_truth"]),
    #     x="evidence_count",
    #     hue="ground_truth",
    #     fill=True,
    #     common_norm=True,  # scale densities by group proportion so areas reflect true counts
    #     palette={True: "#1f77b4", False: "#ff7f0e"},
    #     alpha=0.6,
    #     linewidth=1.5,
    # )
    # plt.xlabel("evidence_count")
    # plt.title("KDE of evidence_count by ground_truth")
    # plt.legend(title="ground_truth")
    # plt.tight_layout()
    # plt.show()
    
    
    print("Simulating proteomics data...")
    n_obs = 100
    sim_data = simulate_data(
        graph=gt_dag,
        n=n_obs,
        add_feature_var=False,
        seed=seed,
    )
    protein_data = sim_data["Protein_data"]

    # Add missing nodes into data
    all_nodes = list(indra_nx.nodes())
    for i in range(len(all_nodes)):
        if all_nodes[i] not in protein_data.keys():
            protein_data[all_nodes[i]] = np.random.normal(np.random.uniform(5,20), np.random.uniform(.5,3), size=n_obs)
    
    final_sim_data = pd.DataFrame(protein_data)

    print("Estimating posterior network...")
    posterior_network = estimate_posterior_dag(
        final_sim_data, indra_priors=indra_data, 
        prior_strength=5, 
        scoring_function=BICGaussIndraPriors, 
        search_algorithm=SparseHillClimb, 
        n_bootstrap=200, 
        add_high_corr_edges_to_priors=False, 
        corr_threshold=0.9, edge_probability=0.5,
        convert_to_probability=False
    )

    print("Evaluating network performance...")
    result = nx.to_pandas_edgelist(posterior_network.directed)
    result["source"] = result["source"].astype(str)
    result["target"] = result["target"].astype(str)
    result.loc[:, "pred"] = True
    result = result.merge(indra_data, on=['source', 'target'], how='outer')
    result["pred"] = result["pred"].fillna(False)

    print("Recall:", round(recall_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    print("Precision:", round(precision_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    
    causomic_recall = recall_score(
        result["ground_truth"],
        result["pred"]
    )
    causomic_precision = precision_score(
        result["ground_truth"],
        result["pred"]
    )
    
    pc_results = fit_pc(final_sim_data)    
    print("Evaluating PC performance...")
    result = nx.to_pandas_edgelist(pc_results[0])
    result["source"] = result["source"].astype(str)
    result["target"] = result["target"].astype(str)
    result.loc[:, "pred"] = True
    result = result.merge(indra_data, on=['source', 'target'], how='outer')
    result["pred"] = result["pred"].fillna(False)
    result["ground_truth"] = result["ground_truth"].fillna(False)

    # print("Accuracy:", round(accuracy_score(
    #     result["ground_truth"],
    #     result["pred"]
    # ), 2))
    print("Recall:", round(recall_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    print("Precision:", round(precision_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    
    pc_recall = recall_score(
        result["ground_truth"],
        result["pred"]
    )
    pc_precision = precision_score(
        result["ground_truth"],
        result["pred"]
    )
    
    mmhc_results = fit_hc(final_sim_data)
    print("Evaluating MMHC performance...")
    result = nx.to_pandas_edgelist(mmhc_results[0])
    result["source"] = result["source"].astype(str)
    result["target"] = result["target"].astype(str)
    result.loc[:, "pred"] = True
    result = result.merge(indra_data, on=['source', 'target'], how='outer')
    result["pred"] = result["pred"].fillna(False)
    result["ground_truth"] = result["ground_truth"].fillna(False)

    # print("Accuracy:", round(accuracy_score(
    #     result["ground_truth"],
    #     result["pred"]
    # ), 2))
    print("Recall:", round(recall_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    print("Precision:", round(precision_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    
    mmhc_recall = recall_score(
        result["ground_truth"],
        result["pred"]
    )
    mmhc_precision = precision_score(
        result["ground_truth"],
        result["pred"]
    )
    
    # Plot scatterplots for each edge in the ground-truth DAG using final_sim_data
    # edges = [(u, v) for u, v in gt_dag.edges() if u in final_sim_data.columns and v in final_sim_data.columns]

    # if len(edges) == 0:
    #     print("No matching columns in final_sim_data for edges in gt_dag.")
    # else:
    #     n = len(edges)
    #     ncols = 3
    #     nrows = int(np.ceil(n / ncols))
    #     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    #     for idx, (u, v) in enumerate(edges):
    #         r = idx // ncols
    #         c = idx % ncols
    #         ax = axes[r][c]

    #         # scatter + optional linear fit line
    #         sns.scatterplot(x=final_sim_data[u], y=final_sim_data[v], ax=ax, s=20, alpha=0.7)
    #         try:
    #             sns.regplot(x=final_sim_data[u], y=final_sim_data[v], ax=ax, scatter=False, color="red", truncate=False)
    #         except Exception:
    #             pass

    #         ax.set_xlabel(u)
    #         ax.set_ylabel(v)
    #         ax.set_title(f"{u} → {v}")

    #     # turn off any unused subplots
    #     for idx in range(n, nrows * ncols):
    #         r = idx // ncols
    #         c = idx % ncols
    #         axes[r][c].axis("off")

    #     plt.tight_layout()
    #     plt.show()
    
    notears_results, notears_weights = fit_notears(final_sim_data)
    
    print("Evaluating notears performance...")
    result = nx.to_pandas_edgelist(notears_results)
    result["source"] = result["source"].astype(str)
    result["target"] = result["target"].astype(str)
    result.loc[:, "pred"] = True
    result = result.merge(indra_data, on=['source', 'target'], how='outer')
    result["pred"] = result["pred"].fillna(False)
    result["ground_truth"] = result["ground_truth"].fillna(False)

    # print("Accuracy:", round(accuracy_score(
    #     result["ground_truth"],
    #     result["pred"]
    # ), 2))
    print("Recall:", round(recall_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    print("Precision:", round(precision_score(
        result["ground_truth"],
        result["pred"]
    ), 2))
    
    notears_recall = recall_score(
        result["ground_truth"],
        result["pred"]
    )
    notears_precision = precision_score(
        result["ground_truth"],
        result["pred"]
    )
    
    return causomic_recall, causomic_precision, pc_recall, pc_precision, mmhc_recall, mmhc_precision, notears_recall, notears_precision

    
if __name__ == "__main__":

    n_runs = 10

    results = []
    for i in range(n_runs):
        print(f"Starting run {i+1}/{n_runs}...")
        t0 = time.time()
        try:
            causomic_recall, causomic_precision, pc_recall, pc_precision, mmhc_recall, mmhc_precision, notears_recall, notears_precision = run_graph_sim()
            success = True
            error = None
        except Exception:
            causomic_recall = causomic_precision = pc_recall = pc_precision = mmhc_recall = mmhc_precision = notears_recall = notears_precision = None
            success = False
            error = traceback.format_exc()

        duration = time.time() - t0
        results.append({
            "run": i,
            "causomic_recall": causomic_recall,
            "causomic_precision": causomic_precision,
            "pc_recall": pc_recall,
            "pc_precision": pc_precision,
            "mmhc_recall": mmhc_recall,
            "mmhc_precision": mmhc_precision,
            "notears_recall": notears_recall,
            "notears_precision": notears_precision,
            "success": success,
            "error": error,
            "duration_s": duration,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        })

        print(f"Finished run {i+1}/{n_runs} - success={success} - duration={duration:.2f}s")

    df = pd.DataFrame(results)

    out_dir = os.getcwd()
    csv_path = os.path.join(out_dir, "run_graph_sim_results.csv")
    pkl_path = os.path.join(out_dir, "run_graph_sim_results.pkl")

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"Saved {len(df)} runs to {csv_path} and {pkl_path}")
    print(df[["causomic_recall", "causomic_precision", "notears_recall", 
              "notears_precision", "pc_recall", "pc_precision", "mmhc_recall", "mmhc_precision"
              ]].describe(include="all"))