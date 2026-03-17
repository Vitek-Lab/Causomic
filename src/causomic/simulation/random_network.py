import networkx as nx
import numpy as np
import pandas as pd
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

def generate_indra_data(ground_truth_dag, num_incorrect_nodes=20, num_incorrect_edges=40):
    """
    Generate INDRa-compatible data from a ground truth DAG by introducing incorrect nodes and edges.

    Parameters
    ----------
    ground_truth_dag : nx.DiGraph
        The original DAG.
    num_incorrect_nodes : int
        Number of incorrect nodes to add.
    num_incorrect_edges : int
        Number of incorrect edges to add.

    Returns
    -------
    nx.DiGraph
        A modified DAG with incorrect nodes and edges.
    """
    modified_dag = ground_truth_dag.copy()
    for u, v in modified_dag.edges():
        modified_dag.edges[u, v]["ground_truth"] = True

    existing_nodes = list(modified_dag.nodes)

    # Add incorrect nodes
    for i in range(num_incorrect_nodes):
        new_node = f"X{i}"
        modified_dag.add_node(new_node)
        # Optionally connect the new node to existing nodes
        if existing_nodes:
            target_node = np.random.choice(existing_nodes)
            modified_dag.add_edge(new_node, target_node)
            modified_dag.edges[new_node, target_node]["ground_truth"] = False

    # Add incorrect edges
    all_nodes = list(modified_dag.nodes)
    for _ in range(num_incorrect_edges):
        src = np.random.choice(all_nodes)
        dst = np.random.choice(all_nodes)
        if src != dst and not modified_dag.has_edge(src, dst):
            modified_dag.add_edge(src, dst)
            modified_dag.edges[src, dst]["ground_truth"] = False

    edges = list(modified_dag.edges)
    for edge in edges:
        gt_edge = modified_dag.edges[edge].get("ground_truth", None)
        if gt_edge == True:
            evidence = np.random.beta(1,1) #3,1.5
        else:
            evidence = np.random.beta(1,5)
        modified_dag.edges[edge]["evidence_count"] = evidence

    # Convert modified_dag edges to a DataFrame
    edges_data = {
        'source': [],
        'target': [],
        'ground_truth': [],
        'evidence_count': []
    }

    for u, v in modified_dag.edges():
        edges_data['source'].append(u)
        edges_data['target'].append(v)
        edges_data['ground_truth'].append(modified_dag.edges[u, v]["ground_truth"])
        edges_data['evidence_count'].append(modified_dag.edges[u, v]["evidence_count"])

    edges_df = pd.DataFrame(edges_data)

    return modified_dag, edges_df

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
    indra_nx, indra_data = generate_indra_data(gt_dag, num_incorrect_nodes=10, num_incorrect_edges=100)
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