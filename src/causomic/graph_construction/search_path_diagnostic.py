import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge
from collections import Counter
from joblib import Parallel, delayed
from tqdm import tqdm


def random_acyclic_subgraph(nodes, allowed_edges, inclusion_prob=0.15, rng=None):
    """Generate a random DAG by greedily adding allowed edges without creating cycles."""
    if rng is None:
        rng = np.random.default_rng()

    dag = DAG()
    dag.add_nodes_from(nodes)

    edges = list(allowed_edges)
    rng.shuffle(edges)

    for u, v in edges:
        if rng.random() > inclusion_prob:
            continue
        dag.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(u, v)

    return dag


def run_single_random_init(data, edge_priors, score_fn, estimator,
                           expert_knowledge, allowed_edges, nodes,
                           inclusion_prob, max_indegree, seed):
    """Single Hill Climb run from a random initial DAG on the full dataset."""
    import logging
    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    rng = np.random.default_rng(seed)
    start_dag = random_acyclic_subgraph(nodes, allowed_edges, inclusion_prob, rng)

    scorer = score_fn(data, edge_priors=edge_priors)
    est = estimator(data=data, allowed_additions=set(allowed_edges))

    estimated_dag = est.estimate(
        scoring_method=scorer,
        start_dag=start_dag,
        expert_knowledge=expert_knowledge,
        max_indegree=max_indegree,
        epsilon=0.01,
        show_progress=False,
    )
    return estimated_dag


def search_path_diagnostic(data, edge_priors, score_fn, estimator,
                           expert_knowledge, K=50, inclusion_prob=0.15,
                           max_indegree=5):
    """
    Run K Hill Climb searches from random initializations on the SAME full dataset.
    Compare edge sets to diagnose search path dependence vs genuine signal.
    """
    nodes = list(data.columns)
    allowed_edges = list(edge_priors.keys())

    dags = Parallel(n_jobs=-2)(
        delayed(run_single_random_init)(
            data, edge_priors, score_fn, estimator,
            expert_knowledge, allowed_edges, nodes,
            inclusion_prob, max_indegree, seed=i
        )
        for i in tqdm(range(K), desc="Random init runs")
    )

    dags = [d for d in dags if d is not None]
    return dags


def compare_dag_sets(dags_random_init, dags_bootstrap, label_a="random_init", label_b="bootstrap"):
    """
    Compare edge stability between two sets of DAGs.
    Returns a DataFrame with per-edge frequencies from each approach.
    """
    def edge_frequencies(dags):
        counts = Counter()
        for dag in dags:
            counts.update(list(dag.edges()))
        n = len(dags)
        return {edge: count / n for edge, count in counts.items()}

    freq_a = edge_frequencies(dags_random_init)
    freq_b = edge_frequencies(dags_bootstrap)

    all_edges = set(freq_a.keys()) | set(freq_b.keys())
    rows = []
    for edge in sorted(all_edges):
        rows.append({
            "source": edge[0],
            "target": edge[1],
            f"freq_{label_a}": freq_a.get(edge, 0.0),
            f"freq_{label_b}": freq_b.get(edge, 0.0),
        })

    df = pd.DataFrame(rows)
    df["abs_diff"] = abs(df[f"freq_{label_a}"] - df[f"freq_{label_b}"])
    df = df.sort_values("abs_diff", ascending=False)
    return df