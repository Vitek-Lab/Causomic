import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.estimators import PC, HillClimbSearch
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple

try:
    from notears.linear import notears_linear  # type: ignore[import-not-found]
except ImportError:
    notears_linear = None

def _model_to_nx(model) -> nx.DiGraph:
    """Convert a pgmpy DAG/PDAG/BayesianModel to a DiGraph."""
    G = nx.DiGraph()
    G.add_nodes_from(model.nodes())
    # Works for PGMDAG/BayesianModel/PDAG with directed edges exposed by .edges()
    G.add_edges_from(list(model.edges()))
    return G

def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score features column-wise (genes)."""
    X = np.asarray(X, dtype=float)
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)

def fit_pc(
    df: pd.DataFrame,
    return_type: str = "pdag",
) -> Tuple[nx.DiGraph, object]:
    """
    PC algorithm (constraint-based).
    Parameters
    ----------
    X : array-like, shape (n_samples, n_genes)
    names : optional list of column (gene) names
    alpha : significance level for CI tests
    ci_test : 'pearsonr' (linear) | 'kci' (nonlinear) | 'chisq' (discrete)
    return_type : 'pdag' (CPDAG) or 'dag' (fully oriented by rules)

    Returns
    -------
    G : nx.DiGraph  (directed edges present in the learned (C)PDAG/DAG)
    model : pgmpy object (PDAG or DAG / BayesianModel)
    """

    pc = PC(df)
    model = pc.estimate(variant="parallel", ci_test="pearsonr", return_type="dag", max_cond_vars=3)
    G = _model_to_nx(model)
    return G, model


def fit_hc(
    df: pd.DataFrame
) -> Tuple[nx.DiGraph, object]:
    """
    Hill Climb Search via pgmpy.
    Parameters
    ----------
    score : 'bic' | 'k2'
    ci_test : 'pearsonr' (continuous) | 'chisq' (discrete) | 'kci'
    max_indegree : optional integer cap during hill-climb

    Returns
    -------
    G : nx.DiGraph
    model : pgmpy BayesianModel
    """

    est = HillClimbSearch(df)
    model = est.estimate(scoring_method="bic-g")
    G = _model_to_nx(model)
    return G, model


def fit_notears(
    df: pd.DataFrame,
    lambda1: float = 0.001,
    loss_type: str = "l2",
    w_threshold: float = 0.0,
    standardize: bool = True,
    max_iter: int = 1000,
) -> Tuple[nx.DiGraph, np.ndarray]:
    """
    NOTEARS (linear). Returns DAG and weighted adjacency W.
    Parameters
    ----------
    lambda1 : l1 penalty (sparsity)
    w_threshold : abs weight cutoff to keep an edge
    standardize : z-score columns before optimization
    loss_type : 'l2' | 'logistic' (use 'l2' for continuous)
    max_iter : optimizer iterations

    Returns
    -------
    G : nx.DiGraph
    W : np.ndarray, shape (p, p), weights from NOTEARS
    """
    if notears_linear is None:
        raise ImportError(
            "NOTEARS support is optional. Install it with `pip install 'causomic[notears]'`."
        )

    X = np.asarray(df, dtype=float)
    if standardize:
        X = _standardize(X)

    p = X.shape[1]

    W = notears_linear(
        X, lambda1=lambda1, loss_type=loss_type, max_iter=max_iter
    )
    
    names = df.columns.tolist()
    G = nx.DiGraph()
    G.add_nodes_from(names)

    # NOTEARS: W[i, j] = j -> i
    for i in range(p):        # child
        for j in range(p):    # parent
            if i == j:
                continue
            if abs(W[i, j]) > w_threshold:
                G.add_edge(names[j], names[i], weight=W[i, j])

    return G, W