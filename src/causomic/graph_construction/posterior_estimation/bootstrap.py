"""Repeated structure search over resampled data.

:func:`run_bootstrap` drives many independent searches in parallel and returns
the DAGs; reducing that population to a single graph is
:mod:`~causomic.graph_construction.posterior_estimation.selection`'s job.

The same machinery serves two regimes, distinguished by the resampling
arguments rather than by separate code paths:

- ``subsample_frac < 1`` with ``replace=True`` -- a genuine bootstrap. Edge
  frequency across runs estimates edge confidence.
- ``subsample_frac = 1`` with ``replace=False`` -- random restarts on the full
  data. Variation reflects search-path dependence, not sampling noise.

Interventional data is supported by carrying per-row arm labels alongside the
resample so that arm membership survives shuffling, with
:func:`_resample_with_arm_floor` protecting arms too small to bootstrap safely.
"""

import logging
from contextlib import contextmanager
from typing import Optional

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge
from tqdm import tqdm

from causomic.graph_construction.posterior_estimation.edge_priors import (
    prepare_indra_priors,
    remove_high_corr_edges_from_blacklist,
)
from causomic.graph_construction.posterior_estimation.hill_climb import (
    random_acyclic_subgraph,
)


@contextmanager
def _tqdm_joblib(tqdm_object):
    """Update ``tqdm`` based on job completion in ``joblib.Parallel``."""
    old_batch_callback = joblib.parallel.BatchCompletionCallBack

    class TqdmBatchCompletionCallback(old_batch_callback):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _resample_with_arm_floor(
    combined: pd.DataFrame,
    arm_col: str,
    frac: float,
    replace: bool,
    floor: int,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Bootstrap-resample ``combined``, holding any arm smaller than ``floor`` rows fixed.

    An arm with fewer rows than ``floor`` carries too little information to usefully
    bootstrap: with-replacement resampling at a typical ``frac<1`` setting collapses it
    to just 1-2 unique rows a meaningful fraction of the time, which isn't enough to
    identify a multi-parent GLM fit and mostly produces candidates that get discarded as
    singular (see the HPN-DREAM interventional consensus run that motivated this - a
    6-row arm resampled at frac=0.65 from a 30-row pool got <=2 rows ~20% of the time).
    Those arms are kept in full, unresampled, on every draw; only arms at or above
    ``floor`` rows are bootstrap-resampled at (``frac``, ``replace``). ``floor=0``
    (the default everywhere this is called) reproduces the original single pooled
    ``.sample(...)`` call exactly, so this is opt-in only.
    """
    if not floor:
        return combined.sample(frac=frac, replace=replace, random_state=rng)

    parts = []
    for _, arm_group in combined.groupby(arm_col, sort=False):
        if len(arm_group) < floor:
            parts.append(arm_group)
        else:
            parts.append(arm_group.sample(frac=frac, replace=replace, random_state=rng))
    return pd.concat(parts)


def process_bootstrap(
    data: pd.DataFrame,
    edge_priors: dict,
    prior_strength: float,
    score_fn: type,
    estimator: type,
    expert_knowledge: ExpertKnowledge,
    seed: int = 0,
    random_init: bool = False,
    subsample_frac: float = 0.65,
    replace: bool = True,
    interventional: bool = False,
    arm_labels: Optional[pd.Series] = None,
    clamped_nodes: Optional[dict] = None,
    arm_resample_floor: int = 0,
) -> Optional[DAG]:
    """
    Process single bootstrap sample for causal discovery with uncertainty quantification.

    Performs causal discovery on a bootstrap resample of the data using constrained
    Hill Climb search with biological priors. This function is designed for parallel
    execution to enable robust uncertainty estimation through bootstrap aggregation.

    The bootstrap procedure helps quantify uncertainty in causal edge discovery by:
    1. Resampling data with replacement
    2. Running constrained causal discovery
    3. Aggregating results across multiple bootstrap samples

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset to resample
    edge_priors : dict
        Dictionary of edge prior probabilities for biological constraints
    prior_strength : float
        Scaling factor for prior influence in scoring
    score_fn : type
        Scoring function class (AICGaussIndraPriors or BICGaussIndraPriors)
    estimator : type
        Causal discovery algorithm class (typically SparseHillClimb)
    expert_knowledge : ExpertKnowledge
        Hard constraints on required/forbidden edges
    seed : int, optional
        Random seed for reproducible bootstrap resampling. Default is 0.
    random_init : bool, optional
        If True, initialize the hill climb search from a random acyclic subgraph
        rather than an empty DAG. This can help escape local optima but increases
        run-to-run variability. Default is False.
    interventional : bool, optional
        Passed through to `score_fn` (only meaningful for a scoring class that
        supports it, e.g. `BICGaussIndraPriors`). Default is False, which never
        changes this function's resampling or scoring-construction code path -
        see Notes.
    arm_labels : Optional[pd.Series], optional
        Per-sample experimental-arm label aligned to `data`'s index. Required
        for `interventional` to take effect - default is None.
    clamped_nodes : Optional[dict], optional
        Passed through to `score_fn` unchanged when `interventional` is True.
    arm_resample_floor : int, optional
        Only meaningful when `arm_labels` is not None. Arms with fewer than this many
        rows are kept in full (unresampled) on every bootstrap draw rather than being
        bootstrap-resampled at (`subsample_frac`, `replace`) like the rest of the data
        - see `_resample_with_arm_floor`. Default is 0, which disables this and
        reproduces the original single pooled `.sample(...)` call exactly.

    Returns
    -------
    Optional[DAG]
        Estimated causal DAG from bootstrap sample, or None if discovery fails

    Examples
    --------
    >>> # Single bootstrap iteration
    >>> dag = process_bootstrap(
    ...     data=proteomics_data,
    ...     edge_priors=indra_priors,
    ...     prior_strength=2.0,
    ...     score_fn=BICGaussIndraPriors,
    ...     estimator=SparseHillClimb,
    ...     expert_knowledge=constraints
    ... )

    Notes
    -----
    This function includes error handling to gracefully manage numerical
    issues or convergence failures that may occur during bootstrap resampling.
    Failed bootstrap samples return None and are excluded from aggregation.

    The logging suppression prevents verbose output during parallel execution
    while maintaining error reporting for debugging.

    With `arm_labels=None` (the default), `resampled_data`/`custom_score`
    construction are exactly what they were before `interventional` existed -
    the two branches below are never merged into one code path so that case
    stays byte-for-byte unchanged.
    """
    import logging

    # try:
    # Suppress INFO logs from pgmpy in this subprocess
    logging.getLogger("pgmpy").setLevel(logging.WARNING)

    rng = np.random.RandomState(seed)
    # subsample_frac<1 with replace=True -> a bootstrap resample (consensus mode);
    # subsample_frac=1 with replace=False -> the full data (best-of-restarts mode).
    if arm_labels is not None:
        # Resample data and arm_labels TOGETHER, in one .sample() call on a
        # combined frame, so a bootstrap resample can never desynchronize which
        # arm label goes with which resampled row. Two separate .sample() calls
        # sharing one RandomState would each advance its internal state and draw
        # DIFFERENT rows on the second call - not the "same" resample.
        combined = data.copy()
        combined["__arm_label__"] = arm_labels
        resampled_combined = _resample_with_arm_floor(
            combined, "__arm_label__", subsample_frac, replace, arm_resample_floor, rng
        )
        resampled_arm_labels = resampled_combined.pop("__arm_label__")
        resampled_data = resampled_combined
    else:
        resampled_data = data.sample(frac=subsample_frac, replace=replace, random_state=rng)
        resampled_arm_labels = None

    # Initialize the custom scoring function. interventional_kwargs stays empty
    # unless interventional=True was explicitly requested, so score_fn classes
    # that don't accept these kwargs at all (today: the AICGauss* pair -
    # BICGaussIndraPriors and BICGaussNoPriors both do) are unaffected by this
    # parameter existing.
    interventional_kwargs = {}
    if interventional:
        interventional_kwargs = dict(
            interventional=True, arm_labels=resampled_arm_labels, clamped_nodes=clamped_nodes
        )
    custom_score = score_fn(
        resampled_data,
        edge_priors=edge_priors,
        prior_strength=prior_strength,
        **interventional_kwargs,
    )

    allowed = set(edge_priors.keys())
    est = estimator(data=resampled_data, allowed_additions=allowed)

    start_dag = None
    if random_init:
        nodes = list(resampled_data.columns)
        start_dag = random_acyclic_subgraph(nodes, allowed, 0.15, np.random.default_rng(seed))

    # Estimate the DAG using the custom scoring function
    estimated_dag = est.estimate(
        scoring_method=custom_score,
        start_dag=start_dag,
        expert_knowledge=expert_knowledge,
        max_indegree=3,
        epsilon=0.0001,
        show_progress=False,
    )
    return estimated_dag


def run_bootstrap(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float,
    scoring_function: type,
    search_algorithm: type,
    expert_knowledge: ExpertKnowledge,
    add_high_corr_edges_to_priors: bool = False,
    corr_threshold: float = 0.8,
    n_bootstrap: int = 100,
    convert_to_probability: bool = True,
    use_source_counts: bool = False,
    random_init: bool = False,
    subsample_frac: float = 0.65,
    replace: bool = True,
    verbose: bool = True,
    interventional: bool = False,
    arm_labels: Optional[pd.Series] = None,
    clamped_nodes: Optional[dict] = None,
    arm_resample_floor: int = 0,
) -> list:
    """
    Run parallel bootstrap analysis for robust causal discovery with INDRA priors.

    Performs bootstrap resampling and causal discovery to quantify uncertainty
    in learned causal relationships using INDRA biological prior knowledge.
    This approach provides confidence estimates for individual edges by examining
    their frequency across bootstrap samples while leveraging biological constraints.

    The function automatically processes INDRA priors to extract edge probabilities
    using power law modeling, then runs parallel bootstrap analysis for efficient
    uncertainty quantification in biologically-informed causal discovery.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset for bootstrap resampling with samples as rows and variables as columns
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with columns:
        - 'source': Source protein/gene symbols
        - 'target': Target protein/gene symbols
        - 'evidence_count': Evidence count for each relationship
    prior_strength : float
        Scaling factor for biological prior influence in scoring functions
    scoring_function : type
        Scoring function class with prior integration (AICGaussIndraPriors or BICGaussIndraPriors)
    search_algorithm : type
        Causal discovery algorithm class (typically SparseHillClimb)
    expert_knowledge : ExpertKnowledge
        Hard constraints on graph structure (required/forbidden edges)
    add_high_corr_edges_to_priors: bool
        If True, identify highly correlated variable pairs in the data and
        remove edges between them from the blacklist. This helps retain
        potentially valid causal edges that might otherwise be excluded.
    n_bootstrap : int
        Number of bootstrap samples to generate for uncertainty quantification
    convert_to_probability : bool
        If True, convert INDRA evidence counts to edge probabilities using power law modeling
    random_init : bool, optional
        If True, initialize each bootstrap hill climb from a random acyclic subgraph
        rather than an empty DAG. Default is False.

    Returns
    -------
    list
        List of estimated DAGs from bootstrap samples.
        Failed samples are excluded (None values filtered out).

    Examples
    --------
    >>> # Prepare INDRA prior data
    >>> indra_df = pd.DataFrame({
    ...     'source': ['AKT1', 'TP53', 'MDM2'],
    ...     'target': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>>
    >>> # Run bootstrap causal discovery with biological priors
    >>> bootstrap_dags = run_bootstrap(
    ...     data=proteomics_data,
    ...     indra_priors=indra_df,
    ...     prior_strength=2.0,
    ...     scoring_function=BICGaussIndraPriors,
    ...     search_algorithm=SparseHillClimb,
    ...     expert_knowledge=ExpertKnowledge(),
    ...     n_bootstrap=100
    ... )
    >>>
    >>> # Analyze edge confidence from bootstrap results
    >>> edge_counts = Counter()
    >>> for dag in bootstrap_dags:
    ...     if dag is not None:  # Filter out failed bootstraps
    ...         edge_counts.update(dag.edges())
    >>> edge_frequencies = {
    ...     edge: count/len([d for d in bootstrap_dags if d is not None])
    ...     for edge, count in edge_counts.items()
    ... }

    Notes
    -----
    Workflow:
    1. Convert INDRA evidence counts to edge probabilities using power law modeling
    2. Run parallel bootstrap resampling with constrained causal discovery
    3. Aggregate results for uncertainty quantification

    The parallel execution uses n_jobs=-2 to reserve one CPU core for system
    processes while maximizing computational throughput. This prevents system
    overload during intensive bootstrap computations.

    Bootstrap aggregation provides several benefits for biological applications:
    - Confidence intervals for individual causal relationships
    - Robust consensus network structure from noisy biological data
    - Uncertainty quantification for causal claims in publications
    - Model stability assessment across data perturbations

    Typical bootstrap sample sizes for biological networks:
    - Small networks (< 20 nodes): 50-100 samples
    - Medium networks (20-100 nodes): 100-500 samples
    - Large networks (> 100 nodes): 200-1000 samples

    The choice depends on computational resources and required precision
    for downstream biological interpretation and hypothesis generation.

    interventional, arm_labels, clamped_nodes, arm_resample_floor are forwarded to
    `process_bootstrap` (and from there to `score_fn`) unchanged - default is
    `interventional=False`, `arm_labels=None`, `arm_resample_floor=0`, which never
    alters this function's own behavior; only the values ultimately reaching
    `process_bootstrap` change.
    """
    if verbose:
        print("INFO: Starting bootstrap causal discovery:")
    if add_high_corr_edges_to_priors:
        if verbose:
            print("INFO: Adding high-corr edges to priors:")
        updated_indra_priors, updated_blacklist = remove_high_corr_edges_from_blacklist(
            data, indra_priors, expert_knowledge.forbidden_edges, corr_threshold, verbose=verbose
        )
        expert_knowledge.forbidden_edges = updated_blacklist
    else:
        updated_indra_priors = indra_priors

    if verbose:
        print("INFO: Calculating edge probabilities.")

    edge_probabilities = prepare_indra_priors(
        updated_indra_priors, convert_to_probability, use_source_counts
    )

    if verbose:
        print("INFO: Running bootstrap.")
    with _tqdm_joblib(tqdm(total=n_bootstrap, desc="Hill Climb runs")):
        bootstrap_dags = Parallel(n_jobs=-2)(
            delayed(process_bootstrap)(
                data,
                edge_probabilities,
                prior_strength,
                scoring_function,
                search_algorithm,
                expert_knowledge,
                seed=i,
                random_init=random_init,
                subsample_frac=subsample_frac,
                replace=replace,
                interventional=interventional,
                arm_labels=arm_labels,
                clamped_nodes=clamped_nodes,
                arm_resample_floor=arm_resample_floor,
            )
            for i in range(n_bootstrap)
        )
    # for _ in range(n_bootstrap):
    #     process_bootstrap(
    #         data,
    #         edge_probabilities,
    #         prior_strength,
    #         scoring_function,
    #         search_algorithm,
    #         expert_knowledge,
    #     )

    return bootstrap_dags
