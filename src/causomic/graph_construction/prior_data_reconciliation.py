"""
Prior-Data Reconciliation for Causal Network Discovery

This module implements sophisticated algorithms for reconciling INDRA-derived
biological prior knowledge with experimental proteomics data to construct
refined causal graphs. The implementation leverages constrained Hill Climb
search algorithms with custom scoring functions that balance data fit with
prior biological knowledge.

The core innovation is the SparseHillClimb algorithm, which restricts the
search space to biologically plausible edges defined by INDRA prior knowledge,
significantly improving computational efficiency while maintaining causal
discovery performance.

Key Components
--------------
- SparseHillClimb: Constrained Hill Climb search with predefined edge sets
- Custom Scoring Functions: AIC/BIC variants with soft INDRA priors
- Bootstrap Framework: Robust causal discovery with uncertainty quantification
- Prior Integration: Seamless combination of biological knowledge and data

Typical Workflow
----------------
1. Define allowed edges from INDRA prior knowledge
2. Compute edge probabilities from biological evidence
3. Run constrained Hill Climb search with custom scoring
4. Perform bootstrap analysis for uncertainty quantification
5. Extract consensus causal network structure

Examples
--------
>>> # Basic prior-data reconciliation
>>> edge_priors = {("AKT1", "MDM2"): 0.8, ("TP53", "MDM2"): 0.9}
>>> allowed_edges = list(edge_priors.keys())
>>>
>>> # Initialize constrained search
>>> search = SparseHillClimb(data, allowed_additions=allowed_edges)
>>> scoring = AICGaussIndraPriors(data, edge_priors=edge_priors)
>>>
>>> # Discover causal network
>>> causal_dag = search.estimate(scoring_method=scoring)

Author: Devon Kohler
Date: 2024
"""

# Standard library imports
import logging
from collections import deque
from typing import Callable, Deque, Generator, Hashable, Iterable, Optional, Tuple
from xml.parsers.expat import model

# Scientific computing imports
import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize as opt

# Parallel processing and progress tracking
from joblib import Parallel, delayed

# Causal discovery and probabilistic graphical models
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch, LogLikelihoodGauss
from pgmpy.estimators.StructureScore import get_scoring_method
from tqdm.auto import trange


class SparseHillClimb(HillClimbSearch):
    """
    Constrained Hill Climb search for causal discovery with prior knowledge.

    This class extends pgmpy's HillClimbSearch to support restricting edge
    additions to a predefined set of biologically plausible relationships.
    Unlike the standard implementation that considers all possible edges,
    this sparse variant dramatically reduces search space complexity while
    incorporating prior biological knowledge.

    The key innovation is constraining the edge addition operations to only
    those relationships supported by prior evidence (e.g., from INDRA database),
    which both speeds up discovery and improves biological plausibility of
    the resulting causal networks.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset with samples as rows and variables as columns
    allowed_additions : Optional[Iterable[Tuple[str, str]]], default=None
        Set of (parent, child) pairs representing biologically plausible edges.
        If None, falls back to standard HillClimbSearch behavior
    use_cache : bool, default=True
        Whether to cache scoring computations for efficiency
    **kwargs
        Additional arguments passed to parent HillClimbSearch class

    Attributes
    ----------
    allowed_additions : Optional[Set[Tuple[str, str]]]
        Set of allowed edge additions for constrained search

    Examples
    --------
    >>> # Define biologically plausible edges from prior knowledge
    >>> allowed_edges = [("AKT1", "MDM2"), ("TP53", "MDM2"), ("MDM2", "TP53")]
    >>>
    >>> # Initialize constrained search
    >>> search = SparseHillClimb(data, allowed_additions=allowed_edges)
    >>>
    >>> # Run causal discovery with biological constraints
    >>> causal_dag = search.estimate(scoring_method="bic")

    Notes
    -----
    This implementation is particularly valuable for biological applications where:
    - Prior knowledge about regulatory relationships exists
    - Computational efficiency is important for large networks
    - Biological plausibility of discovered edges is crucial

    The sparse constraint can reduce search space from O(n²) to O(k) where
    k is the number of allowed edges, providing substantial speedup for
    large biological networks.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        allowed_additions: Optional[Iterable[Tuple[str, str]]] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        super().__init__(data, use_cache=use_cache, **kwargs)
        self.allowed_additions = set(allowed_additions) if allowed_additions else None

    def estimate(
        self,
        scoring_method=None,
        start_dag: Optional[DAG] = None,
        tabu_length: int = 100,
        max_indegree: Optional[int] = None,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        show_progress: bool = True,
        on_step: Optional[Callable[[int, Tuple[str, Tuple[str, str]], float], None]] = None,
    ) -> DAG:
        """
        Estimate causal DAG using constrained Hill Climb search.

        Performs iterative local search through DAG space, constrained by
        allowed edge additions from prior knowledge. Each iteration evaluates
        add, remove, and flip operations, selecting the change that most
        improves the scoring function while respecting biological constraints.

        Parameters
        ----------
        scoring_method : str or scoring class, default=None
            Scoring function to optimize. Can be string ("bic", "aic") or
            custom scoring class instance
        start_dag : Optional[DAG], default=None
            Initial DAG structure. If None, starts with empty graph
        tabu_length : int, default=100
            Length of tabu list to prevent cycling in search
        max_indegree : Optional[int], default=None
            Maximum number of parents allowed per node
        expert_knowledge : Optional[ExpertKnowledge], default=None
            Hard constraints on required/forbidden edges
        epsilon : float, default=1e-4
            Minimum score improvement to continue search
        max_iter : int, default=1000000
            Maximum number of search iterations
        show_progress : bool, default=True
            Whether to display progress bar during search
        on_step : Optional[Callable], default=None
            Callback function called after each search step

        Returns
        -------
        DAG
            Estimated causal directed acyclic graph

        Examples
        --------
        >>> # Basic constrained search
        >>> dag = search.estimate(scoring_method="bic")
        >>>
        >>> # With custom scoring and constraints
        >>> expert = ExpertKnowledge()
        >>> expert.add_required_edge(("AKT1", "MDM2"))
        >>> dag = search.estimate(
        ...     scoring_method=custom_scorer,
        ...     expert_knowledge=expert,
        ...     max_indegree=3
        ... )

        Notes
        -----
        The algorithm terminates when either:
        - No operation improves score by more than epsilon
        - Maximum iterations reached
        - No legal operations remain

        Constraint enforcement significantly reduces computational complexity
        compared to unconstrained search, especially for large biological networks.
        """
        score, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)

        expert_knowledge = expert_knowledge or ExpertKnowledge()

        if not nx.is_directed_acyclic_graph(start_dag):
            raise ValueError("required_edges create a cycle in start_dag.")

        max_indegree = float("inf") if max_indegree is None else max_indegree
        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag

        it = trange(int(max_iter)) if show_progress else range(int(max_iter))
        for t in it:
            best_op, best_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    expert_knowledge.forbidden_edges,
                    expert_knowledge.required_edges,
                ),
                key=lambda x: x[1],
                default=(None, None),
            )

            if show_progress:
                try:
                    it.set_postfix({"Δscore": f"{best_delta:.4f}"})
                except Exception:
                    pass

            if on_step is not None:
                on_step(t, best_op, best_delta)

            if best_op is None or best_delta < epsilon:
                break
            if best_op[0] == "+":
                current_model.add_edge(*best_op[1])
                tabu_list.append(("-", best_op[1]))
            elif best_op[0] == "-":
                current_model.remove_edge(*best_op[1])
                tabu_list.append(("+", best_op[1]))
            else:  # flip
                X, Y = best_op[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_op)

        return current_model

    def _legal_operations(
        self,
        model: DAG,
        score: Callable,
        structure_score: Callable,
        tabu_list: Deque[Tuple[str, Tuple[Hashable, Hashable]]],
        max_indegree: int,
        forbidden_edges: Iterable[Tuple[str, str]],
        required_edges: Iterable[Tuple[str, str]],
    ) -> Generator[Tuple[Tuple[str, Tuple[Hashable, Hashable]], float], None, None]:
        """
        Generate all legal operations with their score improvements.

        Evaluates three types of operations: edge addition, edge removal, and
        edge reversal. For addition operations, restricts candidates to the
        allowed_additions set if provided, dramatically reducing search space
        for biological applications.

        Parameters
        ----------
        model : DAG
            Current DAG structure being evaluated
        score : Callable
            Local scoring function for individual variables
        structure_score : Callable
            Prior probability function for structure changes
        tabu_list : Deque
            Recent operations to avoid cycling
        max_indegree : int
            Maximum allowed parents per node
        forbidden_edges : Iterable[Tuple[str, str]]
            Hard-forbidden edge constraints
        required_edges : Iterable[Tuple[str, str]]
            Hard-required edge constraints

        Yields
        ------
        Tuple[Tuple[str, Tuple[str, str]], float]
            Operation and its score improvement: ((op_type, (parent, child)), delta)
            where op_type is "+", "-", or "flip"

        Notes
        -----
        The key innovation is constraining ADD operations to allowed_additions,
        which reduces complexity from O(n²) to O(k) where k is the number of
        biologically plausible edges. This maintains discovery quality while
        dramatically improving computational efficiency.

        Operations are filtered by:
        - Tabu list (avoid recent operations)
        - Expert knowledge constraints
        - Acyclicity requirements
        - Maximum indegree limits
        - Biological plausibility (for additions)
        """
        tabu = set(tabu_list)
        existing = set(model.edges())

        # --- ADD: iterate only allowed candidates (if provided)
        if self.allowed_additions is not None:
            potential = self.allowed_additions - existing - {(y, x) for (x, y) in existing}
        else:
            # fall back to full scan
            from itertools import permutations

            potential = (
                set(permutations(self.variables, 2)) - existing - {(y, x) for (x, y) in existing}
            )

        forbidden = set(forbidden_edges)
        required = set(required_edges)

        for X, Y in potential:
            op = ("+", (X, Y))
            # cheap checks first; avoid expensive path query early
            if (op in tabu) or ((X, Y) in forbidden):
                continue
            # cycle check
            if nx.has_path(model, Y, X):
                continue
            parents_old = model.get_parents(Y)
            if len(parents_old) + 1 <= max_indegree:
                delta = score(Y, parents_old + [X]) - score(Y, parents_old)
                delta += structure_score("+")
                yield (op, delta)

        # --- REMOVE: only current edges (unchanged)
        for X, Y in list(existing):
            op = ("-", (X, Y))
            if (op in tabu) or ((X, Y) in required):
                continue
            p_old = model.get_parents(Y)
            p_new = [v for v in p_old if v != X]
            delta = score(Y, p_new) - score(Y, p_old)
            delta += structure_score("-")
            yield (op, delta)

        # --- FLIP: only if reverse is allowed (if using allowed_additions)
        for X, Y in list(existing):
            op = ("flip", (X, Y))
            if (op in tabu) or (("flip", (Y, X)) in tabu) or ((X, Y) in required):
                continue
            if self.allowed_additions is not None and (Y, X) not in self.allowed_additions:
                continue
            if (Y, X) in forbidden:
                continue
            # cycle check for flips
            # if any(len(path) > 2 for path in nx.all_simple_paths(model, X, Y)):
            #     continue
            model.remove_edge(X, Y)
            if nx.has_path(model, Y, X):
                model.add_edge(X, Y)
                continue
            model.add_edge(X, Y)
            
            Xp = model.get_parents(X)
            Yp = model.get_parents(Y)
            if len(Xp) + 1 <= max_indegree:
                delta = (score(X, Xp + [Y]) - score(X, Xp)) + (
                    score(Y, [v for v in Yp if v != X]) - score(Y, Yp)
                )
                delta += structure_score("flip")
                yield (op, delta)


class AICGaussIndraPriors(LogLikelihoodGauss):
    """
    AIC scoring with soft INDRA biological priors.

    Extends standard AIC (Akaike Information Criterion) scoring to incorporate
    soft biological priors from INDRA knowledge base. The scoring function
    balances data fit with biological plausibility, encouraging edges with
    strong prior evidence while penalizing model complexity.

    The score combines:
    - Standard AIC: log-likelihood - (df + 2)
    - Prior bonus: Σ log(p_ij / (1 - p_ij)) for edges with prior probability p_ij

    This approach provides a principled way to incorporate biological knowledge
    without hard constraints, allowing data to override weak priors when
    evidence is strong.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset for scoring
    edge_priors : Optional[Dict[Tuple[str, str], float]], default=None
        Dictionary mapping (parent, child) tuples to prior probabilities [0,1].
        Higher values indicate stronger biological evidence for the edge
    prior_strength : float, default=1.0
        Scaling factor for prior influence (λ parameter)
    **kwargs
        Additional arguments passed to LogLikelihoodGauss

    Attributes
    ----------
    edge_priors : Dict[Tuple[str, str], float]
        Edge prior probabilities
    prior_strength : float
        Prior influence scaling parameter

    Examples
    --------
    >>> # Define biological priors from INDRA evidence
    >>> priors = {
    ...     ("AKT1", "MDM2"): 0.8,    # Strong evidence
    ...     ("TP53", "MDM2"): 0.9,    # Very strong evidence
    ...     ("MDM2", "TP53"): 0.7     # Moderate evidence
    ... }
    >>>
    >>> # Initialize AIC scorer with priors
    >>> scorer = AICGaussIndraPriors(data, edge_priors=priors, prior_strength=2.0)
    >>>
    >>> # Use in causal discovery
    >>> search = SparseHillClimb(data)
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    The log-odds transformation log(p/(1-p)) provides symmetric treatment
    of prior evidence: strong positive evidence (p=0.9) gives +log(9),
    while strong negative evidence (p=0.1) gives -log(9).

    Prior strength parameter allows tuning the balance between data fit
    and biological plausibility based on confidence in prior knowledge.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        **kwargs,
    ):
        super(AICGaussIndraPriors, self).__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda

    def local_score(self, variable: str, parents: list) -> float:
        """
        Compute AIC score with biological priors for a variable given its parents.

        Calculates the local score for a variable conditioned on its parent set,
        combining standard AIC penalty with biological prior information.
        Higher scores indicate better model fit and/or stronger prior support.

        Parameters
        ----------
        variable : str
            Target variable to score
        parents : list
            List of parent variable names

        Returns
        -------
        float
            Local AIC score with prior bonus. Higher values are better.

        Notes
        -----
        Score decomposition:
        - Base AIC: log-likelihood - (degrees_of_freedom + 2)
        - Prior bonus: Σ log(p/(1-p)) × prior_strength for each parent edge
        - Final score: AIC - prior_bonus (subtraction because we return negative AIC)

        The log-odds formulation ensures that:
        - p > 0.5 gives positive bonus (encourages edge)
        - p < 0.5 gives negative bonus (discourages edge)
        - p = 0.5 gives zero bonus (neutral)
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Standard AIC score
        aic_score = ll - (df_model + 2)

        # Soft prior component
        prior_bonus = 0
        for parent in parents:
            p = self.edge_priors.get(variable, {}).get(parent, 0.5)
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)

            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

        prior_bonus *= self.prior_strength

        return aic_score + prior_bonus


class BICGaussIndraPriors(LogLikelihoodGauss):
    """
    BIC scoring with soft INDRA biological priors.

    Extends standard BIC (Bayesian Information Criterion) scoring to incorporate
    soft biological priors from INDRA knowledge base. BIC applies stronger
    penalty for model complexity than AIC, making it more conservative in
    edge selection while still benefiting from biological prior knowledge.

    The score combines:
    - Standard BIC: log-likelihood - ((df + 2)/2) × log(n)
    - Prior bonus: Σ log(p_ij / (1 - p_ij)) for edges with prior probability p_ij

    BIC's stronger complexity penalty makes it particularly suitable when
    aiming for sparse, interpretable causal networks with high confidence.

    Parameters
    ----------
    data : pd.DataFrame
        Observational dataset for scoring
    edge_priors : Optional[Dict[Tuple[str, str], float]], default=None
        Dictionary mapping (parent, child) tuples to prior probabilities [0,1]
    prior_strength : float, default=1.0
        Scaling factor for prior influence (λ parameter)
    **kwargs
        Additional arguments passed to LogLikelihoodGauss

    Attributes
    ----------
    edge_priors : Dict[Tuple[str, str], float]
        Edge prior probabilities
    prior_strength : float
        Prior influence scaling parameter

    Examples
    --------
    >>> # Initialize BIC scorer with biological priors
    >>> priors = {("AKT1", "MDM2"): 0.85, ("TP53", "MDM2"): 0.92}
    >>> scorer = BICGaussIndraPriors(data, edge_priors=priors)
    >>>
    >>> # Use in constrained causal discovery
    >>> search = SparseHillClimb(data, allowed_additions=list(priors.keys()))
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    BIC vs AIC trade-offs:
    - BIC: More conservative, stronger complexity penalty, better for sparse networks
    - AIC: More liberal, weaker complexity penalty, better for predictive models

    Choose BIC when interpretability and network sparsity are priorities.
    Choose AIC when predictive performance is the primary concern.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        edge_priors: Optional[dict] = None,
        prior_strength: float = 1.0,
        **kwargs,
    ):
        super().__init__(data, **kwargs)
        self.edge_priors = edge_priors or {}
        self.prior_strength = prior_strength  # This is lambda

    def local_score(self, variable: str, parents: list) -> float:
        """
        Compute BIC score with biological priors for a variable given its parents.

        Calculates the local score using BIC criterion enhanced with biological
        prior information. BIC applies stronger complexity penalty than AIC,
        promoting sparser models while incorporating prior biological knowledge.

        Parameters
        ----------
        variable : str
            Target variable to score
        parents : list
            List of parent variable names

        Returns
        -------
        float
            Local BIC score with prior bonus. Higher values are better.
            Returns -inf if computation fails (singular matrix, etc.)

        Notes
        -----
        Score decomposition:
        - Base BIC: log-likelihood - ((df + 2)/2) × log(n)
        - Prior bonus: Σ log(p/(1-p)) × prior_strength for each parent edge
        - Final score: BIC + prior_bonus

        BIC's log(n) factor creates stronger penalty for complexity than AIC,
        making it more conservative in edge selection. This is beneficial when
        seeking interpretable, sparse causal networks.

        Error handling returns -inf for degenerate cases (singular covariance
        matrices, etc.) to exclude them from consideration.
        """
        try:
            ll, df_model = self._log_likelihood(variable=variable, parents=parents)
        except:
            # statsmodels will raise ValueError if X is singular
            return -np.inf

        # Standard BIC score
        bic_score = ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))

        # Soft prior component
        prior_bonus = 0
        for parent in parents:
            p = self.edge_priors[(parent, variable)]
            p = np.clip(p, 1e-6, 1 - 1e-6)  # Avoid log(0)
            log_odds = np.log(p / (1 - p))
            prior_bonus += log_odds

        prior_bonus *= self.prior_strength
        return bic_score + prior_bonus


def process_bootstrap(
    data: pd.DataFrame,
    edge_priors: dict,
    prior_strength: float,
    score_fn: type,
    estimator: type,
    expert_knowledge: ExpertKnowledge,
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
    """
    import logging

    try:
        # Suppress INFO logs from pgmpy in this subprocess
        logging.getLogger("pgmpy").setLevel(logging.WARNING)

        resampled_data = data.sample(n=len(data), replace=True)

        # Initialize the custom scoring function
        custom_score = score_fn(
            resampled_data, edge_priors=edge_priors, prior_strength=prior_strength
        )

        allowed = set(edge_priors.keys())
        est = estimator(data=resampled_data, allowed_additions=allowed)
        # Estimate the DAG using the custom scoring function
        estimated_dag = est.estimate(
            scoring_method=custom_score,
            expert_knowledge=expert_knowledge,
            max_indegree=3,
            epsilon=0.01,
            show_progress=False,
        )
        print("one bootstrap finished")
        return estimated_dag
    except Exception as e:
        # Handle the exception here
        print(f"An error occurred during bootstrap processing: {e}")
        return None


def calculate_edge_probabilities(indra_priors: pd.DataFrame) -> dict:
    """
    Calculate edge probabilities from INDRA evidence counts using power law modeling.

    Converts raw INDRA evidence counts to edge probabilities by fitting a discrete
    power law distribution to the evidence count data. This approach recognizes that
    biological evidence follows heavy-tailed distributions where few relationships
    have extensive evidence while most have modest support.

    The power law model P(X = k) ∝ k^(-α) provides a principled way to transform
    evidence counts into probabilities that appropriately weight strong evidence
    while not completely dismissing weaker relationships.

    Parameters
    ----------
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with 'evidence_count' column

    Returns
    -------
    dict
        Mapping from evidence count values to cumulative probabilities [0,1].
        Higher evidence counts map to higher probabilities.

    Examples
    --------
    >>> # Process INDRA evidence counts
    >>> indra_df = pd.DataFrame({
    ...     'source_symbol': ['AKT1', 'TP53', 'MDM2'],
    ...     'target_symbol': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>> prob_mapping = calculate_edge_probabilities(indra_df)
    >>> # Returns: {8: 0.2, 15: 0.6, 25: 0.9} (example values)

    Notes
    -----
    Algorithm steps:
    1. Extract evidence counts and find minimum value (xmin)
    2. Fit power law exponent α using maximum likelihood estimation
    3. Compute discrete power law PMF: P(k) = k^(-α) / ζ(α, xmin)
    4. Calculate cumulative distribution function (CDF) values
    5. Return mapping from counts to CDF probabilities

    The power law model is particularly appropriate for biological networks where:
    - Few relationships have extensive experimental validation
    - Many relationships have limited but meaningful evidence
    - Evidence accumulation follows preferential attachment dynamics

    CDF transformation ensures that higher evidence counts receive higher
    probabilities while maintaining proper probability interpretation.
    """
    edge_evidence = indra_priors["evidence_count"].values.astype(int)

    xmin = edge_evidence.min()

    # Discrete Power Law Log-Likelihood
    def powerlaw_log_likelihood(alpha, data, xmin):
        n = len(data)
        log_sum = -alpha * np.sum(np.log(data))
        zeta = np.sum([k ** (-alpha) for k in range(xmin, max(data) + 1)])
        return -(log_sum - n * np.log(zeta))

    # Fit alpha using MLE
    res = opt.minimize_scalar(
        powerlaw_log_likelihood, bounds=(1.01, 10), args=(edge_evidence, xmin), method="bounded"
    )
    alpha_hat = res.x

    # Compute CDF values (discrete power law)
    support = np.arange(xmin, max(edge_evidence) + 1)
    pmf = support ** (-alpha_hat)
    pmf /= pmf.sum()
    cdf_vals = np.cumsum(pmf)

    value_to_cdf = dict(zip(support, cdf_vals))

    return value_to_cdf


def prepare_indra_priors(indra_priors: pd.DataFrame) -> dict:
    """
    Prepare INDRA prior data for causal discovery by converting to edge probabilities.

    Transforms INDRA evidence counts into edge probability dictionary suitable for
    constrained causal discovery algorithms. This function combines power law
    modeling of evidence counts with proper edge formatting for downstream analysis.

    The preparation process ensures that biological prior knowledge is properly
    encoded as soft constraints that can guide but not override strong data evidence
    during causal discovery.

    Parameters
    ----------
    indra_priors : pd.DataFrame
        DataFrame with INDRA prior information containing columns:
        - 'source_symbol': Source protein/gene symbol
        - 'target_symbol': Target protein/gene symbol
        - 'evidence_count': Number of supporting evidence instances

    Returns
    -------
    dict
        Dictionary mapping (source, target) tuples to edge probabilities [0,1].
        Format: {(source_symbol, target_symbol): probability}

    Examples
    --------
    >>> # Prepare INDRA priors for causal discovery
    >>> indra_df = pd.DataFrame({
    ...     'source_symbol': ['AKT1', 'TP53', 'MDM2'],
    ...     'target_symbol': ['MDM2', 'MDM2', 'TP53'],
    ...     'evidence_count': [15, 25, 8]
    ... })
    >>> edge_priors = prepare_indra_priors(indra_df)
    >>> # Returns: {('AKT1', 'MDM2'): 0.6, ('TP53', 'MDM2'): 0.9, ('MDM2', 'TP53'): 0.2}
    >>>
    >>> # Use in constrained causal discovery
    >>> search = SparseHillClimb(data, allowed_additions=list(edge_priors.keys()))
    >>> scorer = BICGaussIndraPriors(data, edge_priors=edge_priors)
    >>> dag = search.estimate(scoring_method=scorer)

    Notes
    -----
    This function serves as the bridge between INDRA biological knowledge and
    causal discovery algorithms by:

    1. Converting evidence counts to probabilities using power law modeling
    2. Formatting edges as (source, target) tuples for algorithm compatibility
    3. Handling missing evidence with default high probability (1.0)
    4. Ensuring consistent edge representation across the pipeline

    The resulting edge probabilities can be used in:
    - Constrained search algorithms (allowed_additions parameter)
    - Scoring functions with biological priors
    - Expert knowledge specification for hard constraints

    Missing evidence counts are filled with probability 1.0 to ensure all
    edges in the prior network are considered, even if evidence is sparse.
    """
    edge_probability_mapper = calculate_edge_probabilities(indra_priors)

    indra_priors["edge_p"] = indra_priors["evidence_count"].map(edge_probability_mapper).fillna(1.0)

    edge_probabilities = {
        (indra_priors.loc[i, "source_symbol"],
            indra_priors.loc[i, "target_symbol"],
        ): indra_priors.loc[i, "edge_p"]
        for i in range(len(indra_priors))
    }

    return edge_probabilities

def remove_high_corr_edges_from_blacklist(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    black_list: set,
    corr_threshold: float = 0.8
) -> set:
    """
    Remove edges between highly correlated variables from the blacklist.

    This function identifies pairs of variables in the dataset that exhibit
    high correlation (above a specified threshold) and removes any edges
    between these variables from the provided blacklist. It then adds the edges
    to the indra_priors DataFrame with a low prior probability (floor of 
    observed probabilities). This is useful in causal discovery to avoid 
    excluding potentially valid edges that may represent true causal 
    relationships rather than mere correlations.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables of interest.
    indra_priors : pd.DataFrame
        DataFrame containing INDRA prior information with columns:
        - 'source_symbol': Source protein/gene symbols
        - 'target_symbol': Target protein/gene symbols
        - 'evidence_count': Evidence count for each relationship
    black_list : set
        A set of (parent, child) tuples representing edges to be blacklisted.
    corr_threshold : float, default=0.9
        The correlation threshold above which edges will be removed from the blacklist.

    Returns
    -------
    set
        Updated blacklist with edges between highly correlated variables removed.

    Examples
    --------
    >>> # Example dataset
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 4, 6, 8, 10],
    ...     'C': [5, 4, 3, 2, 1]
    ... })
    >>>
    >>> # Initial blacklist with edges to be removed if highly correlated
    >>> blacklist = {('A', 'B'), ('B', 'C')}
    >>>
    >>> # Remove edges between highly correlated variables (threshold=0.9)
    >>> updated_blacklist = remove_high_corr_edges_from_blacklist(df, blacklist, corr_threshold=0.9)
    >>> print(updated_blacklist)
    {('B', 'C')}  # Edge ('A', 'B') removed due to high correlation

    Notes
    -----
    - The function computes the absolute correlation matrix of the dataset.
    - It identifies variable pairs with correlation above the specified threshold.
    - Edges between these highly correlated pairs are removed from the blacklist.
    - This helps retain potentially valid causal edges that might otherwise be excluded.
    """

    # Compute absolute correlation matrix
    corr_matrix = data.corr().abs()

    # Find pairs with correlation above threshold (excluding self-pairs)
    high_corr_pairs = set()
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] >= corr_threshold:
                high_corr_pairs.add((i, j))
                high_corr_pairs.add((j, i))  # Both directions

    print(f"High correlation pairs (threshold={corr_threshold}): {len(high_corr_pairs)}")

    # Remove highly correlated edges from blacklist
    updated_blacklist = set(edge for edge in black_list if edge not in high_corr_pairs)

    # Add missing high-corr edges to indra_priors DataFrame
    for (src, tgt) in high_corr_pairs:
        if not (
            ((indra_priors["source_symbol"] == src) & (indra_priors["target_symbol"] == tgt)).any()
        ):
            new_row = {"source_symbol": src, "target_symbol": tgt, "evidence_count": 1}
            indra_priors.loc[len(indra_priors)] = new_row

    return indra_priors, updated_blacklist

def run_bootstrap(
    data: pd.DataFrame,
    indra_priors: pd.DataFrame,
    prior_strength: float,
    scoring_function: type,
    search_algorithm: type,
    expert_knowledge: ExpertKnowledge,
    add_high_corr_edges_to_priors: bool = True,
    corr_threshold: float = 0.8,
    n_bootstrap: int = 100,
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
        - 'source_symbol': Source protein/gene symbols
        - 'target_symbol': Target protein/gene symbols
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

    Returns
    -------
    list
        List of estimated DAGs from bootstrap samples.
        Failed samples are excluded (None values filtered out).

    Examples
    --------
    >>> # Prepare INDRA prior data
    >>> indra_df = pd.DataFrame({
    ...     'source_symbol': ['AKT1', 'TP53', 'MDM2'],
    ...     'target_symbol': ['MDM2', 'MDM2', 'TP53'],
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
    """
    print("INFO: Starting bootstrap causal discovery:")
    if add_high_corr_edges_to_priors:
        print("INFO: Adding high-corr edges to priors:")
        updated_indra_priors, updated_blacklist = remove_high_corr_edges_from_blacklist(
            data, indra_priors, expert_knowledge.forbidden_edges, corr_threshold
            )
        expert_knowledge.forbidden_edges = updated_blacklist
    else:
        updated_indra_priors = indra_priors

    print("INFO: Calculating edge probabilities.")
    edge_probabilities = prepare_indra_priors(updated_indra_priors)

    print("INFO: Running bootstrap.")
    bootstrap_dags = Parallel(n_jobs=-2)(
        delayed(process_bootstrap)(
            data,
            edge_probabilities,
            prior_strength,
            scoring_function,
            search_algorithm,
            expert_knowledge,
        )
        for _ in range(n_bootstrap)
    )

    return bootstrap_dags    

def main():

    import pickle
    import time
    
    with open("data/model_input.pkl", "rb") as f:
        model_input = pickle.load(f)
    # model_input = list(model_input)
    # model_input[3] = 5
    # model_input = tuple(model_input)
    start_time = time.time()
    bootstrap_dags = run_bootstrap(*model_input, 50)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    main()