"""Hill-climb structure search restricted to a prior edge set.

:class:`SparseHillClimb` is pgmpy's ``HillClimbSearch`` with one change that
matters at biological scale: candidate edge additions are drawn from an explicit
``allowed_additions`` set (the INDRA prior network) rather than from all
``n * (n - 1)`` ordered node pairs. On a few hundred proteins that turns each
iteration from quadratic in node count into linear in prior-edge count.

It also fixes a cycle check in the FLIP operation -- see ``_legal_operations``.
"""

import logging
from collections import deque
from typing import (
    Any,
    Callable,
    Deque,
    Generator,
    Hashable,
    Iterable,
    Optional,
    Set,
    Tuple,
)

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, HillClimbSearch
from pgmpy.estimators.ScoreCache import ScoreCache
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
    cache_size : Optional[int], default=None
        Maximum number of local scores held in the score cache. ``None`` sizes it
        from the candidate-edge count, which is what keeps runtime linear in the
        prior size -- see :meth:`_resolve_cache_size`.
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

    # Measured on hub-heavy priors: the distinct (variable, parent-set) keys a whole
    # search touches sit at ~3x the candidate-edge count, because a candidate is
    # re-keyed each time its child's parent set changes. Sizing the cache above that
    # working set is what keeps runtime linear in prior size; pgmpy's fixed 10_000
    # default starts evicting near ~3_500 candidate edges, and once iteration t's
    # entries are gone before iteration t+1 reuses them every candidate pays a fresh
    # GLM fit (measured 22x slower at the point the cache is undersized).
    _CACHE_KEYS_PER_CANDIDATE = 4
    _CACHE_SIZE_FLOOR = 10_000

    def __init__(
        self,
        data: pd.DataFrame,
        allowed_additions: Optional[Iterable[Tuple[str, str]]] = None,
        use_cache: bool = True,
        cache_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(data, use_cache=use_cache, **kwargs)
        self.allowed_additions = set(allowed_additions) if allowed_additions else None
        self.cache_size = cache_size

    def _resolve_cache_size(self) -> int:
        """Score-cache capacity: explicit if given, else scaled to the candidate count.

        At ~248 bytes per entry this stays cheap -- a 20_000-edge prior asks for
        ~80_000 entries (~20 MB), paid once per worker process.
        """
        if self.cache_size is not None:
            return int(self.cache_size)

        if self.allowed_additions is not None:
            n_candidates = len(self.allowed_additions)
        else:
            n_vars = len(self.variables)
            n_candidates = n_vars * (n_vars - 1)

        return max(self._CACHE_SIZE_FLOOR, self._CACHE_KEYS_PER_CANDIDATE * n_candidates)

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
        # Wrap the cache here rather than letting get_scoring_method do it: its
        # ScoreCache is hard-coded to max_size=10_000, which is far below the working
        # set of a large prior. Passing use_cache=False and wrapping ourselves also
        # avoids double-wrapping a ScoreCache handed in as `scoring_method`.
        score, score_c = get_scoring_method(scoring_method, self.data, False)
        if self.use_cache and not isinstance(score_c, ScoreCache):
            score_c = ScoreCache(score_c, self.data, max_size=self._resolve_cache_size())
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

        # Build the constraint sets ONCE. These never change during a search, but
        # `_legal_operations` used to re-copy them on every iteration -- and callers
        # (e.g. causomic.network.estimate_posterior_dag) pass the full O(n^2)
        # complement of the prior as `forbidden_edges`, which is ~4M pairs at 2000
        # columns and cost ~340 ms per iteration just to copy.
        forbidden = frozenset(expert_knowledge.forbidden_edges)
        required = frozenset(expert_knowledge.required_edges)
        if self.allowed_additions is not None:
            # Every forbidden pair is only ever tested against a candidate drawn from
            # `allowed_additions` (additions directly, flips via their reverse), so
            # the overlap decides every check the full set would. Shrinking to it is
            # exact, and collapses the complement-style list to near-nothing.
            forbidden = forbidden & self.allowed_additions

        it = trange(int(max_iter)) if show_progress else range(int(max_iter))
        for t in it:
            best_op, best_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    forbidden,
                    required,
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

        forbidden = (
            forbidden_edges
            if isinstance(forbidden_edges, (set, frozenset))
            else set(forbidden_edges)
        )
        required = (
            required_edges if isinstance(required_edges, (set, frozenset)) else set(required_edges)
        )

        sp_add = structure_score("+")
        sp_remove = structure_score("-")
        sp_flip = structure_score("flip")

        # Group candidate additions by child. Everything a candidate needs except the
        # new parent itself -- the current parent set, the indegree headroom, the base
        # score, and the reachable set for the cycle check -- depends only on the
        # child, so a hub with d candidate parents does that work once instead of d
        # times. The cycle check in particular drops from d bidirectional searches to
        # a single descendant traversal.
        by_child: dict = {}
        for X, Y in potential:
            if X != Y:
                by_child.setdefault(Y, []).append(X)

        for Y, parents_new in by_child.items():
            parents_old = model.get_parents(Y)
            if len(parents_old) + 1 > max_indegree:
                continue

            descendants = None
            base = None
            for X in parents_new:
                op = ("+", (X, Y))
                # cheap checks first; avoid expensive path query early
                if (op in tabu) or ((X, Y) in forbidden):
                    continue
                # cycle check: X reachable from Y means Y~>X, so adding X->Y closes a
                # loop. Same test as has_path(model, Y, X), computed once per child.
                if descendants is None:
                    descendants = nx.descendants(model, Y)
                if X in descendants:
                    continue
                if base is None:
                    base = score(Y, parents_old)
                yield (op, score(Y, parents_old + [X]) - base + sp_add)

        # --- REMOVE: only current edges
        removable: dict = {}
        for X, Y in existing:
            op = ("-", (X, Y))
            if (op in tabu) or ((X, Y) in required):
                continue
            removable.setdefault(Y, []).append(X)

        for Y, parents_drop in removable.items():
            p_old = model.get_parents(Y)
            base = score(Y, p_old)
            for X in parents_drop:
                p_new = [v for v in p_old if v != X]
                yield (("-", (X, Y)), score(Y, p_new) - base + sp_remove)

        # --- FLIP: only if reverse is allowed (if using allowed_additions)
        for X, Y in list(existing):
            op = ("flip", (X, Y))
            if (op in tabu) or (("flip", (Y, X)) in tabu) or ((X, Y) in required):
                continue
            if self.allowed_additions is not None and (Y, X) not in self.allowed_additions:
                continue
            if (Y, X) in forbidden:
                continue
            # Cycle check for the flip X->Y => Y->X. After removing X->Y, adding
            # Y->X creates a cycle iff a directed path X~>Y still exists (X~>Y plus
            # Y->X closes a loop), so we must test that direction. The previous
            # check used has_path(Y, X), which let cycle-creating flips through and
            # produced non-DAG search outputs.
            Xp = model.get_parents(X)
            Yp = model.get_parents(Y)
            if len(Xp) + 1 > max_indegree:
                continue

            model.remove_edge(X, Y)
            if nx.has_path(model, X, Y):
                model.add_edge(X, Y)
                continue
            model.add_edge(X, Y)

            delta = (score(X, Xp + [Y]) - score(X, Xp)) + (
                score(Y, [v for v in Yp if v != X]) - score(Y, Yp)
            )
            yield (op, delta + sp_flip)


def random_acyclic_subgraph(nodes, allowed_edges, inclusion_prob=0.15, rng=None, max_indegree=2):
    """Generate a random DAG by greedily adding allowed edges without creating cycles.

    Parameters
    ----------
    nodes : list
        Node labels for the DAG
    allowed_edges : iterable of (str, str)
        Candidate edges to sample from
    inclusion_prob : float, default=0.15
        Probability of attempting to include each edge
    rng : numpy Generator, optional
        Random number generator for reproducibility
    max_indegree : int, default=2
        Maximum number of parents allowed per node
    """
    if rng is None:
        rng = np.random.default_rng()

    dag = DAG()
    dag.add_nodes_from(nodes)

    edges = list(allowed_edges)
    rng.shuffle(edges)

    for u, v in edges:
        if rng.random() > inclusion_prob:
            continue
        if len(dag.get_parents(v)) >= max_indegree:
            continue
        dag.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(dag):
            dag.remove_edge(u, v)

    return dag
