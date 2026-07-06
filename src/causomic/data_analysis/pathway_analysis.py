"""Pathway over-representation analysis (ORA) and pathway-library utilities.

Fetches gene-set libraries (e.g. via gseapy), runs over-representation analysis,
and provides tools for selecting a diverse, high-coverage subset of pathways and
exporting results for visualization.
"""

import heapq
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Set, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats

try:
    import gseapy

    _GSEAPY_AVAILABLE = True
except ImportError:
    _GSEAPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Pathway data fetching
# ---------------------------------------------------------------------------


def list_pathway_libraries() -> List[str]:
    """Return available Enrichr gene set library names (requires gseapy)."""
    _require_gseapy()
    return gseapy.get_library_name()


def fetch_pathway_library(library_name: str = "KEGG_2021_Human") -> Dict[str, List[str]]:
    """
    Fetch a gene set library from Enrichr as {pathway: [genes]}.

    Parameters
    ----------
    library_name : str
        Any name returned by ``list_pathway_libraries()``.
        Common options: "KEGG_2021_Human", "Reactome_2022",
        "MSigDB_Hallmark_2020", "GO_Biological_Process_2023".

    Returns
    -------
    dict mapping pathway name → list of gene symbols
    """
    _require_gseapy()
    return gseapy.get_library(library_name)


# ---------------------------------------------------------------------------
# Membership matrix
# ---------------------------------------------------------------------------


def build_membership_matrix(
    gene_universe: List[str],
    pathway_dict: Dict[str, List[str]],
    min_pathway_size: int = 5,
    max_pathway_size: int = 500,
) -> pd.DataFrame:
    """
    Build a boolean (G × P) membership matrix.

    Parameters
    ----------
    gene_universe : list of gene symbols that form the background universe
    pathway_dict  : {pathway_name: [gene_symbols]}
    min_pathway_size, max_pathway_size : filter pathways by how many
        universe genes they contain

    Returns
    -------
    pd.DataFrame of shape (len(gene_universe), n_pathways), dtype bool
    """
    universe_set = set(gene_universe)
    filtered = {
        name: genes
        for name, genes in pathway_dict.items()
        if min_pathway_size <= len(set(genes) & universe_set) <= max_pathway_size
    }
    M = pd.DataFrame(False, index=list(gene_universe), columns=list(filtered.keys()))
    for pathway, genes in filtered.items():
        members = [g for g in genes if g in universe_set]
        M.loc[members, pathway] = True
    return M


# ---------------------------------------------------------------------------
# Over-representation analysis (ORA)
# ---------------------------------------------------------------------------


def run_ora(
    network_genes: Union[List[str], Set[str], nx.Graph],
    pathway_dict: Optional[Dict[str, List[str]]] = None,
    background_genes: Optional[List[str]] = None,
    library_name: str = "KEGG_2021_Human",
    min_pathway_size: int = 5,
    max_pathway_size: int = 500,
    fdr_method: Literal["bh", "bonferroni"] = "bh",
) -> pd.DataFrame:
    """
    Run over-representation analysis via a one-sided hypergeometric test.

    For each pathway the p-value is P(X ≥ k) where X follows a hypergeometric
    distribution with parameters::

        N = |background|          (universe size)
        K = |pathway ∩ background| (pathway size in universe)
        n = |network ∩ background| (query set size)
        k = |pathway ∩ network|    (observed overlap)

    Parameters
    ----------
    network_genes : genes in the posterior network.
        Accepts a list/set of gene symbols or a ``networkx.Graph`` whose
        nodes are gene symbols.
    pathway_dict : optional pre-loaded {pathway: [genes]} dict.
        If None, ``library_name`` is fetched from Enrichr via gseapy.
    background_genes : optional universe of genes to test against.
        Defaults to the union of all pathway genes and network genes.
    library_name : Enrichr library used when ``pathway_dict`` is None.
    min_pathway_size, max_pathway_size : pathway size filters (applied
        within the background).
    fdr_method : "bh" (Benjamini–Hochberg) or "bonferroni".

    Returns
    -------
    pd.DataFrame sorted by FDR with columns:
        pathway, overlap, pathway_size, network_size, background_size,
        p_value, fdr, overlap_genes
    """
    # -- resolve network genes -----------------------------------------------
    if isinstance(network_genes, nx.Graph):
        network_genes = set(network_genes.nodes())
    else:
        network_genes = set(network_genes)

    # -- pathway data ---------------------------------------------------------
    if pathway_dict is None:
        pathway_dict = fetch_pathway_library(library_name)

    # -- background -----------------------------------------------------------
    if background_genes is None:
        all_pathway_genes: Set[str] = set(g for genes in pathway_dict.values() for g in genes)
        background_genes = list(all_pathway_genes | network_genes)
    background_set = set(background_genes)

    # -- membership matrix ---------------------------------------------------
    M = build_membership_matrix(background_genes, pathway_dict, min_pathway_size, max_pathway_size)

    N = len(background_set)
    n = len(network_genes & background_set)

    # -- hypergeometric tests ------------------------------------------------
    records = []
    for pathway in M.columns:
        pathway_gene_set = set(M.index[M[pathway]])
        K = len(pathway_gene_set)
        overlap_set = pathway_gene_set & network_genes
        k = len(overlap_set)
        p_val = stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
        records.append(
            {
                "pathway": pathway,
                "overlap": k,
                "pathway_size": K,
                "network_size": n,
                "background_size": N,
                "p_value": p_val,
                "overlap_genes": overlap_set,
            }
        )

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["fdr"] = _apply_fdr(df["p_value"].to_numpy(), method=fdr_method)
    return df.sort_values("fdr").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Greedy diverse pathway selection (ported from pathway_set.py)
# ---------------------------------------------------------------------------


def compute_jaccard_from_membership(M_bool: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Jaccard similarity across pathways.

    Parameters
    ----------
    M_bool : (G × P) boolean array

    Returns
    -------
    (P × P) Jaccard matrix
    """
    M_int = M_bool.astype(int)
    inter = M_int.T @ M_int
    sizes = np.diag(inter).astype(float)
    unions = sizes[:, None] + sizes[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(unions > 0, inter / unions, 0.0)
    np.fill_diagonal(J, 1.0)
    return J


def coverage_greedy_select(
    M: pd.DataFrame,
    s: pd.Series,
    w: Optional[pd.Series] = None,
    K: int = 12,
    lambda_: float = 0.1,
    mu: float = 0.3,
    max_jaccard: Optional[float] = None,
    precomputed_J: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict:
    """
    Greedy submodular pathway selection maximising gene coverage while
    penalising pathway overlap.

    Parameters
    ----------
    M : (G × P) DataFrame of bool/0-1 (index=genes, columns=pathways)
    s : Series (P,) enrichment score per pathway (index = M.columns)
    w : optional Series (G,) per-gene weights
    K : number of pathways to select
    lambda_ : weight for the pathway enrichment score s[p]
    mu : weight for pairwise Jaccard overlap penalty
    max_jaccard : hard upper bound on Jaccard similarity with any already-
        selected pathway (``None`` = no hard constraint)
    precomputed_J : pre-computed (P × P) Jaccard matrix (optional)
    seed : if set, shuffle pathways before selection for reproducibility

    Returns
    -------
    dict with keys:
        selected, gains, covered_genes, coverage_curve,
        unique_genes_added, params, jaccard
    """
    M = M.astype(bool).loc[:, s.index]
    genes = M.index.to_numpy()
    paths = M.columns.to_numpy()
    G, P = M.shape

    wv = (
        np.ones(G, dtype=float)
        if w is None
        else w.reindex(M.index).fillna(0.0).to_numpy(dtype=float)
    )

    Mb = M.to_numpy(dtype=bool)

    if precomputed_J is None:
        if seed is not None:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(P)
            M = M.iloc[:, perm]
            s = s.iloc[perm]
            paths = M.columns.to_numpy()
            Mb = M.to_numpy(dtype=bool)
        J = compute_jaccard_from_membership(Mb)
    else:
        J = precomputed_J
        assert J.shape == (P, P)

    sv = s.to_numpy(dtype=float)
    covered = np.zeros(G, dtype=bool)
    selected_idx: List[int] = []
    selected_names: List[str] = []
    gains: List[float] = []
    cov_curve: List[float] = []
    unique_added_list: List[Set[str]] = []

    def marginal_gain(p_idx: int) -> Tuple[float, np.ndarray]:
        new_genes = Mb[:, p_idx] & (~covered)
        gain_cov = float(np.dot(wv, new_genes))
        pen = mu * float(np.sum(J[p_idx, selected_idx])) if selected_idx and mu != 0.0 else 0.0
        return gain_cov + lambda_ * float(sv[p_idx]) - pen, new_genes

    heap: List[Tuple[float, int, int]] = []
    stamps = np.zeros(P, dtype=np.int64)
    for j in range(P):
        g, _ = marginal_gain(j)
        heapq.heappush(heap, (-g, j, 0))

    total_cov = 0.0
    while len(selected_idx) < K and heap:
        neg_g, j, stamp = heapq.heappop(heap)

        if max_jaccard is not None and selected_idx:
            if np.any(J[j, selected_idx] > max_jaccard):
                continue

        g_now, new_mask = marginal_gain(j)
        if -neg_g != g_now:
            stamps[j] += 1
            heapq.heappush(heap, (-g_now, j, stamps[j]))
            continue

        selected_idx.append(j)
        selected_names.append(paths[j])
        gains.append(g_now)
        newly = new_mask.nonzero()[0]
        covered[newly] = True
        unique_added_list.append(set(genes[newly]))
        total_cov += float(np.dot(wv, new_mask))
        cov_curve.append(total_cov)

    return {
        "selected": selected_names,
        "gains": gains,
        "covered_genes": set(genes[covered]),
        "coverage_curve": cov_curve,
        "unique_genes_added": unique_added_list,
        "params": {"K": K, "lambda_": lambda_, "mu": mu, "max_jaccard": max_jaccard},
        "jaccard": J,
    }


def select_diverse_pathways(
    ora_results: pd.DataFrame,
    pathway_dict: Dict[str, List[str]],
    background_genes: List[str],
    K: int = 12,
    fdr_threshold: float = 0.05,
    lambda_: float = 0.1,
    mu: float = 0.3,
    max_jaccard: Optional[float] = 0.5,
    min_pathway_size: int = 5,
    max_pathway_size: int = 500,
) -> Dict:
    """
    From ORA results, select up to K diverse, non-redundant significant pathways.

    Uses ``coverage_greedy_select`` with ``-log10(fdr)`` as the enrichment
    score so more significant pathways are preferred, while the Jaccard
    penalty discourages highly overlapping pathways.

    Parameters
    ----------
    ora_results : output of ``run_ora``
    pathway_dict : {pathway: [genes]} used for the ORA
    background_genes : universe used in the ORA
    K : max pathways to return
    fdr_threshold : pre-filter; only pathways with fdr ≤ this are considered
    lambda_, mu, max_jaccard : passed to ``coverage_greedy_select``

    Returns
    -------
    dict from ``coverage_greedy_select`` (keys: selected, gains, covered_genes, …)
    """
    sig = ora_results[ora_results["fdr"] <= fdr_threshold].copy()
    if sig.empty:
        return {"selected": [], "gains": [], "covered_genes": set(), "coverage_curve": []}

    sig_pathways = {p: pathway_dict[p] for p in sig["pathway"] if p in pathway_dict}
    M = build_membership_matrix(background_genes, sig_pathways, min_pathway_size, max_pathway_size)
    # align ora_results to membership matrix columns
    sig_aligned = sig.set_index("pathway").reindex(M.columns)
    # enrichment score: -log10(fdr), clipped to avoid inf
    fdr_vals = sig_aligned["fdr"].clip(lower=1e-300).to_numpy()
    scores = pd.Series(-np.log10(fdr_vals), index=M.columns)

    return coverage_greedy_select(M, scores, K=K, lambda_=lambda_, mu=mu, max_jaccard=max_jaccard)


# ---------------------------------------------------------------------------
# Cytoscape export
# ---------------------------------------------------------------------------

_DILI_COLOR = "#e41a1c"  # overrides pathway color so DILI markers stand out
# Palette: tab20-derived, with all red-family hues removed so no pathway
# swatch is confusable with the DILI red above. Substitutes: a teal-green
# and an olive-green from ColorBrewer Dark2.
_PATHWAY_PALETTE: Tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#1b9e77",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#66a61e",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
)
_NO_PATHWAY_COLOR = "#dddddd"  # truly unannotated (not in any pathway)
_OTHER_PATHWAY_COLOR = "#5e8ca0"  # annotated, but no membership in the top-N
_DRUG_COLOR = "#222222"


def export_to_cytoscape(
    graph: nx.DiGraph,
    *,
    drug_name: str,
    drug_targets: Iterable[str],
    dili_markers: Iterable[str],
    selected_pathways: Dict,
    pathway_dict: Dict[str, List[str]],
    ora_results: Optional[pd.DataFrame] = None,
    path: Union[str, Path] = "network.cyjs",
) -> Path:
    """
    Export a posterior network + pathway annotations for Cytoscape Desktop.

    Writes two files:

    * ``<path>``           — the network as ``.cyjs`` (File → Import → Network
      from File…). Nodes carry pre-computed positions from a pathway-grouped
      layout, so the network opens already organized by pathway.
    * ``<stem>_style.xml`` — a native Cytoscape Vizmap style. Cytoscape Desktop
      ignores style blocks embedded in ``.cyjs`` imports, so the style ships
      separately and must be imported once via the Styles panel:
      ``Styles ▸ ☰ menu ▸ Import Styles from File…``. After import, pick
      "<drug_name> pathway style" from the Style dropdown.

    Encoding
    --------
    * A node is added for ``drug_name`` and connected to each gene in
      ``drug_targets`` that is present in ``graph`` (dashed edges).
    * Node shape encodes role:

      - ``drug``         → diamond
      - ``target``       → triangle (drug target only)
      - ``dili``         → hexagon  (DILI marker only)
      - ``target_dili``  → octagon  (both)
      - ``gene``         → ellipse  (neither)

    * Node fill color encodes the gene's **top pathway** among
      ``selected_pathways["selected"]``. When a gene belongs to multiple
      selected pathways, the tie is broken by smallest FDR in ``ora_results``
      (or smallest pathway size if ``ora_results`` is None). Every pathway
      membership is also stored on the node as ``pathways`` so Cytoscape's
      table panel / filters can use it.

    Parameters
    ----------
    graph : networkx.DiGraph
        Posterior network. Node labels should already be gene symbol strings.
    drug_name : str
        Label for the drug node (e.g. "clozapine").
    drug_targets, dili_markers : iterables of gene symbol strings
    selected_pathways : dict
        Output of ``select_diverse_pathways``. Only ``selected_pathways["selected"]``
        is read.
    pathway_dict : dict
        Same ``{pathway: [genes]}`` used in the ORA.
    ora_results : pd.DataFrame, optional
        Output of ``run_ora``. Used to break ties when a gene is in multiple
        selected pathways.
    path : str or Path
        Output ``.cyjs`` path.

    Returns
    -------
    pathlib.Path
        The path that was written.
    """
    out_path = Path(path)
    selected: List[str] = list(selected_pathways.get("selected", []))

    # Coloring uses (user-selected) ∪ (greedy-coverage extension), capped at
    # palette size. Just taking top-N by FDR leaves most of the network gray
    # because the most-significant pathways are usually highly overlapping.
    # The greedy step picks each next pathway to maximize *new* gene coverage
    # of the network, so the legend hits coverage saturation fast.
    palette_size = len(_PATHWAY_PALETTE)
    network_genes_set = {str(n) for n in graph.nodes()}
    color_pathways: List[str] = []
    seen: Set[str] = set()
    covered: Set[str] = set()
    for p in selected:
        if p in pathway_dict and p not in seen:
            color_pathways.append(p)
            seen.add(p)
            covered |= set(pathway_dict[p]) & network_genes_set

    if ora_results is not None and not ora_results.empty and len(color_pathways) < palette_size:
        sig = ora_results[ora_results["fdr"] <= 0.05]
        candidates = [p for p in sig["pathway"].tolist() if p in pathway_dict and p not in seen]
        cand_members = {p: set(pathway_dict[p]) & network_genes_set for p in candidates}
        while len(color_pathways) < palette_size and candidates:
            best, best_new = None, 0
            for p in candidates:
                new_count = len(cand_members[p] - covered)
                if new_count > best_new:
                    best_new = new_count
                    best = p
            if best is None or best_new == 0:
                break
            color_pathways.append(best)
            seen.add(best)
            covered |= cand_members[best]
            candidates.remove(best)

    color_pathways = color_pathways[:palette_size]
    color_set = set(color_pathways)
    pathway_color = {p: _PATHWAY_PALETTE[i % palette_size] for i, p in enumerate(color_pathways)}

    # gene → list of ALL pathway memberships (full pathway_dict),
    # and separately the subset that drive coloring
    all_gene_to_pathways: Dict[str, List[str]] = {}
    sel_gene_to_pathways: Dict[str, List[str]] = {}
    for p, genes in pathway_dict.items():
        for g in genes:
            all_gene_to_pathways.setdefault(g, []).append(p)
            if p in color_set:
                sel_gene_to_pathways.setdefault(g, []).append(p)

    # tie-break: smallest FDR (or fallback to smallest pathway)
    if ora_results is not None and not ora_results.empty:
        fdr_map = ora_results.set_index("pathway")["fdr"].to_dict()

        def _rank(p: str) -> float:
            return float(fdr_map.get(p, 1.0))

    else:

        def _rank(p: str) -> float:
            return float(len(pathway_dict.get(p, [])))

    target_set = {str(g) for g in drug_targets}
    dili_set = {str(g) for g in dili_markers}

    # ---- pathway-grouped layout ------------------------------------------
    # Each selected pathway becomes a cluster center placed evenly around a
    # circle. Genes are assigned to their top_pathway's cluster; genes with no
    # selected-pathway membership go to an "__other__" cluster placed at the
    # center. Within each cluster we run a small spring layout on the induced
    # subgraph so local connectivity is still visible.
    group_of: Dict[str, str] = {}
    for n in graph.nodes():
        name = str(n)
        sel_pws = sel_gene_to_pathways.get(name, [])
        top_sel = min(sel_pws, key=_rank) if sel_pws else None
        group_of[name] = top_sel if top_sel is not None else "__other__"

    groups: Dict[str, List[str]] = {}
    for name, gname in group_of.items():
        groups.setdefault(gname, []).append(name)

    # cluster centers: selected pathways on a ring; "__other__" at the origin
    ring_radius = 1400.0
    cluster_radius = 320.0
    ordered_groups = [g for g in selected if g in groups]
    centers: Dict[str, Tuple[float, float]] = {}
    n_on_ring = len(ordered_groups)
    for i, gname in enumerate(ordered_groups):
        ang = 2.0 * math.pi * i / max(1, n_on_ring)
        centers[gname] = (ring_radius * math.cos(ang), ring_radius * math.sin(ang))
    if "__other__" in groups:
        centers["__other__"] = (0.0, 0.0)

    pos_xy: Dict[str, Tuple[float, float]] = {}
    for gname, members in groups.items():
        cx, cy = centers.get(gname, (0.0, 0.0))
        member_set = set(members)
        # work on the induced subgraph so the local layout reflects in-cluster edges
        sub = graph.subgraph([n for n in graph.nodes() if str(n) in member_set])
        if sub.number_of_nodes() <= 1:
            for n in members:
                pos_xy[n] = (cx, cy)
            continue
        sub_pos = nx.spring_layout(sub, seed=0, k=1.0 / math.sqrt(len(sub)))
        # normalize sub-layout to [-1, 1]
        xs = [p[0] for p in sub_pos.values()]
        ys = [p[1] for p in sub_pos.values()]
        xr = (max(xs) - min(xs)) or 1.0
        yr = (max(ys) - min(ys)) or 1.0
        for n, (x, y) in sub_pos.items():
            nx_ = (x - (max(xs) + min(xs)) / 2) / (xr / 2)
            ny_ = (y - (max(ys) + min(ys)) / 2) / (yr / 2)
            pos_xy[str(n)] = (cx + nx_ * cluster_radius, cy + ny_ * cluster_radius)

    # drug node: centroid of its in-graph targets (or origin)
    target_coords = [pos_xy[t] for t in target_set if t in pos_xy]
    if target_coords:
        dx = sum(c[0] for c in target_coords) / len(target_coords)
        dy = sum(c[1] for c in target_coords) / len(target_coords)
    else:
        dx, dy = 0.0, 0.0

    def _role(gene: str) -> str:
        in_t, in_d = gene in target_set, gene in dili_set
        if in_t and in_d:
            return "target_dili"
        if in_t:
            return "target"
        if in_d:
            return "dili"
        return "gene"

    # -- build nodes ---------------------------------------------------------
    nodes = []
    for n in graph.nodes():
        name = str(n)
        role = _role(name)
        all_pws = all_gene_to_pathways.get(name, [])
        sel_pws = sel_gene_to_pathways.get(name, [])
        top_sel = min(sel_pws, key=_rank) if sel_pws else None
        x, y = pos_xy.get(name, (0.0, 0.0))
        if role in ("dili", "target_dili"):
            fill = _DILI_COLOR
        elif top_sel is not None:
            fill = pathway_color[top_sel]
        elif all_pws:
            fill = _OTHER_PATHWAY_COLOR
        else:
            fill = _NO_PATHWAY_COLOR
        nodes.append(
            {
                "data": {
                    "id": name,
                    "name": name,
                    "role": role,
                    "pathways": all_pws,
                    "n_pathways": len(all_pws),
                    "selected_pathways": sel_pws,
                    "top_pathway": top_sel if top_sel is not None else "",
                    "color": fill,
                },
                "position": {"x": x, "y": y},
            }
        )

    # drug node
    nodes.append(
        {
            "data": {
                "id": drug_name,
                "name": drug_name,
                "role": "drug",
                "pathways": [],
                "n_pathways": 0,
                "selected_pathways": [],
                "top_pathway": "",
                "color": _DRUG_COLOR,
            },
            "position": {"x": dx, "y": dy},
        }
    )

    # ---- legend nodes -----------------------------------------------------
    # Vertical strip to the right of the network. One rectangle per
    # color_pathways entry, plus a title row. A DILI marker row is added at
    # the bottom when DILI nodes are present so the red override is explained.
    has_dili_nodes = bool(dili_set & {str(n) for n in graph.nodes()})
    if color_pathways or has_dili_nodes:
        legend_x = ring_radius * 2.4
        legend_top = ring_radius
        legend_spacing = 80.0
        nodes.append(
            {
                "data": {
                    "id": "__legend_title__",
                    "name": "Pathway legend",
                    "role": "legend_title",
                    "pathways": [],
                    "n_pathways": 0,
                    "selected_pathways": [],
                    "top_pathway": "",
                    "color": "#ffffff",
                },
                "position": {"x": legend_x, "y": legend_top - legend_spacing},
            }
        )
        for i, pw in enumerate(color_pathways):
            nodes.append(
                {
                    "data": {
                        "id": f"__legend__{pw}",
                        "name": pw,
                        "role": "legend",
                        "pathways": [pw],
                        "n_pathways": 1,
                        "selected_pathways": [pw] if pw in selected else [],
                        "top_pathway": pw,
                        "color": pathway_color[pw],
                    },
                    "position": {"x": legend_x, "y": legend_top + i * legend_spacing},
                }
            )
        if has_dili_nodes:
            nodes.append(
                {
                    "data": {
                        "id": "__legend__DILI",
                        "name": "DILI marker (overrides pathway color)",
                        "role": "legend",
                        "pathways": [],
                        "n_pathways": 0,
                        "selected_pathways": [],
                        "top_pathway": "",
                        "color": _DILI_COLOR,
                    },
                    "position": {
                        "x": legend_x,
                        "y": legend_top + len(color_pathways) * legend_spacing + legend_spacing,
                    },
                }
            )
        # "Other pathway" entry — covers genes annotated in pathway_dict but
        # not in the top-N color pathways
        has_other_pw_nodes = any(n["data"]["color"] == _OTHER_PATHWAY_COLOR for n in nodes)
        if has_other_pw_nodes:
            offset = len(color_pathways) + (2 if has_dili_nodes else 1)
            nodes.append(
                {
                    "data": {
                        "id": "__legend__OTHER",
                        "name": "Other pathway (not in top-N)",
                        "role": "legend",
                        "pathways": [],
                        "n_pathways": 0,
                        "selected_pathways": [],
                        "top_pathway": "",
                        "color": _OTHER_PATHWAY_COLOR,
                    },
                    "position": {
                        "x": legend_x,
                        "y": legend_top + offset * legend_spacing,
                    },
                }
            )

    # -- build edges ---------------------------------------------------------
    edges = []
    for i, (u, v, attrs) in enumerate(graph.edges(data=True)):
        edges.append(
            {
                "data": {
                    "id": f"e{i}",
                    "source": str(u),
                    "target": str(v),
                    "type": "graph",
                    "interaction": "regulates",
                    **{k: _jsonable(val) for k, val in attrs.items()},
                }
            }
        )

    graph_node_names = {str(n) for n in graph.nodes()}
    for j, t in enumerate(target_set):
        if t in graph_node_names:
            edges.append(
                {
                    "data": {
                        "id": f"drug_e{j}",
                        "source": drug_name,
                        "target": t,
                        "type": "drug",
                        "interaction": "targets",
                    }
                }
            )

    cyjs = {
        "format_version": "1.0",
        "generated_by": "causomic.data_analysis.export_to_cytoscape",
        "target_cytoscapejs_version": "~2.1",
        "directed": True,
        "multigraph": False,
        "data": {
            "name": f"{drug_name} posterior network",
            "shared_name": f"{drug_name} posterior network",
            "pathway_colors": pathway_color,
        },
        "elements": {"nodes": nodes, "edges": edges},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cyjs, f, indent=2)

    # ---- Vizmap XML (separate file) ---------------------------------------
    # Cytoscape Desktop's .cyjs reader does NOT apply embedded style blocks,
    # and its "Import Styles from File…" dialog wants the native Vizmap XML
    # format. We ship the style as <stem>_style.xml. The user imports it once
    # via: Styles panel ▸ ☰ menu ▸ Import Styles from File… ▸ pick this file.
    # After importing, select "<drug_name> pathway style" from the Style
    # dropdown.
    style_path = out_path.with_name(out_path.stem + "_style.xml")
    with open(style_path, "w") as f:
        f.write(_build_vizmap_xml(drug_name))

    return out_path


def _build_vizmap_xml(drug_name: str) -> str:
    """Cytoscape Desktop Vizmap XML. Discrete mappings on ``role`` and ``type``
    plus a passthrough on the ``color`` column for per-pathway fill."""
    style_name = f"{drug_name} pathway style"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<vizmap documentVersion="3.0">
  <visualStyle name="{style_name}">
    <network>
      <visualProperty default="#FFFFFF" name="NETWORK_BACKGROUND_PAINT"/>
    </network>
    <node>
      <dependency value="false" name="nodeSizeLocked"/>
      <visualProperty default="ELLIPSE" name="NODE_SHAPE">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="drug"         value="DIAMOND"/>
          <discreteMappingEntry attributeValue="target"       value="TRIANGLE"/>
          <discreteMappingEntry attributeValue="dili"         value="HEXAGON"/>
          <discreteMappingEntry attributeValue="target_dili"  value="OCTAGON"/>
          <discreteMappingEntry attributeValue="legend"       value="ROUND_RECTANGLE"/>
          <discreteMappingEntry attributeValue="legend_title" value="ROUND_RECTANGLE"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="#DDDDDD" name="NODE_FILL_COLOR">
        <passthroughMapping attributeName="color" attributeType="string"/>
      </visualProperty>
      <visualProperty default="" name="NODE_LABEL">
        <passthroughMapping attributeName="name" attributeType="string"/>
      </visualProperty>
      <visualProperty default="60.0" name="NODE_WIDTH">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="drug"         value="200.0"/>
          <discreteMappingEntry attributeValue="legend"       value="260.0"/>
          <discreteMappingEntry attributeValue="legend_title" value="260.0"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="60.0" name="NODE_HEIGHT">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="drug"         value="200.0"/>
          <discreteMappingEntry attributeValue="legend"       value="55.0"/>
          <discreteMappingEntry attributeValue="legend_title" value="55.0"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="#555555" name="NODE_BORDER_PAINT">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="legend_title" value="#FFFFFF"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="1.5" name="NODE_BORDER_WIDTH">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="legend_title" value="0.0"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="18" name="NODE_LABEL_FONT_SIZE">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="drug"         value="28"/>
          <discreteMappingEntry attributeValue="legend"       value="16"/>
          <discreteMappingEntry attributeValue="legend_title" value="22"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="#000000" name="NODE_LABEL_COLOR">
        <discreteMapping attributeName="role" attributeType="string">
          <discreteMappingEntry attributeValue="drug" value="#FFFFFF"/>
        </discreteMapping>
      </visualProperty>
    </node>
    <edge>
      <visualProperty default="DELTA" name="EDGE_TARGET_ARROW_SHAPE"/>
      <visualProperty default="#999999" name="EDGE_STROKE_UNSELECTED_PAINT">
        <discreteMapping attributeName="type" attributeType="string">
          <discreteMappingEntry attributeValue="drug" value="#222222"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="#999999" name="EDGE_TARGET_ARROW_UNSELECTED_PAINT">
        <discreteMapping attributeName="type" attributeType="string">
          <discreteMappingEntry attributeValue="drug" value="#222222"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="SOLID" name="EDGE_LINE_TYPE">
        <discreteMapping attributeName="type" attributeType="string">
          <discreteMappingEntry attributeValue="drug" value="LONG_DASH"/>
        </discreteMapping>
      </visualProperty>
      <visualProperty default="1.5" name="EDGE_WIDTH">
        <discreteMapping attributeName="type" attributeType="string">
          <discreteMappingEntry attributeValue="drug" value="3.0"/>
        </discreteMapping>
      </visualProperty>
    </edge>
  </visualStyle>
</vizmap>
"""


def _jsonable(value):
    """Best-effort coerce edge/node attribute values for JSON."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_gseapy() -> None:
    if not _GSEAPY_AVAILABLE:
        raise ImportError(
            "gseapy is required for programmatic pathway fetching. "
            "Install it with: pip install gseapy"
        )


def _apply_fdr(p_values: np.ndarray, method: str = "bh") -> np.ndarray:
    """Benjamini-Hochberg or Bonferroni FDR correction (no external deps)."""
    n = len(p_values)
    if method == "bonferroni":
        return np.minimum(p_values * n, 1.0)
    # Benjamini-Hochberg
    order = np.argsort(p_values)
    ranks = np.argsort(order) + 1
    fdr = np.minimum(1.0, p_values * n / ranks)
    # enforce monotonicity (step-down): scan from largest rank downward
    fdr_sorted = fdr[order]
    fdr_sorted = np.minimum.accumulate(fdr_sorted[::-1])[::-1]
    result = np.empty(n)
    result[order] = fdr_sorted
    return result
