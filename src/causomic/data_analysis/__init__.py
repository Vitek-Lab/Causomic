"""Proteomics data processing and downstream analysis.

Includes MS data preprocessing (normalization, summarization, imputation),
gene-set/correlation analysis, and pathway over-representation analysis.
"""

from causomic.data_analysis.gene_set import (
    gen_correlation_matrix,
    prep_msstats_data,
    test_gene_sets,
)
from causomic.data_analysis.pathway_analysis import (
    build_membership_matrix,
    coverage_greedy_select,
    export_to_cytoscape,
    fetch_pathway_library,
    list_pathway_libraries,
    run_ora,
    select_diverse_pathways,
)
from causomic.data_analysis.proteomics_data_processor import (
    dataProcess,
    format_sim_data,
    imputation,
    normalize_median,
    summarize_data,
)

__all__ = [
    "build_membership_matrix",
    "coverage_greedy_select",
    "dataProcess",
    "export_to_cytoscape",
    "fetch_pathway_library",
    "format_sim_data",
    "gen_correlation_matrix",
    "imputation",
    "list_pathway_libraries",
    "normalize_median",
    "prep_msstats_data",
    "run_ora",
    "select_diverse_pathways",
    "summarize_data",
    "test_gene_sets",
]
