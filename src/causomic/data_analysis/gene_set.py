"""
Gene Set Analysis Module for Mass Spectrometry Data

This module provides functionality for analyzing correlations between genes in different
gene sets using experimental mass spectrometry data. It includes tools for data preparation,
correlation analysis, and gene set enrichment testing.

The module supports:
- Protein name parsing and gene mapping
- MSstats data preparation and formatting
- Correlation matrix generation with multiple methods
- Gene set testing with customizable thresholds
- Gene extraction and pathway searching utilities

Author: causomic Team
"""

import itertools
import json
import re
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


def parse_protein_name(
    data: pd.DataFrame,
    column_name: str = "Protein",
    parse_gene: bool = False,
    gene_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Parse protein names and optionally map them using a gene mapping table.

    This function processes protein identifiers by extracting gene symbols
    and/or applying gene name mappings. It's commonly used to standardize
    protein/gene names for downstream analysis.

    Args:
        data (pd.DataFrame): Input dataframe containing protein information
        column_name (str, optional): Name of the column containing protein names.
            Defaults to 'Protein'.
        parse_gene (bool, optional): If True, extract gene symbol by splitting
            on "_" and keeping the first part. Defaults to False.
        gene_map (pd.DataFrame, optional): Gene mapping dataframe with columns
            'From' and 'To' for name conversion. Defaults to None.

    Returns:
        pd.DataFrame: Modified dataframe with parsed/mapped protein names

    Example:
        >>> # Parse gene symbols from protein names
        >>> data = parse_protein_name(df, parse_gene=True)

        >>> # Apply gene mapping
        >>> gene_mapping = pd.DataFrame({'From': ['PROT1', 'PROT2'],
        ...                              'To': ['GENE1', 'GENE2']})
        >>> data = parse_protein_name(df, gene_map=gene_mapping)
    """
    # Make a copy to avoid modifying original data
    result_data = data.copy()

    # Parse gene symbol if requested
    if parse_gene:
        result_data.loc[:, column_name] = result_data.loc[:, column_name].apply(
            lambda x: str(x).split("_")[0]
        )

    # Apply gene mapping if provided
    if gene_map is not None:
        result_data = pd.merge(
            result_data, gene_map, how="left", left_on=column_name, right_on="From"
        )
        result_data.loc[:, column_name] = result_data.loc[:, "To"]
        # Clean up merge columns
        result_data = result_data.drop(columns=["From", "To"], errors="ignore")

    return result_data


def prep_msstats_data(
    data: pd.DataFrame, parse_gene: bool = False, gene_map: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare MSstats data for correlation analysis by formatting and reshaping.

    This function processes MSstats ProteinLevelData output by parsing protein names,
    applying gene mappings if provided, and reshaping the data into a wide format
    suitable for correlation analysis.

    Args:
        data (pd.DataFrame): Input MSstats ProteinLevelData with columns:
            - Protein: Protein identifier
            - originalRUN: Run/sample identifier
            - LogIntensities: Log-transformed intensity values
        parse_gene (bool, optional): If True, parse the 'Protein' column by
            splitting on "_" and keeping the first part. Defaults to False.
        gene_map (pd.DataFrame, optional): Gene mapping dataframe with columns
            'From' and 'To' for protein name conversion. Defaults to None.

    Returns:
        pd.DataFrame: Wide-formatted dataframe with:
            - Rows: samples/runs (originalRUN)
            - Columns: proteins/genes
            - Values: log intensities

    Raises:
        KeyError: If required columns are missing from input data
        ValueError: If data cannot be pivoted (duplicate entries)

    Example:
        >>> # Basic preparation
        >>> wide_data = prep_msstats_data(msstats_output)

        >>> # With gene parsing and mapping
        >>> gene_mapping = pd.DataFrame({'From': ['PROT1_001', 'PROT2_002'],
        ...                              'To': ['GENE1', 'GENE2']})
        >>> wide_data = prep_msstats_data(msstats_output,
        ...                               parse_gene=True,
        ...                               gene_map=gene_mapping)
    """
    # Validate required columns
    required_columns = ["Protein", "originalRUN", "LogIntensities"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    # Parse protein names and apply gene mapping
    processed_data = parse_protein_name(data, parse_gene=parse_gene, gene_map=gene_map)

    # Select required columns for pivoting
    pivot_data = processed_data[["Protein", "originalRUN", "LogIntensities"]].copy()

    # Check for duplicate entries before pivoting
    duplicates = pivot_data.duplicated(subset=["Protein", "originalRUN"])
    if duplicates.any():
        print(
            f"WARNING: Found {duplicates.sum()} duplicate Protein-Run combinations. "
            "Taking mean values."
        )
        pivot_data = (
            pivot_data.groupby(["Protein", "originalRUN"])["LogIntensities"].mean().reset_index()
        )

    # Pivot to wide format
    try:
        wide_data = pivot_data.pivot(
            index="originalRUN", columns="Protein", values="LogIntensities"
        )
    except ValueError as e:
        raise ValueError(f"Failed to pivot data: {e}. Check for duplicate entries.")

    # Reset column name and ensure clean format
    wide_data.columns.name = None

    return wide_data


def gen_correlation_matrix(
    data: pd.DataFrame, methods: List[str] = ["pearson", "spearman"], abs_corr: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Generate correlation matrices using multiple correlation methods.

    This function computes correlation matrices between all pairs of proteins/genes
    using specified methods and formats them for easy analysis and filtering.

    Args:
        data (pd.DataFrame): Wide-format dataframe from prep_msstats_data with:
            - Rows: samples/runs
            - Columns: proteins/genes
            - Values: intensities
        methods (List[str], optional): List of correlation methods to compute.
            Supported methods: ["pearson", "spearman", "kendall"].
            Defaults to ["pearson", "spearman"].
        abs_corr (bool, optional): If True, calculate absolute correlation values.
            Defaults to True.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary where keys are method names and values
        are dataframes with columns:
            - 'index': Tuple of (protein1, protein2) pairs
            - 'value': Correlation coefficient

    Raises:
        ValueError: If unsupported correlation method is specified

    Example:
        >>> # Generate Pearson and Spearman correlations
        >>> corr_matrices = gen_correlation_matrix(wide_data)

        >>> # Use only Pearson correlation without absolute values
        >>> corr_matrices = gen_correlation_matrix(wide_data,
        ...                                        methods=["pearson"],
        ...                                        abs_corr=False)

        >>> # Access specific correlation matrix
        >>> pearson_corr = corr_matrices['pearson']
    """
    supported_methods = ["pearson", "spearman", "kendall"]
    invalid_methods = [m for m in methods if m not in supported_methods]
    if invalid_methods:
        raise ValueError(
            f"Unsupported correlation methods: {invalid_methods}. "
            f"Supported methods: {supported_methods}"
        )

    correlations = {}

    for method in methods:
        print(f"Computing {method} correlation matrix...")

        # Calculate correlation matrix
        corr_matrix = data.corr(method=method)

        # Convert to long format for easy manipulation
        corr_long = corr_matrix.unstack().reset_index()

        # Create tuple index for protein pairs
        corr_long.loc[:, "index"] = list(
            zip(corr_long["level_0"].values, corr_long["level_1"].values)
        )

        # Clean up and rename columns
        corr_long = corr_long.drop(["level_0", "level_1"], axis=1)
        corr_long = corr_long.rename(columns={0: "value"})

        # Remove self-correlations (correlation = 1)
        corr_long = corr_long.loc[corr_long["value"] < 1].copy()

        # Remove NaN correlations
        corr_long = corr_long.dropna(subset=["value"])

        # Apply absolute correlation if requested
        if abs_corr:
            corr_long.loc[:, "value"] = np.abs(corr_long.loc[:, "value"])

        correlations[method] = corr_long
        print(f"  Generated {len(corr_long)} protein pair correlations")

    return correlations


def test_gene_sets(
    correlation_data: Dict[str, pd.DataFrame],
    measured_genes: List[str],
    gene_set_file_path: str,
    threshold: float = 0.5,
    differential_analysis: Optional[pd.DataFrame] = None,
    comparison: Optional[str] = None,
    fc_pval: Optional[Literal["fc", "pval"]] = None,
    cutoff: Optional[float] = None,
) -> pd.DataFrame:
    """
    Test gene sets for significant correlations between member genes.

    This function analyzes correlation patterns within gene sets to identify
    pathways with high internal correlation structure. It can optionally
    incorporate differential analysis results for additional filtering.

    Args:
        correlation_data (Dict[str, pd.DataFrame]): Output from gen_correlation_matrix
            containing correlation matrices for different methods
        measured_genes (List[str]): List of genes that were measured in the experiment
        gene_set_file_path (str): Path to JSON file containing gene set definitions
        threshold (float, optional): Correlation threshold for significance.
            Defaults to 0.5.
        differential_analysis (pd.DataFrame, optional): Differential analysis results
            with columns 'Protein', 'Label', 'log2FC', 'adj.pvalue'. Defaults to None.
        comparison (str, optional): Specific comparison from differential analysis
            to use for filtering. Defaults to None.
        fc_pval (str, optional): Use 'fc' for fold change or 'pval' for p-value
            filtering. Defaults to None.
        cutoff (float, optional): Cutoff value for differential analysis filtering.
            Defaults to None.

    Returns:
        pd.DataFrame: Results dataframe with columns:
            - pathway: Gene set name
            - correlation: Correlation method used
            - total_genes: Total genes in the gene set
            - measured_genes: Number of measured genes in the set
            - percent_measured: Percentage of genes measured
            - total_tests: Total possible gene pairs
            - sig_corrs: Number of significant correlations
            - percent: Percentage of significant correlations
            - freq_node_percent: Percentage of high-frequency nodes
            - differential_percent: Percentage of differentially expressed genes

    Raises:
        FileNotFoundError: If gene set file doesn't exist
        KeyError: If required columns are missing from differential analysis
        ValueError: If invalid fc_pval parameter is provided

    Example:
        >>> # Basic gene set testing
        >>> results = test_gene_sets(corr_data, measured_genes,
        ...                          "pathways.json", threshold=0.3)

        >>> # With differential analysis filtering
        >>> results = test_gene_sets(corr_data, measured_genes,
        ...                          "pathways.json",
        ...                          differential_analysis=diff_results,
        ...                          comparison="Treatment-Control",
        ...                          fc_pval="fc", cutoff=1.5)
    """
    # Validate inputs
    if fc_pval is not None and fc_pval not in ["fc", "pval"]:
        raise ValueError("fc_pval must be either 'fc' or 'pval'")

    if differential_analysis is not None:
        required_diff_cols = ["Protein", "Label"]
        if fc_pval == "fc":
            required_diff_cols.append("log2FC")
        elif fc_pval == "pval":
            required_diff_cols.append("adj.pvalue")

        missing_cols = [
            col for col in required_diff_cols if col not in differential_analysis.columns
        ]
        if missing_cols:
            raise KeyError(f"Missing columns in differential_analysis: {missing_cols}")

    # Load gene sets
    try:
        with open(gene_set_file_path, "r") as f:
            gene_sets = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Gene set file not found: {gene_set_file_path}")

    # Use Pearson correlation by default
    correlation_matrix = correlation_data.get("pearson", list(correlation_data.values())[0])
    corr_method = "pearson" if "pearson" in correlation_data else list(correlation_data.keys())[0]

    pathways = list(gene_sets.keys())
    result_list = []

    for pathway_name in pathways:
        print(f"Processing pathway: {pathway_name}")

        current_pathway = gene_sets[pathway_name]
        genes_in_pathway = current_pathway.get("geneSymbols", [])

        # Find intersection of pathway genes with measured genes
        measured_genes_in_pathway = list(set(genes_in_pathway) & set(measured_genes))

        if len(measured_genes_in_pathway) > 2:
            # Generate all possible gene pairs
            gene_pairs = list(itertools.combinations(measured_genes_in_pathway, 2))
            total_combinations = len(gene_pairs)

            # Find significant correlations
            significant_correlations = correlation_matrix.loc[
                (correlation_matrix["index"].isin(gene_pairs))
                & (correlation_matrix["value"] > threshold)
            ]
            num_significant = len(significant_correlations)

            # Calculate node frequency to detect hub genes
            node_counts = defaultdict(int)
            for gene_pair in significant_correlations["index"].values:
                for gene in gene_pair:
                    node_counts[gene] += 1

            # Calculate percentage of high-frequency nodes
            if len(node_counts) > 0:
                expected_freq = (num_significant * 2) / len(node_counts)
                high_freq_nodes = sum(count > expected_freq for count in node_counts.values())
                freq_node_percent = high_freq_nodes / len(node_counts)
            else:
                freq_node_percent = 0

            # Calculate differential expression percentage if provided
            if differential_analysis is not None and comparison is not None:
                diff_subset = differential_analysis.loc[
                    (differential_analysis["Protein"].isin(measured_genes_in_pathway))
                    & (differential_analysis["Label"] == comparison)
                ]

                if fc_pval == "fc":
                    significant_diff = diff_subset.loc[abs(diff_subset["log2FC"]) > cutoff]
                elif fc_pval == "pval":
                    significant_diff = diff_subset.loc[diff_subset["adj.pvalue"] < cutoff]
                else:
                    significant_diff = pd.DataFrame()

                percent_differential = (
                    len(significant_diff) / len(measured_genes_in_pathway)
                    if len(measured_genes_in_pathway) > 0
                    else 0
                )
            else:
                percent_differential = np.nan

            # Create result row
            pathway_result = pd.DataFrame(
                {
                    "pathway": [pathway_name],
                    "correlation": [corr_method],
                    "total_genes": [len(genes_in_pathway)],
                    "measured_genes": [len(measured_genes_in_pathway)],
                    "percent_measured": [
                        (
                            len(measured_genes_in_pathway) / len(genes_in_pathway)
                            if len(genes_in_pathway) > 0
                            else 0
                        )
                    ],
                    "total_tests": [total_combinations],
                    "sig_corrs": [num_significant],
                    "percent": [
                        num_significant / total_combinations if total_combinations > 0 else 0
                    ],
                    "freq_node_percent": [freq_node_percent],
                    "differential_percent": [percent_differential],
                }
            )

        else:
            # Handle pathways with insufficient measured genes
            pathway_result = pd.DataFrame(
                {
                    "pathway": [pathway_name],
                    "correlation": ["insufficient_genes"],
                    "total_genes": [len(genes_in_pathway)],
                    "measured_genes": [len(measured_genes_in_pathway)],
                    "percent_measured": [
                        (
                            len(measured_genes_in_pathway) / len(genes_in_pathway)
                            if len(genes_in_pathway) > 0
                            else 0
                        )
                    ],
                    "total_tests": [0],
                    "sig_corrs": [0],
                    "percent": [0],
                    "freq_node_percent": [0],
                    "differential_percent": [np.nan],
                }
            )

        result_list.append(pathway_result)

    # Combine all results
    final_results = pd.concat(result_list, ignore_index=True)

    return final_results


def extract_genes_in_path(
    measured_genes: List[str], gene_set_name: str, gene_set_file_path: str, return_all: bool = False
) -> List[str]:
    """
    Extract genes that are present in a specific gene set/pathway.

    This function retrieves the list of genes associated with a particular
    pathway and optionally filters to only those that were measured in
    the experiment.

    Args:
        measured_genes (List[str]): List of gene names that were measured
            in the experiment
        gene_set_name (str): Name of the specific pathway/gene set to extract
        gene_set_file_path (str): Path to JSON file containing gene set definitions
        return_all (bool, optional): If True, return all genes in the pathway
            regardless of whether they were measured. If False, return only
            measured genes. Defaults to False.

    Returns:
        List[str]: List of genes in the specified pathway. If return_all=False,
        only genes that were measured are returned.

    Raises:
        FileNotFoundError: If gene set file doesn't exist
        KeyError: If specified gene set name is not found

    Example:
        >>> # Get measured genes in glycolysis pathway
        >>> glycolysis_genes = extract_genes_in_path(
        ...     measured_genes,
        ...     'KEGG_GLYCOLYSIS_GLUCONEOGENESIS',
        ...     'pathways.json'
        ... )

        >>> # Get all genes in pathway (regardless of measurement)
        >>> all_glycolysis = extract_genes_in_path(
        ...     measured_genes,
        ...     'KEGG_GLYCOLYSIS_GLUCONEOGENESIS',
        ...     'pathways.json',
        ...     return_all=True
        ... )
    """
    try:
        with open(gene_set_file_path, "r") as f:
            gene_sets = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Gene set file not found: {gene_set_file_path}")

    if gene_set_name not in gene_sets:
        raise KeyError(f"Gene set '{gene_set_name}' not found in {gene_set_file_path}")

    current_pathway = gene_sets[gene_set_name]
    genes_in_pathway = current_pathway.get("geneSymbols", [])

    if return_all:
        result_genes = genes_in_pathway
    else:
        # Return intersection of pathway genes with measured genes
        result_genes = list(set(genes_in_pathway) & set(measured_genes))

    return result_genes


def find_sets_with_gene(
    gene: Union[str, List[str]], gene_set_file_path: str, percent: Optional[float] = None
) -> List[str]:
    """
    Find gene sets/pathways that contain specified gene(s).

    This function searches through all gene sets to find those containing
    the specified gene(s). It supports both exact matching and flexible
    matching based on percentage thresholds.

    Args:
        gene (str or List[str]): Single gene name or list of gene names to search for
        gene_set_file_path (str): Path to JSON file containing gene set definitions
        percent (float, optional): If provided, return pathways where at least
            this many genes from the input list are found. If None, all genes
            must be present for multi-gene searches. For single genes, performs
            regex matching. Defaults to None.

    Returns:
        List[str]: List of pathway names containing the specified gene(s)

    Raises:
        FileNotFoundError: If gene set file doesn't exist
        TypeError: If gene parameter is not string or list

    Example:
        >>> # Find pathways containing a specific gene
        >>> pathways = find_sets_with_gene('TP53', 'pathways.json')

        >>> # Find pathways with regex pattern matching
        >>> pathways = find_sets_with_gene(['TP53'], 'pathways.json')

        >>> # Find pathways containing at least 3 genes from the list
        >>> gene_list = ['TP53', 'BRCA1', 'BRCA2', 'ATM']
        >>> pathways = find_sets_with_gene(gene_list, 'pathways.json', percent=3)

        >>> # Find pathways containing all genes in the list
        >>> pathways = find_sets_with_gene(gene_list, 'pathways.json')
    """
    if isinstance(gene, str):
        gene = [gene]
    elif not isinstance(gene, list):
        raise TypeError("gene parameter must be a string or list of strings")

    try:
        with open(gene_set_file_path, "r") as f:
            gene_sets = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Gene set file not found: {gene_set_file_path}")

    pathways = list(gene_sets.keys())
    matching_pathways = []

    for pathway_name in pathways:
        current_pathway = gene_sets[pathway_name]
        genes_in_pathway = current_pathway.get("geneSymbols", [])

        if len(gene) == 1:
            # Single gene: use regex matching for flexible search
            matching_genes = [g for g in genes_in_pathway if re.search(gene[0], g, re.IGNORECASE)]

            if len(matching_genes) > 0:
                matching_pathways.append(pathway_name)

        elif percent is not None:
            # Multiple genes with percentage threshold
            genes_found = list(set(genes_in_pathway) & set(gene))
            if len(genes_found) >= percent:
                matching_pathways.append(pathway_name)

        else:
            # Multiple genes: require all genes to be present
            genes_found = list(set(genes_in_pathway) & set(gene))
            if len(genes_found) == len(gene):
                matching_pathways.append(pathway_name)

    return matching_pathways


def main() -> None:
    """
    Example usage and testing of the gene set analysis functions.

    This function demonstrates a complete gene set analysis workflow using
    MSstats data and regulatory pathways. It shows data preparation,
    correlation analysis, and gene set testing with differential analysis.
    """
    try:
        # Load and prepare MSstats data
        print("Loading MSstats data...")
        msstats_data = pd.read_csv("data/Talus/processed_data/ProteinLevelData.csv")

        # Filter for specific experimental group
        msstats_data = msstats_data[msstats_data["GROUP"] == "DMSO"]
        print(f"Data shape after filtering: {msstats_data.shape}")

        # Prepare data for correlation analysis
        print("Preparing data for correlation analysis...")
        prepared_data = prep_msstats_data(msstats_data, gene_map=None, parse_gene=True)
        prepared_data = prepared_data.reset_index(drop=True)
        prepared_data.columns.name = None
        print(f"Prepared data shape: {prepared_data.shape}")
        print(f"Number of proteins: {prepared_data.shape[1]}")

        # Generate correlation matrices
        print("Generating correlation matrices...")
        correlation_data = gen_correlation_matrix(prepared_data, methods=["pearson"], abs_corr=True)
        print("Correlation matrix generation completed")

        # Load differential analysis results
        print("Loading differential analysis results...")
        differential_results = pd.read_csv("data/Talus/processed_data/model.csv")
        differential_results = parse_protein_name(
            differential_results, column_name="Protein", parse_gene=True
        )

        # Test regulatory pathways
        print("Testing regulatory pathways...")
        regulatory_results = test_gene_sets(
            correlation_data,
            list(prepared_data.columns),
            "data/gene_sets/regulatory_pathways.json",
            threshold=0.33,
            differential_analysis=differential_results,
            comparison="DMSO-DbET6",
        )

        # Display results
        print("\nRegulatory pathway analysis results:")
        print(regulatory_results)

        # Show top pathways by correlation percentage
        if len(regulatory_results) > 0:
            top_pathways = regulatory_results.nlargest(5, "percent")
            print(f"\nTop 5 pathways by correlation percentage:")
            for _, row in top_pathways.iterrows():
                print(
                    f"  {row['pathway']}: {row['percent']:.3f} "
                    f"({row['sig_corrs']}/{row['total_tests']} significant)"
                )

    except FileNotFoundError as e:
        print(f"Error: Required data file not found - {e}")
        print("Please ensure all data files are in the correct locations")
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Please check your data format and file paths")


if __name__ == "__main__":
    main()
