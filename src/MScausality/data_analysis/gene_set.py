"""Analyze correlations between genes in different gene sets, given some experimental data."""

import pandas as pd
import numpy as np
import json
import itertools
import re
from collections import defaultdict


def parse_protein_name(data, column_name='Protein', parse_gene=False, gene_map=None):
    if parse_gene:
        data.loc[:, column_name] = data.loc[:, column_name].apply(lambda x: x.split("_")[0])

    if gene_map is not None:
        data = pd.merge(data, gene_map, how='left', left_on=column_name, right_on='From')
        data.loc[:, column_name] = data.loc[:, 'To']
    
    return data

def prep_msstats_data(data, parse_gene=False, gene_map=None):
    """
    Prepare MSstats data by parsing protein names and merging with gene map if provided.

    Parameters:
    - gene_map (pd.DataFrame, optional): Gene mapping data with columns 'From' and 'To'.
    - data (pd.DataFrame): Input data which is the ProteinLevelData object from the MSstats `dataProcess` function.
    - parse_gene (bool): If True, parse the 'Protein' column by splitting on "_" and keeping the first part.

    Returns:
    - pd.DataFrame: Wide-formatted MSstats data with columns for each protein and 'originalRUN'.

    Example:
    prep_msstats_data(data, gene_map=gene_mapping, parse_gene=True)
    """

    data = parse_protein_name(data, parse_gene=parse_gene, gene_map=gene_map)

    parse_data = data[['Protein', 'originalRUN', 'LogIntensities']]
    wide_data = parse_data.pivot(index='originalRUN', columns='Protein', values='LogIntensities')
    wide_data = wide_data.iloc[:, 1:]  # Exclude the first column (originalRUN) if needed

    return wide_data

def gen_correlation_matrix(data, methods=["pearson", "spearman"], abs_corr=True):
    """
    Generates a list of different correlation matrices.

    Parameters:
    - data (pd.DataFrame): Output of the prep_msstats_data function.
    - methods (list): List of correlation methods to use (e.g., ["pearson", "spearman"]).
    - abs_corr (bool): If True, calculate the absolute correlation.

    Returns:
    - dict: Dictionary containing correlation matrices for each specified method.

    Example:
    gen_correlation_matrix(data, methods=["pearson", "spearman"], abs_corr=True)
    """

    correlations = {}
    for m in methods:
        cor_mat = data.corr(method=m)

        # Prepare correlation data for easy indexing
        cor_mat = cor_mat.unstack().reset_index()
        # Combine first two columns in one
        cor_mat.loc[:, 'index'] = list(zip(cor_mat['level_0'].values, cor_mat['level_1'].values))
        cor_mat = cor_mat.drop(['level_0', 'level_1'], axis=1)

        # Rename 0 column to value
        cor_mat = cor_mat.rename(columns={0: 'value'})
        cor_mat = cor_mat.loc[cor_mat["value"] < 1]

        if abs_corr:
            cor_mat.loc[:, "value"] = np.abs(cor_mat.loc[:, "value"])

        correlations[m] = cor_mat
        
        print("Correlation matrix for {}:".format(m))

    return correlations

def test_gene_sets(correlation_data, 
                   measured_genes, 
                   gene_set_file_path, 
                   threshold=0.5,
                   differential_analysis=None,
                   comparison=None,
                   fc_pval=None,
                   cutoff=None):
    """
    Checks correlations between genes in gene sets.

    Parameters:
    - correlation_data (dict): Output of gen_correlation_matrix function.
    - measured_genes (list): List of measured genes.
    - gene_set_file_path (str): File path to gene set JSON file.
    - threshold (float): Correlation threshold to count as significant correlation. Default is 0.5.
    - differential_analysis (pd.DataFrame): Differential analysis results to use for filtering.
    - comparison (str): Comparison to use for filtering differential analysis results.
    - fc_pval (float): Use fold change (fc) or p-value (pval) cutoff for differential analysis. Default is None.
    - cutoff (float): Cutoff value for fold change or p-value. Default is None.

    Returns:
    - pd.DataFrame: Results containing pathway information and correlation statistics.

    Example:
    test_gene_sets(correlation_data, 'gsea_pathways.json', threshold=0.5)
    """

    with open(gene_set_file_path, 'r') as f:
        gsea = json.load(f)

    pathways = list(gsea.keys())
    correlation_data = correlation_data['pearson']
    corr_type = 'pearson'
    # measured_genes = list(correlation_data.values())[0].columns

    result_list = list()

    for path in pathways:
        print(path)
        current_path = gsea[path]
        genes_in_path = current_path['geneSymbols']

        measured_genes_in_path = list(set(genes_in_path) & set(measured_genes))

        if len(measured_genes_in_path) > 2:
            combination_mat = list(tuple(itertools.combinations(measured_genes_in_path, 2)))
            total_combs = len(combination_mat)

            # Calcuclate all significant correlations
            sig_correlations = correlation_data.loc[
                (correlation_data["index"].isin(combination_mat)) & 
                (correlation_data["value"] > threshold), :]
            sig_tracker = len(sig_correlations)

            # Calculate if only a couple nodes dominate the significant correlations
            count_dict = defaultdict(int)
            for tup in sig_correlations["index"].values:
                for item in tup:
                    count_dict[item] += 1

            if len(count_dict) > 0:
                freq_nodes = sum(np.array(list(count_dict.values())) > 
                                round((sig_tracker*2) / len(count_dict))) / len(count_dict)
            else:
                freq_nodes = 0

            if differential_analysis is not None:
                # Filter significant correlations based on differential analysis results
                subset = differential_analysis.loc[
                    (differential_analysis["Protein"].isin(measured_genes_in_path)) & \
                        (differential_analysis["Label"] == comparison), :]
                
                if fc_pval == "fc":
                    col_name = "log2FC"
                    percent_differential = len(subset.loc[abs(subset[col_name]) > cutoff]
                                           ) / len(measured_genes_in_path)
                elif fc_pval == "pval":
                    col_name = "adj.pvalue"
                    percent_differential = len(subset.loc[subset[col_name] < cutoff]
                                           ) / len(measured_genes_in_path)

            else:
                percent_differential = np.nan

            temp_data = pd.DataFrame({
                'pathway': [path],
                'correlation': [corr_type],
                'total_genes': [len(genes_in_path)],
                'measured_genes': [len(measured_genes_in_path)],
                'percent_measured': [len(measured_genes_in_path) / len(genes_in_path)],
                'total_tests': [total_combs],
                'sig_corrs': [sig_tracker],
                'percent': [sig_tracker / total_combs],
                'freq_node_percent' : [freq_nodes],
                'differential_percent' : [percent_differential]
            })

            result_list.append(temp_data)

        else:
            temp_data = pd.DataFrame({
                'pathway': [path],
                'correlation': ['no measured nodes'],
                'total_genes': [len(genes_in_path)],
                'measured_genes': [0],
                'percent_measured': [0],
                'total_tests': [0],
                'sig_corrs': [0],
                'percent': [0],
                'freq_node_percent' : [0]
            })

            result_list.append(temp_data)

    results = pd.concat(result_list, ignore_index=True)

    return results

def extract_genes_in_path(measured_genes, gene_set_name, gene_set_file_path, return_all=False):
    """
    Return observed genes in a given gene set.

    Parameters:
    - measured_genes (dict): list of gene names that were measured
    - gene_set_name (str): Pathway name to extract genes for.
    - gene_set_file_path (str): File path to GSEA JSON file.
    - return_all (bool): If True, return all genes in the pathway. Default is False.

    Returns:
    - list: List of genes observed in the specified pathway.

    Example:
    extract_genes_in_path(correlation_data, 'KEGG_GLYCOLYSIS_GLUCONEOGENESIS', 'gene_set_pathways.json')
    """

    with open(gene_set_file_path, 'r') as f:
        gsea = json.load(f)

    current_path = gsea.get(gene_set_name, {})
    genes_in_path = current_path.get('geneSymbols', [])

    if return_all:
        measured_genes_in_path = genes_in_path
    else:
        measured_genes_in_path = list(set(genes_in_path) & set(measured_genes))

    return measured_genes_in_path

def find_sets_with_gene(gene, gene_set_file_path, percent=None):
    """
    Return sets that contain a given gene.

    Parameters:
    - gene (list): List of genes to search for in GSEA pathways.
    - gene_set_file_path (str): File path to GSEA JSON file.
    - percent: If a float, will return path if a certain percentage of genes are found in the pathway. Otherwise None will return only if all genes are present. Default is None.

    Returns:
    - list: List of pathway names containing the given gene.

    Example:
    find_sets_with_gene('your_gene', 'gsea_pathways.json')
    """

    with open(gene_set_file_path, 'r') as f:
        gsea = json.load(f)

    pathways = list(gsea.keys())
    result_list = []

    for path in pathways:
        current_path = gsea[path]
        genes_in_path = current_path.get('geneSymbols', [])

        if len(gene) == 1:
            measured_genes_in_path = [g for g in genes_in_path if 
                                      re.search(gene[0], g)]

            if len(measured_genes_in_path) > 0:
                result_list.append(path)
        elif percent is not None:
            measured_genes_in_path = list(set(genes_in_path) & set(gene))
            if len(measured_genes_in_path) > percent:#/ len(gene)
                result_list.append(path)
        else:
            measured_genes_in_path = list(set(genes_in_path) & set(gene))

            if len(measured_genes_in_path) == len(gene):
                result_list.append(path)

    return result_list

def main():
    msstats_data = pd.read_csv("data/Talus/processed_data/ProteinLevelData.csv")
    msstats_data = msstats_data[msstats_data["GROUP"] == "DMSO"]

    input_data = prep_msstats_data(msstats_data, gene_map=None, parse_gene=True)
    input_data = input_data.reset_index(drop=True)
    input_data.columns.name = None
    corr_data = gen_correlation_matrix(input_data, methods=["pearson"], abs_corr=True)
    print('correlation matrix generated')
    
    msstats_model = pd.read_csv("data/Talus/processed_data/model.csv")
    msstats_model = parse_protein_name(msstats_model, column_name ="Protein", parse_gene=True)

    regulatory_paths = test_gene_sets(corr_data, input_data.columns, 
                                      "data/gene_sets/regulatory_pathways.json", 
                                      threshold=0.33, 
                                      differential_analysis=msstats_model,
                                      comparison="DMSO-DbET6")
    print(regulatory_paths)

if __name__ == "__main__":
    main()