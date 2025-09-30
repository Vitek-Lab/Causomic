"""
INDRA Database Query Interface

This module provides functions for querying the INDRA (Integrated Network and
Dynamical Reasoning Assembler) database through Neo4j to retrieve biological
relationships and networks. It supports queries for compounds, genes, and
MeSH terms, with automatic ID resolution and standardized output formatting.

Key functionality:
- Query compound-gene relationships
- Retrieve upstream/downstream gene networks
- Query gene-disease associations via MeSH terms
- Standardize biological identifiers (ChEBI, HGNC, MeSH)
- Format results into structured DataFrames

Author: Devon Kohler
"""

# Standard library imports
import json
import os
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Tuple

# INDRA imports
import indra_cogex

# Third-party imports
import pandas as pd
from dotenv import load_dotenv
from indra.databases.chebi_client import get_chebi_id_from_name, get_chebi_name_from_id
from indra.databases.hgnc_client import get_hgnc_id, get_hgnc_name
from indra.databases.mesh_client import get_mesh_name
from indra.databases.uniprot_client import get_gene_name
from indra.statements import Statement
from indra_cogex.client import Neo4jClient
from indra_cogex.representation import norm_id

# Local imports
from causomic.graph_construction.utils import get_neighbor_network

# Load environment variables
load_dotenv()


def compound_query(*, compounds: Iterable[Tuple[str, str]], client: Neo4jClient) -> List[Statement]:
    """
    Query the INDRA database for relationships between compounds and human genes/proteins.

    Retrieves all relationships where the source is one of the specified compounds
    and the target is a human gene/protein entity. Uses Neo4j Cypher queries to
    traverse the INDRA knowledge graph.

    Parameters
    ----------
    compounds : Iterable[Tuple[str, str]]
        Iterable of tuples containing compound identifiers.
        Each tuple should be (namespace, identifier), e.g., ("chebi", "CHEBI:15377")
    client : Neo4jClient
        Neo4j client instance for executing database queries

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing relationships between
        the specified compounds and human gene/protein entities

    Examples
    --------
    >>> compounds = [("chebi", "CHEBI:15377"), ("chebi", "CHEBI:16991")]
    >>> statements = compound_query(compounds=compounds, client=neo4j_client)
    >>> print(f"Found {len(statements)} compound-gene relationships")
    """
    compounds_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in compounds])

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{compounds_nodes_str}]
            AND n1.id <> n2.id
            AND n2.type = "human_gene_protein"
        RETURN p
        """
    )

    return client.query_relations(query)


def get_ids(ids: Iterable[str], type: str) -> List[Tuple[str, str]]:
    """
    Resolve entity names/IDs to standardized CURIEs (Compact URIs).

    Converts a list of entity names or identifiers to standardized namespace:ID
    format using appropriate database clients. Handles failed resolutions gracefully.

    Parameters
    ----------
    ids : Iterable[str]
        List or iterable of entity names or IDs to resolve
    type : str
        Type of entity to resolve. Supported values:
        - "chebi": Chemical entities (resolved via ChEBI database)
        - "gene": Gene entities (resolved via HGNC database)

    Returns
    -------
    List[Tuple[str, str]]
        List of tuples containing (namespace, resolved_id).
        Only successfully resolved IDs are included.

    Examples
    --------
    >>> chemical_names = ["glucose", "ATP", "caffeine"]
    >>> curies = get_ids(chemical_names, "chebi")
    >>> print(curies)  # [("chebi", "CHEBI:17234"), ("chebi", "CHEBI:15422"), ...]

    >>> gene_names = ["EGFR", "TP53", "BRCA1"]
    >>> curies = get_ids(gene_names, "gene")
    >>> print(curies)  # [("hgnc", "3236"), ("hgnc", "11998"), ...]
    """
    parsed_ids = set()
    failed = set()

    if type == "chebi":
        for id in ids:
            chebi_id = get_chebi_id_from_name(id)
            if chebi_id:
                parsed_ids.add(chebi_id)
            else:
                failed.add(id)

        curies = [("chebi", chebi_id) for chebi_id in parsed_ids if chebi_id is not None]

    elif type == "gene":
        for id in ids:
            gene_id = get_hgnc_id(id)
            if gene_id:
                parsed_ids.add(gene_id)
            else:
                failed.add(id)

        curies = [("hgnc", gene_id) for gene_id in parsed_ids if gene_id is not None]

    else:
        raise ValueError(f"Unsupported entity type: {type}. Supported types: 'chebi', 'gene'")

    return curies


def format_query_results(queries: List[Statement]) -> pd.DataFrame:
    """
    Format INDRA statements into a standardized pandas DataFrame.

    Processes INDRA Relation statements to extract key information including
    entity IDs, human-readable names, relationship types, evidence counts,
    belief scores, and publication references. Handles various data structures
    and missing fields gracefully.

    Parameters
    ----------
    queries : List[Statement]
        List of INDRA Relation statements to format

    Returns
    -------
    pd.DataFrame
        DataFrame with standardized columns:
        - source_id: Source entity identifier
        - source_symbol: Human-readable source name
        - relation: Relationship/statement type
        - target_id: Target entity identifier
        - target_symbol: Human-readable target name
        - stmt_hash: Unique statement hash (if available)
        - evidence_count: Number of supporting evidences
        - belief: Belief score (0-1, if available)
        - source_counts: Aggregated source counts
        - pmid: Publication references dictionary

    Notes
    -----
    Only processes statements where both source and target namespaces
    are supported by the ID mapping functions. Unsupported statements
    are silently filtered out.

    Examples
    --------
    >>> statements = compound_query(compounds=compounds, client=client)
    >>> df = format_query_results(statements)
    >>> print(df.columns.tolist())
    ['source_id', 'source_symbol', 'relation', 'target_id', 'target_symbol', ...]
    """
    columns = [
        "source_id",
        "source_symbol",
        "relation",
        "target_id",
        "target_symbol",
        "stmt_hash",
        "evidence_count",
        "belief",
        "source_counts",
        "pmid",
    ]

    # Mapping from namespace to name resolution function
    id_mapper: Dict[str, callable] = {
        "HGNC": get_hgnc_name,
        "CHEBI": get_chebi_name_from_id,
        "MESH": get_mesh_name,
        "UP": get_gene_name,
    }

    # Custom field mapping for different relation types
    relation_mapper: Dict[str, str] = {"gene_disease_association": "papers"}

    # Process INDRA statements and extract relevant data
    rows = []
    for relation in queries:
        # Filter for supported relation types and namespaces
        if (
            isinstance(relation, indra_cogex.representation.Relation)
            and relation.source_ns in id_mapper
            and relation.target_ns in id_mapper
        ):

            # Extract statement type or relation type
            stmt_type = relation.data.get("stmt_type") or relation.rel_type

            # Get appropriate evidence count field
            evidence_field = relation_mapper.get(relation.rel_type, "evidence_count")
            evidence_count = relation.data.get(evidence_field)

            # Extract publication references from statement JSON
            pmid = None
            if "stmt_json" in relation.data:
                try:
                    stmt_json = json.loads(relation.data["stmt_json"])
                    if (
                        stmt_json.get("evidence")
                        and len(stmt_json["evidence"]) > 0
                        and "text_refs" in stmt_json["evidence"][0]
                    ):
                        pmid = stmt_json["evidence"][0]["text_refs"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    pmid = None

            # Calculate source counts sum
            source_counts = None
            if "source_counts" in relation.data:
                try:
                    source_counts = sum(json.loads(relation.data["source_counts"]).values())
                except (json.JSONDecodeError, TypeError):
                    source_counts = None

            rows.append(
                (
                    relation.source_id,
                    id_mapper[relation.source_ns](relation.source_id),
                    stmt_type,
                    relation.target_id,
                    id_mapper[relation.target_ns](relation.target_id),
                    relation.data.get("stmt_hash"),
                    evidence_count,
                    relation.data.get("belief"),
                    source_counts,
                    pmid,
                )
            )

    return pd.DataFrame(rows, columns=columns)


def pull_compound_data(compound_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve and format compound-gene interaction data from INDRA database.

    High-level function that combines ID resolution, database querying, and
    result formatting for compound data. Handles the complete workflow from
    compound names/IDs to structured relationship data.

    Parameters
    ----------
    compound_ids : List[str]
        List of compound identifiers or names (e.g., ChEBI IDs, compound names)
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing compound-gene relationships with
        standardized columns including source/target information, evidence
        counts, and publication references

    Examples
    --------
    >>> compounds = ["glucose", "CHEBI:15377", "caffeine"]
    >>> df = pull_compound_data(compounds, neo4j_client)
    >>> print(f"Found {len(df)} compound-gene interactions")
    >>> print(df[['source_symbol', 'relation', 'target_symbol']].head())
    """
    query_ids = get_ids(compound_ids, "chebi")
    query_results = compound_query(compounds=query_ids, client=client)
    data = format_query_results(query_results)
    return data


def pull_downstream_network(gene_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve downstream gene network from INDRA database.

    Finds all genes that are downstream targets (regulated by) the specified
    input genes. Uses the neighbor network functionality with downstream-only
    filtering and evidence count thresholding.

    Parameters
    ----------
    gene_ids : List[str]
        List of gene identifiers or names (e.g., HGNC symbols, gene names)
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing downstream gene relationships with
        evidence support and relationship metadata

    Examples
    --------
    >>> regulators = ["TP53", "EGFR", "MYC"]
    >>> downstream = pull_downstream_network(regulators, neo4j_client)
    >>> print(f"Found {len(downstream)} downstream relationships")
    """
    query_ids = get_ids(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids, client=client, upstream=False, downstream=True, minimum_evidence_count=1
    )

    data = format_query_results(query_results)
    return data


def pull_upstream_network(gene_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve upstream gene network from INDRA database.

    Finds all genes that are upstream regulators (regulate) the specified
    input genes. Useful for identifying potential causal factors and
    regulatory mechanisms affecting genes of interest.

    Parameters
    ----------
    gene_ids : List[str]
        List of gene identifiers or names for which to find upstream regulators
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing upstream regulatory relationships with
        evidence support and relationship metadata

    Raises
    ------
    ValueError
        If gene_ids is empty or contains invalid identifiers

    Examples
    --------
    >>> targets = ["BRCA1", "BRCA2", "ATM"]
    >>> upstream = pull_upstream_network(targets, neo4j_client)
    >>> print(f"Found {len(upstream)} upstream regulators")
    >>> print(upstream.groupby('source_symbol').size().head())

    Notes
    -----
    Only retrieves relationships with at least one supporting evidence.
    Downstream relationships are excluded to focus on regulatory inputs.
    """
    query_ids = get_ids(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids, client=client, upstream=True, downstream=False, minimum_evidence_count=1
    )

    data = format_query_results(query_results)
    return data


def mesh_query(query_ids: List[Tuple[str, str]], client: Neo4jClient) -> List[Statement]:
    """
    Query Neo4j database for gene-disease associations using MeSH terms.

    Retrieves relationships between human gene/protein entities and specified
    MeSH (Medical Subject Headings) disease terms. Supports both INDRA
    relations and gene-disease associations.

    Parameters
    ----------
    query_ids : List[Tuple[str, str]]
        List of tuples containing MeSH identifiers in (namespace, id) format,
        e.g., [("MESH", "D000544"), ("MESH", "D001943")]
    client : Neo4jClient
        Neo4j client instance for executing database queries

    Returns
    -------
    List[Statement]
        List of relationship objects representing gene-disease associations

    Examples
    --------
    >>> mesh_terms = [("MESH", "D000544"), ("MESH", "D001943")]  # Alzheimer's, Breast Neoplasms
    >>> associations = mesh_query(mesh_terms, neo4j_client)
    >>> print(f"Found {len(associations)} gene-disease associations")
    """
    nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in query_ids])

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel|gene_disease_association]->(n2:BioEntity)
        WHERE
            n2.id IN [{nodes_str}]
            AND n1.id <> n2.id
            AND n1.type = "human_gene_protein"
            AND n1.id CONTAINS "hgnc"
        RETURN p
        """
    )

    return client.query_relations(query)


def pull_mesh_data(mesh_ids: List[str], client: Neo4jClient) -> pd.DataFrame:
    """
    Retrieve and format gene-disease association data using MeSH terms.

    High-level function that combines MeSH ID formatting, database querying,
    and result formatting for disease-gene associations. Provides a complete
    workflow from MeSH disease terms to structured relationship data.

    Parameters
    ----------
    mesh_ids : List[str]
        List of MeSH (Medical Subject Headings) identifier strings for diseases
        or biological processes, e.g., ["D000544", "D001943"]
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame containing gene-disease associations with
        evidence support, belief scores, and publication references

    Examples
    --------
    >>> diseases = ["D000544", "D001943", "D002292"]  # Alzheimer's, Breast Cancer, Cardiomyopathy
    >>> associations = pull_mesh_data(diseases, neo4j_client)
    >>> print(f"Found {len(associations)} gene-disease associations")
    >>> top_genes = associations.groupby('source_symbol')['evidence_count'].sum().sort_values(ascending=False)
    >>> print("Top associated genes:", top_genes.head())

    Notes
    -----
    Automatically converts MeSH IDs to proper namespace format before querying.
    Only returns associations with human gene/protein entities.
    """
    query_ids = [("MESH", mesh_id) for mesh_id in mesh_ids]
    query_results = mesh_query(query_ids=query_ids, client=client)
    data = format_query_results(query_results)
    return data
