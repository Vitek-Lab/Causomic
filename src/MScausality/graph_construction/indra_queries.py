from textwrap import dedent
from typing import Iterable, Tuple, List

from MScausality.graph_construction.utils import get_neighbor_network
import indra_cogex
from indra_cogex.client import Neo4jClient
from indra_cogex.representation import norm_id
from indra.databases.chebi_client import get_chebi_id_from_name, \
    get_chebi_name_from_id
from indra.databases.mesh_client import get_mesh_name
from indra.databases.hgnc_client import get_hgnc_name, get_hgnc_id
from indra.databases.uniprot_client import get_gene_name
from indra.statements import Statement

import pandas as pd

import os
import json
from dotenv import load_dotenv 
load_dotenv()

def compound_query(
    *,
    compounds: Iterable[Tuple[str, str]],
    client: Neo4jClient
    ) -> List[Statement]:

    """
    Queries the INDRA database for relationships between specified compounds and 
    human gene/protein entities.
    Args:
        compounds (Iterable[Tuple[str, str]]): An iterable of tuples, 
            where each tuple contains identifiers for a compound (e.g., 
            namespace and ID).
            client (Neo4jClient): An instance of Neo4jClient used to execute the 
            query.
    Returns:
        List[Statement]: A list of INDRA Statement objects representing the 
            relationships found between the given compounds and human gene/protein 
            entities.
    """
    
    compounds_nodes_str = ", ".join(["'%s'" % norm_id(*node) for \
                                     node in compounds])

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

def get_ids(ids, type):
    """
    Retrieve standardized CURIEs (Compact URIs) for a list of entity names or 
    IDs based on the specified type.
    Args:
        ids (Iterable[str]): A list or iterable of entity names or IDs to be resolved.
        type (str): The type of entity to resolve. Supported values are:
            - "chebi": Chemical entities, resolved using `get_chebi_id_from_name`.
            - "gene": Gene entities, resolved using `get_hgnc_id`.
    Returns:
        List[Tuple[str, str]]: A list of tuples, each containing the namespace 
            ("chebi" or "hgnc") and the resolved ID.
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
        
        curies = [("chebi", chebi_id) for chebi_id in parsed_ids if \
                  chebi_id is not None]

    elif type == "gene":
        for id in ids:
            gene_id = get_hgnc_id(id)
            if gene_id:
                parsed_ids.add(gene_id)
            else:
                failed.add(id)
        
        curies = [("hgnc", gene_id) for gene_id in parsed_ids \
                  if gene_id is not None]

    return curies

def format_query_results(
        queries: List[Statement]
        ) -> pd.DataFrame:
    """
    Format a list of INDRA statements into a pandas DataFrame with standardized columns.
    This function processes a list of INDRA `Relation` statements, extracting relevant
    information such as source and target IDs, symbols, relation types, evidence counts,
    belief scores, and publication references. It uses namespace-specific mapping functions
    to convert IDs to human-readable names and handles different relation types and data
    structures within the statements.
    Args:
        queries (List[Statement]): A list of INDRA `Relation` statements to be formatted.
    Returns:
        pd.DataFrame: A DataFrame with the following columns:
            - source_id: Identifier of the source entity.
            - source_symbol: Human-readable name of the source entity.
            - relation: Type of relation or statement.
            - target_id: Identifier of the target entity.
            - target_symbol: Human-readable name of the target entity.
            - stmt_hash: Unique hash of the statement, if available.
            - evidence_count: Number of evidences or papers supporting the statement.
            - belief: Belief score associated with the statement, if available.
            - source_counts: Sum of source counts from the statement data, if available.
            - pmid: Dictionary of publication references (e.g., PMID), if available.
    Notes:
        - Only statements with both source and target namespaces present in `id_mapper` are processed.
        - The function expects certain keys to be present in the statement data; missing keys are handled gracefully.
        - The function relies on external mapping functions such as `get_hgnc_name`, `get_chebi_name_from_id`, etc.
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
        "pmid"
    ]

    id_mapper = {
        "HGNC": get_hgnc_name,
        "CHEBI": get_chebi_name_from_id,
        "MESH": get_mesh_name,
        "UP": get_gene_name
    }
    
    relation_mapper = {
        "gene_disease_association": "papers"
    }

    
    # Loop over INDRA statements and extract relevant data
    rows = []
    for relation in queries:
        if (type(relation) == indra_cogex.representation.Relation) & \
            (relation.source_ns in id_mapper.keys()) & \
                (relation.target_ns in id_mapper.keys()):
            rows.append(
                (
        relation.source_id,
        id_mapper[relation.source_ns](relation.source_id),
        relation.data["stmt_type"] if "stmt_type" in relation.data.keys() else relation.rel_type,
        relation.target_id,
        id_mapper[relation.target_ns](relation.target_id),
        relation.data["stmt_hash"] if "stmt_hash" in relation.data.keys() else None,
        relation.data[relation_mapper.get(relation.rel_type, 'evidence_count')
                      ] if relation_mapper.get(relation.rel_type, 'evidence_count'
                                               ) in relation.data.keys() else None,
        relation.data["belief"] if "belief" in relation.data.keys() else None,
        json.loads(relation.data["source_counts"]) \
            if "source_counts" in relation.data.keys() else None,
        json.loads(relation.data['stmt_json']
                )['evidence'][0]['text_refs'] if \
                    "stmt_json" in relation.data.keys() \
                        and len(json.loads(relation.data['stmt_json']
                                        )['evidence']) > 0 and \
                                            'text_refs' in json.loads(
                                                relation.data['stmt_json'])['evidence'][0].keys() else None
    )
            )

    df = pd.DataFrame(rows, columns=columns)

    return df

def pull_compound_data(compound_ids: List[str], 
                       client: Neo4jClient) -> pd.DataFrame:
    
    """
    Retrieve and format compound data from the INDRA database.
    Given a list of compound IDs, this function queries the INDRA database for information
    related to those compounds, formats the results, and returns them as a pandas DataFrame.
    Args:
        compound_ids (List[str]): A list of compound identifiers (e.g., ChEBI IDs).
        client (Neo4jClient): An instance of the Neo4jClient used to connect and query the database.
    Returns:
        pd.DataFrame: A DataFrame containing the formatted compound data retrieved from the database.
    """
    
    query_ids = get_ids(compound_ids, "chebi")
    
    query_results = compound_query(compounds=query_ids,
                                   client=client)
    
    data = format_query_results(query_results)
    return data

def pull_downstream_network(gene_ids: List[str],
                            client: Neo4jClient) -> pd.DataFrame:
    
    query_ids = get_ids(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids,
        client=client,
        upstream=False,
        downstream=True,
        minimum_evidence_count=1
    )

    data = format_query_results(query_results)

    return data

def pull_upstream_network(gene_ids: List[str],
                            client: Neo4jClient) -> pd.DataFrame:
    
    """
    Retrieve the upstream network for a list of gene IDs using a Neo4j client.
    This function queries a Neo4j database to find upstream neighbors (regulators or influencers)
    of the specified gene IDs. It formats and returns the results as a pandas DataFrame.
    Args:
        gene_ids (List[str]): A list of gene identifiers for which to retrieve the upstream network.
        client (Neo4jClient): An instance of Neo4jClient used to connect and query the database.
    Returns:
        pd.DataFrame: A DataFrame containing the formatted upstream network information for the given genes.
    Raises:
        ValueError: If gene_ids is empty or invalid.
        Exception: If the query or formatting fails.
    Note:
        - Only upstream neighbors are retrieved (downstream is set to False).
        - Only neighbors with at least one supporting evidence are included.
    """
    
    query_ids = get_ids(gene_ids, "gene")

    query_results = get_neighbor_network(
        nodes=query_ids,
        client=client,
        upstream=True,
        downstream=False,
        minimum_evidence_count=1
    )

    data = format_query_results(query_results)

    return data

def mesh_query(query_ids: List[str], client: Neo4jClient):

    """
    Queries the Neo4j database for relationships between human gene/protein entities and a set of specified MeSH IDs.
    Args:
        query_ids (List[str]): A list of MeSH IDs to query for associated relationships.
        client (Neo4jClient): An instance of Neo4jClient used to execute the query.
    Returns:
        Any: The result of the Neo4j query, typically a list of paths representing relationships between human gene/protein entities and the specified MeSH IDs.
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

def pull_mesh_data(mesh_ids: List[str], 
                       client: Neo4jClient) -> pd.DataFrame:
    
    """
    Retrieve and format data for a list of MeSH IDs from the INDRA database.
    Args:
        mesh_ids (List[str]): A list of MeSH (Medical Subject Headings) identifier strings to query.
        client (Neo4jClient): An instance of Neo4jClient used to connect and execute queries on the Neo4j database.
    Returns:
        pd.DataFrame: A pandas DataFrame containing the formatted results of the MeSH queries.
    Raises:
        Any exceptions raised by `mesh_query` or `format_query_results` will propagate up to the caller.
    """

    
    query_ids = [("MESH", mesh_id) for mesh_id in mesh_ids]
    
    query_results = mesh_query(query_ids = query_ids,
                                   client=client)
    
    data = format_query_results(query_results)
    return data
