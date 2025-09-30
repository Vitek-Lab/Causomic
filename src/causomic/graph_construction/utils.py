"""
Graph Construction Utilities for INDRA Network Queries

This module provides utility functions for constructing biological networks
from the INDRA (Integrated Network and Dynamical Reasoning Assembler) database.
It supports various types of network queries including neighbor networks,
multi-step pathways, and confounding relationships.

Key functionality:
- Direct neighbor network queries (upstream/downstream)
- Multi-step pathway construction (2-4 steps)
- Confounding and mediating relationship detection
- Flexible node and relationship filtering
- Evidence count thresholding for reliability

Author: Devon Kohler
"""

# Standard library imports
from textwrap import dedent
from typing import Dict, Iterable, List, Optional, Tuple

from indra.databases.hgnc_client import get_hgnc_id, get_uniprot_id
from indra.statements import Statement

# INDRA imports
from indra_cogex.client import Neo4jClient
from indra_cogex.client.enrichment.utils import minimum_evidence_helper
from indra_cogex.representation import norm_id

# Database client imports
from protmapper import uniprot_client


def get_neighbor_network(
    *,
    nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    upstream: bool,
    downstream: bool,
    minimum_evidence_count: int,
) -> List[Statement]:
    """
    Retrieve direct neighbor network for specified nodes.

    Queries the INDRA database for direct relationships (one step) between
    the input nodes and their neighbors. Can retrieve upstream regulators,
    downstream targets, or both depending on parameters.

    Parameters
    ----------
    nodes : Iterable[Tuple[str, str]]
        Input nodes as (namespace, identifier) tuples, e.g., [("hgnc", "1234")]
    client : Neo4jClient
        Neo4j client instance for database connectivity
    upstream : bool
        Whether to include upstream neighbors (regulators of input nodes)
    downstream : bool
        Whether to include downstream neighbors (targets of input nodes)
    minimum_evidence_count : int
        Minimum number of evidences required for relationships

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing neighbor relationships

    Raises
    ------
    Exception
        If both upstream and downstream are False

    Examples
    --------
    >>> nodes = [("hgnc", "3236"), ("hgnc", "11998")]  # EGFR, TP53
    >>> neighbors = get_neighbor_network(
    ...     nodes=nodes, client=neo4j_client,
    ...     upstream=True, downstream=False, minimum_evidence_count=2
    ... )
    >>> print(f"Found {len(neighbors)} upstream regulators")
    """
    nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in nodes])

    # Construct query pattern based on direction requirements
    if upstream and downstream:
        q = "p=(n2:BioEntity)-[r1:indra_rel]->(n1:BioEntity)-[r2:indra_rel]->(n3:BioEntity)"
    elif upstream and not downstream:
        q = "p=(n2:BioEntity)-[r1:indra_rel]->(n1:BioEntity)"
    elif not upstream and downstream:
        q = "p=(n1:BioEntity)-[r1:indra_rel]->(n2:BioEntity)"
    else:
        raise Exception("Either upstream or downstream must be True")

    query = f"""\
        MATCH {q}
        WHERE
            n1.id IN [{nodes_str}]
            AND n2.type = "human_gene_protein"
            AND n1.id <> n2.id
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
        RETURN p
    """

    return client.query_relations(query)


def get_two_step_root(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    minimum_evidence_count: int,
) -> List[Statement]:
    """
    Find two-step pathways between root and downstream nodes.

    Retrieves pathways of the form: root_node -> intermediate_node -> downstream_node
    where all relationships have sufficient evidence support. Useful for discovering
    indirect regulatory mechanisms and signaling cascades.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Starting nodes as (namespace, identifier) tuples
    downstream_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    minimum_evidence_count : int
        Minimum evidence count required for each relationship

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing two-step pathways

    Examples
    --------
    >>> root = [("hgnc", "3236")]  # EGFR
    >>> targets = [("hgnc", "11998")]  # TP53
    >>> pathways = get_two_step_root(
    ...     root_nodes=root, downstream_nodes=targets,
    ...     client=neo4j_client, minimum_evidence_count=1
    ... )
    >>> print(f"Found {len(pathways)} two-step pathways")
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])
    downstream_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in downstream_nodes])

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n3:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n2.id IN [{downstream_nodes_str}]
            AND n1.id <> n2.id
            AND n1.id <> n3.id
            AND n2.id <> n3.id
            AND n3.type = "human_gene_protein"
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
            {minimum_evidence_helper(minimum_evidence_count, "r2")}
        RETURN p
        """
    )

    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def get_three_step_root(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    relation: Iterable[str],
    minimum_evidence_count: int,
    mediators: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Statement]:
    """
    Find three-step pathways between root and downstream nodes.

    Retrieves pathways of the form: root -> intermediate1 -> intermediate2 -> downstream
    with optional constraint on intermediate nodes (mediators). Allows filtering
    by specific relationship types and evidence thresholds.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Starting nodes as (namespace, identifier) tuples
    downstream_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types (e.g., ['Phosphorylation', 'Activation'])
    minimum_evidence_count : int
        Minimum evidence count required for each relationship
    mediators : Optional[Iterable[Tuple[str, str]]], default=None
        Optional constraint on intermediate nodes. If provided, intermediate
        nodes must be from this set.

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing three-step pathways

    Examples
    --------
    >>> pathways = get_three_step_root(
    ...     root_nodes=[("hgnc", "3236")],
    ...     downstream_nodes=[("hgnc", "11998")],
    ...     client=neo4j_client,
    ...     relation=['Phosphorylation', 'Activation'],
    ...     minimum_evidence_count=2,
    ...     mediators=[("hgnc", "5290"), ("hgnc", "5291")]  # PI3K family
    ... )
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])
    downstream_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in downstream_nodes])

    # Handle optional mediator constraints
    if mediators is not None:
        mediators_str = ", ".join(["'%s'" % norm_id(*node) for node in mediators])
        mediators_str1 = f"AND n3.id IN [{mediators_str}]"
        mediators_str2 = f"AND n4.id IN [{mediators_str}]"
    else:
        mediators_str1 = ""
        mediators_str2 = ""

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n3:BioEntity)-[r3:indra_rel]->(n4:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n2.id IN [{downstream_nodes_str}]
            AND n1.id <> n2.id
            AND n1.id <> n3.id
            AND n1.id <> n4.id
            AND n2.id <> n3.id
            AND n2.id <> n4.id
            AND n3.id <> n4.id
            AND n3.type = "human_gene_protein"
            AND n4.type = "human_gene_protein"
            AND r1.stmt_type IN {relation}
            AND r2.stmt_type IN {relation}
            AND r3.stmt_type IN {relation}
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
            {minimum_evidence_helper(minimum_evidence_count, "r2")}
            {minimum_evidence_helper(minimum_evidence_count, "r3")}
            {mediators_str1}
            {mediators_str2}
        RETURN p
        """
    )

    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def get_four_step_root(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    relation: Iterable[str],
    minimum_evidence_count: int,
    mediators: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Statement]:
    """
    Find four-step pathways between root and downstream nodes.

    Retrieves complex pathways of the form: root -> int1 -> int2 -> int3 -> downstream
    with optional constraints on intermediate nodes. Useful for discovering
    complex regulatory cascades and multi-step signaling pathways.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Starting nodes as (namespace, identifier) tuples
    downstream_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types for all steps
    minimum_evidence_count : int
        Minimum evidence count required for each relationship
    mediators : Optional[Iterable[Tuple[str, str]]], default=None
        Optional constraint on intermediate nodes

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing four-step pathways

    Notes
    -----
    Four-step pathways can be computationally expensive to query and may
    return many results. Consider using evidence count filters and mediator
    constraints to focus on high-confidence pathways.
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])
    downstream_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in downstream_nodes])

    # Handle optional mediator constraints for all three intermediate nodes
    if mediators is not None:
        mediators_str = ", ".join(["'%s'" % norm_id(*node) for node in mediators])
        mediators_str1 = f"AND n3.id IN [{mediators_str}]"
        mediators_str2 = f"AND n4.id IN [{mediators_str}]"
        mediators_str3 = f"AND n5.id IN [{mediators_str}]"
    else:
        mediators_str1 = ""
        mediators_str2 = ""
        mediators_str3 = ""

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n3:BioEntity)-[r3:indra_rel]->(n4:BioEntity)-[r4:indra_rel]->(n5:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n2.id IN [{downstream_nodes_str}]
            AND n1.id <> n2.id
            AND n1.id <> n3.id
            AND n1.id <> n4.id
            AND n1.id <> n5.id
            AND n2.id <> n3.id
            AND n2.id <> n4.id
            AND n2.id <> n5.id
            AND n3.id <> n4.id
            AND n3.id <> n5.id
            AND n4.id <> n5.id
            AND n3.type = "human_gene_protein"
            AND n4.type = "human_gene_protein"
            AND n5.type = "human_gene_protein"
            AND r1.stmt_type IN {relation}
            AND r2.stmt_type IN {relation}
            AND r3.stmt_type IN {relation}
            AND r4.stmt_type IN {relation}
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
            {minimum_evidence_helper(minimum_evidence_count, "r2")}
            {minimum_evidence_helper(minimum_evidence_count, "r3")}
            {minimum_evidence_helper(minimum_evidence_count, "r4")}
            {mediators_str1}
            {mediators_str2}
            {mediators_str3}
        RETURN p
        """
    )

    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def get_two_step_root_known_med(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    relation: Iterable[str],
    minimum_evidence_count: int,
    mediators: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Statement]:
    """
    Find two-step pathways with known specific mediators.

    Similar to get_two_step_root but with stricter constraints on
    intermediate nodes. Useful when testing specific hypotheses about
    known regulatory intermediates or validating pathway models.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Starting nodes as (namespace, identifier) tuples
    downstream_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types for both steps
    minimum_evidence_count : int
        Minimum evidence count required for each relationship
    mediators : Optional[Iterable[Tuple[str, str]]], default=None
        Required intermediate nodes to constrain pathways

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing constrained two-step pathways

    Notes
    -----
    This function is particularly useful for hypothesis testing when specific
    intermediate proteins are expected to mediate between root and downstream nodes.
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])
    downstream_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in downstream_nodes])

    if mediators is not None:
        mediators_str = ", ".join(["'%s'" % norm_id(*node) for node in mediators])
        mediators_str = f"AND n3.id IN [{mediators_str}]"
    else:
        mediators_str = ""

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n3:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n2.id IN [{downstream_nodes_str}]
            AND n1.id <> n2.id
            AND n1.id <> n3.id
            AND n2.id <> n3.id
            AND n3.type = "human_gene_protein"
            AND r1.stmt_type IN {relation}
            AND r2.stmt_type IN {relation}
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
            {minimum_evidence_helper(minimum_evidence_count, "r2")}
            {mediators_str}
        RETURN p
        """
    )

    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def get_one_step_root_up(
    *, root_nodes: Iterable[Tuple[str, str]], client: Neo4jClient
) -> List[Statement]:
    """
    Find upstream regulators of root nodes.

    Retrieves proteins that directly regulate the specified root nodes.
    Useful for identifying potential upstream causal factors or regulatory
    inputs to a pathway of interest.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing upstream relationships

    Notes
    -----
    This function does not filter by relationship type or evidence count,
    returning all upstream relationships to maximize discovery of potential
    regulatory inputs.
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])

    query = dedent(
        f"""\
        MATCH p=(n2:BioEntity)-[r1:indra_rel]->(n1:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n1.id <> n2.id
            AND n2.type = "human_gene_protein"
        RETURN p
        """
    )

    return client.query_relations(query)


def get_one_step_root_down(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    relation: Iterable[str],
    minimum_evidence_count: int,
) -> List[Statement]:
    """
    Find direct downstream relationships from root to target nodes.

    Retrieves direct causal relationships from root nodes to specified
    downstream targets. Useful for validating known direct regulatory
    connections or exploring immediate downstream effects.

    Parameters
    ----------
    root_nodes : Iterable[Tuple[str, str]]
        Source nodes as (namespace, identifier) tuples
    downstream_nodes : Iterable[Tuple[str, str]]
        Target nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types to query
    minimum_evidence_count : int
        Minimum evidence count required for relationships

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing direct downstream relationships
    """
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])
    downstream_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in downstream_nodes])

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n2.id IN [{downstream_nodes_str}]
            AND n1.id <> n2.id
            AND r1.stmt_type IN {relation}
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
        RETURN p
        """
    )

    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def get_id(ids: Iterable[str], id_type: str) -> List[Tuple[str, str]]:
    """
    Convert protein/gene identifiers to HGNC format.

    Standardizes various identifier types (UniProt, gene symbols) to HGNC
    identifiers for consistent use in network queries. Essential for ensuring
    identifier compatibility across different data sources.

    Parameters
    ----------
    ids : Iterable[str]
        Collection of identifiers to convert
    id_type : str
        Type of input identifiers ("uniprot" or "gene")

    Returns
    -------
    List[Tuple[str, str]]
        List of (namespace, identifier) tuples in HGNC format

    Examples
    --------
    >>> # Convert UniProt IDs to HGNC
    >>> uniprot_ids = ["P31749", "P04637"]  # AKT1, TP53
    >>> hgnc_curies = get_id(uniprot_ids, "uniprot")
    >>> # Returns: [("hgnc", "391"), ("hgnc", "11998")]

    Notes
    -----
    Failed conversions are tracked but not returned. Consider logging
    conversion failures for debugging identifier mapping issues.
    """
    if id_type == "uniprot":
        uniprot_ids = set(ids)

        hgnc_ids = set()
        failed = set()
        for uniprot_id in uniprot_ids:
            hgnc_id = uniprot_client.get_hgnc_id(uniprot_id)
            if hgnc_id:
                hgnc_ids.add(hgnc_id)
            else:
                failed.add(uniprot_id)

    elif id_type == "gene":
        hgnc_ids = set()
        failed = set()
        for gene_id in ids:
            hgnc_id = get_hgnc_id(gene_id)
            get_uniprot_id(gene_id)
            if hgnc_id:
                hgnc_ids.add(hgnc_id)
            else:
                failed.add(gene_id)

    hgnc_curies = [("hgnc", gene_id) for gene_id in hgnc_ids if gene_id is not None]

    return hgnc_curies


def query_between_relationships(
    nodes: Iterable[Tuple[str, str]], client: Neo4jClient, relation: Iterable[str]
) -> List[Statement]:
    """
    Query all relationships between nodes in a set.

    Finds all direct causal relationships connecting any pair of nodes
    within the specified set. Useful for discovering internal connectivity
    within a protein set or pathway.

    Parameters
    ----------
    nodes : Iterable[Tuple[str, str]]
        Collection of nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types to query

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing internal relationships

    Examples
    --------
    >>> # Find relationships within a protein complex
    >>> complex_proteins = [("HGNC", "391"), ("HGNC", "6973"), ("HGNC", "11998")]
    >>> internal_rels = query_between_relationships(
    ...     nodes=complex_proteins,
    ...     client=neo4j_client,
    ...     relation=["Phosphorylation", "Activation"]
    ... )
    """
    nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in nodes])
    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r:indra_rel]->(n2:BioEntity)
        WHERE 
            n1.id IN [{nodes_str}]
            AND n2.id IN [{nodes_str}]
            AND n1.id <> n2.id
            AND r.stmt_type IN {relation}
        RETURN p
    """
    )
    return client.query_relations(query)


def query_confounder_relationships(
    nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient,
    minimum_evidence_count: int,
    mediators: Optional[Iterable[Tuple[str, str]]] = None,
) -> List[Statement]:
    """
    Find potential confounding relationships between node pairs.

    Identifies proteins that have common causes (fork structures) with nodes
    in the specified set. These represent potential confounders that could
    bias causal inferences if not properly controlled.

    Parameters
    ----------
    nodes : Iterable[Tuple[str, str]]
        Collection of nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    minimum_evidence_count : int
        Minimum evidence count required for relationships
    mediators : Optional[Iterable[Tuple[str, str]]], default=None
        Optional constraint on confounding nodes

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing confounding relationships

    Notes
    -----
    Confounders create spurious correlations and must be identified for
    valid causal inference. This function finds fork structures where
    a common cause influences multiple measured proteins.
    """
    if mediators is not None:
        mediators_str = ", ".join(["'%s'" % norm_id(*node) for node in mediators])
        mediators_str = f"AND n3.id IN [{mediators_str}]"
    else:
        mediators_str = ""

    nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in nodes])
    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)<-[r1:indra_rel]-(n3:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{nodes_str}]
            AND n2.id IN [{nodes_str}]
            AND n1.id <> n2.id
            AND NOT n3.id IN [{nodes_str}]
            AND n3.type = "human_gene_protein"
            {minimum_evidence_helper(minimum_evidence_count, "r1")}
            {minimum_evidence_helper(minimum_evidence_count, "r2")}
            {mediators_str}
        RETURN p
    """
    )
    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]


def query_mediator_relationships(
    nodes: Iterable[Tuple[str, str]], client: Neo4jClient, relation: Iterable[str]
) -> List[Statement]:
    """
    Find mediating relationships between node pairs.

    Identifies two-step pathways (chains) connecting pairs of nodes in the
    specified set through intermediate proteins. These mediators represent
    potential causal mechanisms linking measured proteins.

    Parameters
    ----------
    nodes : Iterable[Tuple[str, str]]
        Collection of nodes as (namespace, identifier) tuples
    client : Neo4jClient
        Neo4j client instance for database connectivity
    relation : Iterable[str]
        List of allowed relationship types for both steps

    Returns
    -------
    List[Statement]
        List of INDRA Statement objects representing mediating pathways

    Notes
    -----
    Mediators create indirect causal effects and are important for
    understanding the mechanisms underlying observed correlations.
    This function identifies chain structures: A -> mediator -> B.
    """
    nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in nodes])
    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r1:indra_rel]->(n3:BioEntity)-[r2:indra_rel]->(n2:BioEntity)
        WHERE
            n1.id IN [{nodes_str}]
            AND n2.id IN [{nodes_str}]
            AND n1.id <> n2.id
            AND NOT n3 IN [{nodes_str}]
            AND r1.stmt_type IN {relation}
            AND r2.stmt_type IN {relation}
        RETURN p
    """
    )
    return [
        relation
        for path in client.query_tx(query)
        for relation in client.neo4j_to_relations(path[0])
    ]
