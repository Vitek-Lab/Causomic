from indra.databases.hgnc_client import get_uniprot_id, get_hgnc_id, get_hgnc_name
from indra_cogex.client.subnetwork import _paths_to_stmts, indra_shared_upstream_subnetwork
from indra_cogex.client import Neo4jClient
from indra_cogex.representation import norm_id

from textwrap import dedent
from typing import Iterable, List, Tuple
from indra.statements import Statement


from protmapper import uniprot_client

import pandas as pd
import numpy as np
import networkx as nx
import json

import matplotlib.pyplot as plt

import os
import time
from dotenv import load_dotenv 
load_dotenv() 


def get_two_step_root(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    downstream_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient
) -> List[Statement]:
    """Return the INDRA Statement subnetwork induced by paths of length
    two between nodes A and B in a query with intermediate nodes X such
    that paths look like A-X-B.

    Parameters
    ----------
    nodes :
        The nodes to query (A and B are one of these nodes in
        the following examples).
    client :
        The Neo4j client.
    first_forward:
        If true, query A->X otherwise query A<-X
    second_forward:
        If true, query X->B otherwise query X<-B

    Returns
    -------
    :
        The INDRA statement subnetwork induced by the query
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
            AND n3.type = "human_gene_protein"
            AND r1.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
            AND r2.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
        RETURN p
        """
    )


    print(query)
    return _paths_to_stmts(client=client, query=query)
    # return client.query_relations(query)

def get_one_step_root_up(
    *,
    root_nodes: Iterable[Tuple[str, str]],
    client: Neo4jClient
    ) -> List[Statement]:
    """Return the INDRA Statement subnetwork induced by paths of length
    two between nodes A and B in a query with intermediate nodes X such
    that paths look like A-X-B.

    Parameters
    ----------
    nodes :
        The nodes to query (A and B are one of these nodes in
        the following examples).
    client :
        The Neo4j client.
    first_forward:
        If true, query A->X otherwise query A<-X
    second_forward:
        If true, query X->B otherwise query X<-B

    Returns
    -------
    :
        The INDRA statement subnetwork induced by the query
    """
    
    root_nodes_str = ", ".join(["'%s'" % norm_id(*node) for node in root_nodes])

    query = dedent(
        f"""\
        MATCH p=(n2:BioEntity)-[r1:indra_rel]->(n1:BioEntity)
        WHERE
            n1.id IN [{root_nodes_str}]
            AND n1.id <> n2.id
            AND n2.type = "human_gene_protein"
            AND r1.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
        RETURN p
        """
    )


    print(query)
    return _paths_to_stmts(client=client, query=query)
    # return client.query_relations(query)

def get_id(ids, id_type):
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


def get_root_neighbors(root_ids, downstream_ids, id_type, client, evidence_count=5):

    root_hgnc_id = get_id(root_ids, id_type)
    downstream_hgnc_id = get_id(downstream_ids, id_type)
    # get the network
    # for i in range(levels):
    neighbors = get_two_step_root(
        root_nodes=root_hgnc_id,
        downstream_nodes=downstream_hgnc_id,
        client=client)

    columns = [
        "source_hgnc_id",
        "source_hgnc_symbol",
        # "source_uniprot_id",
        "relation",
        "target_hgnc_id",
        "target_hgnc_symbol",
        # "target_uniprot_id",
        "stmt_hash",
        "evidence_count"
    ]

    rows = []
    skipped = 0
    for relation in neighbors:
        print(skipped)
        rows.append(
            (
                relation.source_id,
                get_hgnc_name(relation.source_id),
                # source_uniprot,
                relation.data["stmt_type"],
                relation.target_id,
                get_hgnc_name(relation.target_id),
                # target_uniprot,
                relation.data["stmt_hash"],
                sum(json.loads(relation.data["source_counts"]).values())
            )
        )
        skipped+=1

    df = pd.DataFrame(rows, columns=columns)
    df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    df = df.loc[df["evidence_count"] >= evidence_count]
    df = df[df["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]
    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]

    # neighbors = get_one_step_root_up(
    #     root_nodes=root_hgnc_id,
    #     client=client)
    
    # rows = []
    # skipped = 0
    # for relation in neighbors:
    #     print(skipped)
    #     rows.append(
    #         (
    #             relation.source_id,
    #             get_hgnc_name(relation.source_id),
    #             # source_uniprot,
    #             relation.data["stmt_type"],
    #             relation.target_id,
    #             get_hgnc_name(relation.target_id),
    #             # target_uniprot,
    #             relation.data["stmt_hash"],
    #             sum(json.loads(relation.data["source_counts"]).values())
    #         )
    #     )
    #     skipped+=1

    # df2 = pd.DataFrame(rows, columns=columns)
    # df2.drop_duplicates(subset=["stmt_hash"], inplace=True)
    # df2 = df2.loc[df2["evidence_count"] >= evidence_count]
    # df2 = df2[df2["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]
    # df2 = df2[(-pd.isna(df2["source_hgnc_symbol"])) & (-pd.isna(df2["target_hgnc_symbol"]))]
    # df = pd.concat([df, df2])

    ids = np.unique(np.concatenate([root_hgnc_id, 
                                    downstream_hgnc_id]))
    ids = np.unique(np.concatenate([df.loc[:, "source_hgnc_id"].values, 
                                    df.loc[:, "target_hgnc_id"].values]))
    # ids = [30581, 1103]
    hgnc_curies = [("hgnc", gene_id) for gene_id in ids if gene_id is not None]
    
    t0 = time.time()
    neighbors = indra_shared_upstream_subnetwork(
        nodes=hgnc_curies,
        client=client,
    )
    t1 = time.time()
    print(t1-t0)

    print(len(neighbors))

    rows = []
    skipped = 0
    for relation in neighbors:
        print(skipped)
        rows.append(
            (
                relation.source_id,
                get_hgnc_name(relation.source_id),
                # source_uniprot,
                relation.data["stmt_type"],
                relation.target_id,
                get_hgnc_name(relation.target_id),
                # target_uniprot,
                relation.data["stmt_hash"],
                sum(json.loads(relation.data["source_counts"]).values())
            )
        )
        skipped+=1
    temp_df = pd.DataFrame(rows, columns=columns)
    temp_df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    temp_df = temp_df.loc[temp_df["evidence_count"] >= evidence_count]

    df = pd.concat([df, temp_df])
    return df

def build_root_network(initial_root_nodes, initial_downstream_nodes, id_type, client, evidence_count=1):

    # if type(evidence_count) != list:
    #     evidence_count = [evidence_count] * levels

    # for i in range(levels):
    neighbors = get_root_neighbors(
        initial_root_nodes, 
        initial_downstream_nodes,
        id_type, 
        client, 
        evidence_count
        )
    
    # Concat target and source symbols into a list and remove duplicates
    # source_nodes = neighbors["source_hgnc_symbol"].unique()
    # target_nodes = neighbors["target_hgnc_symbol"].unique()
    # initial_nodes = list(set(source_nodes).union(set(target_nodes)))

    return neighbors

def main():
    client = Neo4jClient(url=os.getenv("API_URL"), 
                        auth=(os.getenv("USER"), 
                            os.getenv("PASSWORD"))
                    )
    
    network = build_root_network(
        ["BRD2", "BRD3", "BRD4"], 
        ['BAZ2B', 'ENO1', 'ENO3', 
         'NIBAN2', 'PEBP1', 'SERPINF1', 
         'SLC26A6', 'SSH3', 'TMTC3'],
         "gene", client, 
         evidence_count=1)
    
    graph = nx.DiGraph()
    for i in range(len(network)):
        graph.add_edge(network.iloc[i, 0], network.iloc[i, 1])
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    nx.draw_networkx(graph, pos)

    plt.show()

if __name__ == "__main__":
    main()