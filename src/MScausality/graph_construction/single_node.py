
from indra.databases.hgnc_client import get_uniprot_id, get_hgnc_id, get_hgnc_name

from indra_cogex.client import Neo4jClient

from MScausality.graph_construction.utils import get_neighbor_network, query_confounder_relationships

from protmapper import uniprot_client

import pandas as pd
import networkx as nx
import json

import matplotlib.pyplot as plt

import os
from dotenv import load_dotenv 
load_dotenv() 

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


def get_neighbors(ids, id_type, client, evidence_count=5,
                  upstream=True, downstream=True):

    hgnc_id = get_id(ids, id_type)
    # get the network
    # for i in range(levels):
    if upstream:
        neighbors_u = get_neighbor_network(
            nodes=hgnc_id,
            client=client,
            upstream=upstream,
            downstream=False,
            minimum_evidence_count=evidence_count)
    else:
        neighbors_u = list()
    if downstream: 
        neighbors_d = get_neighbor_network(
            nodes=hgnc_id,
            client=client,
            upstream=False,
            downstream=downstream,
            minimum_evidence_count=evidence_count)
    else:
        neighbors_d = list()

    neighbors = neighbors_u + neighbors_d

    columns = [
        "source_hgnc_id",
        "source_hgnc_symbol",
        "relation",
        "target_hgnc_id",
        "target_hgnc_symbol",
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
    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]

    return df

def get_counfounders(ids, id_type, client, evidence_count=5):
    hgnc_id = get_id(ids, id_type)
    confounders = query_confounder_relationships(hgnc_id, client, evidence_count)

    columns = [
        "source_hgnc_id",
        "source_hgnc_symbol",
        "relation",
        "target_hgnc_id",
        "target_hgnc_symbol",
        "stmt_hash",
        "evidence_count"
    ]

    rows = []
    skipped = 0
    for relation in confounders:
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
    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]

    return df


def build_network(initial_nodes, id_type, client, evidence_count=5, levels=3,
                  upstream=True, downstream=True):

    if type(evidence_count) != list:
        evidence_count = [evidence_count] * levels
    if type(upstream) != list:
        upstream = [upstream] * levels
    if type(downstream) != list:
        downstream = [downstream] * levels

    neighbors_con_list = list()

    # Get neighbors for each level
    for i in range(levels):
        neighbors = get_neighbors(initial_nodes, 
                                  id_type, 
                                  client, 
                                  evidence_count[i],
                                  upstream[i],
                                  downstream[i])
        
        # Concat target and source symbols into a list and remove duplicates
        new_nodes = neighbors.loc[:, ["source_hgnc_symbol", 
                                      "target_hgnc_symbol"]].drop_duplicates()
        source_nodes = new_nodes["source_hgnc_symbol"].unique()
        target_nodes = new_nodes["target_hgnc_symbol"].unique()
        initial_nodes = list(set(source_nodes).union(set(target_nodes)))

        neighbors_con_list.append(neighbors)

    # Get upstream confounders
    # TODO: Update extraction to actually get both edges of confounding
    # confounders = get_counfounders(initial_nodes, id_type, client, evidence_count[0])
    # neighbors_con_list.append(confounders)

    neighbors = pd.concat(neighbors_con_list)
    neighbors = neighbors.drop_duplicates().reset_index(drop=True)

    return neighbors

def main():
    client = Neo4jClient(url=os.getenv("API_URL"), 
                        auth=(os.getenv("USER"), 
                            os.getenv("PASSWORD"))
                    )
    
    network = build_network(["MYC"], "gene", client, 
                            evidence_count=[100, 100, 70,50], levels=2,
                            upstream=True, downstream=[True, False, False, False])
    
    print(network)
    graph = nx.DiGraph()
    for i in range(len(network)):
        graph.add_edge(network.loc[i, "source_hgnc_symbol"], 
                       network.loc[i, "target_hgnc_symbol"])
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    nx.draw_networkx(graph, pos)

    plt.show()
    print(os.getcwd())
    network.reset_index(drop=True).to_csv(
            "data/INDRA_networks/MYC/4_step_MYC.tsv",
            sep="\t",
            index=False)

if __name__ == "__main__":
    main()