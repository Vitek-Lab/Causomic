from indra.databases.hgnc_client import get_hgnc_name
from indra_cogex.client.subnetwork import indra_shared_upstream_subnetwork
from indra_cogex.client import Neo4jClient

import pandas as pd
import numpy as np
import networkx as nx
import json

import matplotlib.pyplot as plt

from MScausality.graph_construction.utils import get_id, query_confounder_relationships, get_two_step_root, get_two_step_root_known_med

import os
import time
from dotenv import load_dotenv 
load_dotenv() 

def get_root_neighbors(root_ids, downstream_ids, id_type, client, evidence_count=5):

    root_hgnc_id = get_id(root_ids, id_type)
    downstream_hgnc_id = get_id(downstream_ids, id_type)
    # get the network
    # for i in range(levels):
    neighbors = get_two_step_root_known_med(#get_two_step_root
        root_nodes=root_hgnc_id,
        downstream_nodes=downstream_hgnc_id,
        client=client,
        minimum_evidence_count=evidence_count)

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
    df = df.drop_duplicates()
    # df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    # df = df.loc[df["evidence_count"] >= evidence_count]
    # df = df[df["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]
    # df = df[(-pd.isna(df["source_hgnc_symbol"])) & \
    #         (-pd.isna(df["target_hgnc_symbol"]))]

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
    # ids = np.unique(np.concatenate([df.loc[:, "source_hgnc_id"].values, 
    #                                 df.loc[:, "target_hgnc_id"].values]))
    # ids = [30581, 1103]
    hgnc_curies = [("hgnc", gene_id) for gene_id in ids if gene_id is not None]
    
    t0 = time.time()
    neighbors = query_confounder_relationships(
        nodes=hgnc_curies,
        client=client,
        minimum_evidence_count=evidence_count
    )
    t1 = time.time()
    print(t1-t0)

    rows = []
    skipped = 0
    for relation in neighbors:
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
    temp_df = temp_df.drop_duplicates()
    # temp_df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    # temp_df = temp_df.loc[temp_df["evidence_count"] >= evidence_count]

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
        ['ENOB', 'BAZ2B', 'ENOA', 'DCNL1', 'YAP1',
       'MPRD', 'ALG2', 'EURL', 'MPC2', 'LAMA1', 'RIPR2', 'DAZP1', 'ITAM',
       'UTP15', 'UB2Q1', 'DOCK6', 'HACL2', 'CHM4C', 'UB2G2', 'TKT',
       'P85A', 'HSDL2', 'UTP4', 'TIM8B', 'PDE1C', 'WDR43', 'BC11A',
       'RN3P2', 'PRP18', 'TIA1', 'PAIP1', 'BHE40', 'K2C1B', 'PHAR2',
       'ZBT21', 'MED8', 'DPYL2', 'PKP3', 'ZN184', 'HYCCI', 'ARSB',
       'DGCR8', 'LCMT1', 'EMRE', 'MCRI2', 'BL1S4', 'ZFP62', 'TXNIP',
       'HEAT3', 'H2A2C', 'GTR5', 'RPAB5', 'PSA', 'BRI3B', 'FUBP1',
       'RNBP6', 'MTX3', 'AGRIN', 'WDR75', 'ELP2', 'SPP24', 'NDUA1',
       'MTMRC', 'NOL8', 'PIP30', 'MTMR9', 'LRC58', 'ACTY', 'ITAL',
       'INP5K', 'GDIB', 'WDR81', 'MAPK2', 'TTC7A', 'DJB12', 'CHSS1',
       '2A5G', 'SDHB', 'UBP3', 'ANM3', 'ZN131', 'PP1R8', 'HLX', 'DNJB6',
       'ZN141', 'ROBO1', 'ZN214', 'K1C14', 'ENOG', 'AL1A1', 'IGSF1',
       'PGK1', 'GTPB4', 'VP13B', 'NDKA', 'TXN4B', 'BAG1'],
         "gene", client, 
         evidence_count=3)
    temp = network.loc[:, ["source_hgnc_symbol", "target_hgnc_symbol"]].reset_index(drop=True)
    print(temp)
    graph = nx.DiGraph()
    for i in range(len(temp)):
        if (temp.iloc[i, 0] is not None) & (temp.iloc[i, 1] is not None):
            graph.add_edge(temp.iloc[i, 0], temp.iloc[i, 1])
    pos = nx.nx_agraph.graphviz_layout(graph, prog="neato")
    nx.draw_networkx(graph, pos)

    plt.show()

if __name__ == "__main__":
    main()