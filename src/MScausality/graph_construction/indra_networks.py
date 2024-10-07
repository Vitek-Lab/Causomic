"""Pull different networks from INDRA"""

import json
import time
import os
from typing import Iterable, Optional, Set

import click
import numpy as np
import pandas as pd
from indra.databases.hgnc_client import get_hgnc_name, get_uniprot_id, get_hgnc_id
from protmapper import uniprot_client

from indra_cogex.client import Neo4jClient, autoclient

from MScausality.graph_construction.utils import query_between_relationships, query_confounder_relationships, query_mediator_relationships

from dotenv import load_dotenv 
load_dotenv()

@autoclient()
def analysis_uniprot(
    ids: Iterable[str],
    *,
    client: Optional[Neo4jClient] = None,
    minimum_evidence_count: Optional[int] = 1,
    relation_types: Optional[Set[str]] = ["IncreaseAmount", "DecreaseAmount"],
    id_type: str = "uniprot",
):
    """
    This analysis takes a list of target proteins and gets a subnetwork
    of INDRA increase amount and decrease amount statements where
    both the source and target are in the list. It uses a default
    minimum evidence count of 1, which filters out the most obscure
    and likely incorrect statements (e.g., due to technical reader errors).

    .. todo::

        - consider direct versus indirect interactions (i.e., is physical contact
          involved). Should we remove these, or try and keep these "bypass" edges?
        - consider cycles. Look at canonical-ness, e.g., use curated pathway databases
          as a guide to figure out what the "forward" pathways are. This can be
          thought of like an optimization problem - what is the smallest set of least
          important edges to remove.
        - small cycles where A->B and B->A.
    """

    if id_type == "uniprot":
        uniprot_ids = set(ids)
        click.echo(f"Querying {len(uniprot_ids):,} UniProt identifiers")

        hgnc_ids = set()
        failed = set()
        for uniprot_id in uniprot_ids:
            hgnc_id = uniprot_client.get_hgnc_id(uniprot_id)
            if hgnc_id:
                hgnc_ids.add(hgnc_id)
            else:
                failed.add(uniprot_id)
        if failed:
            click.echo(f"Failed to get HGNC ID for UniProts: {sorted(failed)}")

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
        if failed:
            click.echo(f"Failed to get HGNC ID for Gene: {sorted(failed)}")


    hgnc_curies = [("hgnc", gene_id) for gene_id in hgnc_ids if gene_id is not None]
    neighbors = query_between_relationships(
        nodes=hgnc_curies,
        client=client,
        relation=relation_types
    )
    print(len(neighbors))
    columns = [
        "source_hgnc_id",
        "source_hgnc_symbol",
        # "source_uniprot_id",
        "relation",
        "target_hgnc_id",
        "target_hgnc_symbol",
        # "target_uniprot_id",
        "stmt_hash",
        "evidence_count",
        "source_counts",
        "reason_added"
    ]

    start = time.time()
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
                sum(json.loads(relation.data["source_counts"]).values()),
                relation.data["source_counts"],
                ## TODO test if this works
                "direct"
            )
        )
        skipped+=1
    end = time.time()
    print(end - start)

    df = pd.DataFrame(rows, columns=columns)
    df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    df = df.loc[df["evidence_count"] > minimum_evidence_count]

    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]
    df = df[df["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]

    ids = np.unique(np.concatenate([df["source_hgnc_id"].astype(str).values, df["target_hgnc_id"].values]))
    print(len(ids))

    # hgnc_curies = [("hgnc", gene_id) for gene_id in ids if gene_id is not None]
    neighbors = query_confounder_relationships(
        nodes=hgnc_curies,
        client=client,
        relation=relation_types
    )
    print(len(neighbors))
    start = time.time()
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
                sum(json.loads(relation.data["source_counts"]).values()),
                relation.data["source_counts"],
                "confounder"
            )
        )
        skipped+=1
    end = time.time()
    print(end - start)
    temp_df = pd.DataFrame(rows, columns=columns)
    temp_df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    temp_df = temp_df.loc[temp_df["evidence_count"] > minimum_evidence_count]

    neighbors = query_mediator_relationships(
        nodes=hgnc_curies,
        client=client,
        relation=relation_types
    )
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
                sum(json.loads(relation.data["source_counts"]).values()),
                relation.data["source_counts"],
                "mediator"
            )
        )
        skipped+=1

    temp_df2 = pd.DataFrame(rows, columns=columns)
    temp_df2.drop_duplicates(subset=["stmt_hash"], inplace=True)
    temp_df2 = temp_df.loc[temp_df["evidence_count"] > minimum_evidence_count]

    df = pd.concat([df, temp_df, temp_df2])
    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]

    # Add in if source and target were observed or latent
    df["source_observed"] = df["source_hgnc_id"].isin(hgnc_ids)
    df["target_observed"] = df["target_hgnc_id"].isin(hgnc_ids)

    df = df.reset_index(drop=True)

    return df


def main():

    client = Neo4jClient(url=os.getenv("API_URL"), 
                         auth=(os.getenv("USER"), 
                               os.getenv("PASSWORD"))
                        )
    
    ids = ['NACA', 'BRD2', 'BAZ2A', 'DDX49', 'PCBP2', 'EIF3F', 'SRP14', 'CHD4',
           'SNX3', 'HDAC1', 'RO60', 'SRSF9', 'ARPC3', 'NONO', 'HDGF', 'BTF3',
           'EIF3H', 'EIF3E', 'RNPS1', 'RHOA', 'SRRM1', 'RACK1', 'ERP29', 
           'EIF3D', 'SRSF2', 'SUMO2', 'SF3A1', 'WDR1', 'CBX3']
    # ids=["MYC"]
    # ids = ['PARD6A', 'ARF3', 'DR1', 'RAN']

    data = analysis_uniprot(
        ids=ids,
        client=client,
        minimum_evidence_count=1,
        id_type="gene")
    
    print(data)

if __name__ == "__main__":
    main()