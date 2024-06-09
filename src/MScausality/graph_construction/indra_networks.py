"""Pull different networks from INDRA"""

import json
import time
import os
from pathlib import Path
from typing import Iterable, Optional, Set, Union
from textwrap import dedent

import bioregistry
import click
import numpy as np
import pandas as pd
import pystow
from indra.databases.hgnc_client import get_hgnc_name, get_uniprot_id, get_hgnc_id
from indra.databases.chebi_client import get_chebi_name_from_id
from indra.databases.mesh_client import get_mesh_name
from protmapper import uniprot_client

from indra_cogex.client import Neo4jClient, autoclient
from indra_cogex.client.subnetwork import indra_shared_upstream_subnetwork, indra_shared_downstream_subnetwork, indra_mediated_subnetwork, indra_subnetwork_relations
from indra_cogex.representation import get_nodes_str
from indra_cogex.client.utils import minimum_evidence_helper
from dotenv import load_dotenv 
load_dotenv() 

OUTPUT_MODULE = pystow.module("indra", "cogex", "analysis", "devon")


def get_query_from_file(fname: str) -> Set[str]:
    """Get the query UniProt set from one of the files in this directory by name."""
    prefix, *lines = HERE.joinpath(fname).read_text().splitlines()
    lines = {line.strip() for line in lines if line.strip()}
    norm_prefix = bioregistry.normalize_prefix(prefix)
    if norm_prefix is None:
        raise ValueError(f"invalid prefix: {prefix}")
    if norm_prefix == "hgnc":
        return {
            uniprot_id.strip()
            for hgnc_id in lines
            for uniprot_id in get_uniprot_id(hgnc_id).split(",")
        }
    elif norm_prefix == "uniprot":
        return lines
    else:
        raise ValueError(f"unhandled prefix: {norm_prefix}")


def get_query_from_tsv(fname, *, column, sep=",") -> Set[str]:
    """Get a gene list query from a TSV file"""
    df = pd.read_csv(fname, sep=sep)
    return _get_query_from_df(df, column=column)


def get_query_from_xlsx(fname, *, column: str) -> Set[str]:
    df = pd.read_excel(fname)
    return _get_query_from_df(df, column=column)


def _get_query_from_df(df: pd.DataFrame, column) -> Set[str]:
    df = df[df[column].notna()]
    lines = set(df[column])
    if column.lower().replace("_", "").removesuffix("id") == "hgnc":
        return {
            uniprot_id.strip()
            for hgnc_id in lines
            for uniprot_id in get_uniprot_id(hgnc_id).split(",")
        }
    elif column.lower().replace("_", "").removesuffix("id") in {"uniprot", "uniprotkb"}:
        return lines
    else:
        raise ValueError(f"unhandled column: {column}")


@autoclient()
def analysis_uniprot(
    ids: Iterable[str],
    *,
    data_path: Optional[Path] = None,
    analysis_id: str,
    client: Optional[Neo4jClient] = None,
    minimum_evidence_count: Optional[int] = None,
    id_type: str = "uniprot",
):
    """
    This analysis takes a list of target proteins and gets a subnetwork
    of INDRA increase amount and decrease amount statements where
    both the source and target are in the list. It uses a default
    minimum evidence count of 8, which filters out the most obscure
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
    analysis_module = OUTPUT_MODULE.module(analysis_id)

    def _get_uniprot_from_hgnc(hgnc_id: str) -> Optional[str]:
        uniprot_id = get_uniprot_id(hgnc_id)
        if "," not in uniprot_id:
            return uniprot_id
        for uu in uniprot_id.split(","):
            if uu in uniprot_ids:
                return uu
        return None

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
    neighbors = indra_subnetwork_relations(
        nodes=hgnc_curies,
        client=client,
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

    df = df[(-pd.isna(df["source_hgnc_symbol"])) & (-pd.isna(df["target_hgnc_symbol"]))]
    df = df[df["relation"].isin(["IncreaseAmount", "DecreaseAmount"])]

    ids = np.unique(np.concatenate([df["source_hgnc_id"].astype(str).values, df["target_hgnc_id"].values]))
    print(len(ids))

    # hgnc_curies = [("hgnc", gene_id) for gene_id in ids if gene_id is not None]
    neighbors = indra_shared_upstream_subnetwork(
        nodes=hgnc_curies,
        client=client,
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

    neighbors = indra_mediated_subnetwork(
        nodes=hgnc_curies,
        client=client,
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
    df = pd.concat([df, temp_df, temp_df2])

    # Add in if source and target were observed or latent
    df["source_observed"] = df["source_hgnc_id"].isin(hgnc_ids)
    df["target_observed"] = df["target_hgnc_id"].isin(hgnc_ids)

    return df


def _read_df(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path).resolve()
    df = pd.read_csv(path, sep=",")
    columns = list(df.columns)
    columns[0] = "uniprot"
    df.columns = columns
    df["hgnc"] = df["uniprot"].map(uniprot_client.get_hgnc_id)
    df = df[df["hgnc"].notna()]
    df = df[["hgnc", *columns]]
    return df

def pull_small_molecules(
        ids: Iterable[str],
        *,
        data_path: Optional[Path] = None,
        analysis_id: str,
        client: Optional[Neo4jClient] = None,
        minimum_evidence_count: Optional[int] = None,
        id_type: str = "uniprot",):
    
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
    nodes_str = get_nodes_str(hgnc_curies)
    evidence_line = minimum_evidence_helper(minimum_evidence_count)

    query = dedent(
        f"""\
        MATCH p=(n1:BioEntity)-[r:indra_rel]->(n2:BioEntity)
        WHERE 
            n2.id IN [{nodes_str}]
            AND n1.type = "small_molecule"
            {evidence_line}
        RETURN p
        """
    )#AND r.stmt_type IN ['IncreaseAmount', 'DecreaseAmount']
    query_results = client.query_relations(query)
    
    rows = []
    for relation in query_results:
        
        if relation.source_ns == "MESH":
            rows.append(
                (
                    relation.source_id,
                    get_mesh_name(str(relation.source_id)),
                    # source_uniprot,
                    relation.data["stmt_type"],
                    relation.target_id,
                    get_hgnc_name(relation.target_id),
                    # target_uniprot,
                    relation.data["stmt_hash"],
                    sum(json.loads(relation.data["source_counts"]).values()),
                    relation.data["source_counts"],
                )
            )        
        else:
            rows.append(
                (
                    relation.source_id,
                    get_chebi_name_from_id(str(relation.source_id)),
                    # source_uniprot,
                    relation.data["stmt_type"],
                    relation.target_id,
                    get_hgnc_name(relation.target_id),
                    # target_uniprot,
                    relation.data["stmt_hash"],
                    sum(json.loads(relation.data["source_counts"]).values()),
                    relation.data["source_counts"],
                )
            )

    columns = [
        "source_chebi_id",
        "source_chebi_symbol",
        # "source_uniprot_id",
        "relation",
        "target_hgnc_id",
        "target_hgnc_symbol",
        # "target_uniprot_id",
        "stmt_hash",
        "evidence_count",
        "source_counts",
    ]

    df = pd.DataFrame(rows, columns=columns)
    df.drop_duplicates(subset=["stmt_hash"], inplace=True)
    df = df.loc[-pd.isna(df["source_chebi_symbol"])]

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
    # ids = ['PARD6A', 'ARF3', 'DR1', 'RAN']

    data = analysis_uniprot(
        ids=ids,
        analysis_id="small_mol",
        client=client,
        minimum_evidence_count=1,
        id_type="gene")

if __name__ == "__main__":
    main()