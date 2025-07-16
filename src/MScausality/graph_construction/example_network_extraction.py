from MScausality.graph_construction.utils import get_one_step_root_down, \
    query_confounder_relationships, get_three_step_root, get_two_step_root,\
        get_two_step_root_known_med

from MScausality.graph_construction.indra_queries import pull_compound_data,\
    pull_mesh_data, get_ids, format_query_results

import os
import pandas as pd
from indra_cogex.client import Neo4jClient

def extract_trog_network(client):
    """
    Extracts a gene interaction network relevant to troglitazone and drug-induced liver injury (DILI).

    This function performs the following steps:
    1. Loads gene targets for the compound troglitazone.
    2. Loads gene targets associated with DILI (using MeSH ID D056486).
    3. Identifies one-step, two-step, and three-step relationships between troglitazone targets and DILI targets,
       with configurable minimum evidence counts for each step.
    4. Combines all identified relationships into a single network, removing duplicate edges.
    5. Searches for potential latent confounder relationships among all genes in the network, with a configurable
       minimum evidence count.
    6. Merges confounder relationships into the main network and removes duplicates.

    Args:
        client: An INDRA Neo4j client object used to query gene and compound data.

    Returns:
        pd.DataFrame: A DataFrame containing the combined network of gene relationships, including confounders.
                      Each row represents an edge with source and target gene identifiers.
    """



    # Load compound targets
    print("Loading compound targets")
    compound_ids = ["troglitazone"]
    compound_df = pull_compound_data(compound_ids, client)
    compound_targets = compound_df["target_symbol"].unique()

    # Load DILI targets
    print("Loading DILI targets")
    gene_disease_df = pull_mesh_data(['D056486'], client)
    dili_targets = gene_disease_df[
        (gene_disease_df["relation"] == "gene_disease_association")
        ].drop_duplicates()["source_symbol"].values

    # Two steps between targets and DILI
    print("Running one-step query")
    one_step_relations = format_query_results(
        get_one_step_root_down(
            root_nodes=get_ids(compound_targets, "gene"), 
            downstream_nodes=get_ids(dili_targets, "gene"), 
            client=client, minimum_evidence_count=2) # Edit evidence count as needed
    )


    print("Running two-step query")
    two_step_relations = format_query_results(
        get_two_step_root_known_med(
            root_nodes=get_ids(compound_targets, "gene"), 
            downstream_nodes=get_ids(dili_targets, "gene"), 
            client=client, minimum_evidence_count=5 # Edit evidence count as needed
            )
    )

    print("Running three-step query")
    three_step_relations = format_query_results(
        get_three_step_root(
            root_nodes=get_ids(compound_targets, "gene"), 
            downstream_nodes=get_ids(dili_targets, "gene"), 
            client=client, minimum_evidence_count=5) # Edit evidence count as needed
    )


    # Combine all relations
    all_network_nodes = pd.concat([one_step_relations, two_step_relations, 
                                   three_step_relations])
    all_network_nodes = all_network_nodes.drop_duplicates(
        subset=["source_id", "target_id"]
    ).reset_index(drop=True)

    # Look for latent confounders
    print("Finding confounders")
    confounder_relations = format_query_results(
        query_confounder_relationships(get_ids(all_network_nodes, "gene"), 
                                    client,
                                    minimum_evidence_count=20)
        )

    # Combine confounders with the rest of the network
    all_network_nodes = pd.concat([all_network_nodes, confounder_relations])
    all_network_nodes = all_network_nodes.drop_duplicates(
        subset=["source_id", "target_id"]
    ).reset_index(drop=True)
    return compound_df, gene_disease_df, all_network_nodes


if __name__ == "__main__":
    client = Neo4jClient()
    compound_df, gene_disease_df, all_network_nodes = extract_trog_network(client)
