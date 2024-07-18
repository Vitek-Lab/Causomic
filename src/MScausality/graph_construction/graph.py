
import networkx as nx
import pandas as pd
import numpy as np
import copy

from y0.graph import NxMixedGraph
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.algorithm.identify import Identification, identify
from y0.dsl import P, Variable

# from mfes import run_mfes_heuristic
from MScausality.graph_construction.graph_reduction import mfas_greedy_min_set

import pickle

import matplotlib.pyplot as plt

class GraphBuilder:

    def __init__(self, indra_stmts: pd.DataFrame, experimental_data: pd.DataFrame, is_msstats_format: bool):
        self.indra_stmts = indra_stmts
        self.experimental_data = experimental_data
        self.is_msstats_format = is_msstats_format

    def prep_experimental_data(self, data_type="TMT", protein_format=None):
        """
        Prepares experimental data for graph building.

        :param data_type: One of 'TMT', 'LF'
        :param protein_format: One of None, 'UniProtKB_AC/ID', 'Gene_Namce_Organism'
        :return:
        """
        if data_type == "TMT":
            self.experimental_data.loc[:, "Run"] = self.experimental_data.loc[:,
                                                   "Run"] + "_" + self.experimental_data.loc[:, "Channel"]
        if data_type == "LF":
            # Rename LogIntensities to Abundance
            self.experimental_data = self.experimental_data.rename(columns={"LogIntensities": "Abundance", "RUN": "Run"})

        self.experimental_data = self.experimental_data.loc[:, ["Protein", "Abundance", "Run"]]

        if protein_format == "UniProtKB_AC/ID":
            self.experimental_data.loc[:, "Protein"] = self.experimental_data.loc[:, "Protein"
                                                       ].str.split("|").str[-1].str.split("_").str[0]
        elif protein_format == "Gene_Name_Organism":
            self.experimental_data.loc[:, "Protein"] = self.experimental_data.loc[:, "Protein"
                                                       ].str.split("_").str[0]

        self.experimental_data = self.experimental_data.groupby(["Protein", "Run"])["Abundance"].sum().reset_index()
        self.experimental_data = pd.pivot_table(data=self.experimental_data,
                                                index='Run', columns='Protein',
                                                values='Abundance')

    def prep_indra_stmts(self, source_name="", target_name=""):

        if source_name != "":
            self.indra_stmts = self.indra_stmts.rename(columns={source_name: "source"})
        if target_name != "":
            self.indra_stmts = self.indra_stmts.rename(columns={target_name: "target"})

        self.indra_stmts = self.indra_stmts.drop_duplicates()

        self.raw_indra_stmts = self.indra_stmts.copy()

        self.indra_stmts = self.indra_stmts.loc[:, ["source", "target", 
                                                    "evidence_count", 
                                                    "relation"]]


        self.indra_stmts = self.indra_stmts.groupby(
            ["source", "target", "relation"])["evidence_count"].sum().reset_index()

        filtered_indra = self.indra_stmts

        filtered_indra.loc[:, "source_measured"] = filtered_indra.loc[
            :, "source"].isin(self.experimental_data.columns)
        filtered_indra.loc[:, "target_measured"] = filtered_indra.loc[
            :, "target"].isin(self.experimental_data.columns)

        self.indra_stmts = filtered_indra
        

    def build_full_graph(self,
              data_type="TMT",
              protein_format=None,
              source_name="",
              target_name=""):

        print("Preparing experimental data...")
        if self.is_msstats_format:
            self.prep_experimental_data(data_type, protein_format)
        else:
            print("Data already in wide format...")

        print("Preparing INDRA statements...")
        self.prep_indra_stmts(source_name, target_name)

        print("Building graph...")
        graph = nx.from_pandas_edgelist(self.indra_stmts, 
                                        source='source',
                                        target='target', 
                                        edge_attr=None, 
                                        create_using=nx.DiGraph())
        
        # Add in node is observed or latent
        attrs = {node: (True if node in self.experimental_data.columns else False) for node in list(graph)}
        nx.set_node_attributes(graph, attrs, name="observed")
        
        self.full_graph = graph

        observed_nodes = sum([graph.nodes[i]["observed"] for i in graph.nodes])
        latent_nodes = len(graph.nodes) - observed_nodes

        self.n_obs_nodes = observed_nodes
        self.n_latent_nodes = latent_nodes

    def build_dag(self):
        print("Fixing cycles...")
        graph = self.reduce_to_dag(self.full_graph)

        self.graph = graph

    def identify_latent_nodes(self):
        """
        Takes experimental data of observed nodes and returns list of proteins that were unobserved. Data must be in
        long format with protein names as columns.

        :return: latent_nodes: list - list of strings indicating nodes that were not measured in experimental data
        """

        obs_nodes = self.experimental_data.columns
        all_nodes = list(self.graph)

        latent_nodes = [i for i in all_nodes if i not in obs_nodes and i != "\\n"]
        attrs = {node: (True if node not in obs_nodes and node != "\\n" else False) for node in all_nodes}

        nx.set_node_attributes(self.graph, attrs, name="hidden")

        self.latent_nodes = latent_nodes

    def create_latent_graph(self):
        """
        Takes latent nodes, removes them from the causal graph, and adds latent edges in their place. Will return a
        NxMixedGraph from the y0 package which can be used to find identifiable queries.

        Returns
        -------
        y0.graph.NxMixedGraph
            A y0 graph which includes observed nodes and latent confounders
        """
        self.identify_latent_nodes()

        simplified_graph = simplify_latent_dag(copy.deepcopy(self.graph), "hidden")
        self.simplified_graph = simplified_graph
        y0_graph = NxMixedGraph()
        y0_graph = y0_graph.from_latent_variable_dag(simplified_graph.graph, "hidden")

        # y0_graph = y0_graph.from_latent_variable_dag(self.graph, "hidden")

        nodes = list(y0_graph.nodes())
        directed_edges = list(y0_graph.directed.edges())
        remove = list()
        ## Remove node if it isnt in directed edges
        for node in nodes:
            found = [item for item in directed_edges if ((item[0] == node) | (item[1] == node))]
            # print(found)
            if len(found) == 0:
                remove.append(node)

        y0_graph = y0_graph.remove_nodes_from(remove)

        self.causal_graph = y0_graph

    def find_all_identifiable_pairs(self):

        def get_nodes_combinations(graph):
            all_pairs_it = nx.all_pairs_shortest_path(graph)
            potential_nodes = list()
            for i in all_pairs_it:
                temp_pairs = list(zip([i[0] for _ in range(len(i[1].keys()))], i[1].keys()))
                for j in temp_pairs:
                    if j[0] != j[1]:
                        potential_nodes.append(j)
            return potential_nodes

        potential_nodes = get_nodes_combinations(self.causal_graph.directed)

        identify = list()
        not_identify = list()
        i=1
        for pair in potential_nodes:
            try:
                is_ident = Identification.from_expression(graph=self.causal_graph, query=P(pair[1] @ pair[0]))
                identify.append((pair[0], pair[1]))
            except:
                not_identify.append((pair[0], pair[1]))

            # print(i)
            i+=1

        self.identified_edges = {"Identifiable": identify, "NonIdentifiable": not_identify}

    def plot_latent_graph(self, figure_size=(4, 3), title=None):

        ## Create new graph and specify color and shape of observed vs latent edges
        temp_g = nx.DiGraph()

        for d_edge in list(self.causal_graph.directed.edges):
            temp_g.add_edge(d_edge[0], d_edge[1], color="black", style='-', size=30)
        for u_edge in list(self.causal_graph.undirected.edges):
            if temp_g.has_edge(u_edge[0], u_edge[1]):
                temp_g.add_edge(u_edge[1], u_edge[0], color="red", style='--', size=1)
            else:
                temp_g.add_edge(u_edge[0], u_edge[1], color="red", style='--', size=1)

        ## Extract edge attributes
        pos = nx.spring_layout(temp_g)
        edges = temp_g.edges()
        colors = [temp_g[u][v]['color'] for u, v in edges]
        styles = [temp_g[u][v]['style'] for u, v in edges]
        arrowsizes = [temp_g[u][v]['size'] for u, v in edges]

        ## Plot
        fig, ax = plt.subplots(figsize=figure_size)
        nx.draw_networkx_nodes(temp_g, pos=pos, node_size=1000, margins=[.1, .1], alpha=.7)
        nx.draw_networkx_labels(temp_g, pos=pos, font_weight='bold')
        nx.draw_networkx_edges(temp_g, pos=pos, ax=ax, connectionstyle='arc3, rad = 0.1',
                               edge_color=colors, width=3, style=styles, arrowsize=arrowsizes)
        if title is not None:
            ax.set_title(title)
        plt.show()


    def reduce_to_dag(self, graph):
        """
        Reduces a graph to a directed acyclic graph (DAG) by removing edges that create cycles. Minimum arc set problem.

        Parameters
        ----------
        graph: nx.DiGraph
            A directed graph

        Returns
        -------
        nx.DiGraph
            A directed acyclic graph (DAG)
        """

        # Remove self loops (if they are there)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # Add weights as edge attributes
        graph = add_weights(graph, self.experimental_data, self.indra_stmts)

        # Remove cycles
        # cycles = list(nx.simple_cycles(graph))
        # graph = run_mfes_heuristic(graph, is_labeled=True)
        graph, arc_set, broken_correlations = mfas_greedy_min_set(graph, self.experimental_data)
        self.removed_edges = arc_set
        self.broken_correlations = broken_correlations
        # graph = minimum_arc_set(graph)
        
        return graph

def add_weights(graph, experimental_data, indra_stmts):
        
    edges = list(graph.edges)

    correlations = dict()
    evidence = dict()

    for edge in edges:
        source = edge[0]
        target = edge[1]
        if source in experimental_data.columns and target in experimental_data.columns:
            edge_data = experimental_data.loc[:, [source, target]]
            edge_corr = edge_data.corr().iloc[0, 1]

            correlations[(source, target)] = abs(edge_corr)
        else:
            correlations[(source, target)] = 0

        evidence[(source, target)] = indra_stmts.loc[
            ((indra_stmts["source"] == source) & 
            (indra_stmts["target"] == target)), "evidence_count"].sum()

    # Normalize dictionary values
    max_corr = max(correlations.values())
    min_corr = min(correlations.values())

    for key, value in correlations.items():
        normalized_value = (value - min_corr) / (max_corr - min_corr)
        correlations[key] = normalized_value

    max_evidence = max(evidence.values())
    min_evidence = min(evidence.values())

    for key, value in evidence.items():
        normalized_value = (value - min_evidence) / (max_evidence - min_evidence)
        evidence[key] = normalized_value

    # For all edges calculate correlation in experimental data
    for edge in edges:
        source = edge[0]
        target = edge[1]
        graph[source][target]["correlation"] = correlations[(source, target)]

        graph[source][target]["evidence_count"] = evidence[(source, target)]

        graph[source][target]["weight"] = 1 - ((evidence[(source, target)]*.5) + (correlations[(source, target)]*.5))
        graph[source][target]["orig_edges"] = [(source, target)]

    correlation_weights = [graph.edges[edge]['correlation'] for edge in graph.edges()]
    evidence_weights = [graph.edges[edge]['evidence_count'] for edge in graph.edges()]
    edge_weights = [graph.edges[edge]['weight'] for edge in graph.edges()]
    
    return graph

def main():

    morf_ccni_obs_network = pd.read_csv("data/INDRA_networks/Talus_networks/CCCNNNNNNAAGWT_UNKNOWN.tsv", sep="\t")
    msstats_data = pd.read_csv("data/Talus/processed_data/ProteinLevelData.csv")

    morf_ccni_obs_dag = GraphBuilder(morf_ccni_obs_network, msstats_data, True)

    morf_ccni_obs_dag.build_full_graph(data_type="LF",
                            protein_format="Gene_Name_Organism",
                            source_name="source_hgnc_symbol",
                            target_name="target_hgnc_symbol")
    morf_ccni_obs_dag.build_dag()

    morf_ccni_obs_dag.create_latent_graph()
    morf_ccni_obs_dag.plot_latent_graph(figure_size=(12, 8))

if __name__ == "__main__":
    main()





    # def fix_cycles(self, graph):
        
    #     # Remove self loops (if they are there)
    #     graph.remove_edges_from(nx.selfloop_edges(graph))

    #     # Remove cycles
    #     # cycles = list(nx.simple_cycles(graph))
    #     cycles=None
    #     # print(len(cycles))
    #     print(len(graph.edges))
    #     graph = remove_cycles(graph, self.indra_stmts, cycles,
    #                           self.raw_indra_stmts, self.experimental_data)
    #     return graph

# def remove_cycles(graph, indra_stmts, cycles,
#                   raw_indra_stmts, experimental_data):
#     try:
#         cycles = nx.find_cycle(graph)
#         cycles = list(sum(cycles, ()))
#         indexes = np.unique(cycles, return_index=True)[1]
#         cycles = [cycles[index] for index in sorted(indexes)]
#         print(cycles)
#         cycle_checker = True
#     except nx.exception.NetworkXNoCycle:
#         cycle_checker = False
#         print(0)

#     counter = 0
#     while cycle_checker:
#         if len(cycles) == 2:
#             edges = indra_stmts[
#                 ((indra_stmts["source"] == cycles[0]) &
#                  (indra_stmts["target"] == cycles[1])) |
#                 ((indra_stmts["source"] == cycles[1]) &
#                   (indra_stmts["target"] == cycles[0]))]

#             edges_raw = raw_indra_stmts[
#                 ((raw_indra_stmts["source"] == cycles[0]) &
#                  (raw_indra_stmts["target"] == cycles[1])) |
#                 ((raw_indra_stmts["source"] == cycles[1]) &
#                  (raw_indra_stmts["target"] == cycles[0]))].drop_duplicates()
            
#             edges = edges.sort_values(by="evidence_count", ascending=False).reset_index(drop=True)

#             # drop edge with less evidence
#             if edges.loc[0, "evidence_count"] != edges.loc[1, "evidence_count"]:
#                 try:
#                     graph.remove_edge(edges.loc[1, "source"],
#                                       edges.loc[1, "target"])
#                     counter+=1
#                 except:
#                     pass
#             elif edges.loc[0, "source_measured"] & \
#                     edges.loc[0, "target_measured"]:
#                 cycle_data = experimental_data.loc[:, [edges.loc[0, "source"], edges.loc[0, "target"]]]
#                 cycle_corr = cycle_data.corr().iloc[0,1]
#                 if cycle_corr > 0:
#                     edges_to_drop = edges_raw[edges_raw["relation"] != "IncreaseAmount"].reset_index(drop=True)
#                     edges_to_keep = edges_raw[edges_raw["relation"] == "IncreaseAmount"].reset_index(drop=True)
#                 else:
#                     edges_to_drop = edges_raw[edges_raw["relation"] == "IncreaseAmount"].reset_index(drop=True)
#                     edges_to_keep = edges_raw[edges_raw["relation"] != "IncreaseAmount"].reset_index(drop=True)

#                 if len(edges_to_keep.loc[:, ["source", "target"]].drop_duplicates()) == 2:
#                     raise "cycle could not be resolved"
#                     break
#                 else:
#                     for edge in range(len(edges_to_drop)):
#                         try:
#                             graph.remove_edge(edges_to_drop.loc[edge, "source"],
#                                               edges_to_drop.loc[edge, "target"])
#                             counter += 1
#                         except nx.exception.NetworkXError:
#                             pass
#             else:
#                 # TODO: test how many times this actually happens
#                 try:
#                     graph.remove_edge(edges.loc[0, "source"],
#                                       edges.loc[0, "target"])
#                     graph.remove_edge(edges.loc[1, "source"],
#                                       edges.loc[1, "target"])
#                     counter += 2
#                     print("yes")
#                 except:
#                     pass

#         else:
#             evidence_in_cycle = dict()
#             for node in range(len(cycles)):
                
#                 source_node = cycles[node]
#                 if node+1 < len(cycles):
#                     target_node = cycles[node + 1]
#                 else:
#                     target_node = cycles[0]

#                 stmt = indra_stmts[
#                     (indra_stmts["source"] == source_node) &
#                     (indra_stmts["target"] == target_node)
#                     ].reset_index(drop=True)
#                 try:
#                     evidence_in_cycle[
#                             (source_node, target_node)
#                             ] = stmt.loc[0, "evidence_count"]
#                 except:
#                     evidence_in_cycle[
#                         (source_node, target_node)
#                     ] = 1000

#             # Get all keys with minimum value in dictionary
#             min_evidence_node = [k for k, v in evidence_in_cycle.items() \
#                                  if v == min(evidence_in_cycle.values())]
#             if len(min_evidence_node) == 1:
#                 try:
#                     graph.remove_edge(min_evidence_node[0][0],
#                                       min_evidence_node[0][1])
#                     counter += 1
#                 except:
#                     pass
#             else:
#                 cycle_correlations = dict()
#                 for i in range(len(min_evidence_node)):
#                     try:
#                         cycle_data = experimental_data.loc[:, [min_evidence_node[i][0], min_evidence_node[i][1]]]
#                         cycle_correlations[min_evidence_node[i]] = cycle_data.corr().iloc[0,1]
#                         counter += 1
#                     except:
#                         pass
#                 if len(cycle_correlations) > 0:
#                     min_corr_node = min(cycle_correlations, key=lambda y: abs(cycle_correlations[y]))
#                     try:
#                         graph.remove_edge(min_corr_node[0],
#                                           min_corr_node[1])
#                         counter += 1
#                     except:
#                         pass
#                 else:
#                     try:
#                         graph.remove_edge(min_evidence_node[0][0],
#                                           min_evidence_node[0][1])
#                         counter += 1
#                     except:
#                         pass
#         try:
#             cycles = nx.find_cycle(graph)
#             cycles = list(sum(cycles, ()))
#             indexes = np.unique(cycles, return_index=True)[1]
#             cycles = [cycles[index] for index in sorted(indexes)]
#             print("cycle removed")
#             print(cycles)
#         except nx.exception.NetworkXNoCycle:
#             cycle_checker = False
#             print(counter)

#     return graph

# def add_low_evidence_edges(graph, indra_stmts, experimental_data, evidence, min_corr = .5):

#     potential_edges = indra_stmts[indra_stmts["evidence_count"] <= evidence].reset_index(drop=True)
#     potential_edges = potential_edges.loc[:,
#                       ["source", "target", "relation"]].drop_duplicates(ignore_index=True)
#     test_adds = 0
#     for i in range(len(potential_edges)):
#         edge_in_graph = graph.has_edge(potential_edges.loc[i, "source"],
#                                        potential_edges.loc[i, "target"])
#         source_measured = potential_edges.loc[i, "source"] in experimental_data.columns
#         target_measured = potential_edges.loc[i, "target"] in experimental_data.columns

#         if (not edge_in_graph) & (source_measured & target_measured) & \
#                 (potential_edges.loc[i, "source"] != potential_edges.loc[i, "target"]):
#             obs_corr = experimental_data.loc[:, [potential_edges.loc[i, "source"],
#                                 potential_edges.loc[i, "target"]]].corr().iloc[0, 1]
#             if (abs(obs_corr) > min_corr) & \
#                     (((potential_edges.loc[i, "relation"] == "IncreaseAmount") & (obs_corr > 0)) | \
#                      (((potential_edges.loc[i, "relation"] == "DecreaseAmount") & (obs_corr < 0)))):

#                 if not graph.has_node(potential_edges.loc[i, "target"]):
#                     graph.add_edge(potential_edges.loc[i, "source"],
#                                    potential_edges.loc[i, "target"])
#                     test_adds += 1
#                 elif not graph.has_node(potential_edges.loc[i, "source"]):
#                     graph.add_edge(potential_edges.loc[i, "source"],
#                                    potential_edges.loc[i, "target"])
#                     test_adds += 1
#                 else:
#                     if potential_edges.loc[i, "target"] not in nx.ancestors(graph, potential_edges.loc[i, "source"]):
#                         graph.add_edge(potential_edges.loc[i, "source"],
#                                        potential_edges.loc[i, "target"])
#                         test_adds += 1
#     return graph