
import networkx as nx
import pandas as pd
import numpy as np
import copy

from y0.graph import NxMixedGraph
from y0.algorithm.simplify_latent import simplify_latent_dag
from y0.identify import is_identifiable
from y0.dsl import P, Variable

import matplotlib.pyplot as plt

class GraphBuilder:

    def __init__(self, indra_statements: pd.DataFrame, experimental_data: pd.DataFrame):
        self.indra_statements = indra_statements
        self.experimental_data = experimental_data

    def prep_experimental_data(self, data_type="TMT", protein_format=None):
        """
        Prepares experimental data for graph building.

        :param data_type: One of 'TMT', 'LFQ'
        :param protein_format: One of None, 'UniProtKB_AC/ID', 'Gene_Namce_Organism'
        :return:
        """
        if data_type == "TMT":
            self.experimental_data.loc[:, "Run"] = self.experimental_data.loc[:,
                                                   "Run"] + "_" + self.experimental_data.loc[:, "Channel"]

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

    def prep_indra_stmts(self, evidence_filter, source_name="", target_name=""):

        if source_name != "":
            self.indra_statements = self.indra_statements.rename(columns={source_name: "source"})
        if target_name != "":
            self.indra_statements = self.indra_statements.rename(columns={target_name: "target"})

        self.raw_indra_statements = self.indra_statements.copy()

        self.indra_statements = self.indra_statements.loc[:, ["source",
                                                              "target",
                                                              "evidence_count",
                                                              "relation"]]

        self.indra_statements = self.indra_statements.groupby(
            ["source", "target", "relation"])["evidence_count"].sum().reset_index()

        if evidence_filter > 0:
            filtered_indra = self.indra_statements[
                self.indra_statements["evidence_count"] > evidence_filter
                ].reset_index(drop=True)
        else:
            filtered_indra = self.indra_statements

        filtered_indra.loc[:, "source_measured"] = filtered_indra.loc[
            :, "source"].isin(self.experimental_data.columns)
        filtered_indra.loc[:, "target_measured"] = filtered_indra.loc[
            :, "target"].isin(self.experimental_data.columns)

        self.indra_statements = filtered_indra
        

    def fix_cycles(self, graph):
        
        # Remove self loops (if they are there)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # Remove cycles
        cycles = list(nx.simple_cycles(graph))
        graph = remove_cycles(graph, self.indra_statements, cycles,
                              self.raw_indra_statements, self.experimental_data)
        return graph

    def build(self,
              data_type="TMT",
              protein_format=None,
              evidence_filter=1,
              source_name="",
              target_name=""):


        self.prep_experimental_data(data_type, protein_format)

        self.prep_indra_stmts(evidence_filter, source_name, target_name)

        graph = nx.from_pandas_edgelist(self.indra_statements, 
                                        source='source',
                                        target='target', 
                                        edge_attr=None, 
                                        create_using=nx.DiGraph())
        
        graph = self.fix_cycles(graph)
        graph = add_low_evidence_edges(graph,
                                       self.raw_indra_statements,
                                       self.experimental_data)

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

        y0_graph = y0_graph.from_latent_variable_dag(self.graph, "hidden")
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
            is_ident = is_identifiable(self.causal_graph, P(pair[1] @ pair[0]))
            if is_ident:
                identify.append((pair[0], pair[1]))
            else:
                not_identify.append((pair[0], pair[1]))

            print(i)
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


def remove_cycles(graph, indra_stmts, cycles,
                  raw_indra_stmts, experimental_data):
    
    for i in range(len(cycles)):
        if len(cycles[i]) == 2:
            edges = indra_stmts[
                ((indra_stmts["source"] == cycles[i][0]) &
                 (indra_stmts["target"] == cycles[i][1])) |
                ((indra_stmts["source"] == cycles[i][1]) &
                  (indra_stmts["target"] == cycles[i][0]))]

            edges_raw = raw_indra_stmts[
                ((raw_indra_stmts["source"] == cycles[i][0]) &
                 (raw_indra_stmts["target"] == cycles[i][1])) |
                ((raw_indra_stmts["source"] == cycles[i][1]) &
                 (raw_indra_stmts["target"] == cycles[i][0]))]
            
            edges = edges.sort_values(by="evidence_count", ascending=False).reset_index(drop=True)

            # drop edge with less evidence
            if edges.loc[0, "evidence_count"] != edges.loc[1, "evidence_count"]:
                graph.remove_edge(edges.loc[1, "source"],
                                  edges.loc[1, "target"])
            elif edges.loc[0, "source_measured"] & \
                    edges.loc[0, "target_measured"]:
                cycle_data = experimental_data.loc[:, [edges.loc[0, "source"], edges.loc[0, "target"]]]
                cycle_corr = cycle_data.corr().iloc[0,1]
                if cycle_corr > 0:
                    edges_to_drop = edges_raw[edges_raw["relation"] != "IncreaseAmount"].reset_index(drop=True)
                    edges_to_keep = edges_raw[edges_raw["relation"] == "IncreaseAmount"].reset_index(drop=True)
                else:
                    edges_to_drop = edges_raw[edges_raw["relation"] == "IncreaseAmount"].reset_index(drop=True)
                    edges_to_keep = edges_raw[edges_raw["relation"] != "IncreaseAmount"].reset_index(drop=True)

                if len(edges_to_keep.loc[:, ["source", "target"]].drop_duplicates()) == 2:
                    raise "cycle could not be resolved"
                    break
                else:
                    for edge in range(len(edges_to_drop)):
                        try:
                            graph.remove_edge(edges_to_drop.loc[edge, "source"],
                                              edges_to_drop.loc[edge, "target"])
                        except nx.exception.NetworkXError:
                            pass
            else:
                # TODO: test how many times this actually happens
                graph.remove_edge(edges.loc[0, "source"],
                                  edges.loc[0, "target"])
                graph.remove_edge(edges.loc[1, "source"],
                                  edges.loc[1, "target"])

        else:
            evidence_in_cycle = dict()
            for node in range(len(cycles[i])):
                
                source_node = cycles[i][node]
                if node+1 < len(cycles[i]):    
                    target_node = cycles[i][node + 1]
                else:
                    target_node = cycles[i][0]

                stmt = indra_stmts[
                    (indra_stmts["source"] == source_node) &
                    (indra_stmts["target"] == target_node)
                    ].reset_index(drop=True)

                if stmt.loc[0, "source_measured"] & \
                    stmt.loc[0, "target_measured"]:
                    evidence_in_cycle[
                        (source_node, target_node)
                        ] = stmt.loc[0, "evidence_count"]

            # Get all keys with minimum value in dictionary
            min_evidence_node = [k for k, v in evidence_in_cycle.items() \
                                 if v == min(evidence_in_cycle.values())]
            if len(min_evidence_node) == 1:
                graph.remove_edge(min_evidence_node[0][0],
                                  min_evidence_node[0][1])
            else:
                cycle_correlations = dict()
                for i in range(len(min_evidence_node)):
                    cycle_data = experimental_data.loc[:, [min_evidence_node[i][0], min_evidence_node[i][1]]]
                    cycle_correlations[min_evidence_node[i]] = cycle_data.corr().iloc[0,1]

                min_corr_node = min(cycle_correlations, key=lambda y: abs(cycle_correlations[y]))
                graph.remove_edge(min_corr_node[0],
                                  min_corr_node[1])

    return graph

def add_low_evidence_edges(graph, indra_stmts, experimental_data, min_corr = .5):

    potential_edges = indra_stmts[indra_stmts["evidence_count"] == 1].reset_index(drop=True)
    potential_edges = potential_edges.loc[:,
                      ["source", "target", "relation"]].drop_duplicates(ignore_index=True)
    test_adds = 0
    for i in range(len(potential_edges)):
        edge_in_graph = graph.has_edge(potential_edges.loc[i, "source"],
                                       potential_edges.loc[i, "target"])
        source_measured = potential_edges.loc[i, "source"] in experimental_data.columns
        target_measured = potential_edges.loc[i, "target"] in experimental_data.columns

        if (not edge_in_graph) & (source_measured & target_measured):
            obs_corr = experimental_data.loc[:, [potential_edges.loc[i, "source"],
                                potential_edges.loc[i, "target"]]].corr().iloc[0, 1]
            if (abs(obs_corr) > min_corr) & \
                    (((potential_edges.loc[i, "relation"] == "IncreaseAmount") & (obs_corr > 0)) | \
                     (((potential_edges.loc[i, "relation"] == "DecreaseAmount") & (obs_corr < 0)))):

                if not graph.has_node(potential_edges.loc[i, "target"]):
                    graph.add_edge(potential_edges.loc[i, "source"],
                                   potential_edges.loc[i, "target"])
                    test_adds += 1
                elif not graph.has_node(potential_edges.loc[i, "source"]):
                    graph.add_edge(potential_edges.loc[i, "source"],
                                   potential_edges.loc[i, "target"])
                    test_adds += 1
                else:
                    if potential_edges.loc[i, "target"] not in nx.ancestors(graph, potential_edges.loc[i, "source"]):
                        graph.add_edge(potential_edges.loc[i, "source"],
                                       potential_edges.loc[i, "target"])
                        test_adds += 1
    return graph



def main():

    experimental_data = pd.read_csv("/Users/kohler.d/Library/CloudStorage/OneDrive-NortheasternUniversity/Northeastern/Research/MS_data/Single_cell/Leduc/MSstats/MSstats_summarized.csv")
    indra_statements = pd.read_csv("../../data/sox_pathway.csv")

    graph = GraphBuilder(indra_statements, experimental_data)
    graph.build(data_type="TMT",
                protein_format="UniProtKB_AC/ID",
                evidence_filter=1,
                source_name="source_hgnc_symbol",
                target_name="target_hgnc_symbol")
    graph.create_latent_graph()
    graph.find_all_identifiable_pairs()
    graph.plot_latent_graph(figure_size=(12, 8))

if __name__ == "__main__":
    main()
            
        
# test = nx.DiGraph()
# test.add_edge("X", "Y")
# test.add_edge("C", "X")
# test.add_edge("C", "Y")
# all_nodes = ["X", "Y", "C"]
# attrs = {node: (True if node not in ["X", "Y"] and node != "\\n" else False) for node in all_nodes}
#
# nx.set_node_attributes(test, attrs, name="hidden")
# # simplified_graph = simplify_latent_dag(copy.deepcopy(test), "hidden")
# y0_graph = NxMixedGraph()
# y0_graph = y0_graph.from_latent_variable_dag(test)
#
# for node, data in test.nodes.items():
#     print(node, data)
#     print(data["hidden"])
#     if data["hidden"]:
#         for a, b in itt.combinations(test.successors(node), 2):
#             print(a,b)
#
# is_identifiable(graph=y0_graph, query=P(Variable("Y") @ Variable("X")))
#
