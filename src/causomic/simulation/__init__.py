"""Synthetic graph and data generation for method development and testing.

Provides hand-built example DAGs, procedural random/structured DAG generation
(including INDRA-style misspecification), cyclic-graph generation, and routines
to simulate proteomics-style data over a causal graph.
"""

from causomic.simulation.cyclic_network import (
    generate_cyclic_graph,
    simulate_cyclic_data,
)
from causomic.simulation.example_graphs import (
    backdoor,
    frontdoor,
    mediator,
    signaling_network,
)
from causomic.simulation.proteomics_simulator import (
    add_missing,
    build_igf_network,
    generate_coefficients,
    simulate_data,
)
from causomic.simulation.random_network import (
    generate_indra_data,
    generate_random_dag,
    generate_structured_dag,
    ground_truth_interventional_effect,
    indra_dag_to_evidence_graph,
)

__all__ = [
    "add_missing",
    "backdoor",
    "build_igf_network",
    "frontdoor",
    "generate_coefficients",
    "generate_cyclic_graph",
    "generate_indra_data",
    "generate_random_dag",
    "generate_structured_dag",
    "ground_truth_interventional_effect",
    "indra_dag_to_evidence_graph",
    "mediator",
    "signaling_network",
    "simulate_cyclic_data",
    "simulate_data",
]
