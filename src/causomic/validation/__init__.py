"""Validation and benchmarking utilities.

Tools for comparing learned networks against baselines (PC, hill-climbing,
NOTEARS) and for running end-to-end benchmark and model-validation workflows.
"""

from causomic.validation.network_comparison import fit_hc, fit_notears, fit_pc

__all__ = [
    "fit_hc",
    "fit_notears",
    "fit_pc",
]
