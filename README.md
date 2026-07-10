<p align="center">
  <img src="https://github.com/Vitek-Lab/Causomic/blob/main/data/images/logo.png" height="150">
</p>

# Causomic

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![codecov](https://codecov.io/gh/Vitek-Lab/Causomic/branch/main/graph/badge.svg)](https://codecov.io/gh/Vitek-Lab/Causomic)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-development-orange.svg)](https://github.com/Vitek-Lab/Causomic)

**Causal inference methods for -omics research**

Causomic is a Python package for causal inference on -omics data (proteomics, transcriptomics, metabolomics, phosphoproteomics, etc.). Its goal is to predict the effects of interventions (e.g., drug treatments, protein inhibitions) on biological systems by combining prior-knowledge interaction networks with deep probabilistic causal models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Data Requirements](#data-requirements)
- [Main Components](#main-components)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

A fundamental challenge in biological experimentation is understanding how interventions (e.g., drug treatments, protein inhibitions) affect complex biological systems. Traditional machine learning approaches, particularly black box models, attempt to predict these effects without explicitly modeling the underlying causal relationships. This can be problematic when explainability is crucial (e.g., identifying disease-driving pathways) or when models incorrectly infer that downstream proteins causally influence upstream targets. Causomic addresses these limitations by:

1. **Integrating prior biological knowledge** from biological network databases (e.g., INDRA)
2. **Building causal graphs** that represent protein relationships and reconciling them with experimental data
3. **Training deep probabilistic models** with variational Bayesian inference (Pyro/PyTorch)
4. **Predicting intervention effects** on downstream proteins with uncertainty quantification

The package is particularly useful for:
- Drug discovery and target identification
- Understanding protein pathway dynamics
- Predicting off-target effects of interventions
- Analyzing perturbation experiments in proteomics

## Features

### 🧬 **Prior Knowledge Network (PKN) Construction**
- Integration with INDRA (Integrated Network and Dynamical Reasoning Assembler)
- Automatic extraction and filtering of protein interaction networks
- Reconciliation of a prior network with experimental data via bootstrapped structure learning

### 📊 **Causal Modeling**
- Bayesian probabilistic models using Pyro (latent-variable structural causal models)
- Support for both observational and interventional data
- Native handling of missing data
- Uncertainty quantification for predictions

### 🎯 **Intervention Prediction**
- Predict downstream effects of protein inhibitions
- Estimate pathway-level responses
- Validate predictions against experimental data

### 🔬 **MS Data Processing**
- Integration with proteomics (MSstats) output format
- Normalization, summarization, and imputation utilities
- Gene-set correlation and pathway over-representation analysis (ORA)

### 🧪 **Simulation**
- Generate example graphs exhibiting different causal structures
- Simulate realistic proteomics data over causal graphs
- Procedural DAG generation with INDRA-style misspecification for method development

## Installation

### Prerequisites
- Python 3.11
- PyTorch 2.0.1
- Pyro-PPL

### Install from source
```bash
git clone https://github.com/Vitek-Lab/Causomic.git
cd Causomic
pip install -e .
```

### Optional dependencies
- `dev` — testing and linting tools (`pytest`, `black`, `isort`):
  `pip install -e ".[dev]"`
- **INDRA-CoGEx** — required only for the Neo4j-backed CoGEx query features
  (`causomic.network`, `causomic.graph_construction.utils_neo4j`, and
  `neo4j_indra_queries`). It is not on PyPI; install it from source if you need
  those features:
  ```bash
  pip install git+https://github.com/gyorilab/indra_cogex.git
  ```
  The rest of the package works without it.

## Getting Started

The end-to-end workflow follows three steps:

1. **Learn the causal graph** — build a prior-knowledge network (e.g. from INDRA)
   and reconcile it with your data into a causal DAG.
2. **Train the structural causal model** — fit the latent-variable model (`LVM`)
   to your protein-level data over that graph.
3. **Predict interventions** — query the trained model for the downstream
   effect of an intervention (e.g. inhibiting a target protein).

📓 **The complete, runnable walkthrough lives in the
[User Manual notebook](vignettes/user_manual.ipynb).** It covers both a
simulated ground-truth system and a real INDRA network (EGFR inhibition),
from graph construction through interventional inference. Start there.

## Data Requirements

### Input Data Format
Causomic expects data in different formats depending on where in the pipeline
you start. The causal model and graph construction expect data in wide format
with genes as columns, samples as rows, and values being quantitative
experimental measurements.

### Preprocessing with MSstats (R)

If you are using MS-based proteomics data, we recommend running the data through
the MSstats pipeline via `dataProcess`. The resulting `ProteinLevelData` object
can be passed directly into Causomic. A Python-side `dataProcess` is available in
`causomic.data_analysis` for simulated and summarized data.

## Main Components

### 🕸️ **Graph Construction** (`causomic.graph_construction`)
Build, filter, and query biological interaction graphs, and reconcile a prior
network with experimental data.
- `prepare_graph`, `add_evidence_info`, `filter_graph_by_evidence_count`
- `query_drug_targets`, `query_effect_nodes`, `query_forward_paths`, `query_confounders`
- `prepare_indra_priors`, `run_bootstrap`, `calculate_edge_probabilities`

### 🎯 **Causal Modeling** (`causomic.causal_model`)
Probabilistic structural causal models for intervention prediction.
- `LVM` — latent-variable model (fit / intervention interface)
- `ProteomicPerturbationModel`, `StochasticEdgeProteomicModel` — underlying Pyro models

### 📈 **Data Analysis** (`causomic.data_analysis`)
Proteomics preprocessing and downstream analysis.
- `dataProcess`, `normalize_median`, `summarize_data`, `imputation`
- `gen_correlation_matrix`, `test_gene_sets`, `prep_msstats_data`
- `run_ora`, `fetch_pathway_library`, `select_diverse_pathways`, `export_to_cytoscape`

### 🧪 **Simulation** (`causomic.simulation`)
Synthetic graph and data generation for testing and method development.
- `mediator`, `backdoor`, `frontdoor`, `signaling_network` — example graphs
- `simulate_data`, `generate_coefficients`, `build_igf_network`
- `generate_structured_dag`, `generate_indra_data`, `generate_cyclic_graph`

### 🚀 **High-Level Entry Points**
- `causomic.network` — network estimation helpers (`extract_indra_prior`,
  `consensus_dag`, `estimate_posterior_dag`, `filter_to_causal_subgraph`, …)
- `causomic.workflows` — packaged pipelines (`run_toxicity_detection_workflow`)

  Some of these helpers query INDRA-CoGEx and require the optional
  [INDRA-CoGEx install](#optional-dependencies).

## Documentation

### User Manual
The primary documentation is the runnable notebook:
- [User Manual](vignettes/user_manual.ipynb) — complete workflow, from graph
  construction to interventional inference, on both simulated and real data.

### API Reference
Detailed API documentation lives in the source-code docstrings. Key modules:

- `causomic.causal_model.LVM` — latent-variable causal model
- `causomic.causal_model.models` — underlying Pyro model definitions
- `causomic.graph_construction.utils_nx` — network construction and querying
- `causomic.graph_construction.prior_data_reconciliation` — prior/data reconciliation
- `causomic.data_analysis.proteomics_data_processor` — data preprocessing
- `causomic.simulation` — synthetic graph and data generation

## Contributing

We welcome contributions! Please also see [CONTRIBUTING.md](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/Vitek-Lab/Causomic.git
cd Causomic
pip install -e ".[dev]"
```

### Code Style & Tests
We use Black for formatting and isort for import sorting, and pytest for tests:
```bash
black --check src/ tests/
isort --check-only src/ tests/
pytest
```

## Citation

If you use Causomic in your research, please cite:

```bibtex
@software{kohler2024causomic,
  title={Causomic: Causal inference methods for -omics research},
  author={Kohler, Devon},
  year={2024},
  url={https://github.com/Vitek-Lab/Causomic},
  version={0.9.0}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Devon Kohler
- **Email**: kohler.d@northeastern.edu
- **Institution**: Northeastern University
- **GitHub**: [@devonjkohler](https://github.com/devonjkohler)

## Acknowledgments

- [INDRA](https://indra.readthedocs.io/) - Integrated Network and Dynamical Reasoning Assembler
- [MSstats](https://www.bioconductor.org/packages/release/bioc/html/MSstats.html) - Statistical tools for proteomics
- [Pyro](https://pyro.ai/) - Probabilistic programming framework
- [NetworkX](https://networkx.org/) - Network analysis library
