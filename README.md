# CausOmic

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Development Status](https://img.shields.io/badge/status-development-orange.svg)](https://github.com/devonjkohler/CausOmic)

**Causal inference methods for mass spectrometry (MS)-based proteomics**

CausOmic is a Python package designed to perform causal inference using different types of omics data, including proteomics, transcriptomics, metabolomics, phosphoproteomics, ect. The primary goal is to predict the effects of interventions (e.g., drug treatments, protein inhibitions) on biological systems by leveraging causal modeling techniques and protein interaction networks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Requirements](#data-requirements)
- [Main Components](#main-components)
- [Usage Examples](#usage-examples)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

A fundamental challenge in biological experimentation is understanding how interventions (e.g., drug treatments, protein inhibitions) affect complex biological systems. Traditional machine learning approaches, particularly black box models, attempt to predict these effects without explicitly modeling the underlying causal relationships. This can be problematic when explainability is crucial (e.g., identifying disease-driving pathways) or when models incorrectly infer that downstream proteins causally influence upstream targets. CausOmic addresses these limitations by:

1. **Integrating prior biological knowledge** from biological network databases
2. **Building causal graphs** that represent protein relationships
3. **Training deep probabilistic models** with variational Bayesian inference (Pyro/PyTorch)
4. **Predicting intervention effects** on downstream proteins

The package is particularly useful for:
- Drug discovery and target identification
- Understanding protein pathway dynamics
- Predicting off-target effects of interventions
- Analyzing perturbation experiments in proteomics

## Features

### 🧬 **Prior Knowledge Network (PKN) Construction**
- Integration with INDRA (Integrated Network and Dynamical Reasoning Assembler)
- Automatic extraction of protein interaction networks
- Support for GSEA-driven pathway analysis
- Posterior network estimation using PKN and experimental data

### 📊 **Causal Modeling**
- Bayesian probabilistic models using Pyro
- Latent variable models for handling missing data
- Support for both observational and interventional data
- Uncertainty quantification for predictions

### 🎯 **Intervention Prediction**
- Predict effects of protein inhibitions
- Estimate downstream pathway responses
- Quantify prediction uncertainty
- Validation against experimental data

### 🔬 **MS Data Processing**
- Integration with proteomics (MSstats) output format
- Data normalization and preprocessing utilities
- Handling of protein-level summarized data

### **Simulation**
- Generate example graphs which exhibit different causal structures
- Simulate data over causal graphs using real world data generating processes
- Leverage simulations for method validations

## Installation

### Prerequisites
- Python 3.9 or higher
- PyTorch
- Pyro-PPL

### Install from source
```bash
git clone https://github.com/devonjkohler/CausOmic.git
cd CausOmic
pip install -e .
```

### Dependencies
The main dependencies include:
- `pyro-ppl==1.8.5` - Probabilistic programming
- `torch` - Deep learning framework
- `networkx` - Graph manipulation
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting
- `y0` - Network analysis utilities

## Quick Start

```python
from causomic.data_analysis.proteomics_data_processor import dataProcess

from causomic.simulation.example_graphs import mediator
from causomic.simulation.proteomics_simulator import simulate_data

# 1. Load your data (we use simulation)
med_graph = mediator(add_independent_nodes=False, output_node=False)
    
simulated_data = simulate_data(
      med_graph['Networkx'], 
      coefficients=med_graph['Coefficients'], 
      add_error=False,
      mnar_missing_param=[-3, 0.4],  # Missing not at random
      add_feature_var=True, 
      n=100, 
      seed=2
)

# 2. Preprocess data (assuming MS proteomics data)
input_data = dataProcess(
    simulated_data["Feature_data"], 
    normalization=False, 
    summarization_method="TMP", 
    MBimpute=False, 
    sim_data=True
)

# 4. Fit causal model
from causomic.causal_model.LVM import LVM

lvm = LVM(backend="pyro", num_steps=2000, verbose=True)
lvm.fit(input_data, med_graph["causomic"])

model = ProteomicPerturbationModel(
    n_obs=len(data),
    root_nodes=['target_protein'],
    downstream_nodes=['downstream_protein1', 'downstream_protein2']
)

# 5. Make predictions
intervention_value = 7.0
lvm.intervention({"X": intervention_value}, "Z")
```

## Data Requirements

### Input Data Format
CausOmic expects data in the MSstats `ProteinLevelData` format:

| Column | Description |
|--------|-------------|
| `Protein` | Protein identifier |
| `LogIntensities` | Log-normalized protein intensities |
| `Condition` | Experimental condition |
| `BioReplicate` | Biological replicate identifier |
| `Run` | MS run identifier |

### Preprocessing with MSstats (R)
Before using CausOmic, process your raw MS data with MSstats in R:

```r
# R code example
library(MSstats)
processed_data <- dataProcess(
    raw_data, 
    annotation_file,
    # ... other parameters
)
# Export ProteinLevelData for CausOmic
write.csv(processed_data$ProteinLevelData, "protein_level_data.csv")
```

## Main Components

### 📈 **Data Analysis** (`causomic.data_analysis`)
- Data normalization and preprocessing
- Statistical utilities for proteomics data
- Integration with MSstats workflows

### 🕸️ **Graph Construction** (`causomic.graph_construction`)
- INDRA network queries and processing
- Protein interaction network building
- Graph filtering and validation utilities

### 🎯 **Causal Modeling** (`causomic.causal_model`)
- Probabilistic models for causal inference
- Bayesian parameter estimation
- Intervention effect prediction
- Latent variable models for missing data

### 🧪 **Simulation** (`causomic.simulation`)
- Synthetic data generation for testing
- Model validation utilities
- Simulation studies for method development

## Usage Examples

### Example Datasets
The package includes several example datasets:
- **Talus Bio**: Small molecule inhibition experiments
- **IGF Pathway**: Insulin-like growth factor signaling
- **MYC Pathway**: MYC oncogene signaling networks
- **Synthetic Data**: Simulated proteomics experiments

### Jupyter Notebooks
Comprehensive examples are available in the `vignettes/` directory:
- `user_manual.ipynb` - Complete workflow tutorial
- `graph_building/` - Network construction examples
- `simulations/` - Simulation studies
- `methods_paper/` - Methodology validation

## Documentation

### User Manual
The primary documentation is available as a Jupyter notebook:
- [User Manual](vignettes/user_manual.ipynb) - Complete workflow and API reference

### API Reference
Detailed API documentation is available in the source code docstrings. Key modules:

- `causomic.causal_model.models` - Core causal models
- `causomic.graph_construction.utils` - Network utilities
- `causomic.data_analysis.normalization` - Data preprocessing
- `causomic.simulation` - Synthetic data generation

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
git clone https://github.com/devonjkohler/CausOmic.git
cd CausOmic
pip install -e ".[dev]"
```

### Code Style
We use Black for code formatting and isort for import sorting:
```bash
black src/
isort src/
```

## Citation

If you use CausOmic in your research, please cite:

```bibtex
@software{kohler2024causomic,
  title={CausOmic: Causal inference methods for mass spectrometry-based proteomics},
  author={Kohler, Devon},
  year={2024},
  url={https://github.com/devonjkohler/CausOmic},
  version={0.0.1-dev}
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
