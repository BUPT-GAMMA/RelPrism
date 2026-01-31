# RelPrism: Multi-Faceted Self-Supervised Learning on Relational Databases

This repository contains the official implementation of **RelPrism: Multi-Faceted Self-Supervised Learning on Relational Databases**.

## Requirements
To install the required dependencies, run the following commands to create a conda environment:

```bash
conda env create -n relprism python=3.10
conda activate relprism

# Install core dependencies
pip install relbench[full]
pip install pyg-lib -f https://data.pyg.org/whl/torch-<torch_version>+<cuda_version>.html
# Note: Replace <torch_version> and <cuda_version> with your PyTorch and CUDA versions
# Example: torch-2.0.0+cu118

pip install pytorch-frame==0.2.3
pip install sentence-transformers==3.3.1
pip install scikit-learn==1.5.2
pip install torch-geometric
pip install numpy pandas tqdm pyyaml
```

## Data Preparation

We have prepared the processed data for pre-training and adaptation in the `data` folder.

### Using Pre-processed Data

The pre-processed data for `rel-f1` dataset is already included in the `data` folder. Simply run the code with the default settings.

### Processing New Datasets

If you wish to conduct experiments on other datasets, our data processing pipeline follows the approach described in [Griffin](https://github.com/yanxwb/Griffin/tree/processing_data). Please refer to their documentation for detailed data preparation steps.


## Usage

### Pre-Training and Adaptation

To perform pre-training followed by downstream task adaptation:

```bash
python main.py
```