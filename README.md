# FairU-GSL

Code for "FairU-GSL: Fairness-Aware Graph Structure Learning under Uncertainty".

## Overview

- `train.py`: the main training entry for GSL models (GRCN, IDGL, PROGNN, PROSE, SLAPS, SUBLIME).
- `src/`: core implementation modules.
  - `FairUnGSLmodule.py`: the FairU-GSL plug-in, implementing the cross-group threshold offset, stochastic floor sampling, and the fairness-aware modulation `phi`.
  - `fairness_utils.py`: dataset loading, sensitive-attribute handling, and ΔSP / ΔEO metrics.
  - `fairness_loss.py`: differentiable Statistical Parity and Equal Opportunity surrogates.
  - `adversarial.py`: adversarial discriminator and the associated loss.
  - `fair_contrastive.py`: a contrastive fairness head used by the contrastive models.
  - `measure_utils.py`: entropy estimation and edge-compression helpers.
  - `gcn.py`: a vanilla GCN classifier head shared across models.
- `models/`: the six GSL models (GRCN, IDGL, PROGNN, PROSE, SLAPS, SUBLIME) used as plug-in points for FairU-GSL.
- `dataset/`: eight node-classification fairness benchmarks bundled with the repository.

## Installation

**Note:** FairU-GSL depends on [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse) and [DEEP GRAPH LIBRARY (DGL)](https://www.dgl.ai/pages/start.html).  To streamline the installation, FairU-GSL does **NOT** install these libraries for you. Please install it by yourself for running FairU-GSL:

- torch>=1.12.0
- torch_geometric>=2.1.0
- torch_sparse>=0.6.12
- dgl>=0.9.0

Then install the remaining dependencies:

```
pip install -r requirements.txt
```

## Datasets

All eight fairness benchmarks are bundled under `dataset/`: NBA, BailA, CreditA, GermanA, Pokec_n, Pokec_z, Syn-1, and Syn-2.

## Usage

You can train FairU-GSL by following the 3 steps below.

### Step 1: Choose a backbone and a dataset

**models**:

- `GRCN`, `IDGL`, `PROGNN`, `PROSE`, `SLAPS`, `SUBLIME` — run through `train.py --backbone <NAME>`.

**Datasets**: `nba`, `bailA`, `creditA`, `germanA`, `pokec_n`, `pokec_z`, `syn1`, `syn2`.

### Step 2: Run training

For the models, e.g. GRCN on BailA:

```
python train.py --backbone GRCN --dataset bailA
```


### Step 3: Read the results

Each run prints per-epoch metrics (ACC / AUC / F1 / ΔSP / ΔEO) to stdout and writes a per-run JSON summary under `results/`.
