# Distilling Context into Parameters for Chronos-2

This project proposes adapting the Doc-to-LoRA framework to the Chronos-2 time-series foundation model to distill long historical traffic contexts into lightweight LoRA adapters. By using a hypernetwork to generate these parameters in a single forward pass, the goal is to significantly reduce inference latency and memory requirements for transportation forecasting without sacrificing predictive performance.

**Implementation Plan**
See [Implementation Plan](IMPLEMENTATION_PLAN.md) for implementation details.

## How to run

**1. Setup environment**:
```bash
uv sync
```

**2. Download data**:
- Download [PEMS-BAY dataset](https://mega.nz/file/dN5VQaob#m9E9WQbgtwYFIWveEmFQPI8I9Z_spBJkZW7LT2GGuGE)
- Unzip
- Place `pems-bay.h5` in root folder `data`

**3. Run pipline** with default parameters:
```bash
uv run src/main.py dataset_cfg=dataset/PEMS-BAY
```