# TERRA-GAN: Human-Guided Deep Learning for Bare Earth Model Generation

## Overview

TERRA-GAN is a deep learning framework that automatically generates bare earth models from Digital Surface Models (DSMs) by removing man-made structures and occlusions. (Important note: The implemented version uses image processing to generate masks, although in practice you would use a segmentation model like [Segment Anything](https://github.com/facebookresearch/segment-anything). The system uses Generative Adversarial Networks (GANs) with partial convolutions to create realistic terrain in masked regions.

## Key Innovations

1. **Boundary-Aware Loss**: Improves transitions between original and AI-generated terrain
2. **Human-Guided Fine-Tuning**: Web-based annotation portal allows human feedback to refine results
3. **Spatially-Aware Data Handling**: Prevents data leakage between adjacent terrain tiles

## Requirements

### Software
- Python 3.10+
- PyTorch 2.0+ (tested with 2.5.1)
- MLflow 2.8.0+
- CUDA Toolkit (for GPU acceleration)
- Additional packages in `requirements.txt`

### Hardware
- NVIDIA GPU with 16GB+ VRAM (developed on RTX 4070Ti)
- 32GB+ RAM (64GB recommended)
- 100GB+ storage space

## Quick Start

```bash
# Clone repository
git clone https://github.com/FKGSOFTWARE/TERRA-GAN.git
cd TERRA-GAN

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Data Setup

The data used for this project was sourced from [Ordinance Survey - Aerial Digimap](https://digimap.edina.ac.uk/). It is designed to use their data, where we download data through their portal. This is done by selecting download data > inputing the grid reference, e.g. "NJ05" > defining the data to download ("Aerial Imagery": "High Resolution 25cm", "Height and Terrain": "Digital Surface Model (2m)") > downloading and placing the zip folders in the input directory.

1. Place raw data zip files in designated directories:
   - Training data: `./data/raw_data/experiment_training_input/`
   - Evaluation data: `./data/raw_data/experiment_human_eval_input/`

2. Ensure the baseline model is in `./_BASELINE_MODEL/BASELINE_MODEL.pth` (if using pre-trained weights)

## Running Experiments

### Start MLflow UI (Optional - if running experiment scripts, this should be automatic)
```bash
./start_mlflow.sh [PORT]  # Default port is 5000
```

### Standard Experiment (Baseline & Human-Guided)
```bash
./run_experiment.sh
```
You will be prompted for an experiment name (e.g., `EXPERIMENT_00_BASELINE`) and to confirm human annotation steps.

### Ablation Study (No Human Guidance)
```bash
./ablation_experiment.sh
```

### Boundary-Aware Loss Experiment
1. Set `training.loss_weights.boundary` to 0.5 in `config.yaml`, needs to be > 0 to be included.
2. Run `./run_experiment.sh`
3. Name it `EXPERIMENT_04_ADDITION_BOUNDARY-AWARE-LOSS`

## Configuration

Key settings in `config.yaml`:
- `training`: Epochs, batch sizes, learning rates, loss weights
- `evaluation`: Checkpoint paths, metric thresholds
- `mask_processing`: Parameters for feature detectors
- `data`: Paths for raw, processed, and output data

## Evaluation

```bash
# Calculate terrain metrics
python evaluate_terrain.py \
    --original-masks path/to/original_masks \
    --final-annotations path/to/human_annotations \
    --output-file path/to/metrics.json

# Compare experiments statistically
python result_metrics_statistical_significance.py \
    --experiments path/to/exp1_metrics.json path/to/exp2_metrics.json \
    --output path/to/stats_comparison.json
```

## Code Structure

- `main_pipeline.py`: Orchestrates the pipeline modes
- `mvp_gan/`: Core GAN implementation
  - `src/models/`: Generator and Discriminator models
  - `src/training/`: Training loops
  - `src/evaluation/`: Metrics and evaluation
- `utils/`: Helper modules
  - `api/`: Portal client for human feedback
  - `data_splitting.py`: Spatially-aware data splitting
  - `experiment_tracking.py`: MLflow integration
  - `visualization/`: DSM colorization
- Evaluation scripts: `evaluate_terrain.py`, `plot_*.py`
