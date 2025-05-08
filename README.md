# Roof Primitive Regression and Point Cloud Processing

This repository contains code for roof primitive regression and point cloud processing using deep learning techniques. The project is implemented in Python and PyTorch, with additional utilities for evaluation and visualization.

## Features

- **Point Cloud Processing**: Implements point cloud sampling, grouping, and feature extraction using neural networks.
- **Roof Primitive Regression**: Predicts geometric parameters of roof primitives from point cloud data.
- **Visualization**: Uses `pyvista` for 3D visualization of predictions and ground truth.
- **Evaluation Metrics**: Includes metrics like RMSE, reconstruction scores, and parameter errors for model evaluation.

## Repository Structure

- `model.py`: Contains the neural network models, including `Pct` and `Pct_reg`, for point cloud processing and regression.
- `eval_roof_primitive_reg.py`: Script for evaluating the model's performance, including visualization and metric computation.
- `util.py`: Utility functions for point cloud sampling and grouping.
- `data/`: Directory for storing input point cloud data.
- `results/`: Directory for saving evaluation results and visualizations.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- PyVista
- Matplotlib
- Pandas
- Scipy

Install the required packages using:

```bash
pip install -r requirements.txt
