# Roof Primitive Regression and Point Cloud Processing

This repository contains code for roof primitive regression and point cloud processing using deep learning techniques. The project is implemented in Python and PyTorch, with additional utilities for evaluation and visualization.

## Roof Primitives

Design of the roof primitives, including the parameters and visualizations.
![alt text](https://github.com/Zhixinli2887/Building-Reconstruction/edit/main/image/primitives.png)

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

## Reconstruction Results


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
```

## Citations
```bash
@ARTICLE{Li2019-iz,
  title   = "{GEOMETRIC} {OBJECT} {BASED} {BUILDING} {RECONSTRUCTION} {FROM}
             {SATELLITE} {IMAGERY} {DERIVED} {POINT} {CLOUDS}",
  author  = "Li, Zhixin and Xu, Bo and Shan, Jie",
  journal = "International Archives of the Photogrammetry, Remote Sensing \&
             Spatial Information Sciences",
  year    =  2019
}

@ARTICLE{Zhang2021-as,
  title     = "Optimal model fitting for building reconstruction from point
               clouds",
  author    = "Zhang, Wenyuan and Li, Zhixin and Shan, Jie",
  journal   = "IEEE J. Sel. Top. Appl. Earth Obs. Remote Sens.",
  publisher = "Institute of Electrical and Electronics Engineers (IEEE)",
  volume    =  14,
  pages     = "9636--9650",
  year      =  2021
}

@ARTICLE{Li2022-zb,
  title     = "{RANSAC}-based multi primitive building reconstruction from {3D}
               point clouds",
  author    = "Li, Zhixin and Shan, Jie",
  journal   = "ISPRS J. Photogramm. Remote Sens.",
  publisher = "Elsevier BV",
  month     =  jan,
  year      =  2022,
  language  = "en"
}
```
