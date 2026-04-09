### 1. Incompressible Flow: NACA 0012 Airfoil Surrogate

Computational Fluid Dynamics (CFD) is incredibly accurate, but it is painfully slow. Waiting upwards of 10 to 15 minutes for a single 2D RANS simulation to converge makes rapid design iteration, aerodynamic optimization, and real-time control system design computationally prohibitive. 

I built this project to explore how Machine Learning can augment traditional solvers. By engineering neural networks as **high-fidelity surrogate models**, we can train them on massive Navier-Stokes datasets and then deploy them for real-time inference.

This repository contains an end-to-end pipeline for training Physics-Informed Neural Networks (PINNs) to predict aerodynamic fields in a fraction of a second. Instead of just letting a standard neural network guess the answers based on data alone, I engineered custom loss functions that force the AI to obey fundamental momentum conservation laws and integration rules. 

The result is a suite of robust surrogate models that execute orders of magnitude faster than ANSYS Fluent, successfully predicting complex non-linear physics while resisting catastrophic failure during edge-case extrapolation.

---
## The Portfolio Projects
I tackled two distinct flow regimes to prove this concept. You can dive into the detailed engineering reports, mesh strategies, and failure analyses for each project below:

### 1. Incompressible Flow: NACA 0012 Airfoil
* **The Goal:** Predict Lift, Drag, and a 300-point spatial Pressure Coefficient ($C_p$) distribution across a sweeping Angle of Attack (AoA), including post-stall regimes.
* **The Result:** Achieved a > 2,400x end-to-end script speedup (with core inference taking < 0.05 seconds). Bounded global lift error to < 3.0% during unseen post-stall extrapolation, while identifying critical drag limitations in standard MLP architectures.
* **The Engineering Challenge:** Solving "Integration Drift." Standard neural networks failed to conserve momentum between the local pressure curves and the global forces. I solved this by engineering a custom Physics-Informed 1D-CNN (PINN-CNN) that predicts the spatial curves and mathematically integrates them in-graph (via non-uniform Trapezoidal rules) to accurately predict global Lift, while a parallel Direct MLP handles Drag.
* **Read the full report:** [Airfoil Surrogate Documentation](README_Airfoil.md)

### 2. Compressible Flow: 2D Axisymmetric CD Nozzle (Diagnostic Study)
* **The Goal:** Predict macroscopic thrust and the 1D centerline Mach distribution of a converging-diverging nozzle based solely on the Nozzle Pressure Ratio (NPR).
* **The Result:** Exposed the strict mathematical limitations of standard Deep Learning on compressible fluid discontinuities. Documented a highly technical failure analysis detailing the "Smoothing Effect" on normal shockwaves and the impact of data starvation.
* **The Engineering Challenge:** Capturing normal shockwaves. Standard neural networks trained with Mean Squared Error (MSE) completely fail to capture the Rankine-Hugoniot jump conditions of a shock, instead predicting unphysical oscillations. This project explains the need for Gradient-Aware Loss and Hard Momentum Integration to constrain the model and achieve a true compressible ROM.
* **Read the full report:** [Nozzle Diagnostic Analysis](README_Nozzle.md)

---

## Try the Demo (No ANSYS License Required)
To make it as easy as possible to see these speedups in action, I have included a consolidated inference script at the root of this repository so you can test the pre-trained models locally.

1. **Create and activate the Conda environment:**
```bash
conda env create -f environment.yml
conda activate cfdml
```
2. **Run the Airfoil Inference Demo for any Angle of Attack (e.g., 7.5 degrees):**
```bash
 python predict_airfoil_demo.py --aoa 7.5
```
*What happens next:*  The script will instantly load the saved .keras models, predict the spatial $C_p$ distribution, calculate Lift and Drag, and generate an engineering plot in the reports/demo_outputs/ folder—all within a few seconds.

## Repository Structure
This codebase is organized to clearly separate the CFD data pipeline from the Machine Learning research and training:
* `/scripts/` - The Data Pipeline. Contains the automated dataset generation (Fluent TUI scripts), CFD batch runners, and ETL preprocessing scripts.
* `/projects/` - The Machine Learning Hub. Contains all AI development, including intermediate architecture searches, baseline models, NASA wind-tunnel validation, and the **final PINN training loops**.
* `/models/` - Deployment Ready. The saved Neural Network weights (.keras) and statistical scalers (.pkl) ready for deployment.
* `/reports/` - Generated engineering plots, and the outputs from the demo script.