# Accelerating Aerodynamics: Physics-Informed ML Surrogates for CFD
Computational Fluid Dynamics (CFD) is incredibly accurate, but it is painfully slow. Waiting upwards of 10 minutes for a single 2D simulation to converge makes rapid design iteration and real-time control system optimization almost impossible. I built this project to see if I could bypass that bottleneck.
This repository contains the end-to-end pipeline for training Physics-Informed Neural Networks (PINNs) to predict aerodynamic fields in milliseconds. Instead of just letting a standard neural network guess the answers based on data, I engineered custom loss functions that force the AI to obey fundamental momentum conservation laws. 
The result is a surrogate model that is up to 246,000x faster than ANSYS Fluent, with minimal loss in physical accuracy.

---

## The Portfolio Projects
I tackled two distinct flow regimes to prove this concept. You can dive into the detailed engineering reports, mesh strategies, and failure analyses for each project below:

### 1. Compressible Flow: 2D Axisymmetric CD Nozzle
* **The Goal:** Predict macroscopic thrust and the 1D centerline Mach distribution based solely on the Nozzle Pressure Ratio (NPR).
* **The Result:** Achieved a 30,907x speedup (inference drops from ~10 minutes to 0.018 seconds).
* **The Engineering Challenge:** Standard neural networks struggle violently with the sharp discontinuities of normal shockwaves. I documented a detailed failure analysis of this "Smoothing Effect" and successfully stabilized the predictions using a momentum-conservation physics loss constraint.
* **Read the full report:** [Nozzle Surrogate Documentation](README_Nozzle.md)

### 2. Incompressible Flow: NACA 0012 Airfoil
* **The Goal:** Predict Lift, Drag, and a 200-point spatial Pressure Coefficient (Cp) distribution across a sweeping Angle of Attack (AoA).
* **The Result:** Achieved a ~93,424x Faster speedup with less than 2.0% global force error.
* **The Engineering Challenge:** Decoupling macroscopic forces from microscopic fluid dynamics. I built a multi-task architecture where a Direct MLP handles the global forces, while an Integral Physics-Informed CNN predicts the spatial curves.
* **Read the full report:** [Airfoil Surrogate Documentation](README_Airfoil.md)

---

## Try the Demo (No ANSYS License Required)
I wanted to make it as easy as possible to see these speedups in action. I have included a consolidated inference script at the root of this repository so you can test the pre-trained models locally.

1. **Create and activate the Conda environment:**
```bash
 conda env create -f environment.yml
 conda activate cfdml
```
2. **Run the Airfoil Inference Demo for any Angle of Attack (e.g., 7.5 degrees):**
```bash
 python predict_airfoil_demo.py --aoa 7.5
```
*What happens next:* The script will instantly load the saved Keras models, predict the Cp distribution, calculate Lift and Drag, and generate an engineering plot in the reports/demo_outputs/ folder—all in under 0.05 seconds.

## Repository Structure
This codebase is organized to clearly separate the CFD data pipeline from the Machine Learning research and training:
* `/scripts/` - The Data Pipeline. Contains the automated dataset generation (Fluent TUI scripts), CFD batch runners, and ETL preprocessing scripts.
* `/experiments/` - The Machine Learning Hub. Contains all AI development, including intermediate architecture searches, baseline models, NASA wind-tunnel validation, and the **final PINN training loops**.
* `/models/` - The saved Neural Network weights (.keras) and statistical scalers (.pkl) ready for deployment.
* `/reports/` - Generated engineering plots, CFD timing logs, and the outputs from the demo script.