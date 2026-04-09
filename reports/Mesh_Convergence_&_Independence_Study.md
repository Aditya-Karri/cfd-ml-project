# Mesh Convergence & Independence Study

## 1. Objective
To perform a grid-convergence study on a 2D NACA 0012 airfoil using Ansys Fluent to ensure spatial discretization errors are minimized and the chosen mesh is computationally efficient.

## 2. Geometry & Solver Setup
* **Computational Domain:** A C-type fluid domain was constructed, the far-field boundaries were extended to 20m x 20m (representing 20 chord lengths). The geometry features a face split at 0.25m and 1m (the length of the airfoil) to allow for edge refinements.
* **Mesh Topology:** A fully structured, face-mapped Quad4 mesh was chosen to perfectly align elements with the streamwise flow and minimize numerical diffusion achieving accurate y-plus targeting without unnecessary cell count inflation.
* **Solver Settings:** * Pressure-Based Coupled Solver (Pseudo-Transient)
  * Turbulence Model: SST k-omega
  * Freestream Conditions: Velocity magnitude = 87.64 m/s (components determined by AoA).
  * Turbulence Intensity = 0.052%, Viscosity Ratio = 0.009 (matched to NASA validation data).

The case was tested at a high Angle of Attack (12°) at a Reynolds Number of 6e6 to stress-test the boundary layer under an adverse pressure gradient. 

## 3. Mesh Refinement Hierarchy
Three structured meshes were generated using a constant Grid Refinement Factor of approximately 1.41 (doubling the total 2D cell count at each step). The first cell height was mathematically locked across all three grids via bias factor adjustments to maintain y-plus < 1.

| Metric                    | Coarse Mesh           | Medium Mesh           | Fine Mesh
| :---                      | :---                  | :---                  | :---
| **Total Cells**           | 97,500                | 192,500               | 390,000
| **Edge 1 (Wall Normal)**  | 125 div (Bias: 1.3e6) | 175 div (Bias: 9.0e5) | 250 div (Bias: 6.0 e5)
| **Edge 2 (Wake)**         | 60 divisions          | 85 divisions          | 120 divisions
| **Edge 3 (Farfield)**     | 250 divisions         | 350 divisions         | 500 divisions
| **Edge 4 (Airfoil)**      | 80 divisions          | 115 divisions         | 160 divisions

## 4. Results Table
| Mesh Level | Cell Count | Cl	   | Cd       | % delta Cd
| :---       | :---   | :---       | :---     | :---
| **Coarse** | 97,500	  | 1.2419 | 0.017259 | Medium to Coarse: 2.076%
| **Medium** | 192,500	  | 1.2444 | 0.016908 | Fine to Medium: 0.517%
| **Fine**   | 390,000	  | 1.2447 | 0.016821 |

## 5. Grid Convergence Index (GCI) Analysis
To formally quantify the discretization error, Roache's GCI method was applied to the drag coefficient (C_d).
* **Average Refinement Ratio (r):** 1.41
* **Apparent Order of Accuracy (p):** 4.03
* **Fine Grid GCI (GCI_12):** 0.21%
* **Medium Grid GCI (GCI_23):** 0.85%

## 6. Mesh Type Chosen and Why
**Decision:** The **Medium Mesh (192,500 cells)** was selected for the final dataset generation.
**Why:** The results demonstrate monotonic convergence. The difference in aerodynamic forces between the Medium and Fine meshes is negligible (0.517% for drag). Medium Mesh has a GCI of 0.85%. Because the discretization uncertainty is less than 1%, the Medium mesh provides grid-independent accuracy while saving approximately 50% in computational cost compared to the Fine mesh.

## 7. Convergence vs. Correctness
* **Convergence:** The Coarse mesh achieved strict residual convergence (< 1e-06). For the Medium and Fine meshes, the continuity residuals leveled off, but the actual quantities of interest (Cl and Cd) were completely frozen to the 5th decimal place over dozens of iterations, and momentum/turbulence residuals dropped into the 1e-08 to 1e-09 range. The simulations were manually halted as macro-physical convergence was definitively achieved.
* **Correctness:** While the math is converged, correctness depends on whether our physics models match reality. Since the SST k-omega model is an industry-standard assumption for attached and mildly separated flow, we can confidently assume these results are structurally "correct" for this specific operational envelope.