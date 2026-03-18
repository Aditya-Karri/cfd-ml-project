import csv
import numpy as np
import os

# Define path for the new Nozzle CSV
file_path = r"D:\cfd-ml-project\cases_to_run_nozzle.csv"

# Define NPR ranges using adaptive sampling
# Define range: NPR 2 to 18
npr_values = np.concatenate([
    np.arange(2.0, 6.0, 0.5),   # 2.0 to 5.5: Over-expanded (highly dynamic internal shocks, 0.5 steps)
    np.arange(6.0, 12.0, 1.0),  # 6.0 to 11.0: Moving towards perfectly expanded (1.0 steps)
    np.arange(12.0, 18.1, 2.0)  # 12.0 to 18.0: Under-expanded (internal flow is locked/stable, 2.0 steps)
])

# Round to 1 decimal place to prevent floating-point errors (e.g., 3.099999)
npr_values = np.round(npr_values, 1)

# Remove any accidental duplicates and sort
npr_values = np.unique(npr_values)

print(f"Generating {len(npr_values)} intelligently spaced Nozzle CFD cases...")

with open(file_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["case_id", "npr"])  # Header
    
    for npr in npr_values:
        # Create a clean ID name (e.g., "npr_3.1")
        case_id = f"npr_{npr:.1f}"
        writer.writerow([case_id, f"{npr:.1f}"])

print(f"'cases_to_run_nozzle.csv' successfully generated at {file_path}")