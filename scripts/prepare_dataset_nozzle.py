import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Nozzle_sweep")
CASES_CSV = os.path.join(PROJECT_ROOT, "cases_to_run_nozzle.csv")

SAVE_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "nozzle_dataset.npz")
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

# Define the mathematical nozzle domain (x = -0.5 m to x = 0.5 m)
NUM_POINTS = 400
x_uniform = np.linspace(-0.5, 0.5, NUM_POINTS)

# Master Lists for 1D Data
X_npr = []
Y_thrust = []
Y_mach = []
Y_wall_p = []

# Master Lists for 2D Field Data
field_x = None
field_y = None
Field_P = []
Field_Mach = []

print("Starting data extraction and interpolation")

# Read the run matrix
if not os.path.exists(CASES_CSV):
    print(f"Error: Run matrix file not found at {CASES_CSV}")
    exit()

cases_df = pd.read_csv(CASES_CSV)

for index, row in cases_df.iterrows():
    case_id = row['case_id']
    npr = row['npr']
    case_path = os.path.join(DATA_DIR, case_id)
    
    #  Extract Converged Thrust
    thrust_hist_file = os.path.join(case_path, f"thrust_history_{case_id}.out")
    try:
        hist_data = pd.read_csv(thrust_hist_file, sep='\s+', skiprows=3, names=['iter', 'thrust'], on_bad_lines='skip')
        hist_data['thrust'] = pd.to_numeric(hist_data['thrust'], errors='coerce')
        hist_data = hist_data.dropna()
        final_thrust = abs(hist_data['thrust'].tail(50).mean())
    except Exception as e:
        print(f"Warning: Failed to read thrust data for {case_id} | Error: {e}")
        continue

    # Extract & Interpolate Centerline Mach
    mach_file = os.path.join(case_path, f"axis_mach_{case_id}.csv")
    try:
        mach_df = pd.read_csv(mach_file, header=0)
        mach_df.columns = mach_df.columns.str.strip().str.replace('"', '')
        
        x_col = [c for c in mach_df.columns if 'x-coordinate' in c][0]
        m_col = [c for c in mach_df.columns if 'mach' in c][0]
        
        mach_df = mach_df.sort_values(by=x_col) 
        f_mach = interp1d(mach_df[x_col], mach_df[m_col], kind='linear', fill_value="extrapolate")
        mach_uniform = f_mach(x_uniform)
    except Exception as e:
        print(f"Warning: Failed to read centerline Mach data for {case_id} | Error: {e}")
        continue
        
    # Extract & Interpolate Wall Pressure
    wall_p_file = os.path.join(case_path, f"wall_p_{case_id}.csv")
    try:
        wall_df = pd.read_csv(wall_p_file, header=0)
        wall_df.columns = wall_df.columns.str.strip().str.replace('"', '')
        
        x_col = [c for c in wall_df.columns if 'x-coordinate' in c][0]
        p_col = [c for c in wall_df.columns if 'pressure' in c][0]
        
        wall_df = wall_df.sort_values(by=x_col)
        f_press = interp1d(wall_df[x_col], wall_df[p_col], kind='linear', fill_value="extrapolate")
        press_uniform = f_press(x_uniform)
    except Exception as e:
        print(f"Warning: Failed to read wall pressure for {case_id} | Error: {e}")
        continue

    # Extract 2D Field Data (Pressure & Mach)
    field_file = os.path.join(case_path, f"field_data_{case_id}.csv")
    try:
        field_df = pd.read_csv(field_file, header=0)
        field_df.columns = field_df.columns.str.strip().str.replace('"', '')
        
        fx_col = [c for c in field_df.columns if 'x-coordinate' in c][0]
        fy_col = [c for c in field_df.columns if 'y-coordinate' in c][0]
        fp_col = [c for c in field_df.columns if 'pressure' in c][0]
        fm_col = [c for c in field_df.columns if 'mach' in c][0]
        
        # Sort by X then Y to guarantee every case has the exact same node ordering
        field_df = field_df.sort_values(by=[fx_col, fy_col])
        
        # Save the mesh coordinates only once (from the first successful case)
        if field_x is None:
            field_x = field_df[fx_col].values
            field_y = field_df[fy_col].values
            
        Field_P.append(field_df[fp_col].values)
        Field_Mach.append(field_df[fm_col].values)
    except Exception as e:
        print(f"Warning: Failed to read 2D field data for {case_id} | Error: {e}")
        continue

    # Append valid case data to master lists
    X_npr.append(npr)
    Y_thrust.append(final_thrust)
    Y_mach.append(mach_uniform)
    Y_wall_p.append(press_uniform)

# Convert to structured Numpy Arrays
X_npr = np.array(X_npr).reshape(-1, 1)  
Y_thrust = np.array(Y_thrust).reshape(-1, 1) 
Y_mach = np.array(Y_mach)               
Y_wall_p = np.array(Y_wall_p)           
Field_P = np.array(Field_P)
Field_Mach = np.array(Field_Mach)

print("\n Data Extraction Summary")
print(f"Successfully processed {len(X_npr)} cases.")
print(f"  X_npr shape:       {X_npr.shape}")
print(f"  Y_thrust shape:    {Y_thrust.shape}")
print(f"  1D Mach shape:     {Y_mach.shape}")
print(f"  1D Wall P shape:   {Y_wall_p.shape}")
print(f"  2D Field Mesh:     {field_x.shape[0]} nodes extracted.")
print(f"  2D Field P shape:  {Field_P.shape}")
print(f"  2D Field M shape:  {Field_Mach.shape}")

# Save dataset
np.savez(
    SAVE_PATH, 
    npr=X_npr, 
    thrust=Y_thrust, 
    mach=Y_mach, 
    wall_p=Y_wall_p, 
    x_grid=x_uniform,
    field_x=field_x,
    field_y=field_y,
    field_p=Field_P,
    field_mach=Field_Mach
)
print(f"Dataset successfully saved to: {SAVE_PATH}")

# Plot check
plt.figure(figsize=(12, 5))

# Plot a diverse sampling of NPRs to verify physical trends
indices_to_plot = np.linspace(0, len(X_npr)-1, 5, dtype=int)
for idx in indices_to_plot:
    plt.plot(x_uniform, Y_mach[idx], label=f"NPR = {X_npr[idx][0]:.1f}")

plt.title("Centerline Mach Number Distribution (Interpolated)")
plt.xlabel("Axial Distance (m)")
plt.ylabel("Mach Number")
plt.axvline(x=0.0, color='k', linestyle='--', label='Throat (x=0.0 m)')
plt.xlim([-0.5, 0.5])
plt.legend()
plt.grid(True, alpha=0.5)
plt.tight_layout()
plt.show()