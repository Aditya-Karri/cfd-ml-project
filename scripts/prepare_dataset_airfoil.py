import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Configuration
PROJECT_ROOT = r"D:\cfd-ml-project"
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Airfoil_sweep")
OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "datasets", "airfoil_dataset.npz")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

NUM_POINTS = 150
beta = np.linspace(0, np.pi, NUM_POINTS)
x_grid = 0.5 * (1 - np.cos(beta))

# data containers
X_data = []      
y_cp_data = []   
y_cl_data = []   
y_cd_data = []   
case_ids = []    

all_items = glob.glob(os.path.join(RAW_DATA_DIR, '*'))
case_dirs = [d for d in all_items if os.path.isdir(d)]

print(f"Scanning {RAW_DATA_DIR}")
print(f"Found {len(case_dirs)} case directories.")

for case_dir in case_dirs:
    case_id = os.path.basename(case_dir)
    
    # parse aoa from folder name
    try:
        clean_name = case_id.lower().replace('minus', '-').replace('deg', '')
        aoa = float(clean_name)
        re_num = 6e6 
    except ValueError:
        print(f"Could not parse angle for {case_id}. Moving to next case.")
        continue

    # read force history
    hist_files = glob.glob(os.path.join(case_dir, "history_*.out"))
    if not hist_files:
        print(f"No history file found for {case_id}.")
        continue
        
    cl_history = []
    cd_history = []
    try:
        with open(hist_files[0], 'r') as f:
            lines = f.readlines()
            
        # scan bottom-up for the last 20 valid iterations
        for line in reversed(lines):
            parts = line.strip().split()
            if len(parts) >= 3:
                try:
                    cl_history.append(float(parts[1]))
                    cd_history.append(float(parts[2]))
                    if len(cl_history) >= 20:
                        break
                except ValueError:
                    continue
                    
        if not cl_history or not cd_history:
            print(f"No valid Cl/Cd data extracted for {case_id}.")
            continue
            
        # Average the last iterations
        cl = np.mean(cl_history)
        cd = np.mean(cd_history)
        
    except Exception as e:
        print(f"Failed to read force history for {case_id}: {e}")
        continue

    # read pressure distributions
    files_upper = glob.glob(os.path.join(case_dir, "cp_upper_*.csv"))
    files_lower = glob.glob(os.path.join(case_dir, "cp_lower_*.csv"))
    
    if not files_upper or not files_lower:
        print(f"Missing Cp files for {case_id}.")
        continue
        
    try:
        # load fluent exports, handle space delimiting
        df_u = pd.read_csv(files_upper[0], sep=r'\s+', on_bad_lines='skip')
        df_l = pd.read_csv(files_lower[0], sep=r'\s+', on_bad_lines='skip')
        
        df_u.columns = [str(c).strip() for c in df_u.columns]
        df_l.columns = [str(c).strip() for c in df_l.columns]
        
        # fallback if fluent bumped headers down
        if not any('x-coordinate' in c for c in df_u.columns):
            df_u = pd.read_csv(files_upper[0], sep=r'\s+', skiprows=1, on_bad_lines='skip')
            df_u.columns = [str(c).strip() for c in df_u.columns]
        if not any('x-coordinate' in c for c in df_l.columns):
            df_l = pd.read_csv(files_lower[0], sep=r'\s+', skiprows=1, on_bad_lines='skip')
            df_l.columns = [str(c).strip() for c in df_l.columns]

        x_col_u = [c for c in df_u.columns if 'x-coordinate' in c][0]
        cp_col_u = [c for c in df_u.columns if 'pressure-coefficient' in c][0]
        x_col_l = [c for c in df_l.columns if 'x-coordinate' in c][0]
        cp_col_l = [c for c in df_l.columns if 'pressure-coefficient' in c][0]

        # force numeric to drop text artifacts
        df_u[x_col_u] = pd.to_numeric(df_u[x_col_u], errors='coerce')
        df_u[cp_col_u] = pd.to_numeric(df_u[cp_col_u], errors='coerce')
        df_l[x_col_l] = pd.to_numeric(df_l[x_col_l], errors='coerce')
        df_l[cp_col_l] = pd.to_numeric(df_l[cp_col_l], errors='coerce')

        df_u = df_u.dropna(subset=[x_col_u, cp_col_u]).sort_values(x_col_u)
        df_l = df_l.dropna(subset=[x_col_l, cp_col_l]).sort_values(x_col_l)
        
        # interpolate onto uniform grid using linear extrapolation to prevent leading edge spikes
        interp_u = interp1d(df_u[x_col_u], df_u[cp_col_u], kind='linear', fill_value="extrapolate")
        interp_l = interp1d(df_l[x_col_l], df_l[cp_col_l], kind='linear', fill_value="extrapolate")
        
        cp_combined = np.concatenate([interp_u(x_grid), interp_l(x_grid)])
        
    except Exception as e:
        print(f"Cp processing failed for {case_id}: {e}")
        continue

    X_data.append([aoa, re_num])
    y_cp_data.append(cp_combined)
    y_cl_data.append(cl)
    y_cd_data.append(cd)
    case_ids.append(case_id)

# Save and Plot
if len(X_data) > 0:
    X = np.array(X_data)
    y_cp = np.array(y_cp_data)
    y_cl = np.array(y_cl_data)
    y_cd = np.array(y_cd_data)

    print(f"\nProcessing complete. Valid cases extracted: {len(X)}")
    np.savez(OUTPUT_FILE, X=X, y_cp=y_cp, y_cl=y_cl, y_cd=y_cd, x_grid=x_grid, case_ids=case_ids)
    print(f"Dataset successfully saved to {OUTPUT_FILE}")
    
    # Lift Curve plot
    plt.figure(figsize=(8,5))
    sort_idx = np.argsort(X[:, 0])
    plt.plot(X[sort_idx, 0], y_cl[sort_idx], 'o-', color='blue', label='CFD Data')
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel('Cl')
    plt.grid(True)
    plt.title(f'Lift Curve ({len(X)} cases)')
    
    lift_plot_path = os.path.join(PROJECT_ROOT, "reports", "dataset_lift_curve.png")
    plt.savefig(lift_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved lift curve plot: {lift_plot_path}")
    
    # Cp Distributions plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.turbo(np.linspace(0, 1, len(X))) 
    
    for i in sort_idx:
        aoa = X[i, 0]
        cp_curve = y_cp[i]
        color = colors[i]
        
        mid = len(cp_curve) // 2
        cp_upper = cp_curve[:mid]
        cp_lower = cp_curve[mid:]
        
        plt.plot(x_grid, cp_upper, color=color, linewidth=1.2, label=f'{aoa:.1f}°')
        plt.plot(x_grid, cp_lower, color=color, linewidth=1.2, linestyle='--')

    plt.gca().invert_yaxis()
    
    # adjust limits for clarity
    current_bottom, current_top = plt.ylim()
    plt.ylim(1.5, current_top)

    plt.xlabel('x/c', fontsize=12)
    plt.ylabel('$C_p$', fontsize=12)
    plt.title('Pressure Coefficient Distribution across AoA sweep', fontsize=14, pad=15)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title="AoA", fontsize='small', ncol=2)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    
    cp_plot_path = os.path.join(PROJECT_ROOT, "reports", "dataset_cp_distribution.png")
    plt.savefig(cp_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved Cp distribution plot: {cp_plot_path}")

else:
    print("\nNo valid cases were processed")