import os
import subprocess
import math
import shutil
import csv
import sys
import time

# CONFIGURATION
FLUENT_CMD = r"D:\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe"
PROJECT_ROOT = r"D:\cfd-ml-project"

# Paths
TEMPLATE_JOU = os.path.join(PROJECT_ROOT, "journals", "run_airfoil_template.jou")
CASES_CSV = os.path.join(PROJECT_ROOT, "cases_to_run.csv")

# OUTPUT DIRECTORIES
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Airfoil_sweep")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "Airfoil_logs")

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def calc_vectors(aoa_degrees):
    rad = math.radians(float(aoa_degrees))
    return math.cos(rad), math.sin(rad), - math.sin(rad)

def run_case(case_data):
    case_id = case_data["case_id"]
    aoa = case_data["aoa"]

    # prepare journal
    vx, vy, vy_neg = calc_vectors(aoa)

    if not os.path.exists(TEMPLATE_JOU):
        print(f"Error: template {TEMPLATE_JOU} is missing")
        return False, 0.0

    with open(TEMPLATE_JOU, "r") as f:
        template = f.read()

    jou_content = template.replace("<VX>", f"{vx:.6f}")\
                            .replace("<VY>", f"{vy:.6f}")\
                            .replace("<VY_NEG>", f"{vy_neg:.6f}")\
                            .replace("<OUTPUT_PREFIX>", case_id)
    
    run_jou_path = os.path.join(LOG_DIR, f"run_{case_id}.jou")
    with open (run_jou_path, "w") as f:
        f.write(jou_content)
    
    log_path = os.path.join(LOG_DIR, f"log_{case_id}.txt")
    
    cmd = f'"{FLUENT_CMD}" 2ddp -wait -g -i "{run_jou_path}"'

    print(f" Launching Fluent... Logs: {log_path}")
    start_time = time.time()
    
    with open(log_path, "w") as log_handle:
        try:
            subprocess.run(cmd, shell=True, stdout=log_handle, stderr=subprocess.STDOUT, timeout=72000)
        except subprocess.TimeoutExpired:
            print(" Timeout Expired")
            return False, 0.0
            
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"solve time: {elapsed_time:.1f} seconds")

    junk_files = ["cl-rfile.out", "cd-rfile.out", "cl-rfile", "cd-rfile"]
    print(f"cleaning up junk files for {case_id}")
    for junk in junk_files:
        junk_path = os.path.join(PROJECT_ROOT, junk) 
        
        if os.path.exists(junk_path):
            try:
                os.remove(junk_path)
            except PermissionError:
                print(f" could not delete {junk} (file is locked by windows)")
        
    # check convergence
    with open(log_path, "r") as f:
        log_content = f.read().lower()
        # Check for either the standard message OR our custom monitor name
        if "solution is converged" in log_content or "cl-converge" in log_content:
            print(" Converged (Forces Stabilized)")
        else:
            print(" Reached max iterations (Did not fully stabilize)")
    
    # move files
    case_dir = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)

    files_to_move = [
        f"cp_upper_{case_id}.csv",
        f"cp_lower_{case_id}.csv",
        f"history_{case_id}.out",
        f"naca0012_{case_id}.cas.h5",
        f"naca0012_{case_id}.dat.h5"
    ]

    moved_count = 0
    for fname in files_to_move:
        src = os.path.join(PROJECT_ROOT, fname)
        dst = os.path.join(case_dir, fname)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
        else:
            if not os.path.exists(dst):
                print(f" missing files: {fname}")
    if moved_count > 0:
        print(f" Moved {moved_count} files to {case_dir}/")
        return True, elapsed_time
    else:
        print(" No output files found.")
        return False, 0.0
if __name__ == "__main__":
    if not os.path.exists(CASES_CSV):
        print(f" Error: {CASES_CSV} not found")
        sys.exit(1)
    
    print(f" Batch Run Started. Outputs will go to: {OUTPUT_DIR}")
    
    total_cfd_time = 0.0
    successful_runs = 0
    case_timings = []  # New list to hold individual case times
    
    with open(CASES_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            success, case_time = run_case(row)
            if success:
                total_cfd_time += case_time
                successful_runs += 1
                # Save the individual case data
                case_timings.append({
                    "case_id": row['case_id'],
                    "aoa": row['aoa'],
                    "time": case_time
                    })
    # save CFD runtime report
    if successful_runs > 0:
        avg_time = total_cfd_time / successful_runs
        timing_file_path = os.path.join(PROJECT_ROOT, "airfoil_cfd_run_time.txt")
        
        with open(timing_file_path, "w") as f:
            f.write("CFD BATCH EXECUTION TIMING REPORT\n")
            
            # Print Individual Case Timings
            f.write("INDIVIDUAL CASE TIMINGS:\n")
            f.write(f"{'Case ID':<15} | {'AoA (deg)':<10} | {'Time (s)':<10}\n")

            for ct in case_timings:
                f.write(f"{ct['case_id']:<15} | {ct['aoa']:<10} | {ct['time']:.2f}\n")
            
            f.write("SUMMARY:\n")
            f.write(f"Total Successful Cases Run : {successful_runs}\n")
            f.write(f"Total Compute Time (s)     : {total_cfd_time:.2f}\n")
            f.write(f"Average CFD Time/Case (s)  : {avg_time:.2f}\n")
            
        print(f"\n Batch complete! Average CFD time per case: {avg_time:.2f} seconds")
        print(f" Timing report saved to: {timing_file_path}")
    else:
        print("\n No cases completed successfully. Timing report not generated")