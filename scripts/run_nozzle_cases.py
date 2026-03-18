import os
import subprocess
import shutil
import csv
import sys
import time

# Configuration and Paths
FLUENT_CMD = r"D:\ANSYS Inc\ANSYS Student\v252\fluent\ntbin\win64\fluent.exe"
PROJECT_ROOT = r"D:\cfd-ml-project"

TEMPLATE_JOU = os.path.join(PROJECT_ROOT, "journals", "run_nozzle_template.jou")
CASES_CSV = os.path.join(PROJECT_ROOT, "cases_to_run_nozzle.csv")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Nozzle_sweep")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "nozzle_logs")

# Ensure necessary directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def run_case(case_data):
    """Executes a single Fluent simulation based on the provided NPR parameters."""
    case_id = case_data['case_id']
    npr = float(case_data['npr'])
    
    # Calculate boundary condition pressures (assuming 100,000 Pa back pressure)
    total_pressure = npr * 100000.0
    initial_gauge = total_pressure * 0.98 
    
    print(f"\nStarting {case_id} (NPR={npr}, P0={total_pressure} Pa, P_initial={initial_gauge} Pa)")
    
    # Prepare the Journal File
    if not os.path.exists(TEMPLATE_JOU):
        print(f"Error: Template missing at {TEMPLATE_JOU}")
        return False, 0.0
        
    with open(TEMPLATE_JOU, 'r') as file:
        template = file.read()
    
    # Replace case-specific parameters into the template
    jou_content = template.replace("<TOTAL_PRESSURE>", f"{total_pressure:.1f}") \
                          .replace("<INITIAL_GAUGE>", f"{initial_gauge:.1f}") \
                          .replace("<OUTPUT_PREFIX>", case_id)
    
    run_jou_path = os.path.join(LOG_DIR, f"run_{case_id}.jou")
    with open(run_jou_path, 'w') as file:
        file.write(jou_content)
    
    log_path = os.path.join(LOG_DIR, f"log_{case_id}.txt")
    cmd = f'"{FLUENT_CMD}" 2ddp -g -t4 -i "{run_jou_path}"'
    
    print(f"Launching Fluent... Logging to: {log_path}")
    
    case_start = time.time()
    
    with open(log_path, "w") as log_handle:
        try:
            subprocess.run(cmd, shell=True, stdout=log_handle, stderr=subprocess.STDOUT, timeout=7200)
        except subprocess.TimeoutExpired:
            print("Process timed out after 7200 seconds.")
            return False, 0.0

    case_end = time.time()
    case_duration = case_end - case_start
    print(f"Solve Time: {case_duration:.2f} seconds ({case_duration/60:.2f} minutes)")

    # check convergence
    with open(log_path, "r") as file:
        if "solution is converged" in file.read():
            print("Solution converged successfully.")
        else:
            print("Maximum iterations reached or convergence failed. Check log.")

    case_dir = os.path.join(OUTPUT_DIR, case_id)
    os.makedirs(case_dir, exist_ok=True)
    moved_count = 0

    # Handle the Fluent default history file by renaming it
    raw_history_file = os.path.join(PROJECT_ROOT, "thrust-rfile.out")
    clean_history_file = os.path.join(case_dir, f"thrust_history_{case_id}.out")

    if os.path.exists(raw_history_file):
        shutil.move(raw_history_file, clean_history_file)
        moved_count += 1
    else:
        print("Missing file: thrust-rfile.out (History not saved)")

    # move remaining files
    files_to_move = [
        f"wall_p_{case_id}.csv", 
        f"axis_mach_{case_id}.csv",
        f"field_data_{case_id}.csv",
        f"thrust_{case_id}.txt",
        f"nozzle_{case_id}.cas.h5",
        f"nozzle_{case_id}.dat.h5"
    ]
    
    for fname in files_to_move:
        src = os.path.join(PROJECT_ROOT, fname)
        dst = os.path.join(case_dir, fname)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
        else:
            if not os.path.exists(dst):
                print(f"Expected output file not found: {fname}")

    if moved_count > 0:
        print(f"Successfully moved {moved_count} files to {case_dir}/")
        return True, case_duration
    else:
        print("No output files were generated.")
        return False, 0.0


if __name__ == "__main__":
    if not os.path.exists(CASES_CSV):
         print(f"Error: Run matrix file not found: {CASES_CSV}")
         sys.exit(1)

    print(f"Nozzle Batch Run Initiated. Output directory: {OUTPUT_DIR}")
    
    successful_runs = 0
    total_cfd_time = 0.0
    case_timings = []

    # Read run matrix and execute cases
    with open(CASES_CSV, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            success, duration = run_case(row)
            if success:
                successful_runs += 1
                total_cfd_time += duration
                case_timings.append({
                    'case_id': row['case_id'], 
                    'npr': row['npr'], 
                    'time': duration
                })

    print("Nozzle batch sweep complete")
    
    # Save CFD runtime report
    if successful_runs > 0:
        avg_time = total_cfd_time / successful_runs
        timing_file_path = os.path.join(PROJECT_ROOT, "nozzle_cfd_run_time.txt")
        
        with open(timing_file_path, "w") as f:
            f.write("CFD BATCH EXECUTION TIMING REPORT\n")
            
            # Print Individual Case Timings
            f.write("INDIVIDUAL CASE TIMINGS:\n")
            f.write(f"{'Case ID':<15} | {'NPR':<10} | {'Time (s)':<10}\n")
            for ct in case_timings:
                f.write(f"{ct['case_id']:<15} | {ct['npr']:<10} | {ct['time']:.2f}\n")
            
            # Print Summary
            f.write("SUMMARY:\n")
            f.write(f"Total Successful Cases Run : {successful_runs}\n")
            f.write(f"Total Compute Time (s)     : {total_cfd_time:.2f}\n")
            f.write(f"Average CFD Time/Case (s)  : {avg_time:.2f}\n")
            
        print(f"\nBatch complete! Average CFD time per case: {avg_time:.2f} seconds.")
        print(f"Timing report saved to: {timing_file_path}")
    else:
        print("\n Cases not completed. Timing report not generated.")