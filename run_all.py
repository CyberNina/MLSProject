import os
import subprocess
import time
import sys
from datetime import datetime

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# The file where all outputs will be saved (overwritten every time)
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline_report.txt")

# =============================================================================
# LOGGER CLASS: Intercepts output and writes it to both terminal and .txt file
# =============================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # "w" ensures the file is cleared and overwritten on each run
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # Print to terminal
        self.log.write(message)       # Write to txt file

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect standard output to our Logger
sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout

# =============================================================================
# PIPELINE DEFINITION
# =============================================================================
PIPELINE_SCRIPTS = [
    ("1. Data Loading and Windowing", os.path.join("src", "main.py")),
    ("2. Random Forest Training", os.path.join("src", "train_rf.py")),
    ("3. Deep Learning Training", os.path.join("src", "train_dl.py")),
    ("4. Real-Time Simulation", os.path.join("src", "predict.py")),
    ("5. Adversarial Robustness Test", os.path.join("attacks", "advers_attack.py")),
    ("6. Generating Presentation Plots", os.path.join("src", "visualize_results.py"))
]

def run_script(step_name, script_path):
    print("\n" + "-" * 70)
    print(f">> Executing: {step_name}")
    full_path = os.path.join(BASE_DIR, script_path)
    
    if not os.path.exists(full_path):
        print(f"\n[ERROR] Missing file: {full_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    # Run the script capturing output in real-time to send to Logger
    process = subprocess.Popen(
        [sys.executable, full_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Read line by line as the child script prints
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] Script '{script_path}' failed with an error.")
        sys.exit(1)
        
    print(f"\n[SUCCESS] Completed in {time.time() - start_time:.1f} seconds.")

def main():
    print("=" * 85)
    print(f" STARTING PIPELINE: CRYPTOJACKING DETECTION")
    print(f" Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" This output will be saved to: {LOG_FILE}")
    print("=" * 85)
    
    t0 = time.time()
    for step, path in PIPELINE_SCRIPTS:
        run_script(step, path)
        
    print("\n" + "=" * 85)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {time.time() - t0:.1f} SECONDS.")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()