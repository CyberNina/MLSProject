import os
import subprocess
import time
import sys
from datetime import datetime

# --- CONFIGURAZIONE PATH ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Il file dove verranno salvati tutti gli output (sovrascritto ogni volta)
LOG_FILE = os.path.join(RESULTS_DIR, "pipeline_report.txt")

# =============================================================================
# CLASSE LOGGER: Intercetta l'output e lo scrive sia a video che nel file .txt
# =============================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        # La "w" garantisce che il file venga svuotato e sovrascritto ad ogni avvio
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # Stampa sul terminale
        self.log.write(message)       # Scrive nel file txt

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Reindirizziamo l'output standard verso il nostro Logger
sys.stdout = Logger(LOG_FILE)
sys.stderr = sys.stdout

# =============================================================================
# DEFINIZIONE DELLA PIPELINE
# =============================================================================
PIPELINE_SCRIPTS = [
    ("1. Caricamento e Windowing", os.path.join("src", "main.py")),
    ("2. Addestramento Random Forest", os.path.join("src", "train_rf.py")),
    ("3. Addestramento Deep Learning", os.path.join("src", "train_dl.py")),
    ("4. Simulazione Real-Time", os.path.join("src", "predict.py")),
    ("5. Test di Robustezza Avversaria", os.path.join("attacks", "advers_attack.py")),
    ("6. Generazione Grafici PowerPoint", os.path.join("src", "visualize_results.py"))
]

def run_script(step_name, script_path):
    print("\n" + "»" * 70)
    print(f">> Esecuzione: {step_name}")
    full_path = os.path.join(BASE_DIR, script_path)
    
    if not os.path.exists(full_path):
        print(f"\n[ERRORE] File mancante: {full_path}")
        sys.exit(1)
        
    start_time = time.time()
    
    # Esegue lo script catturando l'output in tempo reale per mandarlo al Logger
    process = subprocess.Popen(
        [sys.executable, full_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Legge riga per riga man mano che lo script figlio stampa
    for line in process.stdout:
        print(line, end="")
        
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERRORE] Lo script '{script_path}' ha generato un errore.")
        sys.exit(1)
        
    print(f"\n[✅ OK] Eseguito in {time.time() - start_time:.1f} secondi.")

def main():
    print("=" * 85)
    print(f" 🚀 AVVIO PIPELINE: CRYPTOJACKING DETECTION")
    print(f" 📅 Data e Ora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" 📝 Questo output verrà salvato in: {LOG_FILE}")
    print("=" * 85)
    
    t0 = time.time()
    for step, path in PIPELINE_SCRIPTS:
        run_script(step, path)
        
    print("\n" + "=" * 85)
    print(f" 🎉 PIPELINE COMPLETATA CON SUCCESSO IN {time.time() - t0:.1f} SECONDI!")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    main()