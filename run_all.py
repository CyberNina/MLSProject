import subprocess
import sys
import os
import time

# Trova il percorso assoluto della cartella principale (LabML)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Lista ordinata di tutti gli script da eseguire
scripts_to_run = [
    ("1. Filtraggio Dati Grezzi", "src/main.py"),
    ("2. Feature Engineering (Windowing)", "src/build_features.py"),
    ("3. Addestramento Random Forest (Baseline)", "src/train_rf.py"),
    ("4. Addestramento Deep Learning (MLP)", "src/train_dl.py"),
    ("5. Simulazione Real-Time (RF vs DL)", "src/predict.py"),
    ("6. Test di Robustezza (Attacchi Avversari)", "attacks/advers_attack.py"),
    ("7. Generazione Grafici Finali", "src/visualize_results.py")
]

def run_pipeline():
    print("="*90)
    print("AVVIO PIPELINE AUTOMATIZZATA: CRYPTOJACKING DETECTION")
    print("="*90)
    
    start_time_total = time.time()

    for step_name, script_path in scripts_to_run:
        script_full_path = os.path.join(BASE_DIR, script_path)
        
        if not os.path.exists(script_full_path):
            print(f"\n[ERRORE CRITICO] File non trovato: {script_path}")
            print("Interruzione della pipeline.")
            sys.exit(1)

        print(f"\n>> Esecuzione: {step_name}")
        print(f">> Script: {script_path}")
        print("-" * 50)
        
        start_time_step = time.time()
        
        try:
            # Esegue il comando nel terminale usando lo stesso Python del Virtual Environment
            result = subprocess.run([sys.executable, script_full_path], check=True)
            
            elapsed_step = time.time() - start_time_step
            print(f"\n[✅ OK] {step_name} completato in {elapsed_step:.1f} secondi.")
            
        except subprocess.CalledProcessError as e:
            # Se uno script va in errore (es. manca un file), si ferma tutto
            print(f"\n[❌ ERRORE] L'esecuzione di {script_path} è fallita!")
            print("Interruzione della pipeline per evitare reazioni a catena.")
            sys.exit(1)

    elapsed_total = time.time() - start_time_total
    print("\n" + "="*90)
    print(f" PIPELINE COMPLETATA CON SUCCESSO IN {elapsed_total:.1f} SECONDI! ")
    print("Tutti i modelli sono stati addestrati e i risultati salvati.")
    print("="*70)

if __name__ == "__main__":
    run_pipeline()