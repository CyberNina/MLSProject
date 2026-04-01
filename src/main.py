import os
import sys
import pandas as pd

# Setup path per importare utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loader import load_and_process_data

# Percorsi dinamici
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_traffic.csv")


def main():
    print("=" * 70)
    print("FASE 1 — CARICAMENTO, FILTRAGGIO E FEATURE ENGINEERING")
    print("=" * 70)

    # ==========================================================================
    # NOTA: rispetto alla versione precedente, main.py è molto più semplice.
    # Tutta la logica di filtraggio MAC, windowing e calcolo Delta_Time
    # è stata spostata dentro utils/loader.py (_load_single_file).
    # Questo file si occupa solo di orchestrare il caricamento e salvare
    # il dataset finale su disco.
    #
    # I 4 MAC dei Raspberry Pi sono definiti in loader.py come costante
    # KNOWN_RASPBERRY_MACS — non vengono più inferiti con mode() dal dato,
    # eliminando il rischio di selezionare il MAC del gateway al posto
    # del dispositivo target.
    # ==========================================================================

    # 1. Carica, filtra, calcola feature (tutto in loader.py)
    try:
        df = load_and_process_data(DATA_DIR)
    except ValueError as e:
        print(f"\n[ERRORE CRITICO] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERRORE INATTESO] {e}")
        sys.exit(1)

    # 2. Verifica che il dataset non sia vuoto
    if df.empty:
        print("[ERRORE CRITICO] Il dataset finale è vuoto.")
        sys.exit(1)

    # 3. Verifica colonne attese
    expected_cols = {'Time_Mean', 'Time_Var', 'Length_Mean',
                     'Length_Min', 'Length_Max', 'Length_Var', 'Label'}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[ERRORE CRITICO] Colonne mancanti nel dataset: {missing}")
        sys.exit(1)

    # 4. Report finale
    counts = df['Label'].value_counts()
    total = len(df)
    print(f"\n--- Riepilogo Dataset Finale ---")
    print(f"  Finestre totali:   {total:>8,}")
    print(f"  Benign    (0):     {counts.get(0, 0):>8,}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Malicious (1):     {counts.get(1, 0):>8,}  ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Feature:           {[c for c in df.columns if c != 'Label']}")

    # 5. Salvataggio
    print(f"\n>> Salvataggio in: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("[✅ OK] Fase 1 completata con successo!")


if __name__ == "__main__":
    main()