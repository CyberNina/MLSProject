import pandas as pd
import glob
import os

# =============================================================================
# MAC ADDRESS NOTI DEI DISPOSITIVI RASPBERRY PI NEL DATASET
# Hardcoded perché noti a priori dall'analisi del dataset.
# Non si usa mode() per evitare selezione casuale in caso di parità.
#
# dc:a6:32:ed:e8:9d → Raspberry Pi (benign: download, interactive, video, webbrowsing)
# 88:e9:fe:52:ae:7b → Raspberry Pi (benign: idle)
# dc:a6:32:67:66:4b → Raspberry Pi (malicious: WebminePool_Aggressive, Stealthy, Webmine_*)
# dc:a6:32:68:35:8a → Raspberry Pi (malicious: Binary, WebminePool_Robust)
# =============================================================================
KNOWN_RASPBERRY_MACS = [
    "dc:a6:32:ed:e8:9d",
    "88:e9:fe:52:ae:7b",
    "dc:a6:32:67:66:4b",
    "dc:a6:32:68:35:8a",
]

# Colonne raw usate dal dataset (Wireshark export)
COLS_TO_USE = ['Time', 'Length', 'Hw_src', 'HW_dst']

# Rinomina per coerenza interna alla pipeline
RENAME_MAP = {
    'Time':   'Timestamp',
    'Length': 'Packet_Length',
    'Hw_src': 'Src_MAC',
    'HW_dst': 'Dst_MAC',
}

# Dimensione finestra rolling (pacchetti)
WINDOW_SIZE = 10


def _load_single_file(filepath, label):
    """
    Carica un singolo CSV, rinomina le colonne, assegna la label,
    filtra per MAC noti, ordina per Timestamp e calcola le feature
    di windowing INTERNAMENTE al file.

    Il windowing viene fatto per file separato per evitare che il
    diff() tra l'ultimo pacchetto di un file e il primo del successivo
    produca Delta_Time nell'ordine di ore/giorni (confine tra capture).

    Restituisce un DataFrame con le feature windowed oppure None se
    il file non è utilizzabile.
    """
    try:
        df = pd.read_csv(filepath, usecols=COLS_TO_USE, header=0)
    except ValueError as ve:
        print(f"   [ERR] Colonne mancanti in {os.path.basename(filepath)}: {ve}")
        return None
    except Exception as e:
        print(f"   [ERR] {os.path.basename(filepath)}: {e}")
        return None

    # Rinomina colonne
    df = df.rename(columns=RENAME_MAP)

    # Converti Timestamp a numerico, scarta righe non valide
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])

    # Converti Packet_Length a numerico, scarta righe non valide
    # (es. righe "Reassembled" presenti in alcuni export Wireshark)
    df['Packet_Length'] = pd.to_numeric(df['Packet_Length'], errors='coerce')
    df = df.dropna(subset=['Packet_Length'])

    # Assegna label
    df['label'] = label

    # Filtra per MAC noti: tieni solo pacchetti dove Src O Dst
    # è uno dei Raspberry Pi conosciuti
    mask = (df['Src_MAC'].isin(KNOWN_RASPBERRY_MACS)) | \
           (df['Dst_MAC'].isin(KNOWN_RASPBERRY_MACS))
    df = df[mask].copy()

    if df.empty:
        print(f"   [WARN] {os.path.basename(filepath)}: nessuna riga dopo filtraggio MAC.")
        return None

    # Ordina per Timestamp DENTRO il file
    df = df.sort_values('Timestamp').reset_index(drop=True)

    # ==========================================================================
    # FEATURE ENGINEERING (WINDOWING) per file singolo
    #
    # NOTA PROGETTUALE — Scelta "worst-case / side-channel":
    # Si escludono deliberatamente Protocol, IP sorgente/destinazione e porte.
    # In scenari reali avanzati il malware può usare VPN, proxy o Domain
    # Fronting (es. tunnel su Cloudflare), rendendo l'analisi L3/L4 inefficace.
    # Il modello è costretto a lavorare solo sulla FORMA del traffico
    # (timing e dimensioni dei pacchetti), dimostrando robustezza anche in
    # condizioni di totale cecità sul payload e sui protocolli applicativi.
    # ==========================================================================

    # Delta_Time: tempo tra pacchetti consecutivi DENTRO questo file.
    # fillna(0) solo sulla prima riga (nessun pacchetto precedente).
    df['Delta_Time'] = df['Timestamp'].diff().fillna(0)

    # Sanity check: delta negativi non devono esistere dopo sort.
    # Se ci sono (clock skew, duplicati) li azzeriamo.
    df['Delta_Time'] = df['Delta_Time'].clip(lower=0)

    # Rolling window sulle feature
    roll = df['Delta_Time'].rolling(window=WINDOW_SIZE)
    roll_len = df['Packet_Length'].rolling(window=WINDOW_SIZE)

    time_mean = roll.mean()
    time_var  = roll.var().fillna(0)

    len_mean  = roll_len.mean()
    len_min   = roll_len.min()
    len_max   = roll_len.max()
    len_var   = roll_len.var().fillna(0)

    # Label della finestra: 1 se almeno un pacchetto nella finestra è malicious.
    # Questo è corretto perché il windowing avviene DENTRO un file già
    # etichettato uniformemente (tutto 0 o tutto 1), quindi max() == label.
    labels = df['label'].rolling(window=WINDOW_SIZE).max()

    df_feat = pd.DataFrame({
        'Time_Mean':   time_mean,
        'Time_Var':    time_var,
        'Length_Mean': len_mean,
        'Length_Min':  len_min,
        'Length_Max':  len_max,
        'Length_Var':  len_var,
        'Label':       labels,
    })

    # Rimuovi le prime (WINDOW_SIZE - 1) righe dove la finestra non è piena
    df_feat = df_feat.dropna()
    df_feat['Label'] = df_feat['Label'].astype(int)

    rows_in  = len(df)
    rows_out = len(df_feat)
    print(f"   [OK] {os.path.basename(filepath)}: "
          f"{rows_in} pacchetti → {rows_out} finestre (label={label})")

    return df_feat


def load_and_process_data(data_path):
    """
    Carica tutti i file CSV benign e malicious, applica il windowing
    per file separato, e restituisce il dataset finale unito e mescolato.

    Lo shuffle finale è necessario per evitare che il dataset abbia
    tutti i malicious all'inizio e tutti i benign alla fine (conseguenza
    del gap temporale di 142 giorni tra le capture).
    """
    print(f"--- Caricamento e Feature Engineering da: {data_path} ---")
    print(f"    Finestra rolling: {WINDOW_SIZE} pacchetti")
    print(f"    MAC Raspberry Pi noti: {len(KNOWN_RASPBERRY_MACS)}\n")

    df_list = []

    # --- MALICIOUS (Label = 1) ---
    # Un unico pattern cattura tutti i file malicious della cartella.
    # I due pattern separati (Webmine* e WebminePool*) causavano il
    # caricamento doppio dei file WebminePool, gonfiando i malicious.
    malicious_pattern = os.path.join(data_path, "malicious", "Raspberry_*.csv")

    print(">> Caricamento MALICIOUS...")
    loaded_paths = set()
    for filepath in sorted(glob.glob(malicious_pattern)):
        if filepath in loaded_paths:
            print(f"   [SKIP] Duplicato ignorato: {os.path.basename(filepath)}")
            continue
        loaded_paths.add(filepath)
        result = _load_single_file(filepath, label=1)
        if result is not None:
            df_list.append(result)

    # --- BENIGN (Label = 0) ---
    benign_pattern = os.path.join(data_path, "benign", "Raspberry_*benign.csv")

    print("\n>> Caricamento BENIGN...")
    for filepath in sorted(glob.glob(benign_pattern)):
        result = _load_single_file(filepath, label=0)
        if result is not None:
            df_list.append(result)

    # --- UNIONE ---
    if not df_list:
        raise ValueError(
            "Errore critico: nessun dato caricato. "
            "Controlla i percorsi e i nomi dei file."
        )

    df_total = pd.concat(df_list, ignore_index=True)

    # Verifica distribuzione classi
    counts = df_total['Label'].value_counts()
    total  = len(df_total)
    print(f"\n--- Distribuzione classi (pre-shuffle) ---")
    print(f"  Benign    (0): {counts.get(0, 0):>7,}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Malicious (1): {counts.get(1, 0):>7,}  ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Totale:        {total:>7,}")

    if counts.get(0, 0) == 0 or counts.get(1, 0) == 0:
        raise ValueError(
            "Errore critico: una delle due classi è assente nel dataset finale. "
            "Controlla i file sorgente."
        )

    # Shuffle globale con seed fisso per riproducibilità.
    # NECESSARIO: senza shuffle, il dataset avrebbe tutto il malicious
    # (giu 2021) prima e tutto il benign (nov 2021) dopo, a causa del
    # gap temporale tra le capture. Il modello imparerebbe il periodo,
    # non le caratteristiche dell'attacco.
    df_total = df_total.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"\n>> Shuffle completato. Dataset pronto ({total:,} finestre).")
    return df_total


def filter_by_mac(df, target_macs=None):
    """
    Mantenuto per compatibilità, ma con la nuova architettura il
    filtraggio MAC avviene già dentro _load_single_file().
    Se chiamato, usa KNOWN_RASPBERRY_MACS come default.
    """
    if target_macs is None:
        target_macs = KNOWN_RASPBERRY_MACS

    if isinstance(target_macs, str):
        target_macs = [target_macs]

    initial = len(df)
    mask = (df['Src_MAC'].isin(target_macs)) | (df['Dst_MAC'].isin(target_macs))
    df_filtered = df[mask].copy()

    print(f">> Filtraggio MAC: {initial:,} → {len(df_filtered):,} righe")
    return df_filtered