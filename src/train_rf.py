import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_traffic.csv")

# Output
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
DEV_FILE     = os.path.join(BASE_DIR, "data", "dev_set.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH  = os.path.join(MODEL_DIR, "rf_model.pkl")

# =============================================================================
# SCALER CONDIVISO — usato da RF, DL e predict.py.
# Un unico scaler fittato sul training set garantisce che RF e DL
# operino sullo stesso spazio delle feature, rendendo il confronto valido.
# =============================================================================
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Seed per riproducibilità
RANDOM_STATE = 42


def train_and_evaluate():
    print("=" * 70)
    print("FASE 3 — ADDESTRAMENTO RANDOM FOREST (BASELINE)")
    print("=" * 70)

    # --- 1. CARICAMENTO ---
    if not os.path.exists(DATA_FILE):
        print(f"[ERRORE] File non trovato: {DATA_FILE}")
        print("Esegui prima main.py")
        return

    print(f"Caricamento dataset da: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Totale finestre: {len(df):,}")

    counts = df['Label'].value_counts()
    total  = len(df)
    print(f"  Benign    (0): {counts.get(0,0):>7,} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Malicious (1): {counts.get(1,0):>7,} ({counts.get(1,0)/total*100:.1f}%)")

    # --- 2. SPLIT STRATIFICATO HOLDOUT (10%) ---
    # ==========================================================================
    # NOTA METODOLOGICA — Split stratificato invece di temporale:
    # Le capture malicious (giu 2021) e benign (nov 2021) sono temporalmente
    # separate di 142 giorni e provengono da dispositivi diversi.
    # Uno split temporale netto metterebbe tutto il malicious in train e
    # tutto il benign in test (o viceversa), rendendo il task banale e
    # non rappresentativo. Si usa quindi uno split stratificato per classe
    # dopo lo shuffle eseguito in loader.py, garantendo che train e test
    # abbiano la stessa distribuzione di classi. Questa è una scelta
    # ingegneristica deliberata documentata nella relazione.
    # ==========================================================================

    from sklearn.model_selection import StratifiedShuffleSplit

    X = df.drop(columns=['Label'])
    y = df['Label']

    # Primo split: 90% dev, 10% holdout — stratificato per classe
    sss_holdout = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                         random_state=RANDOM_STATE)
    train_dev_idx, holdout_idx = next(sss_holdout.split(X, y))

    X_dev      = X.iloc[train_dev_idx]
    y_dev      = y.iloc[train_dev_idx]
    X_holdout  = X.iloc[holdout_idx]
    y_holdout  = y.iloc[holdout_idx]

    # Salva holdout e dev set su disco
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    df_dev     = df.iloc[train_dev_idx].reset_index(drop=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    df_holdout.to_csv(HOLDOUT_FILE, index=False)
    df_dev.to_csv(DEV_FILE, index=False)

    h_counts = df_holdout['Label'].value_counts()
    print(f"\nHoldout set salvato ({len(df_holdout):,} finestre):")
    print(f"  Benign    (0): {h_counts.get(0,0):>6,} ({h_counts.get(0,0)/len(df_holdout)*100:.1f}%)")
    print(f"  Malicious (1): {h_counts.get(1,0):>6,} ({h_counts.get(1,0)/len(df_holdout)*100:.1f}%)")
    print(f"Dev set salvato  ({len(df_dev):,} finestre)")

    # --- 3. SPLIT INTERNO DEV (80% train, 20% validation) ---
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                      random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss_val.split(X_dev, y_dev))

    X_train = X_dev.iloc[train_idx]
    y_train = y_dev.iloc[train_idx]
    X_val   = X_dev.iloc[val_idx]
    y_val   = y_dev.iloc[val_idx]

    print(f"\nTraining set:   {len(X_train):,} finestre")
    print(f"Validation set: {len(X_val):,} finestre")

    # --- 4. NORMALIZZAZIONE ---
    # ==========================================================================
    # NOTA: StandardScaler su Random Forest non influenza le predizioni
    # (gli alberi usano soglie ordinali, invarianti alla scala).
    # Lo scaler viene comunque fittato qui perché è CONDIVISO con il
    # modello DL: train_dl.py e predict.py caricheranno questo stesso
    # scaler.pkl, garantendo che RF e DL operino sullo stesso spazio
    # delle feature e rendendo il confronto tra i due modelli valido.
    # ==========================================================================
    print("\n--- Normalizzazione (scaler condiviso RF+DL) ---")
    scaler = StandardScaler()
    # Aggiungi .values qui sotto
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled   = scaler.transform(X_val.values)
    X_holdout_scaled = scaler.transform(X_holdout.values)

    # Salva scaler condiviso
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvato in: {SCALER_PATH}")

    # --- 5. ADDESTRAMENTO RANDOM FOREST ---
    print("\n--- Addestramento Random Forest ---")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'   # compensa lo sbilanciamento 90/10
    )
    clf.fit(X_train_scaled, y_train)

    # --- 6. VALUTAZIONE SUL VALIDATION SET ---
    print("\n--- Risultati Validation Set ---")
    y_pred_val = clf.predict(X_val_scaled)
    print(classification_report(y_val, y_pred_val,
                                 target_names=['Benign', 'Malicious'], digits=4))

    # --- 7. VALUTAZIONE SULL'HOLDOUT ---
    print("\n--- Risultati Holdout Set (mai visto durante il training) ---")
    y_pred_holdout = clf.predict(X_holdout_scaled)
    print(classification_report(y_holdout, y_pred_holdout,
                                 target_names=['Benign', 'Malicious'], digits=4))

    cm = confusion_matrix(y_holdout, y_pred_holdout)
    print("Matrice di Confusione (Holdout):")
    print(cm)

    # --- 8. FEATURE IMPORTANCE ---
    print("\n--- Feature Importance (Random Forest) ---")
    feature_names = list(X.columns)
    importances   = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {name:<15} {imp:.4f}  {bar}")

    # --- 9. SALVATAGGIO MODELLO ---
    joblib.dump(clf, RF_MODEL_PATH)
    print(f"\n[✅ OK] Modello RF salvato in: {RF_MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()