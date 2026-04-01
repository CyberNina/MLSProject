import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, recall_score,
                              precision_score)

# --- CONFIGURAZIONE ---
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
DATA_FILE     = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")

# =============================================================================
# SCALER CONDIVISO — lo stesso fittato in train_rf.py e caricato in train_dl.py.
# RF e DL ricevono esattamente gli stessi dati preprocessati:
# il confronto tra i due modelli è scientificamente valido.
# =============================================================================
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
RANDOM_STATE  = 42


# --- ARCHITETTURA RETE NEURALE ---
# Deve essere identica a quella definita in train_dl.py
class SimpleMLP(nn.Module):
    def __init__(self, input_dim):
        super(SimpleMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


def load_models(input_dim):
    """Carica RF, DL e scaler condiviso. Restituisce (rf, dl, scaler)."""
    for path, name in [(RF_MODEL_PATH, "rf_model.pkl"),
                       (DL_MODEL_PATH,  "mlp_model.pth"),
                       (SCALER_PATH,    "scaler.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File mancante: {name}\n"
                "Assicurati di aver eseguito train_rf.py e train_dl.py"
            )

    rf_clf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    dl_model = SimpleMLP(input_dim)
    dl_model.load_state_dict(torch.load(DL_MODEL_PATH, weights_only=True))
    dl_model.eval()  # disabilita Dropout per l'inferenza
    return rf_clf, dl_model, scaler


def predict_rf(rf_clf, scaler, X):
    """Predizioni RF: restituisce (classi, probabilità classe 1)."""
    # Se X è un DataFrame, usa .values
    X_to_scale = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler.transform(X_to_scale)
    preds    = rf_clf.predict(X_scaled)
    probs    = rf_clf.predict_proba(X_scaled)[:, 1]
    return preds, probs


def predict_dl(dl_model, scaler, X):
    """Predizioni DL: restituisce (classi, probabilità classe 1)."""
    # Se X è un DataFrame, usa .values
    X_to_scale = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler.transform(X_to_scale)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = dl_model(X_tensor)
        probs  = torch.sigmoid(logits).numpy().flatten()
    preds = (probs > 0.5).astype(int)
    return preds, probs


def print_metrics(name, y_true, y_pred, y_prob):
    """Stampa classification report, confusion matrix e AUC-ROC."""
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred,
                                 target_names=['Benign', 'Malicious'], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Matrice di Confusione:")
    print(cm)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")
    try:
        auc = roc_auc_score(y_true, y_prob)
        print(f"  AUC-ROC: {auc:.4f}")
    except Exception:
        pass


def predict_threats():
    print("=" * 85)
    print("   SISTEMA DI RILEVAMENTO MINACCE — SIMULAZIONE REAL-TIME (RF vs DL)")
    print("=" * 85)

    # --- 1. CARICAMENTO DATI ---
    if not os.path.exists(DATA_FILE):
        print(f"[ERRORE] Holdout non trovato: {DATA_FILE}")
        print("Esegui prima train_rf.py")
        return

    df_holdout = pd.read_csv(DATA_FILE)
    X_full     = df_holdout.drop(columns=['Label'])
    y_full     = df_holdout['Label'].values

    # --- 2. CARICAMENTO MODELLI ---
    print("Caricamento modelli...", end=" ")
    try:
        rf_clf, dl_model, scaler = load_models(input_dim=X_full.shape[1])
    except FileNotFoundError as e:
        print(f"\n[ERRORE] {e}")
        return
    print("Fatto.")

    # --- 3. SIMULAZIONE REAL-TIME ---
    # ==========================================================================
    # Il campione demo mantiene la distribuzione REALE dell'holdout
    # (~90% benign, ~10% malicious) — non viene bilanciato artificialmente.
    # Questo rende l'accuratezza della demo rappresentativa delle condizioni
    # reali di deployment, dove il traffico malevolo è minoritario.
    # ==========================================================================
    print("\n" + "=" * 85)
    print("   SIMULAZIONE REAL-TIME — 20 finestre a distribuzione reale")
    print("=" * 85)

    np.random.seed(RANDOM_STATE)
    demo_idx = np.random.choice(len(df_holdout), size=min(20, len(df_holdout)),
                                 replace=False)
    df_demo  = df_holdout.iloc[demo_idx].reset_index(drop=True)
    X_demo   = df_demo.drop(columns=['Label'])
    y_demo   = df_demo['Label'].values

    rf_preds_demo, rf_probs_demo = predict_rf(rf_clf, scaler, X_demo)
    dl_preds_demo, dl_probs_demo = predict_dl(dl_model, scaler, X_demo)

    print(f"\n{'ID':<3} | {'Realtà':<10} | {'Random Forest':<25} | {'Deep Learning':<25}")
    print("-" * 80)

    rf_correct = dl_correct = 0
    for i in range(len(y_demo)):
        real_str = "Malware" if y_demo[i] == 1 else "Benign"

        rf_label = "🔴 MALWARE" if rf_preds_demo[i] == 1 else "🟢 BENIGN"
        rf_conf  = (rf_probs_demo[i] if rf_preds_demo[i] == 1
                    else 1 - rf_probs_demo[i]) * 100
        rf_ok    = "✅" if rf_preds_demo[i] == y_demo[i] else "❌"
        if rf_preds_demo[i] == y_demo[i]:
            rf_correct += 1

        dl_label = "🔴 MALWARE" if dl_preds_demo[i] == 1 else "🟢 BENIGN"
        dl_conf  = (dl_probs_demo[i] if dl_preds_demo[i] == 1
                    else 1 - dl_probs_demo[i]) * 100
        dl_ok    = "✅" if dl_preds_demo[i] == y_demo[i] else "❌"
        if dl_preds_demo[i] == y_demo[i]:
            dl_correct += 1

        print(f"{i:<3} | {real_str:<10} | "
              f"{rf_label:<11} {rf_conf:>5.1f}% {rf_ok} | "
              f"{dl_label:<11} {dl_conf:>5.1f}% {dl_ok}")

    print("-" * 80)
    print(f"Accuratezza demo: RF={rf_correct}/{len(df_demo)} | "
          f"DL={dl_correct}/{len(df_demo)}")
    print("⚠️  Su 20 campioni non ha valore statistico — vedi verifica sotto.")

    # --- 4. VERIFICA SCIENTIFICA SULL'INTERO HOLDOUT ---
    print("\n" + "=" * 85)
    print("   VERIFICA SCIENTIFICA — INTERO HOLDOUT SET")
    print("   (RF e DL valutati sugli stessi dati preprocessati con lo stesso scaler)")
    print("=" * 85)

    rf_preds_full, rf_probs_full = predict_rf(rf_clf, scaler, X_full)
    dl_preds_full, dl_probs_full = predict_dl(dl_model, scaler, X_full)

    print_metrics("RANDOM FOREST", y_full, rf_preds_full, rf_probs_full)
    print_metrics("DEEP LEARNING", y_full, dl_preds_full, dl_probs_full)

    # --- 5. RIEPILOGO COMPARATIVO ---
    print("\n" + "=" * 85)
    print("   RIEPILOGO COMPARATIVO RF vs DL")
    print("=" * 85)

    metrics = {
        "Recall (Malicious)":    (recall_score(y_full, rf_preds_full),
                                   recall_score(y_full, dl_preds_full)),
        "Precision (Malicious)": (precision_score(y_full, rf_preds_full),
                                   precision_score(y_full, dl_preds_full)),
        "F1 (Malicious)":        (f1_score(y_full, rf_preds_full),
                                   f1_score(y_full, dl_preds_full)),
        "AUC-ROC":               (roc_auc_score(y_full, rf_probs_full),
                                   roc_auc_score(y_full, dl_probs_full)),
    }

    print(f"\n{'Metrica':<25} {'Random Forest':>15} {'Deep Learning':>15}")
    print("-" * 57)
    for metric, (rf_val, dl_val) in metrics.items():
        winner = "← RF" if rf_val > dl_val else "← DL"
        print(f"{metric:<25} {rf_val:>15.4f} {dl_val:>15.4f}   {winner}")

    print("\n[✅ OK] Simulazione completata.")


if __name__ == "__main__":
    predict_threats()