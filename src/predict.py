import pandas as pd
import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score, recall_score,
                              precision_score)

# --- CONFIGURATION ---
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR     = os.path.join(BASE_DIR, "models")
DATA_FILE     = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")

RF_MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
DL_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")

# =============================================================================
# SHARED SCALER — the same one fitted in train_rf.py and loaded in train_dl.py.
# RF and DL receive exactly the same preprocessed data:
# the comparison between the two models is scientifically valid.
# =============================================================================
SCALER_PATH   = os.path.join(MODEL_DIR, "scaler.pkl")
RANDOM_STATE  = 42


# --- NEURAL NETWORK ARCHITECTURE ---
# Must be identical to the one defined in train_dl.py
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
    """Loads RF, DL, and the shared scaler. Returns (rf, dl, scaler)."""
    for path, name in [(RF_MODEL_PATH, "rf_model.pkl"),
                       (DL_MODEL_PATH,  "mlp_model.pth"),
                       (SCALER_PATH,    "scaler.pkl")]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {name}\n"
                "Make sure you have run train_rf.py and train_dl.py"
            )

    rf_clf = joblib.load(RF_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    dl_model = SimpleMLP(input_dim)
    dl_model.load_state_dict(torch.load(DL_MODEL_PATH, weights_only=True))
    dl_model.eval()  # disables Dropout for inference
    return rf_clf, dl_model, scaler


def predict_rf(rf_clf, scaler, X):
    """RF Predictions: returns (classes, class 1 probabilities)."""
    # If X is a DataFrame, use .values
    X_to_scale = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler.transform(X_to_scale)
    preds    = rf_clf.predict(X_scaled)
    probs    = rf_clf.predict_proba(X_scaled)[:, 1]
    return preds, probs


def predict_dl(dl_model, scaler, X):
    """DL Predictions: returns (classes, class 1 probabilities)."""
    # If X is a DataFrame, use .values
    X_to_scale = X.values if isinstance(X, pd.DataFrame) else X
    X_scaled = scaler.transform(X_to_scale)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        logits = dl_model(X_tensor)
        probs  = torch.sigmoid(logits).numpy().flatten()
    preds = (probs > 0.5).astype(int)
    return preds, probs


def print_metrics(name, y_true, y_pred, y_prob):
    """Prints classification report, confusion matrix, and AUC-ROC."""
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred,
                                 target_names=['Benign', 'Malicious'], digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
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
    print("   THREAT DETECTION SYSTEM — REAL-TIME SIMULATION (RF vs DL)")
    print("=" * 85)

    # --- 1. DATA LOADING ---
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] Holdout set not found: {DATA_FILE}")
        print("Run train_rf.py first.")
        return

    df_holdout = pd.read_csv(DATA_FILE)
    X_full     = df_holdout.drop(columns=['Label'])
    y_full     = df_holdout['Label'].values

    # --- 2. MODEL LOADING ---
    print("Loading models...", end=" ")
    try:
        rf_clf, dl_model, scaler = load_models(input_dim=X_full.shape[1])
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return
    print("Done.")

    # --- 3. REAL-TIME SIMULATION ---
    # ==========================================================================
    # The demo sample maintains the REAL distribution of the holdout set
    # (~90% benign, ~10% malicious) — it is not artificially balanced.
    # This makes the demo accuracy representative of actual deployment
    # conditions, where malicious traffic is the minority.
    # ==========================================================================
    print("\n" + "=" * 85)
    print("   REAL-TIME SIMULATION — 20 samples with real distribution")
    print("=" * 85)

    np.random.seed(RANDOM_STATE)
    demo_idx = np.random.choice(len(df_holdout), size=min(20, len(df_holdout)),
                                 replace=False)
    df_demo  = df_holdout.iloc[demo_idx].reset_index(drop=True)
    X_demo   = df_demo.drop(columns=['Label'])
    y_demo   = df_demo['Label'].values

    rf_preds_demo, rf_probs_demo = predict_rf(rf_clf, scaler, X_demo)
    dl_preds_demo, dl_probs_demo = predict_dl(dl_model, scaler, X_demo)

    print(f"\n{'ID':<3} | {'Reality':<10} | {'Random Forest':<25} | {'Deep Learning':<25}")
    print("-" * 80)

    rf_correct = dl_correct = 0
    for i in range(len(y_demo)):
        real_str = "Malware" if y_demo[i] == 1 else "Benign"

        rf_label = "[MALWARE]" if rf_preds_demo[i] == 1 else "[BENIGN] "
        rf_conf  = (rf_probs_demo[i] if rf_preds_demo[i] == 1
                    else 1 - rf_probs_demo[i]) * 100
        rf_ok    = "OK " if rf_preds_demo[i] == y_demo[i] else "ERR"
        if rf_preds_demo[i] == y_demo[i]:
            rf_correct += 1

        dl_label = "[MALWARE]" if dl_preds_demo[i] == 1 else "[BENIGN] "
        dl_conf  = (dl_probs_demo[i] if dl_preds_demo[i] == 1
                    else 1 - dl_probs_demo[i]) * 100
        dl_ok    = "OK " if dl_preds_demo[i] == y_demo[i] else "ERR"
        if dl_preds_demo[i] == y_demo[i]:
            dl_correct += 1

        print(f"{i:<3} | {real_str:<10} | "
              f"{rf_label:<11} {rf_conf:>5.1f}% {rf_ok} | "
              f"{dl_label:<11} {dl_conf:>5.1f}% {dl_ok}")

    print("-" * 80)
    print(f"Demo Accuracy: RF={rf_correct}/{len(df_demo)} | "
          f"DL={dl_correct}/{len(df_demo)}")
    print("[!] On 20 samples this has no statistical value — see full verification below.")

    # --- 4. SCIENTIFIC VERIFICATION ON FULL HOLDOUT ---
    print("\n" + "=" * 85)
    print("   SCIENTIFIC VERIFICATION — FULL HOLDOUT SET")
    print("   (RF and DL evaluated on the same preprocessed data with the same scaler)")
    print("=" * 85)

    rf_preds_full, rf_probs_full = predict_rf(rf_clf, scaler, X_full)
    dl_preds_full, dl_probs_full = predict_dl(dl_model, scaler, X_full)

    print_metrics("RANDOM FOREST", y_full, rf_preds_full, rf_probs_full)
    print_metrics("DEEP LEARNING", y_full, dl_preds_full, dl_probs_full)

    # --- 5. COMPARATIVE SUMMARY ---
    print("\n" + "=" * 85)
    print("   COMPARATIVE SUMMARY RF vs DL")
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

    print(f"\n{'Metric':<25} {'Random Forest':>15} {'Deep Learning':>15}")
    print("-" * 57)
    for metric, (rf_val, dl_val) in metrics.items():
        winner = "<- RF" if rf_val > dl_val else "<- DL"
        print(f"{metric:<25} {rf_val:>15.4f} {dl_val:>15.4f}   {winner}")

    print("\n[SUCCESS] Simulation completed.")


if __name__ == "__main__":
    predict_threats()