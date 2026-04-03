import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(BASE_DIR, "data", "processed_traffic.csv")

# Output
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
DEV_FILE     = os.path.join(BASE_DIR, "data", "dev_set.csv")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH  = os.path.join(MODEL_DIR, "rf_model.pkl")

# =============================================================================
# SHARED SCALER — used by RF, DL, and predict.py.
# A single scaler fitted on the training set ensures that RF and DL
# operate on the exact same feature space, making the comparison valid.
# =============================================================================
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Seed for reproducibility
RANDOM_STATE = 42


def train_and_evaluate():
    print("=" * 70)
    print("PHASE 3 - RANDOM FOREST TRAINING (BASELINE)")
    print("=" * 70)

    # --- 1. DATA LOADING ---
    if not os.path.exists(DATA_FILE):
        print(f"[ERROR] File not found: {DATA_FILE}")
        print("Run main.py first.")
        return

    print(f"Loading dataset from: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    print(f"Total windows: {len(df):,}")

    counts = df['Label'].value_counts()
    total  = len(df)
    print(f"  Benign    (0): {counts.get(0,0):>7,} ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Malicious (1): {counts.get(1,0):>7,} ({counts.get(1,0)/total*100:.1f}%)")

    # --- 2. STRATIFIED HOLDOUT SPLIT (10%) ---
    # ==========================================================================
    # METHODOLOGICAL NOTE — Stratified split instead of temporal:
    # The malicious (Jun 2021) and benign (Nov 2021) captures are temporally
    # separated by 142 days and come from different devices.
    # A strict temporal split would place all malicious traffic in train and
    # all benign in test (or vice versa), making the task trivial and
    # unrepresentative. We therefore use a stratified split by class
    # after the shuffle performed in loader.py, ensuring train and test
    # have the same class distribution. This is a deliberate engineering
    # choice documented in the report.
    # ==========================================================================

    from sklearn.model_selection import StratifiedShuffleSplit

    X = df.drop(columns=['Label'])
    y = df['Label']

    # First split: 90% dev, 10% holdout — stratified by class
    sss_holdout = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                         random_state=RANDOM_STATE)
    train_dev_idx, holdout_idx = next(sss_holdout.split(X, y))

    X_dev      = X.iloc[train_dev_idx]
    y_dev      = y.iloc[train_dev_idx]
    X_holdout  = X.iloc[holdout_idx]
    y_holdout  = y.iloc[holdout_idx]

    # Save holdout and dev sets to disk
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)
    df_dev     = df.iloc[train_dev_idx].reset_index(drop=True)

    os.makedirs(MODEL_DIR, exist_ok=True)
    df_holdout.to_csv(HOLDOUT_FILE, index=False)
    df_dev.to_csv(DEV_FILE, index=False)

    h_counts = df_holdout['Label'].value_counts()
    print(f"\nHoldout set saved ({len(df_holdout):,} windows):")
    print(f"  Benign    (0): {h_counts.get(0,0):>6,} ({h_counts.get(0,0)/len(df_holdout)*100:.1f}%)")
    print(f"  Malicious (1): {h_counts.get(1,0):>6,} ({h_counts.get(1,0)/len(df_holdout)*100:.1f}%)")
    print(f"Dev set saved    ({len(df_dev):,} windows)")

    # --- 3. DEV INTERNAL SPLIT (80% train, 20% validation) ---
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                      random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss_val.split(X_dev, y_dev))

    X_train = X_dev.iloc[train_idx]
    y_train = y_dev.iloc[train_idx]
    X_val   = X_dev.iloc[val_idx]
    y_val   = y_dev.iloc[val_idx]

    print(f"\nTraining set:   {len(X_train):,} windows")
    print(f"Validation set: {len(X_val):,} windows")

    # --- 4. NORMALIZATION ---
    # ==========================================================================
    # NOTE: StandardScaler on Random Forest does not influence predictions
    # (trees use ordinal thresholds, invariant to scale).
    # The scaler is fitted here anyway because it is SHARED with the
    # DL model: train_dl.py and predict.py will load this exact scaler.pkl,
    # ensuring that RF and DL operate on the same feature space and making
    # the comparison between the two models valid.
    # ==========================================================================
    print("\n--- Normalization (shared RF+DL scaler) ---")
    scaler = StandardScaler()
    # Add .values here
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_val_scaled   = scaler.transform(X_val.values)
    X_holdout_scaled = scaler.transform(X_holdout.values)

    # Save shared scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to: {SCALER_PATH}")

    # --- 5. RANDOM FOREST TRAINING ---
    print("\n--- Random Forest Training ---")
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight='balanced'   # compensates for the 90/10 imbalance
    )
    clf.fit(X_train_scaled, y_train)

    # --- 6. VALIDATION SET EVALUATION ---
    print("\n--- Validation Set Results ---")
    y_pred_val = clf.predict(X_val_scaled)
    print(classification_report(y_val, y_pred_val,
                                 target_names=['Benign', 'Malicious'], digits=4))

    # --- 7. HOLDOUT SET EVALUATION ---
    print("\n--- Holdout Set Results (unseen during training) ---")
    y_pred_holdout = clf.predict(X_holdout_scaled)
    print(classification_report(y_holdout, y_pred_holdout,
                                 target_names=['Benign', 'Malicious'], digits=4))

    cm = confusion_matrix(y_holdout, y_pred_holdout)
    print("Confusion Matrix (Holdout):")
    print(cm)

    # --- 8. FEATURE IMPORTANCE ---
    print("\n--- Feature Importance (Random Forest) ---")
    feature_names = list(X.columns)
    importances   = clf.feature_importances_
    for name, imp in sorted(zip(feature_names, importances),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"  {name:<15} {imp:.4f}  {bar}")

    # --- 9. MODEL SAVING ---
    joblib.dump(clf, RF_MODEL_PATH)
    print(f"\n[SUCCESS] RF Model saved to: {RF_MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()