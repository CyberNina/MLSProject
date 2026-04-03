import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import joblib

# --- CONFIGURATION ---
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEV_FILE  = os.path.join(BASE_DIR, "data", "dev_set.csv")
HOLD_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR   = os.path.join(BASE_DIR, "results", "plots")

DL_MODEL_PATH = os.path.join(MODEL_DIR, "mlp_model.pth")

# =============================================================================
# SHARED SCALER — the same scaler.pkl fitted in train_rf.py.
# It is never re-fitted here: we only use transform() on val and holdout sets.
# This ensures that RF and DL operate on the exact same feature space.
# =============================================================================
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Training parameters
BATCH_SIZE    = 128
LEARNING_RATE = 0.001
EPOCHS        = 30
PATIENCE      = 7
RANDOM_STATE  = 42

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- ARCHITECTURE ---
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
            # No final Sigmoid layer: BCEWithLogitsLoss is used, which
            # incorporates it internally for better numerical stability.
            # Sigmoid is applied manually during inference.
        )

    def forward(self, x):
        return self.network(x)


def train_deep_learning():
    print("=" * 70)
    print("PHASE 4 - DEEP LEARNING TRAINING (MLP)")
    print("=" * 70)

    os.makedirs(IMG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- 1. FILE VERIFICATION ---
    for path, name in [(DEV_FILE, "dev_set.csv"),
                       (HOLD_FILE, "holdout_dataset.csv"),
                       (SCALER_PATH, "scaler.pkl")]:
        if not os.path.exists(path):
            print(f"[ERROR] Missing file: {name}")
            print("Ensure you have run train_rf.py first.")
            return

    # --- 2. LOADING DATA ---
    print("Loading dev set and holdout set...")
    df_dev      = pd.read_csv(DEV_FILE)
    df_holdout  = pd.read_csv(HOLD_FILE)

    X_dev = df_dev.drop(columns=['Label']).values
    y_dev = df_dev['Label'].values

    X_holdout = df_holdout.drop(columns=['Label']).values
    y_holdout = df_holdout['Label'].values

    # --- 3. DEV INTERNAL SPLIT (80% train, 20% validation) ---
    # ==========================================================================
    # NOTE: The same stratified approach used in train_rf.py with the exact same
    # RANDOM_STATE, to guarantee that the train and validation sets are identical
    # between RF and DL. This makes the validation metrics directly comparable
    # between the two models.
    # ==========================================================================
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20,
                                  random_state=RANDOM_STATE)
    train_idx, val_idx = next(sss.split(X_dev, y_dev))

    X_train, y_train = X_dev[train_idx], y_dev[train_idx]
    X_val,   y_val   = X_dev[val_idx],   y_dev[val_idx]

    print(f"Training set:   {len(X_train):,} windows")
    print(f"Validation set: {len(X_val):,} windows")
    print(f"Holdout set:    {len(X_holdout):,} windows")

    # --- 4. NORMALIZATION (Shared scaler, transform only) ---
    print("\nLoading shared RF+DL scaler...")
    scaler = joblib.load(SCALER_PATH)

    X_train_s   = scaler.transform(X_train)
    X_val_s     = scaler.transform(X_val)
    X_holdout_s = scaler.transform(X_holdout)

    # --- 5. TENSOR CONVERSION ---
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train,   dtype=torch.float32).unsqueeze(1)
    X_val_t   = torch.tensor(X_val_s,   dtype=torch.float32)
    y_val_t   = torch.tensor(y_val,     dtype=torch.float32).unsqueeze(1)
    X_hold_t  = torch.tensor(X_holdout_s, dtype=torch.float32)

    # --- 6. CLASS BALANCING (WeightedRandomSampler) ---
    class_counts  = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[int(l)] for l in y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               sampler=sampler)

    # --- 7. MODEL INITIALIZATION ---
    input_dim = X_train_s.shape[1]
    model     = SimpleMLP(input_dim).to(device)

    # BCEWithLogitsLoss: more numerically stable than BCELoss + explicit Sigmoid.
    # Incorporates the sigmoid internally using the log-sum-exp trick.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 8. TRAINING LOOP WITH EARLY STOPPING ---
    print(f"\nDevice: {device}")
    print(f"Starting training ({EPOCHS} epochs max, patience={PATIENCE})...\n")

    best_val_loss     = float('inf')
    patience_counter  = 0
    train_losses      = []
    val_losses        = []
    start_time        = time.time()

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t.to(device))
            val_loss   = criterion(val_logits, y_val_t.to(device)).item()
            # Apply sigmoid to obtain probabilities -> binary predictions
            val_probs  = torch.sigmoid(val_logits).cpu()
            val_preds  = (val_probs > 0.5).float().numpy().flatten()
            val_acc    = accuracy_score(y_val, val_preds)

        val_losses.append(val_loss)

        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Early stopping: saves the best model
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), DL_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(no improvement for {PATIENCE} epochs).")
                break

    elapsed = time.time() - start_time
    print(f"\nTraining time: {elapsed:.1f}s")

    # --- 9. FINAL EVALUATION ON HOLDOUT SET ---
    # Load the best model (not necessarily the last one)
    model.load_state_dict(torch.load(DL_MODEL_PATH, weights_only=True))
    model.eval()

    print("\n--- Holdout Set Results (unseen during training) ---")
    with torch.no_grad():
        hold_logits = model(X_hold_t.to(device))
        hold_probs  = torch.sigmoid(hold_logits).cpu().numpy().flatten()
        hold_preds  = (hold_probs > 0.5).astype(int)

    print(classification_report(y_holdout, hold_preds,
                                 target_names=['Benign', 'Malicious'], digits=4))

    cm = confusion_matrix(y_holdout, hold_preds)
    print("Confusion Matrix (Holdout):")
    print(cm)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"  TN={tn}  FP={fp}  FN={fn}  TP={tp}")
        print(f"  False Positive Rate: {fp/(fp+tn)*100:.2f}%")
        print(f"  False Negative Rate: {fn/(fn+tp)*100:.2f}%")

    # --- 10. PLOTS ---
    # Learning curve
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss', color='#3498db')
    plt.plot(val_losses,   label='Val Loss',   color='#e74c3c')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCEWithLogits)')
    plt.title('Learning Curve - Deep Learning (MLP)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "learning_curve_dl.png"), dpi=300)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Benign', 'Malicious'],
                yticklabels=['Benign', 'Malicious'],
                annot_kws={"size": 14})
    plt.ylabel('Reality')
    plt.xlabel('Prediction')
    plt.title('Confusion Matrix - Deep Learning (MLP)')
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_dl.png"), dpi=300)
    plt.close()

    print(f"\n[SUCCESS] DL Model saved to: {DL_MODEL_PATH}")
    print(f"[SUCCESS] Plots saved to:    {IMG_DIR}")


if __name__ == "__main__":
    train_deep_learning()