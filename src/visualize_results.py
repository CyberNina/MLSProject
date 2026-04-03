import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, recall_score, precision_score, f1_score
import joblib
import os
import torch
import torch.nn as nn
import warnings

# Ignore stylistic warnings from seaborn
warnings.filterwarnings('ignore')

# --- PATH CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FULL = os.path.join(BASE_DIR, "data", "processed_traffic.csv")
DATA_HOLD = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
MODEL_RF  = os.path.join(BASE_DIR, "models", "rf_model.pkl")
MODEL_DL  = os.path.join(BASE_DIR, "models", "mlp_model.pth")
SCALER    = os.path.join(BASE_DIR, "models", "scaler.pkl")
IMG_DIR   = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(IMG_DIR, exist_ok=True)

# --- NEURAL NETWORK ARCHITECTURE ---
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

# Coordinated colors for the presentation
COLOR_RF = '#3498db'     # Blue
COLOR_DL = '#e74c3c'     # Red
COLOR_BENIGN = '#2ecc71' # Green
COLOR_MALWARE = '#e74c3c'# Red

# =============================================================================
# 1. CONTEXT PLOT: Data Distribution (Donut Chart)
# =============================================================================
def plot_data_distribution():
    print(">> Generating: 1. Data Distribution (Donut Chart)...")
    if not os.path.exists(DATA_FULL):
        return

    df = pd.read_csv(DATA_FULL)
    counts = df['Label'].value_counts()
    
    plt.figure(figsize=(8, 6))
    labels = ['Benign (Web Browsing, Video...)', 'Malicious (Cryptojacking)']
    sizes = [counts.get(0, 0), counts.get(1, 0)]
    colors = [COLOR_BENIGN, COLOR_MALWARE]
    explode = (0.05, 0)
    
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=False, startangle=140, textprops={'fontsize': 12, 'weight': 'bold'},
            wedgeprops=dict(width=0.4, edgecolor='w'))
    
    plt.title('Network Traffic Distribution (Ground Truth)', fontsize=16, weight='bold')
    plt.savefig(os.path.join(IMG_DIR, "1_data_distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 2. ANALYSIS PLOT: Feature Importance (Horizontal Bar)
# =============================================================================
def plot_feature_importance():
    print(">> Generating: 2. Random Forest Feature Importance...")
    if not os.path.exists(MODEL_RF) or not os.path.exists(DATA_HOLD):
        return

    clf = joblib.load(MODEL_RF)
    df = pd.read_csv(DATA_HOLD)
    features = [c for c in df.columns if c != 'Label']
    importances = clf.feature_importances_
    
    indices = np.argsort(importances)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], color='#9b59b6', align='center')
    plt.yticks(range(len(indices)), [features[i] for i in indices], fontsize=12)
    plt.xlabel('Relative Importance (Gini Impurity)', fontsize=12, weight='bold')
    plt.title('Feature Importance: What is the algorithm looking at?', fontsize=16, weight='bold')
    
    for i, v in enumerate(importances[indices]):
        plt.text(v + 0.01, i, f"{v:.1%}", va='center', fontsize=11, weight='bold')
        
    plt.xlim(0, max(importances) + 0.1)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(IMG_DIR, "2_feature_importance.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 3. PITFALL PLOT: Baseline Comparison (Bar Chart)
# =============================================================================
def plot_baseline_comparison():
    print(">> Generating: 3. Baseline Metrics Comparison...")
    try:
        df = pd.read_csv(DATA_HOLD)
        rf = joblib.load(MODEL_RF)
        scaler = joblib.load(SCALER)
        
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
        
        dl = SimpleMLP(input_dim=X.shape[1])
        dl.load_state_dict(torch.load(MODEL_DL, weights_only=True))
        dl.eval()
    except Exception as e:
        return

    X_scaled = scaler.transform(X)
    
    rf_preds = rf.predict(X_scaled)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        dl_probs = torch.sigmoid(dl(X_tensor)).numpy().flatten()
    dl_preds = (dl_probs > 0.5).astype(int)

    metrics_rf = [recall_score(y, rf_preds), precision_score(y, rf_preds), f1_score(y, rf_preds)]
    metrics_dl = [recall_score(y, dl_preds), precision_score(y, dl_preds), f1_score(y, dl_preds)]
    
    labels = ['Recall (Malware)', 'Precision', 'F1-Score']
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, metrics_rf, width, label='Random Forest', color=COLOR_RF)
    plt.bar(x + width/2, metrics_dl, width, label='Deep Learning', color=COLOR_DL)

    for i, (v_rf, v_dl) in enumerate(zip(metrics_rf, metrics_dl)):
        plt.text(i - width/2, v_rf + 0.01, f"{v_rf:.1%}", ha='center', weight='bold')
        plt.text(i + width/2, v_dl + 0.01, f"{v_dl:.1%}", ha='center', weight='bold')

    plt.ylabel('Score', fontsize=12, weight='bold')
    plt.title('Peacetime Performance (Baseline without attacks)', fontsize=16, weight='bold')
    plt.xticks(x, labels, fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.savefig(os.path.join(IMG_DIR, "3_baseline_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 4. SCIENTIFIC PLOT: ROC Curve and AUC
# =============================================================================
def plot_roc_curve():
    print(">> Generating: 4. ROC Curve...")
    try:
        df = pd.read_csv(DATA_HOLD)
        rf = joblib.load(MODEL_RF)
        scaler = joblib.load(SCALER)
        
        X = df.drop(columns=['Label']).values
        y = df['Label'].values
        
        dl = SimpleMLP(input_dim=X.shape[1])
        dl.load_state_dict(torch.load(MODEL_DL, weights_only=True))
        dl.eval()
    except Exception as e:
        return

    X_scaled = scaler.transform(X)
    rf_probs = rf.predict_proba(X_scaled)[:, 1]
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    with torch.no_grad():
        dl_probs = torch.sigmoid(dl(X_tensor)).numpy().flatten()

    fpr_rf, tpr_rf, _ = roc_curve(y, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    fpr_dl, tpr_dl, _ = roc_curve(y, dl_probs)
    roc_auc_dl = auc(fpr_dl, tpr_dl)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_rf, tpr_rf, color=COLOR_RF, lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
    plt.plot(fpr_dl, tpr_dl, color=COLOR_DL, lw=2, label=f'Deep Learning (AUC = {roc_auc_dl:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Benign traffic classified as malware)', fontsize=12, weight='bold')
    plt.ylabel('True Positive Rate (Malware correctly detected)', fontsize=12, weight='bold')
    plt.title('ROC Curve - Class Separation Capability', fontsize=16, weight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    
    plt.savefig(os.path.join(IMG_DIR, "4_roc_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 5. RANDOM FOREST CONFUSION MATRIX
# =============================================================================
def plot_confusion_matrix_rf():
    print(">> Generating: 5. Random Forest Confusion Matrix...")
    try:
        df = pd.read_csv(DATA_HOLD)
        clf = joblib.load(MODEL_RF)
        scaler = joblib.load(SCALER)
        
        X = df.drop(columns=['Label']).values
        y_true = df['Label'].values
        X_scaled = scaler.transform(X)
        y_pred = clf.predict(X_scaled)
    except Exception as e:
        return

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Benign', 'Malicious']
    
    plt.figure(figsize=(7, 5))
    # Color matched with RF (Blues)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16, "weight": "bold"})

    plt.ylabel('Reality (True Label)', fontsize=12, weight='bold')
    plt.xlabel('Prediction (Predicted Label)', fontsize=12, weight='bold')
    plt.title('Confusion Matrix - Random Forest', fontsize=16, weight='bold')

    plt.savefig(os.path.join(IMG_DIR, "5_confusion_matrix_rf.png"), dpi=300, bbox_inches='tight')
    plt.close()

# =============================================================================
# 6. DEEP LEARNING CONFUSION MATRIX
# =============================================================================
def plot_confusion_matrix_dl():
    print(">> Generating: 6. Deep Learning Confusion Matrix...")
    try:
        df = pd.read_csv(DATA_HOLD)
        scaler = joblib.load(SCALER)
        
        X = df.drop(columns=['Label']).values
        y_true = df['Label'].values
        X_scaled = scaler.transform(X)
        
        dl = SimpleMLP(input_dim=X.shape[1])
        dl.load_state_dict(torch.load(MODEL_DL, weights_only=True))
        dl.eval()

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            dl_probs = torch.sigmoid(dl(X_tensor)).numpy().flatten()
        y_pred = (dl_probs > 0.5).astype(int)

    except Exception as e:
        return

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Benign', 'Malicious']
    
    plt.figure(figsize=(7, 5))
    # Color matched with DL (Reds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                xticklabels=labels, yticklabels=labels,
                annot_kws={"size": 16, "weight": "bold"})

    plt.ylabel('Reality (True Label)', fontsize=12, weight='bold')
    plt.xlabel('Prediction (Predicted Label)', fontsize=12, weight='bold')
    plt.title('Confusion Matrix - Deep Learning', fontsize=16, weight='bold')

    plt.savefig(os.path.join(IMG_DIR, "6_confusion_matrix_dl.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 70)
    print(" POWERPOINT PRESENTATION PLOT GENERATOR")
    print("=" * 70)
    
    plot_data_distribution()
    plot_feature_importance()
    plot_baseline_comparison()
    plot_roc_curve()
    plot_confusion_matrix_rf()
    plot_confusion_matrix_dl()
    
    print("=" * 70)
    print(f" [SUCCESS] All plots have been generated and saved to: {IMG_DIR}")
    print(" The files are numbered from 1 to 6 to follow the order of your thesis.")
    print(" (Remember that plot 7 is adversarial_robustness.png generated by the attacks)")
    print("=" * 70)

if __name__ == "__main__":
    main()