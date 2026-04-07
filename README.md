# Cryptojacking Detection in IoT Networks via Side-Channel Analysis

This repository contains a Machine Learning framework developed to detect **Cryptojacking** on IoT devices (Raspberry Pi 4). By analyzing solely network metadata (**Side-Channel Analysis**), this approach identifies malicious activity even in encrypted traffic. The system classifies traffic exclusively based on timing and size statistics of network packets.
Two models are trained and compared:

- Random Forest (RF) — baseline, interpretable ensemble model
- Deep Learning MLP — neural network, 2 hidden layers (64 → 32 neurons)

## Data Pipeline - Pre-processing
The automated pipeline transforms raw network captures into a structured dataset through several critical stages:

1.  **Raw Data Loading:** Processing of CSV files generated from network traffic captures (PCAP).
2.  **MAC Filtering:** Strict filtering using a fixed list of 4 known Raspberry Pi MAC addresses to isolate target traffic.
3.  **Feature Engineering:** Extraction of 6 key metrics (Time and Length) calculated over **10-packet rolling windows**.
4.  **Isolated Windowing:** To ensure mathematical accuracy, windows are calculated strictly within individual capture files, preventing cross-file data contamination.
5.  **Scaling:** A shared `StandardScaler` is fitted *only* on the training set to prevent data leakage.

## Project Structure
.
├── run_all.py                # Pipeline orchestrator
├── src/
│   ├── main.py               # Data ingestion & feature engineering
│   ├── train_rf.py           # Random Forest training & scaler serialization
│   ├── train_dl.py           # MLP training via PyTorch (Early Stopping)
│   ├── predict.py            # Comparative evaluation on holdout set
│   └── visualize_results.py  # Statistical visualization & plotting
├── attacks/
│   └── advers_attack.py      # Adversarial perturbation engine
├── utils/
│   └── loader.py             # MAC filtering and windowing logic
├── data/                     # Local storage for CSV captures and splits
├── models/                   # Serialized model weights and scalers
└── results/                  # Performance metrics and ROC/Confusion matrices

## Dataset Summary
The final processed dataset consists of **625,287 windows**.

| Class | Windows | Proportion |
| :--- | :--- | :--- |
| **Benign (0)** | 561,098 | 89.7% |
| **Malicious (1)** | 64,189 | 10.3% |
| **Total** | **625,287** | **100%** |

## 1. Baseline Performance (Standard Conditions)
Results evaluated on a 10% unseen Holdout Set (62,529 windows).

### Random Forest (Baseline)
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Benign | 0.9957 | 0.9973 | 0.9965 |
| **Malicious** | **0.9761** | **0.9623** | **0.9692** |
| **Accuracy** | | | **99.37%** |

### Deep Learning (MLP)
| Class | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| Benign | 0.9908 | 0.9807 | 0.9857 |
| **Malicious** | **0.8451** | **0.9204** | **0.8811** |
| **Accuracy** | | | **97.45%** |

## 2. Feature Importance (Random Forest)
The model relies heavily on temporal rhythms to distinguish mining activity:
* **Time_Mean:** 0.3555
* **Length_Max:** 0.2264
* **Time_Var:** 0.1937
* **Length_Mean:** 0.1262

## 3. Adversarial Robustness Test
Models were subjected to 4 evasion attacks to test resilience in hostile environments.

### Random Forest under Attack

| Attack | Recall | Drop | Status |
| :--- | :---: | :---: | :--- |
| Baseline (no attack) | 96.23% | — | — |
| Timing Jitter 5% | 95.09% | -1.14% | Resistant |
| Timing Jitter 10% | 94.38% | -1.85% | Resistant |
| Timing Jitter 20% | 93.55% | -2.68% | Mild Drop |
| Padding 30 bytes | 82.16% | -14.07% | Vulnerable |
| Padding 50 bytes | 79.36% | -16.87% | Vulnerable |
| Padding 100 bytes | 41.38% | -54.85% | Critical Failure |
| Mimicry 20% | 77.69% | -18.54% | Vulnerable |
| Mimicry 30% | 68.56% | -27.67% | Vulnerable |
| Burst Shaping 20% | 94.33% | -1.90% | Resistant |
| Burst Shaping 40% | 91.46% | -4.77% | Mild Drop |
| **Average** | **81.80%** | **-14.43%** | — |

### Deep Learning under Attack

| Attack | Recall | Drop | Status |
| :--- | :---: | :---: | :--- |
| Baseline (no attack) | 92.04% | — | — |
| Timing Jitter 5% | 92.05% | ~0% | Resistant |
| Timing Jitter 10% | 92.04% | ~0% | Resistant |
| Timing Jitter 20% | 91.84% | -0.20% | Resistant |
| Padding 30 bytes | 91.87% | -0.17% | Resistant |
| Padding 50 bytes | 91.81% | -0.23% | Resistant |
| Padding 100 bytes | 91.21% | -0.83% | Resistant |
| Mimicry 20% | 86.04% | -6.00% | Vulnerable |
| Mimicry 30% | 83.17% | -8.87% | Vulnerable |
| Burst Shaping 20% | 85.28% | -6.76% | Vulnerable |
| Burst Shaping 40% | 59.96% | -32.08% | Critical Failure |
| **Average** | **86.53%** | **-5.51%** | — |

## 4. Final Comparison & Conclusion
| Metric | Random Forest | Deep Learning |
| :--- | :--- | :--- |
| **Standard Recall** | **0.9623** | 0.9204 |
| **Avg. Attack Drop** | -14.43% | **-5.51%** |
| **Robustness**

## To Start 
```bash
# Install dependencies
pip install -r requirements.txt

# Run the entire pipeline
python3 run_all.py

