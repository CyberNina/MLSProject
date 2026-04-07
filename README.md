# Cryptojacking Detection in IoT Networks via Side-Channel Analysis

This repository contains a Machine Learning framework developed to detect **Cryptojacking** on IoT devices (Raspberry Pi 4). By analyzing solely network metadata (**Side-Channel Analysis**), this approach identifies malicious activity even in encrypted traffic.

## Data Pipeline & Pre-processing
The automated pipeline transforms raw network captures into a structured dataset through several critical stages:

1.  **Raw Data Loading:** Processing of CSV files generated from network traffic captures (PCAP).
2.  **MAC Filtering:** Strict filtering using a fixed list of 4 known Raspberry Pi MAC addresses to isolate target traffic.
3.  **Feature Engineering:** Extraction of 6 key metrics (Time and Length) calculated over **10-packet rolling windows**.
4.  **Isolated Windowing:** To ensure mathematical accuracy, windows are calculated strictly within individual capture files, preventing cross-file data contamination.
5.  **Scaling:** A shared `StandardScaler` is fitted *only* on the training set to prevent data leakage.

[Image of a data science pipeline diagram]

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

### Random Forest Robustness
| Attack Type | Recall | Drop vs Baseline | Status |
| :--- | :--- | :--- | :--- |
| Timing Jitter (10%) | 0.9438 | -0.0185 | Resistant |
| **Padding (100 bytes)** | **0.4138** | **-0.5485** | **Vulnerable** |
| Mimicry (30%) | 0.6856 | -0.2767 | Vulnerable |
| Burst Shaping (20%) | 0.9433 | -0.0190 | Resistant |

### Deep Learning (MLP) Robustness
| Attack Type | Recall | Drop vs Baseline | Status |
| :--- | :--- | :--- | :--- |
| Timing Jitter (10%) | 0.9204 | 0.0000 | Resistant |
| **Padding (100 bytes)** | **0.9121** | **-0.0083** | **Resistant** |
| Mimicry (30%) | 0.8317 | -0.0886 | Vulnerable |
| Burst Shaping (20%) | 0.8528 | -0.0676 | Vulnerable |

[Image of adversarial machine learning attacks comparison chart]

## 4. Final Comparison & Conclusion
| Metric | Random Forest | Deep Learning |
| :--- | :--- | :--- |
| **Standard Recall** | **0.9623** | 0.9204 |
| **Avg. Attack Drop** | -14.43% | **-5.51%** |
| **Robustness**
