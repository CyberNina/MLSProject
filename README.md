# Cryptojacking Detection in IoT Networks via Side-Channel Analysis

This repository contains a Machine Learning framework developed to detect **Cryptojacking** on IoT devices (specifically Raspberry Pi 4). By analyzing solely network metadata (**Side-Channel Analysis**) and behavioral patterns, this approach effectively identifies malicious activity even in encrypted or obfuscated traffic.

## Overview
Traditional signature-based detection fails against modern malware. This study proposes a **behavioral approach** using:
* **Feature Engineering:** 10-packet rolling windows extracting mean and variance of Inter-Arrival Time (IAT) and Packet Length.
* **Protocol Blindness:** Deliberate exclusion of IPs, ports, and payloads to ensure a robust side-channel evaluation independent of application-layer encryption.

## Dataset and Methodology
The analysis was performed on a dataset of **625,287 traffic windows**.

* **Windowing:** Data processed using 10-packet rolling windows to capture temporal rhythms.
* **Class Distribution:** Highly imbalanced (89.7% Benign, 10.3% Malicious), reflecting real-world network conditions.
* **Features:** Six key metrics: `Time_Mean`, `Time_Var`, `Length_Mean`, `Length_Min`, `Length_Max`, `Length_Var`.

## Automated Pipeline
Running `run_all.py` sequentially executes the following phases:
1.  **Preprocessing:** Data loading, MAC address filtering, and feature extraction.
2.  **Training:** Trains both a **Baseline Random Forest (RF)** and a **Multi-Layer Perceptron (MLP)** with early stopping and class balancing.
3.  **Evaluation:** Models are tested on a 10% unseen Holdout Set to ensure generalization.
4.  **Adversarial Stress-Test:** Evaluates resilience against 4 evasion attacks: **Padding**, **Jitter**, **Mimicry**, and **Burst Shaping**.
5.  **Visualization:** Automatically generates ROC curves, confusion matrices, and robustness plots.

## Key Findings: The Security Trade-off
The experimental results highlight a critical trade-off between absolute accuracy and adversarial resilience:

| Metric | Random Forest (Baseline) | Deep Learning (MLP) |
| :--- | :--- | :--- |
| **Base Recall** | **~96.2%** | ~91.8% |
| **AUC-ROC** | **0.9946** | 0.9905 |
| **Under Attack** | **Vulnerable** | **Robust** |

**Observations:**
* **Random Forest** excels in static, attack-free conditions with very low false positives. However, it is extremely fragile against evasion tactics; for example, recall plummets to **41%** under a Packet Padding attack.
* **Deep Learning (MLP)** maintains a recall of over **90%** even under severe adversarial attacks. Its ability to learn distributed feature representations makes it the most reliable choice for real-world, hostile deployments.



## To Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the entire pipeline
python3 run_all.py
