# Cryptojacking Detection via Side-Channel Analysis
**Robustness Comparison: Random Forest vs. Deep Learning**

This repository contains a Machine Learning framework developed to detect **Cryptojacking** on IoT devices (Raspberry Pi 4). By analyzing solely network metadata (Side-Channel Analysis) and behavioral patterns, this approach effectively identifies malicious activity even in encrypted traffic.

## Overview
Traditional signature-based detection fails against encrypted or obfuscated malware. This study proposes a **behavioral approach** using:
* **Feature Engineering:** 10-packet rolling windows extracting mean and variance of Inter-Arrival Time (IAT) and Packet Length.
* **Protocol Blindness:** Deliberate exclusion of IPs, ports, and payloads to ensure robust side-channel evaluation.

## Automated Pipeline
Running `run_all.py` sequentially executes:
1.  **Preprocessing:** Data loading and MAC address filtering.
2.  **Training:** Trains both a Baseline Random Forest (RF) and a Multi-Layer Perceptron (MLP) with early stopping.
3.  **Evaluation:** Tests models on a 10% unseen Holdout Set.
4.  **Adversarial Stress-Test:** Evaluates model resilience against 4 evasion attacks (Padding, Jitter, Mimicry, Burst Shaping).
5.  **Visualization:** Automatically generates ROC curves, confusion matrices, and robustness plots.

## Key Findings

| Metric | Random Forest (Baseline) | Deep Learning (MLP) |
| :--- | :--- | :--- |
| **Base Recall** | ~96.2% | ~91.8% |
| **Under Attack** |  Vulnerable |  Robust |
| **AUC-ROC** | 0.9946 | 0.9905 |

**The Security Trade-off:** While **Random Forest** excels in static, attack-free conditions, it is extremely fragile against evasion tactics (e.g., recall plummets to 41% under *Packet Padding*). Conversely, the **Deep Learning** model maintains a recall of over 90% even under severe adversarial attacks, making it the most reliable choice for real-world, hostile deployments.

## To Start it

```bash
# Install dependencies
pip install -r requirements.txt

# Run the entire pipeline
python3 run_all.py
