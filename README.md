# 🛡️ Cryptojacking Detection via Side-Channel Analysis
**Studio Comparativo sulla Robustezza: Random Forest vs. Deep Learning**

Questo repository contiene il framework completo sviluppato per la tesi di laurea in Machine Learning. Il progetto si focalizza sul rilevamento di attacchi di **Cryptojacking** (XMRig mining) su dispositivi IoT (Raspberry Pi 4) analizzando esclusivamente metadati di rete, garantendo l'efficacia del monitoraggio anche su traffico cifrato.



## 📋 Panoramica del Progetto
Il monitoraggio tradizionale basato su firma (Signature-based) fallisce di fronte a malware che utilizzano protocolli cifrati o tecniche di offuscamento. Questo studio propone un approccio **comportamentale** basato su:
- **Feature Engineering**: Finestre rolling di 10 pacchetti per estrarre medie e varianze di Inter-Arrival Time (IAT) e Packet Length.
- **Side-Channel Analysis**: Esclusione deliberata di IP, porte e payload per testare la robustezza del modello in condizioni di "cecità" protocollare.

## 🚀 Struttura della Pipeline (`run_all.py`)
La pipeline automatizzata esegue sequenzialmente:
1. **Pre-processing**: Caricamento dati e pulizia MAC address.
2. **Baseline Training**: Addestramento Random Forest (RF) come benchmark.
3. **Neural Training**: Addestramento di un Multi-Layer Perceptron (MLP) con Early Stopping.
4. **Real-time Simulation**: Test comparativo su Holdout Set (10% dei dati mai visti).
5. **Adversarial Attack**: Stress-test con 4 tipologie di attacchi di evasione (Padding, Jitter, Mimicry, Burst Shaping).
6. **Visualizzazione**: Generazione automatica di matrici di confusione, curve ROC e grafici di robustezza.

## 📊 Risultati e Conclusioni
Dalle analisi condotte, emergono differenze fondamentali nell'approccio alla sicurezza:

| Metrica (Malware) | Random Forest (Baseline) | Deep Learning (MLP) |
|-------------------|--------------------------|----------------------|
| **Recall (Base)** | ~96.2%                   | ~91.8%               |
| **Resilienza** | ❌ Vulnerabile (Drop 15%) | ✅ Robusto (Drop 3%)  |
| **AUC-ROC** | 0.9946                   | 0.9905               |

### Il "Trade-off" della Sicurezza
Mentre il **Random Forest** appare superiore in condizioni statiche, lo studio dimostra che è estremamente fragile di fronte a tecniche di evasione come il *Packet Padding* (dove la Recall crolla al 41%). Il **Deep Learning**, grazie alla capacità di apprendere rappresentazioni distribuite, mantiene una Recall superiore al 90% anche sotto attacco, dimostrandosi la scelta più affidabile per un deployment in ambienti ostili.



## 🛠️ Requisiti e Installazione
```bash
# Installa le dipendenze
pip install -r requirements.txt

# Esegui l'intero studio
python3 run_all.py