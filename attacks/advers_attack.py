import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
import os

# Seed per riproducibilità
np.random.seed(42)
torch.manual_seed(42)

# --- CONFIGURAZIONE ---
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_FILE = os.path.join(BASE_DIR, "data", "holdout_dataset.csv")
RF_MODEL     = os.path.join(BASE_DIR, "models", "rf_model.pkl")
DL_MODEL     = os.path.join(BASE_DIR, "models", "mlp_model.pth")
SCALER_PATH  = os.path.join(BASE_DIR, "models", "scaler.pkl")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)


# --- ARCHITETTURA RETE NEURALE ---
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


# =============================================================================
# ATTACCHI AVVERSARI — NOTA METODOLOGICA
#
# Gli attacchi implementati sono perturbazioni euristiche ispirate a tecniche
# reali di evasione del malware (timing jitter, packet padding, mimicry,
# burst shaping). Non sono gradient-based attacks (FGSM, PGD) in quanto
# il modello opera su feature statistiche aggregate (finestre rolling) e
# non direttamente sui byte dei pacchetti — non esiste un gradiente
# differenziabile rispetto all'input grezzo di rete.
#
# Questo è il modello di minaccia corretto per un sistema di rilevamento
# basato su side-channel: l'attaccante modifica il comportamento del malware
# (timing, dimensioni pacchetti) senza conoscere i parametri del modello
# (black-box attack). I risultati misurano la robustezza del sistema in
# uno scenario realistico di evasione comportamentale.
# =============================================================================


def timing_jitter_attack(df, epsilon=0.05):
    """
    Attacco 1 — Timing Jitter.
    Aggiunge rumore gaussiano ai tempi medi e alla varianza temporale.
    Simula: sleep() randomizzato nel malware per spezzare la periodicità.
    Il rumore è scalato per-campione (non sulla media globale) per
    garantire perturbazioni proporzionali anche su valori estremi.
    """
    df_adv = df.copy()

    # Rumore proporzionale al valore di ogni singola riga (per-sample scaling)
    noise_mean = np.random.normal(0, epsilon, len(df)) * df['Time_Mean'].values
    noise_var  = np.abs(np.random.normal(0, epsilon, len(df))) * df['Time_Var'].values

    df_adv['Time_Mean'] = np.clip(df['Time_Mean'] + noise_mean, 1e-6, None)
    df_adv['Time_Var']  = np.clip(df['Time_Var']  + noise_var,  0,    None)
    return df_adv


def packet_padding_attack(df, max_padding=50):
    """
    Attacco 2 — Packet Padding.
    Aggiunge byte di padding ai pacchetti per alterarne le dimensioni.
    Simula: inserimento di byte spazzatura nel payload per confondere
    i classificatori basati sulla lunghezza dei pacchetti.
    Clip a 1500 byte (MTU Ethernet standard).
    """
    df_adv  = df.copy()
    padding = np.random.uniform(0, max_padding, len(df))

    df_adv['Length_Mean'] = np.clip(df['Length_Mean'] + padding,       0, 1500)
    df_adv['Length_Min']  = np.clip(df['Length_Min']  + padding * 0.5, 0, 1500)
    df_adv['Length_Max']  = np.clip(df['Length_Max']  + padding,       0, 1500)
    df_adv['Length_Var']  = np.clip(df['Length_Var']  + padding * 0.3, 0, None)
    return df_adv


def mimicry_attack(df, df_benign_stats, injection_rate=0.3):
    """
    Attacco 3 — Mimicry (mimetismo).
    Sposta le statistiche del traffico malicious verso quelle del traffico
    benign reale (calcolate dai dati, non hardcoded).
    Simula: il malware imita il comportamento del traffico legittimo
    per sfuggire al rilevamento comportamentale.

    df_benign_stats: dict con le medie delle feature calcolate sui dati
                     benign reali dell'holdout set.
    alpha=0.6: 60% malicious originale + 40% statistiche benign.
    """
    # reset_index garantisce che gli indici siano contigui 0..N-1
    # in modo che np.random.choice(len(df)) produca indici validi per .loc[]
    df_adv   = df.copy().reset_index(drop=True)
    n_inject = int(len(df_adv) * injection_rate)

    if n_inject == 0:
        return df_adv

    inject_idx = np.random.choice(len(df_adv), n_inject, replace=False)
    alpha      = 0.6

    for col in ['Time_Mean', 'Time_Var', 'Length_Mean', 'Length_Var']:
        if col in df_adv.columns and col in df_benign_stats:
            df_adv.loc[inject_idx, col] = (
                alpha * df_adv.loc[inject_idx, col] +
                (1 - alpha) * df_benign_stats[col]
            )
    return df_adv


def burst_shaping_attack(df, burst_prob=0.2, burst_factor=3.0):
    """
    Attacco 4 — Burst Shaping.
    Crea raffiche irregolari di pacchetti per nascondere la periodicità
    del mining (che tende ad avere timing regolare).
    Simula: invio a burst per rompere i pattern temporali rilevabili.
    Usa addizione invece di moltiplicazione per essere efficace anche
    quando la varianza di partenza è vicina a zero.
    """
    df_adv = df.copy().reset_index(drop=True)
    mask   = np.random.rand(len(df_adv)) < burst_prob

    # Addizione di un offset proporzionale al 75° percentile della colonna
    # per garantire effetto anche su valori quasi-zero
    time_offset   = df_adv['Time_Var'].quantile(0.75)  * (burst_factor - 1)
    length_offset = df_adv['Length_Var'].quantile(0.75) * (burst_factor - 1)

    df_adv.loc[mask, 'Time_Var']   = df_adv.loc[mask, 'Time_Var']   + time_offset
    df_adv.loc[mask, 'Length_Var'] = df_adv.loc[mask, 'Length_Var'] + length_offset
    return df_adv


# --- VALUTAZIONE ROBUSTEZZA ---

def evaluate_model_robustness(model_type, df_holdout, df_benign_stats):
    """
    Valuta la robustezza di un modello (RF o DL) sotto attacchi avversari.
    Misura il calo di Recall, Precision e F1 sulla classe malicious.
    Gli attacchi vengono applicati SOLO alle finestre malicious, poi
    il dataset viene ricostruito con i benign originali invariati.
    """
    print(f"\n{'='*70}")
    print(f"   TEST ROBUSTEZZA AVVERSARIALE — {model_type.upper()}")
    print(f"{'='*70}")

    # Verifica file
    for path, name in [(SCALER_PATH, "scaler.pkl"),
                       (RF_MODEL,    "rf_model.pkl"),
                       (DL_MODEL,    "mlp_model.pth")]:
        if not os.path.exists(path):
            print(f"[ERRORE] {name} non trovato. Esegui prima train_rf.py e train_dl.py.")
            return [], 0

    scaler = joblib.load(SCALER_PATH)

    # Funzione di predizione specifica per modello
    # Funzione di predizione specifica per modello
    if model_type == 'rf':
        clf = joblib.load(RF_MODEL)
        def predict(df_input):
            # Assicurati che ci sia .values qui
            X = scaler.transform(df_input.drop(columns=['Label']).values)
            return clf.predict(X)

    else:  # dl
        dummy_X  = df_holdout.drop(columns=['Label']).values
        dl_model = SimpleMLP(input_dim=dummy_X.shape[1])
        dl_model.load_state_dict(torch.load(DL_MODEL, weights_only=True))
        dl_model.eval()

        def predict(df_input):
            # Assicurati che ci sia .values qui
            X        = scaler.transform(df_input.drop(columns=['Label']).values)
            X_tensor = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                logits = dl_model(X_tensor)
                probs  = torch.sigmoid(logits).numpy().flatten()
            return (probs > 0.5).astype(int)

    # Separa benign e malicious
    df_benign    = df_holdout[df_holdout['Label'] == 0].copy()
    df_malicious = df_holdout[df_holdout['Label'] == 1].copy()

    print(f"\nHoldout: {len(df_holdout):,} finestre  "
          f"(benign={len(df_benign):,}, malicious={len(df_malicious):,})")

    # Baseline senza attacchi
    y_pred_base = predict(df_holdout)
    y_true      = df_holdout['Label'].values

    baseline_recall    = recall_score(y_true, y_pred_base)
    baseline_precision = precision_score(y_true, y_pred_base)
    baseline_f1        = f1_score(y_true, y_pred_base)

    print(f"\nBaseline — Recall: {baseline_recall:.4f} | "
          f"Precision: {baseline_precision:.4f} | F1: {baseline_f1:.4f}")

    # Lista attacchi
    attacks = [
        ("Timing Jitter  5%",  lambda d: timing_jitter_attack(d, 0.05)),
        ("Timing Jitter 10%",  lambda d: timing_jitter_attack(d, 0.10)),
        ("Timing Jitter 20%",  lambda d: timing_jitter_attack(d, 0.20)),
        ("Padding  30 bytes",  lambda d: packet_padding_attack(d, 30)),
        ("Padding  50 bytes",  lambda d: packet_padding_attack(d, 50)),
        ("Padding 100 bytes",  lambda d: packet_padding_attack(d, 100)),
        ("Mimicry       20%",  lambda d: mimicry_attack(d, df_benign_stats, 0.20)),
        ("Mimicry       30%",  lambda d: mimicry_attack(d, df_benign_stats, 0.30)),
        ("Burst Shaping 20%",  lambda d: burst_shaping_attack(d, 0.20, 3.0)),
        ("Burst Shaping 40%",  lambda d: burst_shaping_attack(d, 0.40, 4.0)),
    ]

    results = []
    print(f"\n{'Attacco':<22} {'Recall':>8} {'Drop':>8} {'F1':>8} {'Stato'}")
    print("-" * 65)

    for name, attack_fn in attacks:
        # Applica attacco solo alle finestre malicious
        df_mal_adv = attack_fn(df_malicious)

        # Ricostruisce dataset con benign originali + malicious perturbati
        df_test   = pd.concat([df_benign, df_mal_adv], ignore_index=True)
        y_true_adv = df_test['Label'].values
        y_pred_adv = predict(df_test)

        recall    = recall_score(y_true_adv, y_pred_adv)
        precision = precision_score(y_true_adv, y_pred_adv, zero_division=0)
        f1        = f1_score(y_true_adv, y_pred_adv, zero_division=0)
        drop      = baseline_recall - recall
        drop_pct  = (drop / baseline_recall * 100) if baseline_recall > 0 else 0

        if drop < 0.02:
            status = "✅ Resistente"
        elif drop < 0.05:
            status = "⚠️  Lieve calo"
        else:
            status = "❌ Vulnerabile"

        print(f"{name:<22} {recall:>8.4f} {drop:>+8.4f} {f1:>8.4f}  {status}")

        results.append({
            'attack':    name,
            'recall':    recall,
            'precision': precision,
            'f1':        f1,
            'drop':      drop,
            'drop_pct':  drop_pct,
        })

    avg_recall = np.mean([r['recall'] for r in results])
    avg_drop   = np.mean([r['drop']   for r in results])
    print(f"\nMedia — Recall: {avg_recall:.4f} | Drop medio: {avg_drop:+.4f} "
          f"({avg_drop/baseline_recall*100:.1f}% del baseline)")

    return results, baseline_recall


# --- MAIN ---

def main():
    print("=" * 70)
    print("   ADVERSARIAL ROBUSTNESS TEST — RF vs DL")
    print("=" * 70)

    # Verifica file dati
    if not os.path.exists(HOLDOUT_FILE):
        print(f"[ERRORE] {HOLDOUT_FILE} non trovato.")
        print("Esegui prima train_rf.py")
        return

    df_holdout = pd.read_csv(HOLDOUT_FILE)

    # ==========================================================================
    # Statistiche benign calcolate dai dati REALI dell'holdout.
    # Usate nell'attacco Mimicry per spostare le feature malicious
    # verso valori realistici di traffico benigno — non valori hardcoded.
    # ==========================================================================
    df_benign_hold = df_holdout[df_holdout['Label'] == 0]
    df_benign_stats = df_benign_hold[
        ['Time_Mean', 'Time_Var', 'Length_Mean', 'Length_Var']
    ].mean().to_dict()

    print(f"\nStatistiche benign reali (usate per Mimicry attack):")
    for k, v in df_benign_stats.items():
        print(f"  {k}: {v:.6f}")

    # Test RF
    rf_res, rf_base = evaluate_model_robustness('rf', df_holdout, df_benign_stats)
    if not rf_res:
        print("\n[ERRORE] Test RF fallito.")
        return

    # Test DL
    dl_res, dl_base = evaluate_model_robustness('dl', df_holdout, df_benign_stats)
    if not dl_res:
        print("\n[ERRORE] Test DL fallito.")
        return

    # --- CONFRONTO FINALE ---
    print("\n" + "=" * 70)
    print("   CONFRONTO ROBUSTEZZA: RF vs DL")
    print("=" * 70)

    rf_avg_drop = np.mean([r['drop'] for r in rf_res])
    dl_avg_drop = np.mean([r['drop'] for r in dl_res])

    print(f"\n{'Modello':<15} {'Baseline Recall':>16} {'Drop Medio':>12} {'Robustezza'}")
    print("-" * 60)
    rf_label = "✅ Più robusto" if rf_avg_drop <= dl_avg_drop else "❌ Meno robusto"
    dl_label = "✅ Più robusto" if dl_avg_drop <  rf_avg_drop else "❌ Meno robusto"
    print(f"{'Random Forest':<15} {rf_base:>16.4f} {rf_avg_drop:>+12.4f}  {rf_label}")
    print(f"{'Deep Learning':<15} {dl_base:>16.4f} {dl_avg_drop:>+12.4f}  {dl_label}")

    # --- GRAFICO COMPARATIVO ---
    attack_names = [r['attack'] for r in rf_res]
    rf_recalls   = [r['recall'] for r in rf_res]
    dl_recalls   = [r['recall'] for r in dl_res]

    x     = np.arange(len(attack_names))
    width = 0.35

    plt.figure(figsize=(16, 7))
    plt.bar(x - width/2, rf_recalls, width, label='Random Forest',
            color='#3498db', alpha=0.85, edgecolor='black', linewidth=0.5)
    plt.bar(x + width/2, dl_recalls, width, label='Deep Learning',
            color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)

    plt.axhline(y=rf_base, color='#3498db', linestyle='--', alpha=0.6,
                linewidth=2, label=f'RF Baseline ({rf_base:.2%})')
    plt.axhline(y=dl_base, color='#e74c3c', linestyle='--', alpha=0.6,
                linewidth=2, label=f'DL Baseline ({dl_base:.2%})')

    for i, (rf_v, dl_v) in enumerate(zip(rf_recalls, dl_recalls)):
        plt.text(i - width/2, rf_v + 0.005, f'{rf_v:.1%}',
                 ha='center', fontsize=7, color='#1a252f', weight='bold')
        plt.text(i + width/2, dl_v + 0.005, f'{dl_v:.1%}',
                 ha='center', fontsize=7, color='#1a252f', weight='bold')

    plt.xlabel('Attacco Avversariale', fontsize=12, weight='bold')
    plt.ylabel('Recall (Detection Rate)', fontsize=12, weight='bold')
    plt.title('Robustezza agli Attacchi Avversariali — RF vs DL\n'
              '(Black-box behavioral evasion attacks)',
              fontsize=13, weight='bold')
    plt.xticks(x, attack_names, rotation=40, ha='right', fontsize=9)
    plt.ylim([0.0, 1.08])
    plt.legend(fontsize=10, loc='lower left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()

    save_path = os.path.join(RESULTS_DIR, 'adversarial_robustness.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[✅ OK] Grafico salvato: {save_path}")

    # --- CONCLUSIONE ---
    print("\n" + "=" * 70)
    print("   CONCLUSIONE")
    print("=" * 70)

    if rf_avg_drop <= dl_avg_drop:
        diff_pct = ((dl_avg_drop - rf_avg_drop) / max(rf_avg_drop, 1e-9)) * 100
        print(f"\n[+] Random Forest è più robusto del Deep Learning")
        print(f"    Drop medio RF: {rf_avg_drop:+.4f} vs DL: {dl_avg_drop:+.4f}")
        print(f"\n[INFO] Spiegazione:")
        print(f"    RF è un ensemble di alberi con decision boundary discreti.")
        print(f"    Piccole perturbazioni delle feature non attraversano le soglie")
        print(f"    di split, rendendo RF naturalmente robusto a black-box attacks.")
    else:
        diff_pct = ((rf_avg_drop - dl_avg_drop) / max(dl_avg_drop, 1e-9)) * 100
        print(f"\n[+] Deep Learning è più robusto del Random Forest")
        print(f"    Drop medio DL: {dl_avg_drop:+.4f} vs RF: {rf_avg_drop:+.4f}")
        print(f"\n[INFO] Spiegazione:")
        print(f"    L'MLP apprende rappresentazioni distribuite delle feature.")
        print(f"    La generalizzazione su combinazioni di feature non viste")
        print(f"    può renderlo più robusto a perturbazioni locali singole.")

    print(f"\n[✅ OK] Test robustezza completato.")


if __name__ == "__main__":
    main()