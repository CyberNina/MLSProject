import os
import sys
import pandas as pd

# Setup path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.loader import load_and_process_data

# Dynamic paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "processed_traffic.csv")


def main():
    print("=" * 70)
    print("PHASE 1 - DATA LOADING, FILTERING AND FEATURE ENGINEERING")
    print("=" * 70)

    # ==========================================================================
    # NOTE: Compared to the previous version, main.py is much simpler.
    # All the logic for MAC filtering, windowing, and Delta_Time calculation
    # has been moved inside utils/loader.py (_load_single_file).
    # This file only handles orchestrating the loading process and saving
    # the final dataset to disk.
    #
    # The 4 Raspberry Pi MACs are defined in loader.py as a constant
    # KNOWN_RASPBERRY_MACS — they are no longer inferred with mode() from the data,
    # eliminating the risk of selecting the gateway's MAC instead of the
    # target device.
    # ==========================================================================

    # 1. Load, filter, compute features (handled in loader.py)
    try:
        df = load_and_process_data(DATA_DIR)
    except ValueError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        sys.exit(1)

    # 2. Check if the dataset is empty
    if df.empty:
        print("[CRITICAL ERROR] The final dataset is empty.")
        sys.exit(1)

    # 3. Verify expected columns
    expected_cols = {'Time_Mean', 'Time_Var', 'Length_Mean',
                     'Length_Min', 'Length_Max', 'Length_Var', 'Label'}
    missing = expected_cols - set(df.columns)
    if missing:
        print(f"[CRITICAL ERROR] Missing columns in the dataset: {missing}")
        sys.exit(1)

    # 4. Final report
    counts = df['Label'].value_counts()
    total = len(df)
    print(f"\n--- Final Dataset Summary ---")
    print(f"  Total windows:     {total:>8,}")
    print(f"  Benign    (0):     {counts.get(0, 0):>8,}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"  Malicious (1):     {counts.get(1, 0):>8,}  ({counts.get(1,0)/total*100:.1f}%)")
    print(f"  Features:          {[c for c in df.columns if c != 'Label']}")

    # 5. Save to disk
    print(f"\n>> Saving to: {OUTPUT_FILE}")
    df.to_csv(OUTPUT_FILE, index=False)
    print("[SUCCESS] Phase 1 completed successfully.")


if __name__ == "__main__":
    main()