"""
Task A4: Verification of DAECEBO2 (Candidate Fix) against ECEBO
"""
import sys
import os
import difflib
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd

from da_ecebo_and_ecebo_implementation import ECEBO
from da_ecebo_v2_candidate import DAECEBO2
from scripts_stream.task_a_equivalence import get_datasets

def print_diff():
    # Compare lines 380-392 of DAECEBO with revised section of DAECEBO2
    with open('da_ecebo_and_ecebo_implementation.py', 'r', encoding='utf-8') as f:
        orig_lines = f.readlines()[379:393]
    with open('da_ecebo_v2_candidate.py', 'r', encoding='utf-8') as f:
        cand_lines = f.readlines()[156:173]
        
    diff = difflib.unified_diff(
        orig_lines, cand_lines,
        fromfile='da_ecebo_and_ecebo_implementation.py (DAECEBO lines 380-392)',
        tofile='da_ecebo_v2_candidate.py (DAECEBO2 scoring lines)',
        lineterm=''
    )
    print("\n" + "=" * 80)
    print("DIFF: DAECEBO vs DAECEBO2 (CLUSTER SCORING & NORMALIZATION)")
    print("=" * 80)
    for line in diff:
        print(line)
    print("=" * 80 + "\n")

def main():
    print_diff()
    datasets = get_datasets()
    seeds = [42, 43, 44, 45, 46]
    
    results = []
    print("RUNNING TASK A4 VERIFICATION...")
    for d_name, (X, y) in datasets.items():
        print(f"\n--- Dataset: {d_name} (Shape: {X.shape}) ---")
        for seed in seeds:
            ec = ECEBO(random_state=seed)
            da2_base = DAECEBO2(k_windows=1, lambd=1.0, delta_base=0.0, random_state=seed)
            da2_delta = DAECEBO2(k_windows=1, lambd=1.0, delta_base=0.05, random_state=seed)
            
            X_ec, y_ec = ec.fit_resample(X, y)
            X_da2, y_da2 = da2_base.fit_resample(X, y)
            X_da2_d, y_da2_d = da2_delta.fit_resample(X, y)
            
            syn_ec = X_ec[len(X):]
            syn_da2 = X_da2[len(X):]
            syn_da2_d = X_da2_d[len(X):]
            
            # Quota checks: how many samples per cluster were generated?
            # Check array_equal of synthetic samples
            samples_equal_base = np.array_equal(syn_ec, syn_da2)
            samples_equal_delta = np.array_equal(syn_ec, syn_da2_d)
            
            # Max absolute difference
            max_diff_base = float(np.max(np.abs(syn_ec - syn_da2))) if not samples_equal_base else 0.0
            max_diff_delta = float(np.max(np.abs(syn_ec - syn_da2_d))) if not samples_equal_delta else 0.0
            
            status_base = "PASS" if samples_equal_base else "FAIL"
            status_delta = "PASS" if samples_equal_delta else "DIFFERS"
            
            print(f"Seed {seed}: DAECEBO2 (delta=0.0) vs ECEBO: Equal={samples_equal_base} => {status_base} | "
                  f"(delta=0.05) vs ECEBO: Equal={samples_equal_delta} => {status_delta}")
            
            results.append({
                'Dataset': d_name,
                'Seed': seed,
                'A4_Base_Samples_Equal': samples_equal_base,
                'A4_Base_Max_Diff': max_diff_base,
                'A4_Base_Status': status_base,
                'A4_Delta005_Samples_Equal': samples_equal_delta,
                'A4_Delta005_Max_Diff': max_diff_delta,
                'A4_Delta005_Status': status_delta
            })
            
    df_res = pd.DataFrame(results)
    out_path = 'results_stream/A4_equivalence_v2.csv'
    df_res.to_csv(out_path, index=False)
    print(f"\nSaved Task A4 results to {out_path}")

if __name__ == '__main__':
    main()

