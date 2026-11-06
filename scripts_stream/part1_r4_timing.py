"""
Part 1: R4. Re-timed Timing Benchmark (Inside Worker Around fit_resample Only)
=============================================================================
Datasets:
- synth_sea_p5
- synth_hyperplane_p5
- synth_random_rbf_p5
- electricity_ir20
- electricity_ir100
- creditcard
- ecoli2
- yeast3
(Datasets with minority > 5,000 are marked as "too large" and skipped)

Methods:
None, SMOTE, BorderlineSMOTE, ADASYN, G-SMOTE, CCR, A-SUWO, MLOS (majority capped at 20,000),
ECEBO, ECEBO_PSI0, DAECEBO2, DAECEBO2_PSI0.

Output: results_stream/C7_timing_v2.csv
"""
import os
import sys
import subprocess
import pandas as pd
import numpy as np

def main():
    large_datasets = [
        ('ieee_cis', 590540, 20663),
        ('ieee_cis_45', 590540, 20663),
        ('kddcup99_rare', 494021, 5285),
        ('kddcup99_binary', 494021, 107201),
        ('covertype_binary', 581012, 9493),
        ('poker_binary', 829201, 17541),
        ('airlines_ir20', 314074, 14955),
        ('airlines_ir100', 314074, 14955),
    ]

    benchmark_datasets = [
        ('synth_sea_p5', 7835, 391),
        ('synth_hyperplane_p5', 7835, 391),
        ('synth_random_rbf_p5', 7835, 391),
        ('electricity_ir20', 45312, 1907),
        ('electricity_ir100', 45312, 381),
        ('ecoli2', 336, 52),
        ('yeast3', 1484, 163),
        ('creditcard', 284807, 492),
    ]

    methods = [
        'None', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'G-SMOTE', 'CCR', 'A-SUWO',
        'MLOS', 'ECEBO', 'ECEBO_PSI0', 'DAECEBO2', 'DAECEBO2_PSI0'
    ]

    out_csv = 'results_stream/C7_timing_v2.csv'
    os.makedirs('results_stream', exist_ok=True)
    
    # Initialize CSV if not present
    if not os.path.exists(out_csv):
        df_init = pd.DataFrame(columns=['dataset', 'method', 'sampler_seconds', 'import_seconds', 'timed_out'])
        df_init.to_csv(out_csv, index=False)

    # 1. Log large datasets skipped
    for dname, n_rows, n_min in large_datasets:
        row = {
            'dataset': dname,
            'method': 'ALL_METHODS',
            'sampler_seconds': 0.0,
            'import_seconds': 0.0,
            'timed_out': 'too large'
        }
        df_curr = pd.read_csv(out_csv)
        if not ((df_curr['dataset'] == dname) & (df_curr['method'] == 'ALL_METHODS')).any():
            pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
            print(f"Logged {dname}: too large (N_min={n_min} > 5000)")

    # 2. Benchmark runs with 300s timeout per run
    for dname, n_rows, n_min in benchmark_datasets:
        for mname in methods:
            df_curr = pd.read_csv(out_csv)
            if ((df_curr['dataset'] == dname) & (df_curr['method'] == mname)).any():
                print(f"Skipping already recorded {dname} - {mname}")
                continue

            print(f"Running timing for {dname} | {mname}...", flush=True)
            cmd = [sys.executable, 'scripts_stream/part1_r4_worker.py', dname, mname]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                out = proc.stdout.strip()
                err = proc.stderr.strip()
                
                # Look for RESULT line
                res_line = [l for l in out.split('\n') if l.startswith('RESULT:')]
                if res_line:
                    parts = res_line[-1].split(':')
                    s_sec = float(parts[1])
                    i_sec = float(parts[2])
                    status = 'NO'
                else:
                    s_sec = 0.0
                    i_sec = 0.0
                    status = f"ERROR: {err[:50]}"
            except subprocess.TimeoutExpired:
                s_sec = 300.0
                i_sec = 0.0
                status = 'YES (Timeout > 300s)'

            row = {
                'dataset': dname,
                'method': mname,
                'sampler_seconds': round(s_sec, 4),
                'import_seconds': round(i_sec, 4),
                'timed_out': status
            }
            pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
            print(f"  -> {dname} | {mname}: Sampler {s_sec:.4f}s, Import {i_sec:.4f}s, TimedOut: {status}", flush=True)

    print(f"\nAll timing runs finished and saved to {out_csv}")

if __name__ == '__main__':
    main()

