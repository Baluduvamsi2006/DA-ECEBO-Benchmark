import os
import sys
import time
import subprocess
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

out_csv = os.path.join('results_stream', 'C7_timing.csv')
os.makedirs('results_stream', exist_ok=True)

# Write header
with open(out_csv, 'w', encoding='utf-8') as f:
    f.write('dataset,n,minority_count,method,seconds,timed_out\n')

def log_row(dname, n, min_cnt, mname, secs, tout):
    row_str = f"{dname},{n},{min_cnt},{mname},{secs:.4f},{tout}\n"
    with open(out_csv, 'a', encoding='utf-8') as f:
        f.write(row_str)
        f.flush()
    print(f"ROW: {dname:20s} | N={n:6d} | Min={min_cnt:5d} | {mname:16s} | {secs:8.2f}s | TimedOut={tout}")

# 1. Log skipped datasets (> 5,000 minority samples)
skipped = [
    ('ieee_cis', 590540, 20663),
    ('ieee_cis_45', 590540, 20663),
    ('kddcup99_rare', 494021, 5285),
    ('kddcup99_binary', 494021, 107201),
    ('covertype_binary', 581012, 9493),
    ('poker_binary', 829201, 17541),
    ('airlines_ir20', 314074, 14955),
]

print("=== LOGGING SKIPPED DATASETS (Minority > 5,000: 'too large') ===")
for dname, n, min_cnt in skipped:
    log_row(dname, n, min_cnt, 'ALL_METHODS', 0.0, 'too large')

# 2. Datasets to run (Minority <= 5,000)
datasets_to_run = [
    ('synth_sea_p5', 7835, 391),
    ('synth_hyperplane_p5', 10350, 517),
    ('synth_random_rbf_p5', 6837, 341),
    ('electricity_ir100', 26335, 260),
    ('electricity_ir20', 27378, 1303),
    ('creditcard', 284807, 492),
]

methods = [
    'None',
    'SMOTE',
    'BorderlineSMOTE',
    'ADASYN',
    'G-SMOTE',
    'CCR',
    'A-SUWO',
    'MLOS',
    'ECEBO',
    'ECEBO_PSI0',
    'DAECEBO2',
    'DAECEBO2_PSI0'
]

print("\n=== STARTING TIMED RUNS (5-MINUTE TIMEOUT PER CALL) ===")

TIMEOUT_SECS = 300.0

for dname, n, min_cnt in datasets_to_run:
    print(f"\n--- Dataset: {dname} (N={n}, Minority={min_cnt}) ---")
    for mname in methods:
        t0 = time.time()
        try:
            cmd = [sys.executable, 'scripts_stream/task_c7_worker.py', dname, mname]
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECS)
            elapsed = time.time() - t0
            
            # parse output
            out = p.stdout.strip()
            secs = elapsed
            tout = 'NO'
            
            for line in out.splitlines():
                if line.startswith('RESULT:'):
                    parts = line.split(':')
                    secs = float(parts[1])
                    status = parts[2]
                    if status != 'SUCCESS':
                        tout = f'ERROR ({status})'
                    break
                    
            log_row(dname, n, min_cnt, mname, secs, tout)
            
        except subprocess.TimeoutExpired:
            log_row(dname, n, min_cnt, mname, TIMEOUT_SECS, 'YES')
        except Exception as e:
            elapsed = time.time() - t0
            log_row(dname, n, min_cnt, mname, elapsed, f'ERROR ({str(e)[:30]})')

print("\n=== C7 REDUCED PILOT COMPLETE ===")

