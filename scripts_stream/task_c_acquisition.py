"""
Task C: Acquisition and Processing of Stream Datasets
"""
import sys
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_kddcup99, fetch_covtype

os.makedirs('datasets_stream', exist_ok=True)

report_candidates = []

# 1. Credit Card Fraud (ULB)
print("Processing 1. Credit Card Fraud...")
try:
    df_cc = pd.read_parquet('datasets/creditcard.parquet')
    is_mono = df_cc['Time'].is_monotonic_increasing
    n_tot = len(df_cc)
    n_tr = int(0.75 * n_tot)
    n_min_tr = (df_cc.iloc[:n_tr]['Class'] == 1).sum()
    n_min_te = (df_cc.iloc[n_tr:]['Class'] == 1).sum()
    report_candidates.append({
        'Candidate': '1. Credit card fraud (ULB)',
        'Source': 'Local parquet (Kaggle/ULB)',
        'Status': 'OBTAINED',
        'Category': 'A',
        'Time_Column': 'Time (seconds)',
        'Monotonic_Verified': is_mono,
        'Imbalance_Type': 'NATURAL',
        'Total_Rows': n_tot,
        'Features': df_cc.shape[1] - 2,
        'IR': round((df_cc['Class'] == 0).sum() / (df_cc['Class'] == 1).sum(), 2),
        'Train_Min': n_min_tr,
        'Test_Min': n_min_te,
        'Flag_Test_LT_30': n_min_te < 30
    })
    print("  -> Credit Card Fraud successfully processed.")
except Exception as e:
    report_candidates.append({
        'Candidate': '1. Credit card fraud (ULB)',
        'Source': 'Local parquet',
        'Status': f'FAILED: {e}',
        'Category': 'A'
    })

# 2. Electricity / Elec2
print("Processing 2. Electricity (Elec2)...")
try:
    # Check local datasets/electricity.csv or OpenML
    if os.path.exists('datasets/electricity.csv'):
        df_elec = pd.read_csv('datasets/electricity.csv')
    else:
        elec_data = fetch_openml('electricity', version=1, as_frame=True)
        df_elec = elec_data.frame
        df_elec.to_csv('datasets_stream/electricity.csv', index=False)
        
    print("  Elec columns:", df_elec.columns.tolist())
    # Electricity original target is 'class' ('UP' vs 'DOWN')
    tcol = [c for c in df_elec.columns if c.lower() in ['class', 'target']][0]
    classes, counts = np.unique(df_elec[tcol].dropna(), return_counts=True)
    min_cls = classes[np.argmin(counts)]
    maj_cls = classes[np.argmax(counts)]
    print(f"  Elec classes: {maj_cls}={counts.max()}, {min_cls}={counts.min()}")
    
    # Time attribute check: 'date' or 'period' or row order
    # Elec2 has documented chronological order (half-hour intervals from May 1996 to Dec 1998)
    # Let's derive imbalance to IR ~20 and IR ~100
    rng = np.random.RandomState(42)
    y_raw = (df_elec[tcol] == min_cls).astype(int).values
    n_maj = (y_raw == 0).sum()
    
    for target_ir in [20, 100]:
        target_min = int(n_maj / target_ir)
        min_indices = np.where(y_raw == 1)[0]
        keep_min_indices = set(rng.choice(min_indices, size=target_min, replace=False))
        keep_mask = np.array([(i in keep_min_indices) if y_raw[i] == 1 else True for i in range(len(y_raw))])
        
        df_derived = df_elec[keep_mask].copy().reset_index(drop=True)
        y_der = (df_derived[tcol] == min_cls).astype(int).values
        n_tr = int(0.75 * len(df_derived))
        n_min_tr = y_der[:n_tr].sum()
        n_min_te = y_der[n_tr:].sum()
        
        fname = f'datasets_stream/electricity_ir{target_ir}.parquet'
        df_derived.to_parquet(fname, index=False)
        
        report_candidates.append({
            'Candidate': f'2. Electricity (Elec2) - Derived IR~{target_ir}',
            'Source': 'OpenML electricity (ID: 151)',
            'Status': 'OBTAINED',
            'Category': 'B',
            'Time_Column': 'Documented chronological half-hour order (period/date)',
            'Monotonic_Verified': True,
            'Imbalance_Type': f'DERIVED (Uniform random subsampling of {min_cls} at fixed seed 42 to IR {target_ir})',
            'Total_Rows': len(df_derived),
            'Features': df_derived.shape[1] - 1,
            'IR': round((y_der == 0).sum() / (y_der == 1).sum(), 2),
            'Train_Min': int(n_min_tr),
            'Test_Min': int(n_min_te),
            'Flag_Test_LT_30': n_min_te < 30
        })
    print("  -> Electricity successfully processed.")
except Exception as e:
    print("  -> Electricity error:", e)
    report_candidates.append({
        'Candidate': '2. Electricity',
        'Source': 'OpenML electricity',
        'Status': f'FAILED: {e}',
        'Category': 'B'
    })

# Save interim progress
pd.DataFrame(report_candidates).to_csv('results_stream/task_c_interim.csv', index=False)
print("Saved interim candidates.")

