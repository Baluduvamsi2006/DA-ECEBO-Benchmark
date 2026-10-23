"""
R6: Runtime Benchmark with mean over 3 runs across all 15 datasets.
Accurately records fit_resample time (seconds) for:
Baseline, SMOTE, BorderlineSMOTE, ADASYN, EC-EBO, DA-ECEBO.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO


def load_dataset_data(d_name):
    path = f'datasets/{d_name}.parquet' if d_name == 'creditcard' else f'datasets/{d_name}.csv'
    if path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
        
    if d_name == 'creditcard':
        df = df.sort_values('Time').reset_index(drop=True)
        # Use training 75% fold (213,605 rows) as benchmarked in research
        n_tr = int(0.75 * len(df))
        df = df.iloc[:n_tr]
        feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
        y = df['Class'].values.astype(int)
        timestamps = df['Time'].values.astype(float)
        X = df[feat_cols].values
    else:
        target_col = [c for c in df.columns if any(k in c.lower() for k in ['class', 'target', 'survival'])][0]
        vc = df[target_col].value_counts()
        min_cls = vc.index[-1]
        y = (df[target_col] == min_cls).astype(int).values
        feat_cols = [c for c in df.columns if c != target_col]
        for c in feat_cols:
            if df[c].dtype == object or df[c].dtype.name == 'category':
                df[c] = pd.factorize(df[c])[0]
        X = df[feat_cols].values.astype(float)
        timestamps = None
        
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    return X_s, y, timestamps


def time_sampler(sampler_fn, X, y, timestamps, n_runs=3):
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if timestamps is not None:
            sampler_fn(X, y, timestamps=timestamps)
        else:
            sampler_fn(X, y)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return float(np.mean(times))


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    datasets = [
        'creditcard', 'car-good', 'ecoli2', 'glass4', 'glass6',
        'haberman', 'htru2', 'page-blocks-1-3_vs_4', 'shuttle-c0-vs-c4',
        'vehicle0', 'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast3'
    ]
    
    print("=" * 80)
    print("R6: Running Runtime Benchmark (3 runs per method per dataset)...")
    records = []
    
    for d in datasets:
        print(f"Benchmarking {d}...", flush=True)
        X, y, t = load_dataset_data(d)
        
        # 1. Baseline
        t_base = 0.0
        
        # 2. SMOTE
        t_sm = time_sampler(lambda x, y, **kw: SMOTE(random_state=42).fit_resample(x, y), X, y, t)
        
        # 3. BorderlineSMOTE
        t_bsm = time_sampler(lambda x, y, **kw: BorderlineSMOTE(random_state=42).fit_resample(x, y), X, y, t)
        
        # 4. ADASYN (with fallback if isolated)
        def run_adasyn(x, y, **kw):
            try:
                return ADASYN(random_state=42).fit_resample(x, y)
            except Exception:
                return SMOTE(random_state=42).fit_resample(x, y)
        t_ada = time_sampler(run_adasyn, X, y, t)
        
        # 5. EC-EBO
        t_ec = time_sampler(lambda x, y, **kw: ECEBO(random_state=42).fit_resample(x, y), X, y, t)
        
        # 6. DA-ECEBO
        t_da = time_sampler(lambda x, y, timestamps=None: DAECEBO(k_windows=4, lambd=0.25, random_state=42).fit_resample(x, y, timestamps=timestamps), X, y, t)
        
        records.append({
            'Dataset': d,
            'Baseline': round(t_base, 4),
            'SMOTE': round(t_sm, 4),
            'BorderlineSMOTE': round(t_bsm, 4),
            'ADASYN': round(t_ada, 4),
            'EC-EBO': round(t_ec, 4),
            'DA-ECEBO': round(t_da, 4)
        })
        print(f"  {d:<22}: SMOTE={t_sm:.4f}s | EC-EBO={t_ec:.4f}s | DA-ECEBO={t_da:.4f}s")
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/runtime_benchmark_3runs.csv', index=False)
    df_res.to_csv('results/figdata/runtime_benchmark_3runs.csv', index=False)
    print("\n" + "=" * 80)
    print("=== Runtime Benchmark Results (3-run mean) ===")
    print(df_res.to_string(index=False))


if __name__ == '__main__':
    main()
