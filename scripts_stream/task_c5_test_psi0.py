"""
Task C5: Unit Test of DAECEBO2_PSI0 vs ECEBO_PSI0 and Literal-Rule Audit
========================================================================
Outputs: results_stream/C5_psi0_unit_test.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.abspath('.'))
from ecebo_psi0_variant import ECEBO_PSI0
from da_ecebo_v2_candidate import DAECEBO2
from da_ecebo_v2_psi0 import DAECEBO2_PSI0

def load_test_dataset(name):
    if name == 'Creditcard_5k':
        df = pd.read_parquet('datasets/creditcard.parquet')
        frauds = df[df['Class'] == 1]
        non_frauds = df[df['Class'] == 0].sample(n=5000 - len(frauds), random_state=42)
        df_sub = pd.concat([frauds, non_frauds]).sort_index()
        X = df_sub.drop(columns=['Class']).values.astype(np.float64)
        y = df_sub['Class'].values.astype(int)
        return X, y
    else:
        path = os.path.join('datasets', f'{name}.csv')
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values.astype(np.float64)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        return X, y

datasets = ['Ecoli2', 'Glass6', 'Haberman', 'Creditcard_5k']
seeds = [42, 43, 44, 45, 46]

results = []

for dname in datasets:
    print(f"Testing {dname}...")
    X, y = load_test_dataset(dname)
    n_orig = len(X)
    
    for seed in seeds:
        # 1. ECEBO_PSI0
        m_ecebo_psi0 = ECEBO_PSI0(iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=seed)
        X_res_ec, y_res_ec = m_ecebo_psi0.fit_resample(X, y)
        syn_ec = X_res_ec[n_orig:]
        
        # 2. DAECEBO2_PSI0 with k_windows=1, lambd=1.0, delta_base=0.0
        m_da_psi0 = DAECEBO2_PSI0(k_windows=1, lambd=1.0, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=seed)
        X_res_da_psi0, y_res_da_psi0 = m_da_psi0.fit_resample(X, y)
        syn_da_psi0 = X_res_da_psi0[n_orig:]
        
        is_array_equal = bool(np.array_equal(syn_ec, syn_da_psi0))
        max_diff = float(np.max(np.abs(syn_ec - syn_da_psi0)))
        
        # 3. Literal-rule DAECEBO2 (with psi < 0 rule)
        m_da_lit = DAECEBO2(k_windows=1, lambd=1.0, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=seed)
        X_res_da_lit, y_res_da_lit = m_da_lit.fit_resample(X, y)
        syn_da_lit = X_res_da_lit[n_orig:]
        
        # Duplicate share for literal rule
        X_min_orig = X[y == 1]
        nn_real = NearestNeighbors(n_neighbors=1).fit(X_min_orig)
        dists_lit, _ = nn_real.kneighbors(syn_da_lit)
        exact_dup_share_lit = float(np.mean(dists_lit.ravel() < 1e-9))
        
        # Duplicate share for PSI0 rule
        dists_psi0, _ = nn_real.kneighbors(syn_da_psi0)
        exact_dup_share_psi0 = float(np.mean(dists_psi0.ravel() < 1e-9))
        
        status_unit_test = "PASS" if is_array_equal else "FAIL"
        
        print(f"  Seed {seed}: Unit Test = {status_unit_test}, MaxDiff = {max_diff:.2e}, Literal Dup Share = {exact_dup_share_lit:.4f}, PSI0 Dup Share = {exact_dup_share_psi0:.4f}")
        
        results.append({
            "Dataset": dname,
            "Seed": seed,
            "DAECEBO2_PSI0_vs_ECEBO_PSI0_Equal": is_array_equal,
            "Max_Abs_Diff": max_diff,
            "Unit_Test_Status": status_unit_test,
            "Literal_DAECEBO2_Dup_Share": exact_dup_share_lit,
            "PSI0_DAECEBO2_Dup_Share": exact_dup_share_psi0,
        })

df_res = pd.DataFrame(results)
os.makedirs("results_stream", exist_ok=True)
out_csv = "results_stream/C5_psi0_unit_test.csv"
df_res.to_csv(out_csv, index=False)
print(f"\nSaved C5 unit test results to {out_csv}")

