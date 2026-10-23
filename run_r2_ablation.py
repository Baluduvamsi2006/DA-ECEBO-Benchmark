"""
R2: Multi-Seed Ablation Study (10 seeds, mean +/- std, paired Wilcoxon).
Variants:
(i) ECC uniform (uniform quotas across clusters)
(ii) ECC + entropy (= EC-EBO)
(iii) ECC + drift alloc only (lambda = 0.0)
(iv) Full DA-ECEBO (both K=4, lambda=0.25 and K=2, lambda=0.0)
(v) Full with p_drift = 0 (no spatial drift perturbation)
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import compute_metrics
from run_p5_experiments import ECCAblation


def eval_ablation_task(task):
    s, mode, k_win, lam, p_drift, X_tr_s, y_tr, t_tr, X_te_s, y_te = task
    sampler = ECCAblation(mode=mode, k_windows=k_win, lambd=lam, p_drift=p_drift, random_state=s)
    X_res, y_res = sampler.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    probs = clf.predict_proba(X_te_s)[:, 1]
    preds = clf.predict(X_te_s)
    m = compute_metrics(y_te, probs, preds)
    m['Seed'] = s
    m['Mode'] = mode
    m['K_windows'] = k_win
    m['lambda'] = lam
    m['p_drift'] = p_drift
    variant_name = {
        ('ecc_only', 4, 0.25, 0.5): 'ECC Uniform',
        ('ec_ebo', 4, 0.25, 0.5): 'ECC + Entropy (EC-EBO)',
        ('drift_only', 4, 0.0, 0.5): 'ECC + Drift Alloc Only (lambda=0)',
        ('full_da_ecebo', 4, 0.25, 0.5): 'Full DA-ECEBO (K=4, lam=0.25)',
        ('full_da_ecebo', 2, 0.0, 0.5): 'Full DA-ECEBO (K=2, lam=0.0)',
        ('no_drift_steps', 4, 0.25, 0.0): 'Full with p_drift=0'
    }.get((mode, k_win, lam, p_drift), f"{mode}_K{k_win}_L{lam}")
    m['Variant'] = variant_name
    return m


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    print("=" * 80)
    print("R2: Loading CreditCard dataset & preparing chronological 75/25 split...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n_total = len(df)
    tr_sz = int(0.75 * n_total)
    train_df, test_df = df.iloc[:tr_sz], df.iloc[tr_sz:]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = train_df['Class'].values.astype(int)
    t_tr = train_df['Time'].values.astype(float)
    X_te = test_df[feat_cols].values
    y_te = test_df['Class'].values.astype(int)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    seeds = list(range(42, 52))
    variants = [
        ('ecc_only', 4, 0.25, 0.5),
        ('ec_ebo', 4, 0.25, 0.5),
        ('drift_only', 4, 0.0, 0.5),
        ('full_da_ecebo', 4, 0.25, 0.5),
        ('full_da_ecebo', 2, 0.0, 0.5),
        ('no_drift_steps', 4, 0.25, 0.0)
    ]
    
    tasks = []
    for s in seeds:
        for mode, k_win, lam, p_drift in variants:
            tasks.append((s, mode, k_win, lam, p_drift, X_tr_s, y_tr, t_tr, X_te_s, y_te))
            
    print(f"Total ablation tasks: {len(tasks)} (6 variants x 10 seeds)... Running parallel...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        raw_results = list(executor.map(eval_ablation_task, tasks))
    print(f"Multi-seed ablation finished in {time.time() - t0:.2f}s")
    
    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv('results/ablation_multiseed_raw.csv', index=False)
    
    # Summarize mean +/- std
    variant_order = [
        'ECC Uniform',
        'ECC + Entropy (EC-EBO)',
        'ECC + Drift Alloc Only (lambda=0)',
        'Full DA-ECEBO (K=4, lam=0.25)',
        'Full DA-ECEBO (K=2, lam=0.0)',
        'Full with p_drift=0'
    ]
    
    summary = []
    for v in variant_order:
        sub = df_raw[df_raw['Variant'] == v].sort_values('Seed').reset_index(drop=True)
        summary.append({
            'Variant': v,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-F1': f"{sub['Macro-F1'].mean():.4f} +/- {sub['Macro-F1'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}",
            'Raw_F1_mean': sub['F-measure'].mean(),
            'Raw_F1_std': sub['F-measure'].std()
        })
        
    df_sum = pd.DataFrame(summary)
    
    # Paired Wilcoxon signed-rank tests vs Full DA-ECEBO (K=4, lam=0.25)
    full_f1_k4 = df_raw[df_raw['Variant'] == 'Full DA-ECEBO (K=4, lam=0.25)'].sort_values('Seed')['F-measure'].values
    full_f1_k2 = df_raw[df_raw['Variant'] == 'Full DA-ECEBO (K=2, lam=0.0)'].sort_values('Seed')['F-measure'].values
    
    wilcoxon_tests = []
    for v in variant_order:
        if v == 'Full DA-ECEBO (K=4, lam=0.25)':
            continue
        v_f1 = df_raw[df_raw['Variant'] == v].sort_values('Seed')['F-measure'].values
        diff = full_f1_k4 - v_f1
        # Wilcoxon test requires non-zero diffs
        if np.all(diff == 0):
            p_val = 1.0
            stat = 0.0
        else:
            stat, p_val = wilcoxon(full_f1_k4, v_f1, zero_method='wilcox')
        wilcoxon_tests.append({
            'Comparison': f"Full DA[4,0.25] vs {v}",
            'Mean_Diff': np.mean(diff),
            'Statistic': stat,
            'p_value': p_val,
            'Significant_0.05': 'YES' if p_val < 0.05 else 'NO'
        })
        
    df_wilc = pd.DataFrame(wilcoxon_tests)
    df_sum.to_csv('results/ablation_multiseed_summary.csv', index=False)
    df_sum.to_csv('results/figdata/ablation_multiseed_summary.csv', index=False)
    df_wilc.to_csv('results/ablation_wilcoxon_pvalues.csv', index=False)
    df_wilc.to_csv('results/figdata/ablation_wilcoxon_pvalues.csv', index=False)
    
    print("\n" + "=" * 80)
    print("=== Multi-Seed Ablation Summary (10 Seeds) ===")
    print(df_sum[['Variant', 'Minority-F1', 'Precision', 'Recall', 'Macro-F1', 'Macro-Gmean', 'AUC']].to_string(index=False))
    print("\n=== Paired Wilcoxon Signed-Rank Test Results (vs Full DA[4,0.25]) ===")
    print(df_wilc.to_string(index=False))


if __name__ == '__main__':
    main()
