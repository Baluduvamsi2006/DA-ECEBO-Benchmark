"""
S1: Expanding-Window Cross-Validation Grid across ALL 10 SEEDS (seeds 42..51).
Protocol:
- CreditCard chronological train set (213,605 rows)
- Fold 1: train 0..138,843 (65%) -> validate on 138,843..181,564 (20%)
- Fold 2: train 0..181,564 (85%) -> validate on 181,564..213,605 (15%)
- Composite = 0.5 * Val_F1 + 0.5 * Val_Gmean
- Evaluated over ALL 10 seeds: 42..51
- Save results/cv_grid_v3.csv and results/cv_grid_raw_10seeds.csv
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import DAECEBO, compute_metrics


def eval_single_cv_task(task):
    k, lambd, seed, X_tr1_s, y_tr1, t_tr1, X_val1_s, y_val1, X_tr2_s, y_tr2, t_tr2, X_val2_s, y_val2 = task
    
    # Fold 1
    sampler1 = DAECEBO(k_windows=k, lambd=lambd, random_state=seed)
    X_res1, y_res1 = sampler1.fit_resample(X_tr1_s, y_tr1, timestamps=t_tr1)
    clf1 = DecisionTreeClassifier(random_state=seed).fit(X_res1, y_res1)
    probs1 = clf1.predict_proba(X_val1_s)[:, 1]
    preds1 = clf1.predict(X_val1_s)
    m1 = compute_metrics(y_val1, probs1, preds1)
    
    # Fold 2
    sampler2 = DAECEBO(k_windows=k, lambd=lambd, random_state=seed)
    X_res2, y_res2 = sampler2.fit_resample(X_tr2_s, y_tr2, timestamps=t_tr2)
    clf2 = DecisionTreeClassifier(random_state=seed).fit(X_res2, y_res2)
    probs2 = clf2.predict_proba(X_val2_s)[:, 1]
    preds2 = clf2.predict(X_val2_s)
    m2 = compute_metrics(y_val2, probs2, preds2)
    
    # Mean over 2 folds
    f1_mean = 0.5 * (m1['F-measure'] + m2['F-measure'])
    gmean_mean = 0.5 * (m1['Macro-Gmean'] + m2['Macro-Gmean'])
    auc_mean = 0.5 * (m1['AUC'] + m2['AUC'])
    composite = 0.5 * f1_mean + 0.5 * gmean_mean
    
    return {
        'K_windows': k,
        'lambda': lambd,
        'Seed': seed,
        'Val_F1': f1_mean,
        'Val_Gmean': gmean_mean,
        'Val_AUC': auc_mean,
        'Composite_CV': composite
    }


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    print("=" * 80)
    print("S1: Loading CreditCard dataset & preparing expanding-window CV splits (10 seeds)...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n_total = len(df)
    train_size = int(0.75 * n_total)
    train_df = df.iloc[:train_size]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    n_tr = len(train_df)
    f1_tr_end = int(0.65 * n_tr)
    f1_val_end = int(0.85 * n_tr)
    f2_tr_end = int(0.85 * n_tr)
    f2_val_end = n_tr
    
    df_tr1, df_val1 = train_df.iloc[:f1_tr_end], train_df.iloc[f1_tr_end:f1_val_end]
    df_tr2, df_val2 = train_df.iloc[:f2_tr_end], train_df.iloc[f2_tr_end:f2_val_end]
    
    scaler1 = StandardScaler()
    X_tr1_s = scaler1.fit_transform(df_tr1[feat_cols].values)
    X_val1_s = scaler1.transform(df_val1[feat_cols].values)
    y_tr1 = df_tr1['Class'].values.astype(int)
    t_tr1 = df_tr1['Time'].values.astype(float)
    y_val1 = df_val1['Class'].values.astype(int)
    
    scaler2 = StandardScaler()
    X_tr2_s = scaler2.fit_transform(df_tr2[feat_cols].values)
    X_val2_s = scaler2.transform(df_val2[feat_cols].values)
    y_tr2 = df_tr2['Class'].values.astype(int)
    t_tr2 = df_tr2['Time'].values.astype(float)
    y_val2 = df_val2['Class'].values.astype(int)
    
    print(f"Fold 1: train {len(X_tr1_s):,} (frauds: {y_tr1.sum()}) -> val {len(X_val1_s):,} (frauds: {y_val1.sum()})")
    print(f"Fold 2: train {len(X_tr2_s):,} (frauds: {y_tr2.sum()}) -> val {len(X_val2_s):,} (frauds: {y_val2.sum()})")
    
    k_list = [1, 2, 3, 4, 5]
    lambda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    seeds = list(range(42, 52)) # 10 seeds: 42..51
    
    tasks = []
    for k in k_list:
        for lam in lambda_list:
            for s in seeds:
                tasks.append((k, lam, s, X_tr1_s, y_tr1, t_tr1, X_val1_s, y_val1, X_tr2_s, y_tr2, t_tr2, X_val2_s, y_val2))
                
    print(f"Total tasks: {len(tasks)} (25 combos x 10 seeds)... Running with ProcessPoolExecutor...")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        raw_results = list(executor.map(eval_single_cv_task, tasks))
    print(f"Expanding-window CV grid completed in {time.time() - t0:.2f}s")
    
    df_raw = pd.DataFrame(raw_results)
    df_raw.to_csv('results/cv_grid_raw_10seeds.csv', index=False)
    
    # Aggregate across 10 seeds
    summary = []
    for k in k_list:
        for lam in lambda_list:
            sub = df_raw[(df_raw['K_windows'] == k) & (df_raw['lambda'] == lam)]
            n_s = len(sub)
            c_mean = sub['Composite_CV'].mean()
            c_std = sub['Composite_CV'].std()
            c_se = c_std / np.sqrt(n_s) if n_s > 1 else 0.0
            summary.append({
                'K_windows': k,
                'lambda': lam,
                'Val_F1_mean': sub['Val_F1'].mean(),
                'Val_F1_std': sub['Val_F1'].std(),
                'Val_Gmean_mean': sub['Val_Gmean'].mean(),
                'Val_Gmean_std': sub['Val_Gmean'].std(),
                'Val_AUC_mean': sub['Val_AUC'].mean(),
                'Val_AUC_std': sub['Val_AUC'].std(),
                'Composite_CV': c_mean,
                'Composite_CV_std': c_std,
                'Composite_CV_SE': c_se,
                'Composite_CV_str': f"{c_mean:.4f} +/- {c_std:.4f}"
            })
            
    df_cv = pd.DataFrame(summary)
    df_cv.to_csv('results/cv_grid_v3.csv', index=False)
    df_cv.to_csv('results/figdata/cv_grid_v3.csv', index=False)
    print("Saved results/cv_grid_v3.csv and results/figdata/cv_grid_v3.csv")
    
    # 1. Argmax cell
    best_row = df_cv.sort_values('Composite_CV', ascending=False).iloc[0]
    argmax_k = int(best_row['K_windows'])
    argmax_lam = best_row['lambda']
    best_score = best_row['Composite_CV']
    best_se = best_row['Composite_CV_SE']
    
    print("\n" + "=" * 80)
    print(f"ARGMAX CONFIGURATION: K={argmax_k}, lambda={argmax_lam}")
    print(f"Composite CV: {best_row['Composite_CV_str']} (SE = {best_se:.4f}, Val_F1: {best_row['Val_F1_mean']:.4f}, Val_Gmean: {best_row['Val_Gmean_mean']:.4f})")
    
    # 2. 1-Standard-Error Rule:
    # Rule: Find all configs within 1 SE of the best (score >= best_score - best_se).
    # Then pick the simplest (lowest K; if tied, highest lambda / closest to static).
    threshold = best_score - best_se
    within_1se = df_cv[df_cv['Composite_CV'] >= threshold].copy()
    # Sort by simplest: lowest K ascending, then lambda descending (more static/simpler)
    one_se_row = within_1se.sort_values(['K_windows', 'lambda'], ascending=[True, False]).iloc[0]
    one_se_k = int(one_se_row['K_windows'])
    one_se_lam = one_se_row['lambda']
    
    print(f"\n1-SE RULE CONFIGURATION (Threshold >= {threshold:.4f}): K={one_se_k}, lambda={one_se_lam}")
    print(f"Composite CV: {one_se_row['Composite_CV_str']} (SE = {one_se_row['Composite_CV_SE']:.4f})")
    print(f"Candidates within 1-SE ({len(within_1se)} cells):")
    for _, r in within_1se.iterrows():
        print(f"  K={int(r['K_windows'])}, lambda={r['lambda']}: Composite={r['Composite_CV']:.4f} +/- {r['Composite_CV_std']:.4f} (SE={r['Composite_CV_SE']:.4f})")
        
    # 5x5 Pivot Table for Composite Score
    pivot_comp = df_cv.pivot(index='K_windows', columns='lambda', values='Composite_CV')
    pivot_std = df_cv.pivot(index='K_windows', columns='lambda', values='Composite_CV_std')
    print("\n=== 5x5 Composite CV Response Surface (K x lambda, Mean) ===")
    print(pivot_comp.round(4).to_string())
    print("\n=== 5x5 Composite CV Response Surface (K x lambda, Std) ===")
    print(pivot_std.round(4).to_string())
    

if __name__ == '__main__':
    main()
