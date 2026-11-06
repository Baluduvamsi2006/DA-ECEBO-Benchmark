"""
Task F4: Extended Benchmark (B2) with ECEBO_PSI0, G-SMOTE, and CCR
===================================================================
Methods:
  1. None
  2. SMOTE
  3. Borderline-SMOTE
  4. ECEBO (step=0.1)
  5. ECEBO_PSI0 (step=0.1)
  6. G-SMOTE (smote_variants)
  7. CCR (smote_variants)
Scaling: Min-Max (training fold only)
Classifiers: KNN(5), CART, GaussianNB, SVC
Metrics: AUC, Macro-Gmean, Minority-F1, Macro-F1
Folds: 3 repeats x 5 folds = 15 folds per dataset across 43 KEEL datasets
"""
import sys
import os
import time
import multiprocessing as mp
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import smote_variants as sv

from da_ecebo_and_ecebo_implementation import compute_metrics
from ecebo_psi0_variant import ECEBO_PSI0
from scripts_stream.verify_datasets import all_datasets

def holm_bonferroni(p_values):
    p_vals = np.array(p_values, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adjusted_p = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_vals[idx]
        adj = (n - rank) * p
        running_max = max(running_max, adj)
        adjusted_p[idx] = min(running_max, 1.0)
    return adjusted_p

def get_clf(name, seed):
    if name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5)
    elif name == 'CART':
        return DecisionTreeClassifier(random_state=seed)
    elif name == 'GaussianNB':
        return GaussianNB()
    elif name == 'SVC':
        return SVC(random_state=seed)
    raise ValueError(f"Unknown classifier {name}")

def evaluate_models(X_train, y_train, X_test, y_test, seed):
    results = {}
    for clf_name in ['KNN', 'CART', 'GaussianNB', 'SVC']:
        clf = get_clf(clf_name, seed)
        clf.fit(X_train, y_train)
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
        else:
            probs = clf.decision_function(X_test)
            preds = clf.predict(X_test)
        metrics = compute_metrics(y_test, probs, preds)
        results[clf_name] = metrics
    return results

def process_dataset_b2(args):
    dname, X, y = args
    seeds = [42, 43, 44]
    records = []
    
    n_min = (y == 1).sum()
    if n_min < 5:
        return dname, []
        
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr_orig, y_tr = X[train_idx], y[train_idx]
            X_te_orig, y_te = X[test_idx], y[test_idx]
            
            # MinMax scaling fold-wise
            scaler = MinMaxScaler()
            X_tr = scaler.fit_transform(X_tr_orig)
            X_te = scaler.transform(X_te_orig)
            
            # 1. ECEBO_PSI0
            try:
                model_psi0 = ECEBO_PSI0(iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=seed)
                X_res, y_res = model_psi0.fit_resample(X_tr, y_tr)
                res = evaluate_models(X_res, y_res, X_te, y_te, seed)
                for clf_name, m in res.items():
                    records.append({
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': 'minmax',
                        'Method': 'ECEBO_PSI0', 'Step': 0.1, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    })
            except Exception as e:
                pass
                
            # 2. G-SMOTE
            try:
                model_gsmote = sv.G_SMOTE(random_state=seed)
                X_res, y_res = model_gsmote.sample(X_tr, y_tr)
                res = evaluate_models(X_res, y_res, X_te, y_te, seed)
                for clf_name, m in res.items():
                    records.append({
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': 'minmax',
                        'Method': 'G-SMOTE', 'Step': np.nan, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    })
            except Exception as e:
                pass
                
            # 3. CCR
            try:
                model_ccr = sv.CCR(random_state=seed)
                X_res, y_res = model_ccr.sample(X_tr, y_tr)
                res = evaluate_models(X_res, y_res, X_te, y_te, seed)
                for clf_name, m in res.items():
                    records.append({
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': 'minmax',
                        'Method': 'CCR', 'Step': np.nan, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    })
            except Exception as e:
                pass
                
    return dname, records

def main():
    print("Loading datasets for Task F4...")
    datasets_to_run = []
    for dname, (X, y) in all_datasets.items():
        datasets_to_run.append((dname, X, y))
        
    print(f"Total datasets to process: {len(datasets_to_run)}")
    t0 = time.time()
    
    n_cpus = max(1, mp.cpu_count() - 1)
    print(f"Running multiprocessing pool with {n_cpus} workers...")
    with mp.Pool(n_cpus) as pool:
        results = pool.map(process_dataset_b2, datasets_to_run)
        
    all_new_records = []
    for dname, recs in results:
        all_new_records.extend(recs)
        print(f"Dataset {dname}: generated {len(recs)} evaluations")
        
    df_new = pd.DataFrame(all_new_records)
    print(f"Generated {len(df_new)} new evaluations in {time.time()-t0:.1f}s.")
    
    # Load existing records from B_raw.csv
    df_b_raw = pd.read_csv('results_stream/B_raw.csv', keep_default_na=False)
    # Filter to minmax scaling and Step 0.1 or None/SMOTE/BorderlineSMOTE
    mask_existing = (df_b_raw['Scaling'] == 'minmax') & (
        (df_b_raw['Method'].isin(['', 'None', 'SMOTE', 'BorderlineSMOTE'])) |
        (df_b_raw['Method'] == 'EC-EBO_step0.1')
    )
    df_existing = df_b_raw[mask_existing].copy()
    # Normalize Method names
    df_existing['Method'] = df_existing['Method'].replace({'': 'None', 'EC-EBO_step0.1': 'ECEBO'})
    
    # Combine existing and new records
    df_b2_raw = pd.concat([df_existing, df_new], ignore_index=True)
    out_raw_path = 'results_stream/B2_raw.csv'
    df_b2_raw.to_csv(out_raw_path, index=False)
    print(f"Saved B2_raw.csv to {out_raw_path} ({len(df_b2_raw)} rows)")
    
    # Generate B2_summary.csv
    # Average across folds per Dataset, Classifier, Metric, Method
    methods = ['None', 'SMOTE', 'BorderlineSMOTE', 'ECEBO', 'ECEBO_PSI0', 'G-SMOTE', 'CCR']
    classifiers = ['KNN', 'CART', 'GaussianNB', 'SVC']
    metrics = ['AUC', 'Macro-Gmean', 'Minority-F1', 'Macro-F1']
    
    summary_rows = []
    
    for clf in classifiers:
        for metric in metrics:
            # Pivot dataset x method
            sub = df_b2_raw[df_b2_raw['Classifier'] == clf]
            # Fold average per dataset x method
            fold_avg = sub.groupby(['Dataset', 'Method'])[metric].mean().unstack(level='Method')
            
            # Keep datasets that have all 7 methods
            valid_datasets = fold_avg.dropna(subset=[m for m in methods if m in fold_avg.columns])
            n_ds = len(valid_datasets)
            
            # Compute average ranks (lower rank is better, rank 1 = highest metric)
            ranks = valid_datasets.rank(axis=1, ascending=False, method='average')
            avg_ranks = ranks.mean()
            
            # Comparisons for ECEBO and ECEBO_PSI0 against each other method
            comp_stats = {}
            for focus in ['ECEBO', 'ECEBO_PSI0']:
                for opp in methods:
                    if focus == opp:
                        continue
                    diffs = np.round(valid_datasets[focus] - valid_datasets[opp], 4)
                    wins = int(np.sum(diffs > 0))
                    ties = int(np.sum(diffs == 0))
                    losses = int(np.sum(diffs < 0))
                    
                    # Wilcoxon signed rank test
                    non_zero = diffs[diffs != 0]
                    if len(non_zero) > 5:
                        try:
                            res_w = wilcoxon(valid_datasets[focus], valid_datasets[opp], alternative='two-sided')
                            p_val = float(res_w.pvalue)
                        except Exception:
                            p_val = 1.0
                    else:
                        p_val = 1.0
                    comp_stats[f"{focus}_vs_{opp}_WTL"] = f"{wins}/{ties}/{losses}"
                    comp_stats[f"{focus}_vs_{opp}_pval"] = p_val
                    
            row = {
                'Classifier': clf,
                'Metric': metric,
                'N_Datasets': n_ds,
            }
            for m in methods:
                row[f"Rank_{m}"] = round(float(avg_ranks.get(m, np.nan)), 4)
                
            for k, v in comp_stats.items():
                if k.endswith('_pval'):
                    row[k] = v
                else:
                    row[k] = v
                    
            summary_rows.append(row)
            
    df_summary = pd.DataFrame(summary_rows)
    
    # Apply Holm correction across p-values within summary
    pval_cols = [c for c in df_summary.columns if c.endswith('_pval')]
    for pcol in pval_cols:
        holm_col = pcol.replace('_pval', '_Holm_p')
        df_summary[holm_col] = holm_bonferroni(df_summary[pcol].values)
        
    out_summary_path = 'results_stream/B2_summary.csv'
    df_summary.to_csv(out_summary_path, index=False)
    print(f"Saved B2_summary.csv to {out_summary_path} ({len(df_summary)} rows)")

if __name__ == '__main__':
    main()
