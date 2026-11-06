"""
Task B: EC-EBO Direction Check on KEEL Benchmark Datasets
=========================================================
Protocol:
  - 43 verified binary classification datasets from KEEL.
  - Stratified 5-fold CV repeated 3 times (seeds 42, 43, 44).
  - Classifiers: KNN(k=5), DecisionTreeClassifier, GaussianNB, SVC.
  - Feature passes:
      (i) raw features
      (ii) min-max scaled using training fold only
  - Step sizes for EC-EBO:
      0.01 (0.1x), 0.1 (1.0x, current default), 1.0 (10x)
  - Metrics: AUC, G-mean, Minority-F1, Macro-F1.
"""
import sys
import os
import time
import multiprocessing as mp
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
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from da_ecebo_and_ecebo_implementation import ECEBO, compute_metrics
from scripts_stream.verify_datasets import all_datasets

def holm_bonferroni(p_values):
    """Applies Holm-Bonferroni correction to an array of p-values."""
    p_vals = np.array(p_values, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adjusted_p = np.zeros(n)
    running_max = 0.0
    for rank, idx in enumerate(sorted_indices):
        p = p_vals[idx]
        adj = p * (n - rank)
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

def process_dataset(args):
    dname, X, y = args
    seeds = [42, 43, 44]
    step_sizes = [0.01, 0.1, 1.0]
    records = []
    
    # Check if minority has at least 5 samples for 5-fold CV
    n_min = (y == 1).sum()
    if n_min < 5:
        return dname, []
        
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr_orig, y_tr = X[train_idx], y[train_idx]
            X_te_orig, y_te = X[test_idx], y[test_idx]
            
            for scale_pass in ['raw', 'minmax']:
                if scale_pass == 'minmax':
                    scaler = MinMaxScaler()
                    X_tr = scaler.fit_transform(X_tr_orig)
                    X_te = scaler.transform(X_te_orig)
                else:
                    X_tr = X_tr_orig.copy()
                    X_te = X_te_orig.copy()
                    
                # 1. Baseline: None
                res_none = evaluate_models(X_tr, y_tr, X_te, y_te, seed)
                for clf_name, m in res_none.items():
                    row = {
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': scale_pass,
                        'Method': 'None', 'Step': np.nan, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    }
                    records.append(row)
                    
                # 2. SMOTE
                k_smote = min(5, (y_tr == 1).sum() - 1)
                if k_smote >= 1:
                    sm = SMOTE(k_neighbors=k_smote, random_state=seed)
                    try:
                        X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
                    except Exception:
                        X_sm, y_sm = X_tr, y_tr
                else:
                    X_sm, y_sm = X_tr, y_tr
                res_smote = evaluate_models(X_sm, y_sm, X_te, y_te, seed)
                for clf_name, m in res_smote.items():
                    row = {
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': scale_pass,
                        'Method': 'SMOTE', 'Step': np.nan, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    }
                    records.append(row)
                    
                # 3. Borderline-SMOTE
                if k_smote >= 1:
                    bsm = BorderlineSMOTE(k_neighbors=k_smote, random_state=seed)
                    try:
                        X_bsm, y_bsm = bsm.fit_resample(X_tr, y_tr)
                    except Exception:
                        X_bsm, y_bsm = X_sm, y_sm
                else:
                    X_bsm, y_bsm = X_tr, y_tr
                res_bsm = evaluate_models(X_bsm, y_bsm, X_te, y_te, seed)
                for clf_name, m in res_bsm.items():
                    row = {
                        'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': scale_pass,
                        'Method': 'BorderlineSMOTE', 'Step': np.nan, 'Classifier': clf_name,
                        'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                        'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                    }
                    records.append(row)
                    
                # 4. EC-EBO with step sizes
                for step in step_sizes:
                    ec = ECEBO(iters=60, alpha=1.0, k=min(7, (y_tr == 0).sum()), prob=0.2, step=step, random_state=seed)
                    try:
                        X_ec, y_ec = ec.fit_resample(X_tr, y_tr)
                    except Exception as e:
                        X_ec, y_ec = X_tr, y_tr
                    res_ec = evaluate_models(X_ec, y_ec, X_te, y_te, seed)
                    for clf_name, m in res_ec.items():
                        row = {
                            'Dataset': dname, 'Seed': seed, 'Fold': fold, 'Scaling': scale_pass,
                            'Method': f'EC-EBO_step{step}', 'Step': step, 'Classifier': clf_name,
                            'Minority-F1': m['F-measure'], 'Macro-F1': m['Macro-F1'],
                            'Macro-Gmean': m['Macro-Gmean'], 'AUC': m['AUC']
                        }
                        records.append(row)
                        
    return dname, records

def main():
    print("=" * 80)
    print("TASK B: STARTING MULTI-DATASET BENCHMARK")
    print(f"Total datasets to process: {len(all_datasets)}")
    print("=" * 80)
    
    tasks = [(name, X, y) for name, (X, y) in sorted(all_datasets.items())]
    
    t0 = time.time()
    n_workers = min(12, mp.cpu_count())
    print(f"Launching {n_workers} worker processes...")
    
    with mp.Pool(processes=n_workers) as pool:
        all_res = pool.map(process_dataset, tasks)
        
    all_records = []
    for dname, recs in all_res:
        print(f"Dataset {dname} complete: {len(recs)} fold evaluations.")
        all_records.extend(recs)
        
    df_raw = pd.DataFrame(all_records)
    out_raw = 'results_stream/B_raw.csv'
    df_raw.to_csv(out_raw, index=False)
    print(f"\nSaved raw fold evaluations to {out_raw} (Total rows: {len(df_raw):,})")
    print(f"Elapsed time: {time.time() - t0:.1f}s")
    
    # -------------------------------------------------------------
    # Compute Summary Statistics, Rankings, Win/Loss & Wilcoxon
    # -------------------------------------------------------------
    print("\nComputing Summary Statistics, Win/Tie/Loss, Ranks, and Wilcoxon Tests...")
    
    # Fold-averaged performance per dataset x scaling x step x method x classifier
    metrics_list = ['AUC', 'Macro-Gmean', 'Minority-F1', 'Macro-F1']
    classifiers = ['KNN', 'CART', 'GaussianNB', 'SVC']
    scaling_passes = ['raw', 'minmax']
    step_sizes = [0.01, 0.1, 1.0]
    
    summary_rows = []
    
    # For each variant (scaling x step)
    for sc in scaling_passes:
        for step in step_sizes:
            variant_name = f"{sc}_step{step}"
            ec_method_name = f"EC-EBO_step{step}"
            
            for clf in classifiers:
                for met in metrics_list:
                    # Filter dataset-level fold averages
                    # For None, SMOTE, BorderlineSMOTE, and EC-EBO(step)
                    sub = df_raw[(df_raw['Scaling'] == sc) & (df_raw['Classifier'] == clf)]
                    
                    # Compute mean per dataset and method
                    pvt = sub.groupby(['Dataset', 'Method'])[met].mean().unstack()
                    
                    # Ensure all 4 methods exist
                    methods_4 = ['None', 'SMOTE', 'BorderlineSMOTE', ec_method_name]
                    if not all(m in pvt.columns for m in methods_4):
                        continue
                        
                    pvt_4 = pvt[methods_4].dropna()
                    n_datasets = len(pvt_4)
                    
                    # 1. Ranks (higher metric is better; rank 1 = highest value)
                    ranks = pvt_4.rank(axis=1, ascending=False, method='average')
                    avg_rank_ecebo = float(ranks[ec_method_name].mean())
                    avg_rank_none = float(ranks['None'].mean())
                    avg_rank_smote = float(ranks['SMOTE'].mean())
                    avg_rank_bsm = float(ranks['BorderlineSMOTE'].mean())
                    
                    # 2. Win / Tie / Loss of EC-EBO vs each baseline (rounded to 4 decimals)
                    diff_none = np.round(pvt_4[ec_method_name] - pvt_4['None'], 4)
                    diff_smote = np.round(pvt_4[ec_method_name] - pvt_4['SMOTE'], 4)
                    diff_bsm = np.round(pvt_4[ec_method_name] - pvt_4['BorderlineSMOTE'], 4)
                    
                    w_none, t_none, l_none = int((diff_none > 0).sum()), int((diff_none == 0).sum()), int((diff_none < 0).sum())
                    w_smote, t_smote, l_smote = int((diff_smote > 0).sum()), int((diff_smote == 0).sum()), int((diff_smote < 0).sum())
                    w_bsm, t_bsm, l_bsm = int((diff_bsm > 0).sum()), int((diff_bsm == 0).sum()), int((diff_bsm < 0).sum())
                    
                    # 3. Paired Wilcoxon signed-rank test
                    def calc_wilcoxon(diff_series):
                        valid_diff = diff_series[diff_series != 0]
                        if len(valid_diff) >= 5:
                            try:
                                res = wilcoxon(valid_diff, alternative='two-sided')
                                return float(res.pvalue)
                            except Exception:
                                return 1.0
                        return 1.0
                        
                    p_none = calc_wilcoxon(pvt_4[ec_method_name] - pvt_4['None'])
                    p_smote = calc_wilcoxon(pvt_4[ec_method_name] - pvt_4['SMOTE'])
                    p_bsm = calc_wilcoxon(pvt_4[ec_method_name] - pvt_4['BorderlineSMOTE'])
                    
                    row_stat = {
                        'Variant': variant_name,
                        'Scaling': sc,
                        'Step_Size': step,
                        'Classifier': clf,
                        'Metric': met,
                        'N_Datasets': n_datasets,
                        'EC-EBO_AvgRank': round(avg_rank_ecebo, 3),
                        'None_AvgRank': round(avg_rank_none, 3),
                        'SMOTE_AvgRank': round(avg_rank_smote, 3),
                        'BorderlineSMOTE_AvgRank': round(avg_rank_bsm, 3),
                        'vs_None_WTL': f"{w_none}/{t_none}/{l_none}",
                        'vs_SMOTE_WTL': f"{w_smote}/{t_smote}/{l_smote}",
                        'vs_BorderlineSMOTE_WTL': f"{w_bsm}/{t_bsm}/{l_bsm}",
                        'vs_None_Wins': w_none,
                        'vs_None_Ties': t_none,
                        'vs_None_Losses': l_none,
                        'Beats_None_Majority': w_none > (n_datasets / 2.0),
                        'p_vs_None_raw': p_none,
                        'p_vs_SMOTE_raw': p_smote,
                        'p_vs_BorderlineSMOTE_raw': p_bsm
                    }
                    summary_rows.append(row_stat)
                    
    df_summary = pd.DataFrame(summary_rows)
    
    # Apply Holm-Bonferroni correction across tests per comparison baseline
    df_summary['p_vs_None_Holm'] = np.round(holm_bonferroni(df_summary['p_vs_None_raw']), 4)
    df_summary['p_vs_SMOTE_Holm'] = np.round(holm_bonferroni(df_summary['p_vs_SMOTE_raw']), 4)
    df_summary['p_vs_BorderlineSMOTE_Holm'] = np.round(holm_bonferroni(df_summary['p_vs_BorderlineSMOTE_raw']), 4)
    
    out_summary = 'results_stream/B_summary.csv'
    df_summary.to_csv(out_summary, index=False)
    print(f"\nSaved summary benchmark results to {out_summary}")
    
    # Print key headline answer for Task B:
    n_total_settings = len(df_summary)
    n_beat_majority = int(df_summary['Beats_None_Majority'].sum())
    print("\n" + "=" * 80)
    print(f"HEADLINE ANSWER FOR TASK B:")
    print(f"In how many of the 72 settings (6 variants x 4 classifiers x 4 metrics) does EC-EBO beat None on more than half the datasets?")
    print(f"-> {n_beat_majority} out of {n_total_settings} settings ({n_beat_majority/n_total_settings*100:.1f}%)")
    print("=" * 80)

if __name__ == '__main__':
    mp.freeze_support()
    main()

