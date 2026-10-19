"""
Run P1 and P2 experiments for DA-ECEBO research on CreditCard benchmark.
- P1: Fixed pipeline run (seed 42), save raw predictions, compute TP/FP/FN/TN, Precision, Recall, F1, Macro-F1, Macro-Gmean, Specificity, AUC.
- P2: 10 seeds (42..51) across Baseline, SMOTE, BorderlineSMOTE, ADASYN, EC-EBO, DA-ECEBO.
- Verification of Random Forest and KNN without oversampling vs oversampled.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ttest_rel
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, compute_metrics


def run_single_seed_method(args):
    seed, method_name, X_tr_s, y_tr, t_tr, X_te_s, y_te = args
    t0 = time.time()
    
    if method_name == 'Baseline':
        X_res, y_res = X_tr_s, y_tr
    elif method_name == 'SMOTE':
        X_res, y_res = SMOTE(random_state=seed).fit_resample(X_tr_s, y_tr)
    elif method_name == 'BorderlineSMOTE':
        X_res, y_res = BorderlineSMOTE(random_state=seed).fit_resample(X_tr_s, y_tr)
    elif method_name == 'ADASYN':
        X_res, y_res = ADASYN(random_state=seed).fit_resample(X_tr_s, y_tr)
    elif method_name == 'EC-EBO':
        X_res, y_res = ECEBO(random_state=seed).fit_resample(X_tr_s, y_tr)
    elif method_name == 'DA-ECEBO':
        X_res, y_res = DAECEBO(k_windows=4, lambd=0.25, random_state=seed).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    else:
        raise ValueError(f"Unknown method {method_name}")
    
    t_samp = time.time() - t0
    
    # Train decision tree classifier
    clf = DecisionTreeClassifier(random_state=seed)
    clf.fit(X_res, y_res)
    probs = clf.predict_proba(X_te_s)[:, 1]
    preds = clf.predict(X_te_s)
    
    m = compute_metrics(y_te, probs, preds)
    tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
    
    return {
        'Seed': seed,
        'Method': method_name,
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Precision': m['Precision'],
        'Recall': m['Recall'],
        'Minority-F1': m['F-measure'],
        'Macro-F1': m['Macro-F1'],
        'Macro-Gmean': m['Macro-Gmean'],
        'Specificity': m['Specificity'],
        'AUC': m['AUC'],
        'SampTime_s': round(t_samp, 2)
    }


def main():
    os.makedirs('results', exist_ok=True)
    print("=" * 80)
    print("STEP 1: Loading CreditCard dataset & preparing chronological 75/25 split...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n_total = len(df)
    train_size = int(0.75 * n_total)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = train_df['Class'].values.astype(int)
    t_tr = train_df['Time'].values
    
    X_te = test_df[feat_cols].values
    y_te = test_df['Class'].values.astype(int)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    print(f"Total: {n_total} | Train: {len(X_tr)} (Frauds: {y_tr.sum()}) | Test: {len(X_te)} (Frauds: {y_te.sum()})")
    
    # -------------------------------------------------------------
    # P1: Exact Single Pipeline Run (Seed 42)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("P1: Executing Fixed Single-Pipeline Run for DA-ECEBO (Seed 42)...")
    sampler_p1 = DAECEBO(k_windows=4, lambd=0.25, random_state=42)
    t0 = time.time()
    X_res_p1, y_res_p1 = sampler_p1.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    t_samp_p1 = time.time() - t0
    
    clf_p1 = DecisionTreeClassifier(random_state=42)
    clf_p1.fit(X_res_p1, y_res_p1)
    probs_p1 = clf_p1.predict_proba(X_te_s)[:, 1]
    preds_p1 = clf_p1.predict(X_te_s)
    
    tn, fp, fn, tp = confusion_matrix(y_te, preds_p1).ravel()
    m_p1 = compute_metrics(y_te, probs_p1, preds_p1)
    
    print(f"P1 Confirmed Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"P1 Confirmed Metrics: Precision={m_p1['Precision']:.4f}, Recall={m_p1['Recall']:.4f}, "
          f"Minority-F1={m_p1['F-measure']:.4f}, Macro-F1={m_p1['Macro-F1']:.4f}, "
          f"Macro-Gmean={m_p1['Macro-Gmean']:.4f}, AUC={m_p1['AUC']:.4f}")
    
    # Save raw predictions
    np.savez_compressed(
        'results/creditcard_preds_seed42.npz',
        y_true=y_te,
        y_pred=preds_p1,
        y_prob=probs_p1,
        confusion_matrix=np.array([[tn, fp], [fn, tp]])
    )
    print("Saved raw predictions to results/creditcard_preds_seed42.npz")
    
    # -------------------------------------------------------------
    # P2: Multi-Seed Experiment (10 seeds: 42..51)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("P2: Running Multi-Seed Evaluation (10 seeds: 42..51) across 6 methods...")
    seeds = list(range(42, 52))
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    
    tasks = []
    for s in seeds:
        for m_name in methods:
            tasks.append((s, m_name, X_tr_s, y_tr, t_tr, X_te_s, y_te))
            
    print(f"Total runs: {len(tasks)} (using ProcessPoolExecutor)...")
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        results = list(executor.map(run_single_seed_method, tasks))
    print(f"Multi-seed runs completed in {time.time() - t_start:.2f}s")
    
    df_seeds = pd.DataFrame(results)
    df_seeds.to_csv('results/multiseed_creditcard_results.csv', index=False)
    print("Saved results/multiseed_creditcard_results.csv")
    
    # Compute summary table (mean +/- std)
    summary_rows = []
    metrics_to_report = ['Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']
    for m_name in methods:
        sub = df_seeds[df_seeds['Method'] == m_name]
        row = {'Method': m_name}
        for met in metrics_to_report:
            row[f'{met}_mean'] = sub[met].mean()
            row[f'{met}_std'] = sub[met].std()
            row[met] = f"{sub[met].mean():.4f} +/- {sub[met].std():.4f}"
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('results/multiseed_creditcard_summary.csv', index=False)
    print("\n=== P2 Multi-Seed Summary Table (Mean +/- Std over 10 seeds) ===")
    for _, r in df_summary.iterrows():
        print(f"{r['Method']:<18} | Macro-F1: {r['Macro-F1']} | Min-F1: {r['Minority-F1']} | Prec: {r['Precision']} | Rec: {r['Recall']} | AUC: {r['AUC']}")
        
    # Significance checks: DA-ECEBO vs EC-EBO and vs Baseline
    da_sub = df_seeds[df_seeds['Method'] == 'DA-ECEBO'].sort_values('Seed')
    ec_sub = df_seeds[df_seeds['Method'] == 'EC-EBO'].sort_values('Seed')
    bl_sub = df_seeds[df_seeds['Method'] == 'Baseline'].sort_values('Seed')
    
    print("\n=== Significance Testing (10 Paired Seeds) ===")
    for met in ['Minority-F1', 'Macro-F1', 'Precision']:
        # vs EC-EBO
        diff_ec = da_sub[met].values - ec_sub[met].values
        try:
            w_stat_ec, w_p_ec = wilcoxon(diff_ec)
        except Exception:
            w_stat_ec, w_p_ec = np.nan, 1.0
        t_stat_ec, t_p_ec = ttest_rel(da_sub[met].values, ec_sub[met].values)
        
        # vs Baseline
        diff_bl = da_sub[met].values - bl_sub[met].values
        try:
            w_stat_bl, w_p_bl = wilcoxon(diff_bl)
        except Exception:
            w_stat_bl, w_p_bl = np.nan, 1.0
        t_stat_bl, t_p_bl = ttest_rel(da_sub[met].values, bl_sub[met].values)
        
        print(f"Metric: {met:<12} | DA vs EC-EBO: mean diff = {diff_ec.mean():+.4f} (Wilcoxon p={w_p_ec:.4f}, t-test p={t_p_ec:.4f}) | "
              f"DA vs Baseline: mean diff = {diff_bl.mean():+.4f} (Wilcoxon p={w_p_bl:.4f}, t-test p={t_p_bl:.4f})")
              
    # -------------------------------------------------------------
    # P2 Note Verification: Random Forest & KNN without oversampling
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Verifying Random Forest and KNN WITHOUT oversampling vs oversampled on CreditCard...")
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_tr_s, y_tr)
    rf_bl_preds = rf_baseline.predict(X_te_s)
    rf_bl_probs = rf_baseline.predict_proba(X_te_s)[:, 1]
    rf_bl_m = compute_metrics(y_te, rf_bl_probs, rf_bl_preds)
    
    knn_baseline = KNeighborsClassifier(n_neighbors=5, n_jobs=-1).fit(X_tr_s, y_tr)
    knn_bl_preds = knn_baseline.predict(X_te_s)
    knn_bl_probs = knn_baseline.predict_proba(X_te_s)[:, 1]
    knn_bl_m = compute_metrics(y_te, knn_bl_probs, knn_bl_preds)
    
    print(f"RF (Baseline, No Oversampling): Minority-F1 = {rf_bl_m['F-measure']:.4f}, Macro-F1 = {rf_bl_m['Macro-F1']:.4f}, Precision = {rf_bl_m['Precision']:.4f}, Recall = {rf_bl_m['Recall']:.4f}")
    print(f"KNN (Baseline, No Oversampling): Minority-F1 = {knn_bl_m['F-measure']:.4f}, Macro-F1 = {knn_bl_m['Macro-F1']:.4f}, Precision = {knn_bl_m['Precision']:.4f}, Recall = {knn_bl_m['Recall']:.4f}")
    
    rf_knn_df = pd.DataFrame([
        {'Model': 'Random Forest', 'Sampler': 'None (Baseline)', 'Minority-F1': rf_bl_m['F-measure'], 'Macro-F1': rf_bl_m['Macro-F1'], 'Precision': rf_bl_m['Precision'], 'Recall': rf_bl_m['Recall'], 'AUC': rf_bl_m['AUC']},
        {'Model': 'KNN (k=5)', 'Sampler': 'None (Baseline)', 'Minority-F1': knn_bl_m['F-measure'], 'Macro-F1': knn_bl_m['Macro-F1'], 'Precision': knn_bl_m['Precision'], 'Recall': knn_bl_m['Recall'], 'AUC': knn_bl_m['AUC']},
    ])
    rf_knn_df.to_csv('results/rf_knn_verification.csv', index=False)
    print("Saved results/rf_knn_verification.csv")
    print("\nP1 and P2 successfully executed.")


if __name__ == '__main__':
    main()
