"""
R5: Multi-Model Classifier Benchmark across 6 classifiers x methods over 10 seeds on CreditCard.
Optimized 4-worker parallel execution.

Classifiers:
1. KNN (k=5)
2. CART (DecisionTreeClassifier)
3. GaussianNB
4. LinearSVC (scalable linear SVM)
5. RandomForest (n_estimators=50, max_depth=12)
6. LogisticRegression

Methods:
1. Baseline
2. SMOTE
3. BorderlineSMOTE
4. ADASYN
5. EC-EBO
6. DA-ECEBO (K=4, lambda=0.25)
7. DA-ECEBO_v2 (K=2, lambda=0.0)
"""

import os
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, compute_metrics


def get_clf(name, seed):
    if name == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=2)
    elif name == 'CART':
        return DecisionTreeClassifier(random_state=seed)
    elif name == 'GaussianNB':
        return GaussianNB()
    elif name == 'SVM':
        return LinearSVC(dual=False, random_state=seed, max_iter=1000)
    elif name == 'RandomForest':
        return RandomForestClassifier(n_estimators=50, max_depth=12, random_state=seed, n_jobs=2)
    elif name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier {name}")


def worker_run(worker_id, seeds, X_tr_s, y_tr, t_tr, X_te_s, y_te):
    classifiers = ['KNN', 'CART', 'GaussianNB', 'SVM', 'RandomForest', 'LogisticRegression']
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO', 'DA-ECEBO_v2']
    
    out_file = f'results/r5_worker_{worker_id}.csv'
    records = []
    
    print(f"[Worker {worker_id}] Started for seeds: {seeds}", flush=True)
    
    for s in seeds:
        t_s0 = time.time()
        print(f"[Worker {worker_id}] Starting Seed {s}...", flush=True)
        
        # 1. Resample
        resampled_data = {}
        resampled_data['Baseline'] = (X_tr_s, y_tr)
        resampled_data['SMOTE'] = SMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
        resampled_data['BorderlineSMOTE'] = BorderlineSMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
        resampled_data['ADASYN'] = ADASYN(random_state=s).fit_resample(X_tr_s, y_tr)
        resampled_data['EC-EBO'] = ECEBO(random_state=s).fit_resample(X_tr_s, y_tr)
        resampled_data['DA-ECEBO'] = DAECEBO(k_windows=4, lambd=0.25, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        resampled_data['DA-ECEBO_v2'] = DAECEBO(k_windows=2, lambd=0.0, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        
        # 2. Evaluate
        for m_name in methods:
            X_res, y_res = resampled_data[m_name]
            for clf_name in classifiers:
                clf = get_clf(clf_name, s)
                clf.fit(X_res, y_res)
                if hasattr(clf, 'predict_proba'):
                    probs = clf.predict_proba(X_te_s)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                else:
                    probs = clf.decision_function(X_te_s)
                    preds = clf.predict(X_te_s)
                    
                metrics = compute_metrics(y_te, probs, preds)
                metrics['Seed'] = s
                metrics['Method'] = m_name
                metrics['Classifier'] = clf_name
                records.append(metrics)
                
        # Save worker progress
        pd.DataFrame(records).to_csv(out_file, index=False)
        print(f"[Worker {worker_id}] Seed {s} finished in {time.time()-t_s0:.1f}s", flush=True)
        
    print(f"[Worker {worker_id}] ALL SEEDS COMPLETE!", flush=True)


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    
    print("=" * 80)
    print("R5: Loading CreditCard dataset & preparing chronological 75/25 split...")
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
    
    # 4 workers across 10 seeds (42..51)
    seed_groups = [
        [42, 46, 50],
        [43, 47, 51],
        [44, 48],
        [45, 49]
    ]
    
    processes = []
    t0_all = time.time()
    for w_id, s_list in enumerate(seed_groups):
        p = mp.Process(target=worker_run, args=(w_id, s_list, X_tr_s, y_tr, t_tr, X_te_s, y_te))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print(f"\nAll workers joined in {time.time() - t0_all:.1f}s. Aggregating results...")
    
    # Merge all worker outputs
    all_dfs = []
    for w_id in range(len(seed_groups)):
        fpath = f'results/r5_worker_{w_id}.csv'
        if os.path.exists(fpath):
            all_dfs.append(pd.read_csv(fpath))
            
    df_raw = pd.concat(all_dfs, ignore_index=True)
    df_raw.to_csv('results/classifier_benchmark_raw.csv', index=False)
    
    classifiers = ['KNN', 'CART', 'GaussianNB', 'SVM', 'RandomForest', 'LogisticRegression']
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO', 'DA-ECEBO_v2']
    
    summary = []
    for clf_name in classifiers:
        for m_name in methods:
            sub = df_raw[(df_raw['Classifier'] == clf_name) & (df_raw['Method'] == m_name)]
            if len(sub) == 0:
                continue
            summary.append({
                'Classifier': clf_name,
                'Method': m_name,
                'Minority-F1_mean': sub['F-measure'].mean(),
                'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
                'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
                'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
                'Macro-F1': f"{sub['Macro-F1'].mean():.4f} +/- {sub['Macro-F1'].std():.4f}",
                'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
                'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}"
            })
            
    df_sum = pd.DataFrame(summary)
    
    # Mark best per classifier based on Minority-F1
    best_flags = []
    for clf_name in classifiers:
        sub_c = df_sum[df_sum['Classifier'] == clf_name]
        max_f1 = sub_c['Minority-F1_mean'].max()
        for _, row in sub_c.iterrows():
            best_flags.append('BEST' if row['Minority-F1_mean'] == max_f1 else '')
    df_sum['Is_Best'] = best_flags
    
    df_sum.to_csv('results/classifier_benchmark_summary.csv', index=False)
    df_sum.to_csv('results/figdata/classifier_benchmark_summary.csv', index=False)
    print("\n" + "=" * 80)
    print("=== Multi-Model Classifier Benchmark Summary (10 Seeds) ===")
    print(df_sum[['Classifier', 'Method', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC', 'Is_Best']].to_string(index=False))


if __name__ == '__main__':
    mp.freeze_support()
    main()
