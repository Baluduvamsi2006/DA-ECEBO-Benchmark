"""
R5: Multi-Model Classifier Benchmark across 6 classifiers x 6 methods over 10 seeds on CreditCard.
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
"""

import os
import time
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
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif name == 'CART':
        return DecisionTreeClassifier(random_state=seed)
    elif name == 'GaussianNB':
        return GaussianNB()
    elif name == 'SVM':
        return LinearSVC(dual=False, random_state=seed, max_iter=2000)
    elif name == 'RandomForest':
        return RandomForestClassifier(n_estimators=50, max_depth=12, random_state=seed, n_jobs=-1)
    elif name == 'LogisticRegression':
        return LogisticRegression(max_iter=1000, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier {name}")


def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    raw_path = 'results/classifier_benchmark_raw.csv'
    
    # Load existing progress if any
    if os.path.exists(raw_path):
        df_existing = pd.read_csv(raw_path)
        records = df_existing.to_dict('records')
        completed_seeds = set(df_existing['Seed'].unique())
        print(f"Resuming R5: {len(completed_seeds)} seeds already completed: {sorted(completed_seeds)}")
    else:
        records = []
        completed_seeds = set()
        
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
    
    classifiers = ['KNN', 'CART', 'GaussianNB', 'SVM', 'RandomForest', 'LogisticRegression']
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    seeds = list(range(42, 52))
    
    t0_all = time.time()
    for s_idx, s in enumerate(seeds):
        if s in completed_seeds:
            print(f"Skipping Seed {s} (already done).", flush=True)
            continue
            
        t_s0 = time.time()
        print(f"\n--- Seed {s} ({s_idx + 1}/{len(seeds)}) ---", flush=True)
        
        # 1. Resample datasets for this seed
        resampled_data = {}
        resampled_data['Baseline'] = (X_tr_s, y_tr)
        
        t0 = time.time()
        resampled_data['SMOTE'] = SMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
        print(f"  SMOTE: {time.time()-t0:.1f}s", flush=True)
        
        t0 = time.time()
        resampled_data['BorderlineSMOTE'] = BorderlineSMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
        print(f"  Borderline: {time.time()-t0:.1f}s", flush=True)
        
        t0 = time.time()
        resampled_data['ADASYN'] = ADASYN(random_state=s).fit_resample(X_tr_s, y_tr)
        print(f"  ADASYN: {time.time()-t0:.1f}s", flush=True)
        
        t0 = time.time()
        resampled_data['EC-EBO'] = ECEBO(random_state=s).fit_resample(X_tr_s, y_tr)
        print(f"  EC-EBO: {time.time()-t0:.1f}s", flush=True)
        
        t0 = time.time()
        resampled_data['DA-ECEBO'] = DAECEBO(k_windows=4, lambd=0.25, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        print(f"  DA-ECEBO: {time.time()-t0:.1f}s", flush=True)
        
        # 2. Evaluate all classifiers
        for m_name in methods:
            X_res, y_res = resampled_data[m_name]
            for clf_name in classifiers:
                t_clf0 = time.time()
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
                
        print(f"Seed {s} completed in {time.time() - t_s0:.1f}s. Total records: {len(records)}", flush=True)
        # Incremental save after each seed
        df_interim = pd.DataFrame(records)
        df_interim.to_csv(raw_path, index=False)
        
    print(f"\nAll classifier runs completed in {time.time() - t0_all:.2f}s")
    df_raw = pd.DataFrame(records)
    df_raw.to_csv(raw_path, index=False)
    
    # Summary Table: mean +/- std per Classifier x Method
    summary = []
    for clf_name in classifiers:
        for m_name in methods:
            sub = df_raw[(df_raw['Classifier'] == clf_name) & (df_raw['Method'] == m_name)]
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
    main()
