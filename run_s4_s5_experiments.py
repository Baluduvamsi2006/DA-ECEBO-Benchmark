"""
S4(c), S4(d), and S5 Experiments:
S4(c): Rerun runtimes 5 times each, report median + IQR, record actual hill-climb iterations.
S4(d): RBF SVM (sklearn SVC with default RBF kernel) on CreditCard using stratified majority subsample (N_maj=3980, 10:1 ratio).
S5: Haberman test-set bootstrap CI (17 test minority samples) and confirm AUC < 0.5.
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from concurrent.futures import ProcessPoolExecutor
from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, compute_metrics


def get_binary_y(df, target_col):
    vals = df[target_col].values
    if vals.dtype == object or isinstance(vals[0], str):
        vc = df[target_col].value_counts()
        min_cls = vc.index[-1]
        return (df[target_col] == min_cls).astype(int).values
    return vals.astype(int)


def run_s4c_runtimes():
    print("=" * 80)
    print("S4(c): Running runtime benchmarks 5 times each (Median + IQR + Hill-climb iterations)...")
    if os.path.exists('results/runtime_benchmark_5runs.csv'):
        print("  results/runtime_benchmark_5runs.csv already exists; skipping rerun.")
        return
    
    # 1. CreditCard
    df_cc = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    tr_sz = int(0.75 * len(df_cc))
    train_cc = df_cc.iloc[:tr_sz]
    feat_cc = [c for c in df_cc.columns if c not in ['Time', 'Class']]
    X_cc = train_cc[feat_cc].values
    y_cc = train_cc['Class'].values.astype(int)
    t_cc = train_cc['Time'].values.astype(float)
    scaler_cc = StandardScaler()
    X_cc_s = scaler_cc.fit_transform(X_cc)
    
    # 2. HTRU2
    df_ht = pd.read_csv('datasets/htru2.csv')
    target_ht = 'class' if 'class' in df_ht.columns else 'Class'
    feat_ht = [c for c in df_ht.columns if c != target_ht]
    X_ht = df_ht[feat_ht].values
    y_ht = get_binary_y(df_ht, target_ht)
    scaler_ht = StandardScaler()
    X_ht_s = scaler_ht.fit_transform(X_ht)
    
    # 3. Ecoli2
    df_ec = pd.read_csv('datasets/ecoli2.csv')
    target_ec = 'Class' if 'Class' in df_ec.columns else 'class'
    feat_ec = [c for c in df_ec.columns if c != target_ec]
    X_ec = df_ec[feat_ec].values
    y_ec = get_binary_y(df_ec, target_ec)
    scaler_ec = StandardScaler()
    X_ec_s = scaler_ec.fit_transform(X_ec)
    
    datasets = {
        'creditcard': (X_cc_s, y_cc, t_cc),
        'htru2': (X_ht_s, y_ht, np.arange(len(y_ht))),
        'ecoli2': (X_ec_s, y_ec, np.arange(len(y_ec)))
    }
    
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    records = []
    
    for dname, (X, y, t) in datasets.items():
        print(f"  Benchmarking {dname} (N={len(y)}, N_min={np.sum(y)})...")
        for mname in methods:
            times = []
            iters_list = []
            for run_idx in range(5):
                t0 = time.time()
                if mname == 'Baseline':
                    _ = (X, y)
                    iters = 0
                elif mname == 'SMOTE':
                    _ = SMOTE(random_state=42 + run_idx).fit_resample(X, y)
                    iters = 0
                elif mname == 'BorderlineSMOTE':
                    _ = BorderlineSMOTE(random_state=42 + run_idx).fit_resample(X, y)
                    iters = 0
                elif mname == 'ADASYN':
                    _ = ADASYN(random_state=42 + run_idx).fit_resample(X, y)
                    iters = 0
                elif mname == 'EC-EBO':
                    sampler = ECEBO(random_state=42 + run_idx)
                    _ = sampler.fit_resample(X, y)
                    iters = sampler.iters  # configured max iterations per candidate (60)
                elif mname == 'DA-ECEBO':
                    sampler = DAECEBO(k_windows=2, lambd=0.0, random_state=42 + run_idx)
                    _ = sampler.fit_resample(X, y, timestamps=t)
                    iters = sampler.iters  # configured max iterations per candidate (60)
                elapsed = time.time() - t0
                times.append(elapsed)
                iters_list.append(iters)
                
            times = np.array(times)
            med = np.median(times)
            q25 = np.percentile(times, 25)
            q75 = np.percentile(times, 75)
            iqr = q75 - q25
            records.append({
                'Dataset': dname,
                'Method': mname,
                'Median_Time_s': med,
                'IQR_Time_s': iqr,
                'Time_str': f"{med:.4f} (IQR: {iqr:.4f})",
                'HillClimb_Iters': iters_list[0]
            })
            print(f"    {mname:16s}: Median = {med:.4f}s, IQR = {iqr:.4f}s, Iters = {iters_list[0]}")
            
    df_s4c = pd.DataFrame(records)
    df_s4c.to_csv('results/runtime_benchmark_5runs.csv', index=False)
    print("Saved results/runtime_benchmark_5runs.csv")


def _eval_s4d_single_seed(args):
    s, X_tr_s, y_tr_sub, t_tr_sub, X_te_s, y_te = args
    records = []
    samplers = {
        'Baseline': (X_tr_s, y_tr_sub),
        'SMOTE': SMOTE(random_state=s).fit_resample(X_tr_s, y_tr_sub),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=s).fit_resample(X_tr_s, y_tr_sub),
        'ADASYN': ADASYN(random_state=s).fit_resample(X_tr_s, y_tr_sub),
        'EC-EBO': ECEBO(random_state=s).fit_resample(X_tr_s, y_tr_sub),
        'DA-ECEBO': DAECEBO(k_windows=2, lambd=0.0, random_state=s).fit_resample(X_tr_s, y_tr_sub, timestamps=t_tr_sub)
    }
    for mname, (X_res, y_res) in samplers.items():
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=s)
        clf.fit(X_res, y_res)
        probs = clf.decision_function(X_te_s)
        preds = clf.predict(X_te_s)
        m = compute_metrics(y_te, probs, preds)
        m['Seed'] = s
        m['Method'] = mname
        m['Classifier'] = 'SVC_RBF'
        records.append(m)
    return records


def run_s4d_rbf_svm():
    print("\n" + "=" * 80)
    print("S4(d): Running sklearn SVC (default RBF) with stratified majority subsample (N_maj=3980, 10:1 ratio)...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    tr_sz = int(0.75 * len(df))
    train_df, test_df = df.iloc[:tr_sz], df.iloc[tr_sz:]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = train_df['Class'].values.astype(int)
    t_tr = train_df['Time'].values.astype(float)
    X_te = test_df[feat_cols].values
    y_te = test_df['Class'].values.astype(int)
    
    idx_maj = np.where(y_tr == 0)[0]
    idx_min = np.where(y_tr == 1)[0]
    n_min = len(idx_min)  # 398
    
    # Subsample majority to 3,980 (10:1 ratio matching HTRU2 / KEEL reference scale)
    rng_sub = np.random.RandomState(42)
    sub_maj_idx = rng_sub.choice(idx_maj, size=n_min * 10, replace=False)
    keep_idx = np.sort(np.concatenate([idx_min, sub_maj_idx]))
    
    X_tr_sub = X_tr[keep_idx]
    y_tr_sub = y_tr[keep_idx]
    t_tr_sub = t_tr[keep_idx]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_sub)
    X_te_s = scaler.transform(X_te)
    
    print(f"  Subsampled training set: {len(y_tr_sub):,} rows ({np.sum(y_tr_sub == 0):,} majority, {np.sum(y_tr_sub == 1):,} minority; 10:1 ratio)")
    print(f"  Test set: {len(y_te):,} rows ({np.sum(y_te == 1)} frauds)")
    
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    tasks = [(s, X_tr_s, y_tr_sub, t_tr_sub, X_te_s, y_te) for s in range(42, 52)]
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        results_lists = list(executor.map(_eval_s4d_single_seed, tasks))
        
    records = [r for sublist in results_lists for r in sublist]
    print(f"  RBF-SVC 10 seeds completed in {time.time() - t0:.2f}s")
    df_raw = pd.DataFrame(records)
    df_raw.to_csv('results/rbf_svm_benchmark_raw.csv', index=False)
    
    summary = []
    for mname in methods:
        sub = df_raw[df_raw['Method'] == mname]
        summary.append({
            'Classifier': 'SVC (RBF, N_maj=3980)',
            'Method': mname,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}",
            'F1_mean': sub['F-measure'].mean()
        })
    df_sum = pd.DataFrame(summary)
    best_f1 = df_sum['F1_mean'].max()
    df_sum['Is_Best'] = ['BEST' if x == best_f1 else '' for x in df_sum['F1_mean']]
    df_sum.to_csv('results/rbf_svm_benchmark_summary.csv', index=False)
    print("\n=== S4(d) RBF SVC Summary (10 Seeds) ===")
    print(df_sum[['Classifier', 'Method', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC', 'Is_Best']].to_string(index=False))


def run_s5_haberman():
    print("\n" + "=" * 80)
    print("S5: Haberman Chronological Split (17 test minority samples) & 1000-resample Bootstrap CI...")
    df = pd.read_csv('datasets/haberman.csv')
    df = df.sort_values('Year').reset_index(drop=True)
    tr_sz = int(0.75 * len(df))
    train_df, test_df = df.iloc[:tr_sz], df.iloc[tr_sz:]
    feat_cols = [c for c in df.columns if c not in ['Year', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = get_binary_y(train_df, 'Class')
    t_tr = train_df['Year'].values.astype(float)
    X_te = test_df[feat_cols].values
    y_te = get_binary_y(test_df, 'Class')
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    n_min_test = np.sum(y_te == 1)
    print(f"  Train: {len(y_tr)} rows ({np.sum(y_tr == 1)} minority); Test: {len(y_te)} rows ({n_min_test} minority)")
    
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    records = []
    
    # 1. Evaluate Seed 42 models
    models = {}
    models['Baseline'] = DecisionTreeClassifier(random_state=42).fit(X_tr_s, y_tr)
    models['SMOTE'] = DecisionTreeClassifier(random_state=42).fit(*SMOTE(random_state=42).fit_resample(X_tr_s, y_tr))
    models['BorderlineSMOTE'] = DecisionTreeClassifier(random_state=42).fit(*BorderlineSMOTE(random_state=42).fit_resample(X_tr_s, y_tr))
    models['ADASYN'] = DecisionTreeClassifier(random_state=42).fit(*ADASYN(random_state=42).fit_resample(X_tr_s, y_tr))
    models['EC-EBO'] = DecisionTreeClassifier(random_state=42).fit(*ECEBO(random_state=42).fit_resample(X_tr_s, y_tr))
    models['DA-ECEBO'] = DecisionTreeClassifier(random_state=42).fit(*DAECEBO(k_windows=2, lambd=0.0, random_state=42).fit_resample(X_tr_s, y_tr, timestamps=t_tr))
    
    # 2. 1000-resample Bootstrap on the 77 test rows
    n_te = len(y_te)
    B = 1000
    rng_b = np.random.RandomState(42)
    b_indices = [rng_b.choice(n_te, size=n_te, replace=True) for _ in range(B)]
    
    for mname in methods:
        clf = models[mname]
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)
        base_m = compute_metrics(y_te, probs, preds)
        
        b_f1s = []
        b_aucs = []
        for b_idx in b_indices:
            y_b = y_te[b_idx]
            if len(np.unique(y_b)) < 2:
                continue
            pr_b = probs[b_idx]
            pd_b = preds[b_idx]
            m_b = compute_metrics(y_b, pr_b, pd_b)
            b_f1s.append(m_b['F-measure'])
            b_aucs.append(m_b['AUC'])
            
        f1_ci = np.percentile(b_f1s, [2.5, 97.5])
        auc_ci = np.percentile(b_aucs, [2.5, 97.5])
        
        records.append({
            'Method': mname,
            'Test_Minority_Samples': n_min_test,
            'Minority-F1': base_m['F-measure'],
            'Minority-F1_95_CI': f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]",
            'AUC': base_m['AUC'],
            'AUC_95_CI': f"[{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]",
            'AUC_lt_0.5': base_m['AUC'] < 0.5
        })
        
    df_s5 = pd.DataFrame(records)
    df_s5.to_csv('results/haberman_bootstrap_ci.csv', index=False)
    print("\n=== S5 Haberman Bootstrap CI Summary ===")
    print(df_s5[['Method', 'Test_Minority_Samples', 'Minority-F1', 'Minority-F1_95_CI', 'AUC', 'AUC_95_CI', 'AUC_lt_0.5']].to_string(index=False))


def main():
    os.makedirs('results', exist_ok=True)
    run_s4c_runtimes()
    run_s4d_rbf_svm()
    run_s5_haberman()


if __name__ == '__main__':
    main()
