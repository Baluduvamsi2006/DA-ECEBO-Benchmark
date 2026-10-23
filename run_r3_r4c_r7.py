"""
Script to execute R3, R4(c), and R7 experiments:
- R3(a): Bootstrap (1000 resamples) 95% CI for Minority-F1 of Baseline, SMOTE, EC-EBO, DA-ECEBO,
         plus paired difference CI (DA-ECEBO minus EC-EBO, DA-ECEBO minus Baseline).
- R3(b): Rolling-origin protocol check (3 cut points: 60%, 70%, 75% train -> next 25% test).
- R4(c): Haberman chronological split on Year (first 75% / last 25%) over 10 seeds (42..51).
- R7: Electricity market streaming concept drift benchmark (45,312 rows, chronological 75/25 split).
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, compute_metrics


def run_r3_bootstrap_and_rolling():
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/figdata', exist_ok=True)
    print("=" * 80)
    print("R3(a): Computing Bootstrap (1000 resamples) 95% Confidence Intervals on CreditCard test set...")
    
    # Load CreditCard
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
    
    # Fit the 4 key models on seed 42 to get exact test prediction vectors
    print("Fitting models on seed 42 for test prediction vectors...")
    
    # 1. Baseline
    clf_base = DecisionTreeClassifier(random_state=42).fit(X_tr_s, y_tr)
    preds_base = clf_base.predict(X_te_s)
    probs_base = clf_base.predict_proba(X_te_s)[:, 1]
    
    # 2. SMOTE
    X_sm, y_sm = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
    clf_sm = DecisionTreeClassifier(random_state=42).fit(X_sm, y_sm)
    preds_sm = clf_sm.predict(X_te_s)
    probs_sm = clf_sm.predict_proba(X_te_s)[:, 1]
    
    # 3. EC-EBO
    X_ec, y_ec = ECEBO(random_state=42).fit_resample(X_tr_s, y_tr)
    clf_ec = DecisionTreeClassifier(random_state=42).fit(X_ec, y_ec)
    preds_ec = clf_ec.predict(X_te_s)
    probs_ec = clf_ec.predict_proba(X_te_s)[:, 1]
    
    # 4. DA-ECEBO (K=4, lambda=0.25)
    X_da4, y_da4 = DAECEBO(k_windows=4, lambd=0.25, random_state=42).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf_da4 = DecisionTreeClassifier(random_state=42).fit(X_da4, y_da4)
    preds_da4 = clf_da4.predict(X_te_s)
    probs_da4 = clf_da4.predict_proba(X_te_s)[:, 1]
    
    # 5. DA-ECEBO (K=2, lambda=0.0 - new CV grid winner)
    X_da2, y_da2 = DAECEBO(k_windows=2, lambd=0.0, random_state=42).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf_da2 = DecisionTreeClassifier(random_state=42).fit(X_da2, y_da2)
    preds_da2 = clf_da2.predict(X_te_s)
    probs_da2 = clf_da2.predict_proba(X_te_s)[:, 1]

    # Save seed 42 predictions and probabilities for Figure Data
    np.savez_compressed(
        'results/figdata/creditcard_all_models_seed42.npz',
        y_true=y_te,
        preds_base=preds_base, probs_base=probs_base,
        preds_sm=preds_sm, probs_sm=probs_sm,
        preds_ec=preds_ec, probs_ec=probs_ec,
        preds_da4=preds_da4, probs_da4=probs_da4,
        preds_da2=preds_da2, probs_da2=probs_da2
    )
    print("Saved results/figdata/creditcard_all_models_seed42.npz")
    
    # Bootstrap 1000 resamples
    print("Running 1000 bootstrap resamples on test set...")
    n_test = len(y_te)
    rng = np.random.RandomState(42)
    boot_indices = rng.randint(0, n_test, size=(1000, n_test))
    
    f1_base_boot, f1_sm_boot, f1_ec_boot, f1_da4_boot, f1_da2_boot = [], [], [], [], []
    diff_ec4_boot, diff_base4_boot = [], []
    diff_ec2_boot, diff_base2_boot = [], []
    
    for b in range(1000):
        idx = boot_indices[b]
        y_b = y_te[idx]
        
        # compute minority F1 for each
        def get_f1(p):
            tp = np.sum((y_b == 1) & (p[idx] == 1))
            fp = np.sum((y_b == 0) & (p[idx] == 1))
            fn = np.sum((y_b == 1) & (p[idx] == 0))
            denom = 2 * tp + fp + fn
            return (2.0 * tp / denom) if denom > 0 else 0.0
            
        f_b = get_f1(preds_base)
        f_s = get_f1(preds_sm)
        f_e = get_f1(preds_ec)
        f_d4 = get_f1(preds_da4)
        f_d2 = get_f1(preds_da2)
        
        f1_base_boot.append(f_b)
        f1_sm_boot.append(f_s)
        f1_ec_boot.append(f_e)
        f1_da4_boot.append(f_d4)
        f1_da2_boot.append(f_d2)
        diff_ec4_boot.append(f_d4 - f_e)
        diff_base4_boot.append(f_d4 - f_b)
        diff_ec2_boot.append(f_d2 - f_e)
        diff_base2_boot.append(f_d2 - f_b)
        
    def ci_str(arr):
        return f"[{np.percentile(arr, 2.5):.4f}, {np.percentile(arr, 97.5):.4f}]"
        
    bootstrap_df = pd.DataFrame([
        {'Model': 'Baseline', 'Minority-F1_Mean': np.mean(f1_base_boot), '95%_CI': ci_str(f1_base_boot)},
        {'Model': 'SMOTE', 'Minority-F1_Mean': np.mean(f1_sm_boot), '95%_CI': ci_str(f1_sm_boot)},
        {'Model': 'EC-EBO', 'Minority-F1_Mean': np.mean(f1_ec_boot), '95%_CI': ci_str(f1_ec_boot)},
        {'Model': 'DA-ECEBO (K=4, lam=0.25)', 'Minority-F1_Mean': np.mean(f1_da4_boot), '95%_CI': ci_str(f1_da4_boot)},
        {'Model': 'DA-ECEBO (K=2, lam=0.0)', 'Minority-F1_Mean': np.mean(f1_da2_boot), '95%_CI': ci_str(f1_da2_boot)},
        {'Model': 'Diff (DA[4,0.25] minus EC-EBO)', 'Minority-F1_Mean': np.mean(diff_ec4_boot), '95%_CI': ci_str(diff_ec4_boot)},
        {'Model': 'Diff (DA[4,0.25] minus Baseline)', 'Minority-F1_Mean': np.mean(diff_base4_boot), '95%_CI': ci_str(diff_base4_boot)},
        {'Model': 'Diff (DA[2,0.0] minus EC-EBO)', 'Minority-F1_Mean': np.mean(diff_ec2_boot), '95%_CI': ci_str(diff_ec2_boot)},
        {'Model': 'Diff (DA[2,0.0] minus Baseline)', 'Minority-F1_Mean': np.mean(diff_base2_boot), '95%_CI': ci_str(diff_base2_boot)}
    ])
    bootstrap_df.to_csv('results/test_bootstrap_ci.csv', index=False)
    bootstrap_df.to_csv('results/figdata/test_bootstrap_ci.csv', index=False)
    print("Saved results/test_bootstrap_ci.csv")
    print(bootstrap_df.to_string())
    
    # -------------------------------------------------------------
    # R3(b): Rolling-Origin Protocol Check (3 Cut Points)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("R3(b): Executing Rolling-Origin Protocol across 3 chronological cut points...")
    cut_points = [
        ('Cut 1: Train 60% -> Test next 25%', int(0.60 * n_total), int(0.25 * n_total)),
        ('Cut 2: Train 70% -> Test next 25%', int(0.70 * n_total), int(0.25 * n_total)),
        ('Cut 3: Train 75% -> Test next 25%', int(0.75 * n_total), int(0.25 * n_total))
    ]
    
    rolling_records = []
    for cut_name, tr_end, te_len in cut_points:
        te_end = min(n_total, tr_end + te_len)
        df_tr_cut = df.iloc[:tr_end]
        df_te_cut = df.iloc[tr_end:te_end]
        
        X_tr_c = df_tr_cut[feat_cols].values
        y_tr_c = df_tr_cut['Class'].values.astype(int)
        t_tr_c = df_tr_cut['Time'].values.astype(float)
        
        X_te_c = df_te_cut[feat_cols].values
        y_te_c = df_te_cut['Class'].values.astype(int)
        
        scl = StandardScaler()
        X_tr_cs = scl.fit_transform(X_tr_c)
        X_te_cs = scl.transform(X_te_c)
        
        # 1. Baseline
        clf_b = DecisionTreeClassifier(random_state=42).fit(X_tr_cs, y_tr_c)
        m_b = compute_metrics(y_te_c, clf_b.predict_proba(X_te_cs)[:, 1], clf_b.predict(X_te_cs))
        
        # 2. EC-EBO
        X_ec_c, y_ec_c = ECEBO(random_state=42).fit_resample(X_tr_cs, y_tr_c)
        clf_e = DecisionTreeClassifier(random_state=42).fit(X_ec_c, y_ec_c)
        m_e = compute_metrics(y_te_c, clf_e.predict_proba(X_te_cs)[:, 1], clf_e.predict(X_te_cs))
        
        # 3. DA-ECEBO (K=4, lambda=0.25)
        X_da4_c, y_da4_c = DAECEBO(k_windows=4, lambd=0.25, random_state=42).fit_resample(X_tr_cs, y_tr_c, timestamps=t_tr_c)
        clf_d4 = DecisionTreeClassifier(random_state=42).fit(X_da4_c, y_da4_c)
        m_d4 = compute_metrics(y_te_c, clf_d4.predict_proba(X_te_cs)[:, 1], clf_d4.predict(X_te_cs))
        
        # 4. DA-ECEBO (K=2, lambda=0.0)
        X_da2_c, y_da2_c = DAECEBO(k_windows=2, lambd=0.0, random_state=42).fit_resample(X_tr_cs, y_tr_c, timestamps=t_tr_c)
        clf_d2 = DecisionTreeClassifier(random_state=42).fit(X_da2_c, y_da2_c)
        m_d2 = compute_metrics(y_te_c, clf_d2.predict_proba(X_te_cs)[:, 1], clf_d2.predict(X_te_cs))
        
        rolling_records.append({
            'Cut_Point': cut_name,
            'Train_Size': len(df_tr_cut),
            'Test_Size': len(df_te_cut),
            'Test_Frauds': int(y_te_c.sum()),
            'Baseline_F1': m_b['F-measure'],
            'ECEBO_F1': m_e['F-measure'],
            'DAECEBO_F1_K4_L0.25': m_d4['F-measure'],
            'DAECEBO_F1_K2_L0': m_d2['F-measure'],
            'DA4_vs_EC_Diff': m_d4['F-measure'] - m_e['F-measure'],
            'DA2_vs_EC_Diff': m_d2['F-measure'] - m_e['F-measure']
        })
        print(f"  {cut_name:<34} | Frauds: {y_te_c.sum():<3} | DA[4,0.25] F1: {m_d4['F-measure']:.4f} | DA[2,0] F1: {m_d2['F-measure']:.4f} | EC F1: {m_e['F-measure']:.4f} | Base F1: {m_b['F-measure']:.4f}")
        
    rolling_df = pd.DataFrame(rolling_records)
    rolling_df.to_csv('results/rolling_origin_benchmark.csv', index=False)
    rolling_df.to_csv('results/figdata/rolling_origin_benchmark.csv', index=False)
    print("Saved results/rolling_origin_benchmark.csv")


def run_r4c_haberman():
    print("\n" + "=" * 80)
    print("R4(c): Running Chronological Evaluation on Haberman (sorted by Year, first 75% train / last 25% test)...")
    df = pd.read_csv('datasets/haberman.csv')
    # Haberman columns: Age, Year, Nodes, Survival
    target_col = [c for c in df.columns if any(k in c.lower() for k in ['surv', 'class', 'target'])][0]
    vc = df[target_col].value_counts()
    min_cls = vc.index[-1]
    
    # Sort chronologically by Year
    df_sorted = df.sort_values('Year').reset_index(drop=True)
    n = len(df_sorted)
    tr_sz = int(0.75 * n)
    train_df = df_sorted.iloc[:tr_sz]
    test_df = df_sorted.iloc[tr_sz:]
    
    feat_cols = [c for c in df.columns if c != target_col]
    X_tr = train_df[feat_cols].values.astype(float)
    y_tr = (train_df[target_col] == min_cls).astype(int).values
    t_tr = train_df['Year'].values.astype(float)
    
    X_te = test_df[feat_cols].values.astype(float)
    y_te = (test_df[target_col] == min_cls).astype(int).values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    print(f"Haberman Chronological: Train={len(X_tr)} (Year {t_tr.min():.0f}-{t_tr.max():.0f}, {y_tr.sum()} min) | Test={len(X_te)} ({y_te.sum()} min)")
    
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    seeds = list(range(42, 52))
    
    records = []
    for s in seeds:
        for m_name in methods:
            if m_name == 'Baseline':
                X_res, y_res = X_tr_s, y_tr
            elif m_name == 'SMOTE':
                X_res, y_res = SMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'BorderlineSMOTE':
                X_res, y_res = BorderlineSMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'ADASYN':
                X_res, y_res = ADASYN(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'EC-EBO':
                X_res, y_res = ECEBO(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'DA-ECEBO':
                X_res, y_res = DAECEBO(k_windows=4, lambd=0.25, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
                
            clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
            probs = clf.predict_proba(X_te_s)[:, 1]
            preds = clf.predict(X_te_s)
            m = compute_metrics(y_te, probs, preds)
            m['Seed'] = s
            m['Method'] = m_name
            records.append(m)
            
    df_hab = pd.DataFrame(records)
    summary_hab = []
    for m in methods:
        sub = df_hab[df_hab['Method'] == m]
        summary_hab.append({
            'Method': m,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-F1': f"{sub['Macro-F1'].mean():.4f} +/- {sub['Macro-F1'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}"
        })
    df_hab_sum = pd.DataFrame(summary_hab)
    df_hab.to_csv('results/haberman_chronological_raw.csv', index=False)
    df_hab_sum.to_csv('results/haberman_chronological_results.csv', index=False)
    print("Saved results/haberman_chronological_results.csv")
    print(df_hab_sum.to_string(index=False))


def run_r7_electricity():
    print("\n" + "=" * 80)
    print("R7: Running Concept Drift Benchmark on Real-Time Electricity Stream (45,312 rows)...")
    df = pd.read_csv('datasets/electricity.csv')
    # Target is 'class' ('UP' vs 'DOWN')
    target_col = 'class'
    y_raw = df[target_col].values
    min_cls = 'UP'  # 19,237 UP vs 26,075 DOWN
    y_all = (y_raw == min_cls).astype(int)
    
    feat_cols = [c for c in df.columns if c not in [target_col, 'date']]
    for c in feat_cols:
        if df[c].dtype == object or df[c].dtype.name == 'category':
            df[c] = pd.factorize(df[c])[0]
            
    X_all = df[feat_cols].values.astype(float)
    t_all = df['date'].values.astype(float) if 'date' in df.columns else np.arange(len(df), dtype=float)
    
    n = len(df)
    tr_sz = int(0.75 * n)
    X_tr, y_tr, t_tr = X_all[:tr_sz], y_all[:tr_sz], t_all[:tr_sz]
    X_te, y_te = X_all[tr_sz:], y_all[tr_sz:]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    print(f"Electricity: Train={len(X_tr)} (min: {y_tr.sum()}) | Test={len(X_te)} (min: {y_te.sum()})")
    
    methods = ['Baseline', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'EC-EBO', 'DA-ECEBO']
    seeds = list(range(42, 52))
    
    records = []
    for s in seeds:
        for m_name in methods:
            if m_name == 'Baseline':
                X_res, y_res = X_tr_s, y_tr
            elif m_name == 'SMOTE':
                X_res, y_res = SMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'BorderlineSMOTE':
                X_res, y_res = BorderlineSMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'ADASYN':
                X_res, y_res = ADASYN(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'EC-EBO':
                X_res, y_res = ECEBO(random_state=s).fit_resample(X_tr_s, y_tr)
            elif m_name == 'DA-ECEBO':
                X_res, y_res = DAECEBO(k_windows=4, lambd=0.25, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
                
            clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
            probs = clf.predict_proba(X_te_s)[:, 1]
            preds = clf.predict(X_te_s)
            m = compute_metrics(y_te, probs, preds)
            m['Seed'] = s
            m['Method'] = m_name
            records.append(m)
            
    df_elec = pd.DataFrame(records)
    summary_elec = []
    for m in methods:
        sub = df_elec[df_elec['Method'] == m]
        summary_elec.append({
            'Method': m,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-F1': f"{sub['Macro-F1'].mean():.4f} +/- {sub['Macro-F1'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}"
        })
    df_elec_sum = pd.DataFrame(summary_elec)
    df_elec.to_csv('results/electricity_benchmark_raw.csv', index=False)
    df_elec_sum.to_csv('results/electricity_benchmark_results.csv', index=False)
    print("Saved results/electricity_benchmark_results.csv")
    print(df_elec_sum.to_string(index=False))


if __name__ == '__main__':
    run_r3_bootstrap_and_rolling()
    run_r4c_haberman()
    run_r7_electricity()
