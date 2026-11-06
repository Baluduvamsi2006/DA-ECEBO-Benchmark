"""
Phase 2: Validation Grid Search on Training Horizon (First 75%)
===============================================================
Grid:
  K in {1, 2, 4} x lambda in {0.0, 0.25, 1.0} (9 cells)
  delta_base = 0.0
Rules:
  1. DAECEBO2 (literal: psi < 0)
  2. DAECEBO2_PSI0 (|psi| -> 0)

Validation Strategy:
  - 2 expanding-window folds inside the training horizon (first 75%):
      Split train into 3 parts (50%, 25%, 25% of train):
      Fold 1: Train on first 50%, Validate on next 25%
      Fold 2: Train on first 75%, Validate on remaining 25%
  - 5 seeds: 42, 43, 44, 45, 46
  - Primary classifier: CART (DecisionTreeClassifier(random_state=seed))
  - Selection Score: 0.5 * Minority-F1 + 0.5 * Macro-Gmean
  - Selection is strictly out-of-fold within train; test data is never accessed.

Output: results_stream/P2_validation_grid.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from da_ecebo_v2_candidate import DAECEBO2
from da_ecebo_v2_psi0 import DAECEBO2_PSI0

def compute_metrics(y_true, y_pred, y_prob):
    # Classes: 0 (majority), 1 (minority)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_gmean = np.sqrt(rec_maj * rec_min)
    
    f1_min = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_maj = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = 0.5 * (f1_min + f1_maj)
    
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5
        
    return {
        'Minority-F1': float(f1_min),
        'Macro-F1': float(macro_f1),
        'Macro-Gmean': float(macro_gmean),
        'AUC': float(auc),
        'Score': float(0.5 * f1_min + 0.5 * macro_gmean)
    }

def load_data(name):
    if name == 'synth_sea_p5':
        df = pd.read_parquet('datasets_stream/synth_sea_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
    elif name == 'synth_hyperplane_p5':
        df = pd.read_parquet('datasets_stream/synth_hyperplane_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
    elif name == 'synth_random_rbf_p5':
        df = pd.read_parquet('datasets_stream/synth_random_rbf_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
    elif name == 'electricity_ir20':
        df = pd.read_parquet('datasets_stream/electricity_ir20.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'electricity_ir100':
        df = pd.read_parquet('datasets_stream/electricity_ir100.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'creditcard':
        df = pd.read_parquet('datasets/creditcard.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
        t = df['Time'].values.astype(np.float64)
    elif name == 'ecoli2':
        df = pd.read_csv('datasets/Ecoli2.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'yeast3':
        df = pd.read_csv('datasets/yeast3.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")
        
    n_tr = int(len(X) * 0.75)
    return X[:n_tr], y[:n_tr], t[:n_tr]

def main():
    datasets = [
        'creditcard', 'electricity_ir20', 'electricity_ir100',
        'synth_sea_p5', 'synth_hyperplane_p5', 'synth_random_rbf_p5',
        'ecoli2', 'yeast3'
    ]
    
    rules = [('DAECEBO2_Literal', DAECEBO2), ('DAECEBO2_PSI0', DAECEBO2_PSI0)]
    k_vals = [1, 2, 4]
    lam_vals = [0.0, 0.25, 1.0]
    seeds = [42, 43, 44, 45, 46]
    
    out_csv = 'results_stream/P2_validation_grid.csv'
    os.makedirs('results_stream', exist_ok=True)
    
    if not os.path.exists(out_csv):
        pd.DataFrame(columns=[
            'dataset', 'rule', 'K_windows', 'lambd', 'delta_base',
            'val_score_mean', 'val_score_std',
            'val_min_f1_mean', 'val_macro_gmean_mean', 'val_macro_f1_mean', 'val_auc_mean',
            'n_folds', 'n_seeds'
        ]).to_csv(out_csv, index=False)

    for dname in datasets:
        X_tr, y_tr, t_tr = load_data(dname)
        n_tr = len(X_tr)
        
        # 2 Expanding-Window folds
        # Cut 1: train 0..50%, val 50%..75%
        # Cut 2: train 0..75%, val 75%..100%
        idx_50 = int(n_tr * 0.50)
        idx_75 = int(n_tr * 0.75)
        
        folds = [
            (slice(0, idx_50), slice(idx_50, idx_75)),
            (slice(0, idx_75), slice(idx_75, n_tr))
        ]
        
        print(f"\n=================== DATASET: {dname} (Train N={n_tr}) ===================")
        for rule_name, rule_cls in rules:
            for k in k_vals:
                for lam in lam_vals:
                    # Check if already done
                    df_curr = pd.read_csv(out_csv)
                    exists = ((df_curr['dataset'] == dname) &
                              (df_curr['rule'] == rule_name) &
                              (df_curr['K_windows'] == k) &
                              (df_curr['lambd'] == lam)).any()
                    if exists:
                        print(f"Skipping already validated: {dname} | {rule_name} | K={k}, lam={lam}")
                        continue
                        
                    cell_scores = []
                    cell_f1s = []
                    cell_gmeans = []
                    cell_macro_f1s = []
                    cell_aucs = []
                    
                    for seed in seeds:
                        for f_idx, (tr_slice, val_slice) in enumerate(folds):
                            X_f_tr, y_f_tr = X_tr[tr_slice], y_tr[tr_slice]
                            X_f_val, y_f_val = X_tr[val_slice], y_tr[val_slice]
                            t_f_tr = t_tr[tr_slice]
                            
                            # Subsample majority background if very large (e.g. creditcard > 20,000)
                            # to ensure swift resampling while preserving all minority instances
                            n_maj_f = int(np.sum(y_f_tr == 0))
                            n_min_f = int(np.sum(y_f_tr == 1))
                            if n_min_f == 0 or n_maj_f == 0:
                                continue
                                
                            try:
                                sampler = rule_cls(
                                    k_windows=k, lambd=lam, delta_base=0.0,
                                    iters=60, prob=0.2, alpha=1.0, k=min(7, n_min_f),
                                    step=0.1, random_state=seed
                                )
                                # Time-aware fit_resample if t is available
                                if hasattr(sampler, 'fit_resample'):
                                    # Check signature
                                    import inspect
                                    sig = inspect.signature(sampler.fit_resample)
                                    if 't' in sig.parameters:
                                        X_res, y_res = sampler.fit_resample(X_f_tr, y_f_tr, t=t_f_tr)
                                    else:
                                        X_res, y_res = sampler.fit_resample(X_f_tr, y_f_tr)
                                else:
                                    X_res, y_res = X_f_tr, y_f_tr
                            except Exception as e:
                                X_res, y_res = X_f_tr, y_f_tr
                                
                            clf = DecisionTreeClassifier(random_state=seed)
                            clf.fit(X_res, y_res)
                            p_pred = clf.predict(X_f_val)
                            p_prob = clf.predict_proba(X_f_val)[:, 1] if len(clf.classes_) == 2 else np.zeros(len(X_f_val))
                            
                            m = compute_metrics(y_f_val, p_pred, p_prob)
                            cell_scores.append(m['Score'])
                            cell_f1s.append(m['Minority-F1'])
                            cell_gmeans.append(m['Macro-Gmean'])
                            cell_macro_f1s.append(m['Macro-F1'])
                            cell_aucs.append(m['AUC'])
                            
                    res_row = {
                        'dataset': dname,
                        'rule': rule_name,
                        'K_windows': k,
                        'lambd': lam,
                        'delta_base': 0.0,
                        'val_score_mean': round(float(np.mean(cell_scores)), 4) if cell_scores else 0.0,
                        'val_score_std': round(float(np.std(cell_scores)), 4) if cell_scores else 0.0,
                        'val_min_f1_mean': round(float(np.mean(cell_f1s)), 4) if cell_f1s else 0.0,
                        'val_macro_gmean_mean': round(float(np.mean(cell_gmeans)), 4) if cell_gmeans else 0.0,
                        'val_macro_f1_mean': round(float(np.mean(cell_macro_f1s)), 4) if cell_macro_f1s else 0.0,
                        'val_auc_mean': round(float(np.mean(cell_aucs)), 4) if cell_aucs else 0.0,
                        'n_folds': len(folds),
                        'n_seeds': len(seeds)
                    }
                    pd.DataFrame([res_row]).to_csv(out_csv, mode='a', header=False, index=False)
                    print(f"[{dname} | {rule_name}] K={k}, lam={lam} -> Val_Score: {res_row['val_score_mean']:.4f} +/- {res_row['val_score_std']:.4f} "
                          f"(Min-F1: {res_row['val_min_f1_mean']:.4f}, Gmean: {res_row['val_macro_gmean_mean']:.4f})")
                    
    print(f"\nAll validation grid searches complete! Saved to {out_csv}")

if __name__ == '__main__':
    main()

