"""
Part 2: Credit Card 1,000-Resample Test Bootstrap 95% Confidence Intervals
==========================================================================
Evaluates the test set (future 25% chronological slice, N=71,202, N_min=94)
under CART (DecisionTreeClassifier, seed 42) for core methods:
  - None
  - SMOTE
  - ECEBO (literal)
  - ECEBO_PSI0
  - DAECEBO2 (literal, K=4, lambda=0.25)
  - DAECEBO2_PSI0 (K=1, lambda=0.00)

Performs 1,000 bootstrap resamples of the test rows with replacement and reports:
  - Metric estimate on full test set
  - Bootstrap Mean, Std
  - 95% Bootstrap CI: [2.5th percentile, 97.5th percentile]
  for: Macro-Gmean, Minority-F1, Macro-F1, and AUC.

Output: results_stream/P2_creditcard_bootstrap_ci.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.abspath('.'))
from imblearn.over_sampling import SMOTE
from da_ecebo_and_ecebo_implementation import ECEBO
from ecebo_psi0_variant import ECEBO_PSI0
from da_ecebo_v2_candidate import DAECEBO2
from da_ecebo_v2_psi0 import DAECEBO2_PSI0

def compute_fast_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    gmean = np.sqrt(rec_maj * rec_min)
    
    f1_min = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    f1_maj = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) > 0 else 0.0
    macro_f1 = 0.5 * (f1_min + f1_maj)
    
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5
        
    return {
        'Macro-Gmean': float(gmean),
        'Minority-F1': float(f1_min),
        'Macro-F1': float(macro_f1),
        'AUC': float(auc)
    }

def main():
    print("Loading Credit Card dataset...")
    df = pd.read_parquet('datasets/creditcard.parquet')
    n_tr = int(len(df) * 0.75)
    
    df_tr = df.iloc[:n_tr]
    df_te = df.iloc[n_tr:]
    
    feats = [c for c in df.columns if c not in ['Time', 'Class']]
    X_tr = df_tr[feats].values.astype(np.float64)
    y_tr = df_tr['Class'].values.astype(int)
    t_tr = df_tr['Time'].values.astype(np.float64)
    
    X_te = df_te[feats].values.astype(np.float64)
    y_te = df_te['Class'].values.astype(int)
    
    print(f"Train: N={len(X_tr)} (Min={np.sum(y_tr==1)}) | Test: N={len(X_te)} (Min={np.sum(y_te==1)})")
    
    seed = 42
    n_min = int(np.sum(y_tr == 1))
    
    methods = {
        'None': (None, None),
        'SMOTE': (SMOTE(k_neighbors=5, random_state=seed), None),
        'ECEBO': (ECEBO(iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed), None),
        'ECEBO_PSI0': (ECEBO_PSI0(iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed), None),
        'DAECEBO2': (DAECEBO2(k_windows=4, lambd=0.25, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed), t_tr),
        'DAECEBO2_PSI0': (DAECEBO2_PSI0(k_windows=1, lambd=0.00, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed), None)
    }
    
    results = []
    n_test = len(y_te)
    n_boot = 1000
    rng_b = np.random.RandomState(42)
    boot_indices = [rng_b.choice(n_test, size=n_test, replace=True) for _ in range(n_boot)]
    
    for mname, (sampler, t_arg) in methods.items():
        print(f"\nEvaluating {mname}...")
        t0 = time.time()
        if sampler is None:
            X_res, y_res = X_tr, y_tr
        elif t_arg is not None:
            X_res, y_res = sampler.fit_resample(X_tr, y_tr, timestamps=t_arg)
        else:
            X_res, y_res = sampler.fit_resample(X_tr, y_tr)
        t_resamp = time.time() - t0
        print(f"Resampling took {t_resamp:.2f}s (N_res={len(X_res)})")
        
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_res, y_res)
        
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if len(clf.classes_) == 2 else y_pred.astype(float)
        
        # Point estimate
        pt_m = compute_fast_metrics(y_te, y_pred, y_prob)
        print(f"Point estimates: Gmean={pt_m['Macro-Gmean']:.4f}, Min-F1={pt_m['Minority-F1']:.4f}, AUC={pt_m['AUC']:.4f}")
        
        # Bootstrap
        b_metrics = {'Macro-Gmean': [], 'Minority-F1': [], 'Macro-F1': [], 'AUC': []}
        for b_idx in boot_indices:
            yt_b = y_te[b_idx]
            yp_b = y_pred[b_idx]
            ypr_b = y_prob[b_idx]
            bm = compute_fast_metrics(yt_b, yp_b, ypr_b)
            for k in b_metrics:
                b_metrics[k].append(bm[k])
                
        for met in ['Macro-Gmean', 'Minority-F1', 'Macro-F1', 'AUC']:
            arr = np.array(b_metrics[met])
            ci_low = float(np.percentile(arr, 2.5))
            ci_high = float(np.percentile(arr, 97.5))
            results.append({
                'dataset': 'creditcard',
                'classifier': 'CART',
                'method': mname,
                'metric': met,
                'point_estimate': round(pt_m[met], 4),
                'bootstrap_mean': round(float(np.mean(arr)), 4),
                'bootstrap_std': round(float(np.std(arr)), 4),
                'ci_95_lower': round(ci_low, 4),
                'ci_95_upper': round(ci_high, 4),
                'ci_95_string': f"[{ci_low:.4f}, {ci_high:.4f}]",
                'n_resamples': n_boot
            })
            
    df_out = pd.DataFrame(results)
    out_csv = 'results_stream/P2_creditcard_bootstrap_ci.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved Credit Card 1,000-resample bootstrap CI to {out_csv}")
    print(df_out[['method', 'metric', 'point_estimate', 'ci_95_string']].to_string())

if __name__ == '__main__':
    main()

