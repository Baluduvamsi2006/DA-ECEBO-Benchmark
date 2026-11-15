"""
Phase 3 Part 0: Re-run Fixed MLOS Across All Phase 2 Datasets
=============================================================
Re-evaluates MLOS_Fixed across all 8 datasets x 4 classifiers x 10 seeds (320 runs).
Logs each finished run to results_stream/P3_mlos_fixed.csv.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

sys.path.insert(0, os.path.abspath('.'))
from scripts_stream.mlos_fixed import MLOS_Fixed

def compute_all_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_gmean = np.sqrt(rec_maj * rec_min)
    prec_min = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_min_met = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    min_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5
    return {
        'Macro-F1': macro_f1,
        'Macro-Gmean': macro_gmean,
        'Minority-F1': min_f1,
        'Precision': prec_min,
        'Recall': rec_min_met,
        'AUC': auc
    }

def get_classifier(cname, seed):
    if cname == 'CART':
        return DecisionTreeClassifier(random_state=seed)
    elif cname == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif cname == 'GaussianNB':
        return GaussianNB()
    elif cname == 'SVC':
        return SVC(probability=True, random_state=seed, max_iter=2000)
    raise ValueError(f"Unknown classifier {cname}")

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
        df = pd.read_csv('datasets/Ecoli2.csv').sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'yeast3':
        df = pd.read_csv('datasets/yeast3.csv').sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return X, y, t

def main():
    out_csv = 'results_stream/P3_mlos_fixed.csv'
    cols = ['dataset', 'classifier', 'method', 'seed', 'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC', 'N_train', 'N_min_train', 'N_test', 'N_min_test']
    
    if os.path.exists(out_csv):
        df_done = pd.read_csv(out_csv, keep_default_na=False)
    else:
        df_done = pd.DataFrame(columns=cols)
        df_done.to_csv(out_csv, index=False)

    datasets = ['ecoli2', 'yeast3', 'synth_sea_p5', 'synth_hyperplane_p5', 'synth_random_rbf_p5', 'electricity_ir100', 'electricity_ir20', 'creditcard']
    classifiers = ['CART', 'KNN', 'GaussianNB', 'SVC']
    seeds = list(range(42, 52))

    for dname in datasets:
        X, y, t = load_data(dname)
        n_tr = int(0.75 * len(X))
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]

        n_min_tr = int(np.sum(y_tr == 1))
        n_maj_tr = int(np.sum(y_tr == 0))
        n_min_te = int(np.sum(y_te == 1))
        n_te = len(X_te)

        print(f"\nEvaluating MLOS_Fixed on {dname} (Train={n_tr}, Min={n_min_tr})...")

        for seed in seeds:
            df_curr = pd.read_csv(out_csv, keep_default_na=False)
            done_clfs = df_curr[(df_curr['dataset'] == dname) & (df_curr['seed'] == seed)]['classifier'].tolist()
            if set(classifiers).issubset(set(done_clfs)):
                continue

            # Subsample majority for MLOS if > 20000
            if n_maj_tr > 20000:
                rng = np.random.RandomState(seed)
                maj_idx = np.where(y_tr == 0)[0]
                min_idx = np.where(y_tr == 1)[0]
                sub_maj = rng.choice(maj_idx, size=20000, replace=False)
                keep_idx = np.concatenate([sub_maj, min_idx])
                X_in, y_in = X_tr[keep_idx], y_tr[keep_idx]
                delta_n = max(0, 20000 - n_min_tr)
            else:
                X_in, y_in = X_tr, y_tr
                delta_n = max(0, n_maj_tr - n_min_tr)

            sw = MLOS_Fixed(minClass=1, len=1, random_state=seed)
            X_res, y_res = sw.fit_Sample(X_in, y_in, delta_n)

            for cname in classifiers:
                if cname in done_clfs:
                    continue

                if dname == 'creditcard' and cname in ['KNN', 'SVC'] and len(X_res) > 20000:
                    rng_sub = np.random.RandomState(seed)
                    maj_idx = np.where(y_res == 0)[0]
                    min_idx = np.where(y_res == 1)[0]
                    n_c0 = min(len(maj_idx), 10000)
                    n_c1 = min(len(min_idx), 10000)
                    if n_c0 < 10000:
                        n_c1 = min(len(min_idx), 20000 - n_c0)
                    elif n_c1 < 10000:
                        n_c0 = min(len(maj_idx), 20000 - n_c1)
                    sub_maj = rng_sub.choice(maj_idx, size=n_c0, replace=False) if n_c0 > 0 else np.array([], dtype=int)
                    sub_min = rng_sub.choice(min_idx, size=n_c1, replace=False) if n_c1 > 0 else np.array([], dtype=int)
                    keep_idx = np.concatenate([sub_maj, sub_min])
                    rng_sub.shuffle(keep_idx)
                    X_clf_tr, y_clf_tr = X_res[keep_idx], y_res[keep_idx]
                else:
                    X_clf_tr, y_clf_tr = X_res, y_res

                clf = get_classifier(cname, seed)
                try:
                    clf.fit(X_clf_tr, y_clf_tr)
                    y_pred = clf.predict(X_te)
                    if hasattr(clf, 'predict_proba'):
                        y_prob = clf.predict_proba(X_te)[:, 1]
                    elif hasattr(clf, 'decision_function'):
                        y_prob = clf.decision_function(X_te)
                    else:
                        y_prob = y_pred.astype(float)
                    m = compute_all_metrics(y_te, y_pred, y_prob)
                except Exception as e:
                    print(f"Eval error {cname} {dname} seed {seed}: {e}")
                    m = {'Macro-F1': 0.0, 'Macro-Gmean': 0.0, 'Minority-F1': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'AUC': 0.5}

                row = {
                    'dataset': dname,
                    'classifier': cname,
                    'method': 'MLOS_Fixed',
                    'seed': seed,
                    'Macro-F1': round(m['Macro-F1'], 4),
                    'Macro-Gmean': round(m['Macro-Gmean'], 4),
                    'Minority-F1': round(m['Minority-F1'], 4),
                    'Precision': round(m['Precision'], 4),
                    'Recall': round(m['Recall'], 4),
                    'AUC': round(m['AUC'], 4),
                    'N_train': n_tr,
                    'N_min_train': n_min_tr,
                    'N_test': n_te,
                    'N_min_test': n_min_te
                }
                pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
                print(f"[{dname} | {cname} | MLOS_Fixed | seed {seed}] G-mean: {m['Macro-Gmean']:.4f}, Min-F1: {m['Minority-F1']:.4f}, AUC: {m['AUC']:.4f}", flush=True)

    print(f"\nAll done! Results saved to {out_csv}")

if __name__ == '__main__':
    main()
