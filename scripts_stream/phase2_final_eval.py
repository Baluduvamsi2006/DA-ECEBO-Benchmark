"""
Phase 2: Master Benchmark Final Evaluation (10 Seeds, 12 Methods + Sanity Row)
=============================================================================
Datasets:
  1. creditcard (primary, real time column, chronological 75/25) - Natural
  2. electricity_ir20 (imbalance induced by subsampling)
  3. electricity_ir100 (imbalance induced by subsampling)
  4. synth_sea_p5 (synthetic)
  5. synth_hyperplane_p5 (synthetic)
  6. synth_random_rbf_p5 (synthetic, negative control)
  7. ecoli2 (row order)
  8. yeast3 (row order)

Methods:
  1. None
  2. SMOTE
  3. BorderlineSMOTE
  4. ADASYN
  5. G-SMOTE
  6. CCR
  7. A-SUWO
  8. MLOS (majority capped at 20,000 on all datasets)
  9. ECEBO (literal)
  10. ECEBO_PSI0
  11. DAECEBO2 (literal, using validated (K*, lambda*) from P2_validation_grid.csv, delta_base=0.0)
  12. DAECEBO2_PSI0 (using validated (K*, lambda*) from P2_validation_grid.csv, delta_base=0.0)
  13. DAECEBO2_sanity_K1_lam1 (Sanity check: K=1, lambda=1.0, delta_base=0.0, literal; must equal ECEBO)

Classifiers:
  - CART (primary): DecisionTreeClassifier(random_state=seed)
  - KNN: KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
  - GaussianNB: GaussianNB()
  - SVC: SVC(probability=True, random_state=seed, max_iter=2000)

Seeds: 42 to 51 (10 seeds)

Evaluations:
  - Future 25% test set (chronological for stream sets, last 25% for row-order sets)
  - 5 later non-overlapping 10% blocks covering the stream horizon (for stream sets)

Outputs:
  - results_stream/P2_raw.csv (one row per dataset, classifier, method, seed)
  - results_stream/P2_blocks.csv (one row per stream dataset, block, classifier, method, seed)
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score

sys.path.insert(0, os.path.abspath('.'))
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import smote_variants as sv
from scripts_stream.MLOS import MLOS
from da_ecebo_and_ecebo_implementation import ECEBO
from ecebo_psi0_variant import ECEBO_PSI0
from da_ecebo_v2_candidate import DAECEBO2
from da_ecebo_v2_psi0 import DAECEBO2_PSI0

def log_print(msg):
    print(msg, flush=True)
    try:
        with open('results_stream/eval.log', 'a', encoding='utf-8') as f:
            f.write(str(msg) + '\n')
    except Exception:
        pass

def compute_all_metrics(y_true, y_pred, y_prob):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_gmean = np.sqrt(rec_maj * rec_min)
    
    prec_min = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_min_met = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    f1_min = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_maj = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    macro_f1 = 0.5 * (f1_min + f1_maj)
    
    try:
        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5
        
    return {
        'Macro-F1': float(macro_f1),
        'Macro-Gmean': float(macro_gmean),
        'Minority-F1': float(f1_min),
        'Precision': float(prec_min),
        'Recall': float(rec_min_met),
        'AUC': float(auc)
    }

def load_dataset(name):
    if name == 'synth_sea_p5':
        df = pd.read_parquet('datasets_stream/synth_sea_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
        cat_desc = 'synthetic'
    elif name == 'synth_hyperplane_p5':
        df = pd.read_parquet('datasets_stream/synth_hyperplane_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
        cat_desc = 'synthetic'
    elif name == 'synth_random_rbf_p5':
        df = pd.read_parquet('datasets_stream/synth_random_rbf_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
        t = df['Time_Step'].values.astype(np.float64)
        cat_desc = 'synthetic'
    elif name == 'electricity_ir20':
        df = pd.read_parquet('datasets_stream/electricity_ir20.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
        cat_desc = 'imbalance induced by subsampling'
    elif name == 'electricity_ir100':
        df = pd.read_parquet('datasets_stream/electricity_ir100.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
        cat_desc = 'imbalance induced by subsampling'
    elif name == 'creditcard':
        df = pd.read_parquet('datasets/creditcard.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
        t = df['Time'].values.astype(np.float64)
        cat_desc = 'Natural'
    elif name == 'ecoli2':
        df = pd.read_csv('datasets/Ecoli2.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
        cat_desc = 'row order'
    elif name == 'yeast3':
        df = pd.read_csv('datasets/yeast3.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
        cat_desc = 'row order'
    else:
        raise ValueError(f"Unknown dataset {name}")
        
    return X, y, t, cat_desc

def get_best_hyperparams(dname, rule_name):
    # Reads best (K, lambda) from results_stream/P2_validation_grid.csv
    grid_csv = 'results_stream/P2_validation_grid.csv'
    if os.path.exists(grid_csv):
        df_grid = pd.read_csv(grid_csv)
        sub = df_grid[(df_grid['dataset'] == dname) & (df_grid['rule'] == rule_name)]
        if len(sub) > 0:
            best_row = sub.sort_values(by='val_score_mean', ascending=False).iloc[0]
            return int(best_row['K_windows']), float(best_row['lambd'])
    # Default fallbacks if grid row not found
    return 2, 0.25

def sample_dataset(mname, X_tr, y_tr, t_tr, seed, dname):
    n_min = int(np.sum(y_tr == 1))
    n_maj = int(np.sum(y_tr == 0))
    k_nn = max(1, min(5, n_min - 1))
    
    if mname == 'None':
        return X_tr, y_tr
    elif mname == 'SMOTE':
        return SMOTE(k_neighbors=k_nn, random_state=seed).fit_resample(X_tr, y_tr)
    elif mname == 'BorderlineSMOTE':
        return BorderlineSMOTE(k_neighbors=k_nn, random_state=seed).fit_resample(X_tr, y_tr)
    elif mname == 'ADASYN':
        return ADASYN(n_neighbors=k_nn, random_state=seed).fit_resample(X_tr, y_tr)
    elif mname == 'G-SMOTE':
        return sv.G_SMOTE(random_state=seed).sample(X_tr, y_tr)
    elif mname == 'CCR':
        return sv.CCR(random_state=seed).sample(X_tr, y_tr)
    elif mname == 'A-SUWO':
        if n_maj > 20000:
            rng = np.random.RandomState(seed)
            maj_idx = np.where(y_tr == 0)[0]
            min_idx = np.where(y_tr == 1)[0]
            sub_maj = rng.choice(maj_idx, size=20000, replace=False)
            keep_idx = np.concatenate([sub_maj, min_idx])
            X_in, y_in = X_tr[keep_idx], y_tr[keep_idx]
        else:
            X_in, y_in = X_tr, y_tr
        return sv.A_SUWO(random_state=seed).sample(X_in, y_in)
    elif mname == 'MLOS':
        # Majority capped at 20,000 on all datasets
        if n_maj > 20000:
            rng = np.random.RandomState(seed)
            maj_idx = np.where(y_tr == 0)[0]
            min_idx = np.where(y_tr == 1)[0]
            sub_maj = rng.choice(maj_idx, size=20000, replace=False)
            keep_idx = np.concatenate([sub_maj, min_idx])
            X_in, y_in = X_tr[keep_idx], y_tr[keep_idx]
            delta_n = max(0, 20000 - n_min)
        else:
            X_in, y_in = X_tr, y_tr
            delta_n = max(0, n_maj - n_min)
        sw = MLOS(minClass=1, len=1)
        return sw.fit_Sample(X_in, y_in, delta_n)
    elif mname == 'ECEBO':
        return ECEBO(iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed).fit_resample(X_tr, y_tr)
    elif mname == 'ECEBO_PSI0':
        return ECEBO_PSI0(iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed).fit_resample(X_tr, y_tr)
    elif mname == 'DAECEBO2':
        k_val, lam_val = get_best_hyperparams(dname, 'DAECEBO2_Literal')
        m = DAECEBO2(k_windows=k_val, lambd=lam_val, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed)
        return m.fit_resample(X_tr, y_tr, timestamps=t_tr)
    elif mname == 'DAECEBO2_PSI0':
        k_val, lam_val = get_best_hyperparams(dname, 'DAECEBO2_PSI0')
        m = DAECEBO2_PSI0(k_windows=k_val, lambd=lam_val, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed)
        return m.fit_resample(X_tr, y_tr)
    elif mname == 'DAECEBO2_Sanity_K1_lam1':
        m = DAECEBO2(k_windows=1, lambd=1.0, delta_base=0.0, iters=60, prob=0.2, alpha=1.0, k=min(7, n_min), step=0.1, random_state=seed)
        return m.fit_resample(X_tr, y_tr, timestamps=t_tr)
    else:
        raise ValueError(f"Unknown method {mname}")

def get_classifier(cname, seed):
    if cname == 'CART':
        return DecisionTreeClassifier(random_state=seed)
    elif cname == 'KNN':
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    elif cname == 'GaussianNB':
        return GaussianNB()
    elif cname == 'SVC':
        # SVC capped at max_iter=2000
        return SVC(probability=False, random_state=seed, max_iter=2000)
    else:
        raise ValueError(f"Unknown classifier {cname}")

def main():
    datasets = [
        'ecoli2', 'yeast3', 'synth_sea_p5', 'synth_hyperplane_p5',
        'synth_random_rbf_p5', 'electricity_ir100', 'electricity_ir20', 'creditcard'
    ]
    
    methods = [
        'None', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'G-SMOTE', 'CCR', 'A-SUWO',
        'MLOS', 'ECEBO', 'ECEBO_PSI0', 'DAECEBO2', 'DAECEBO2_PSI0', 'DAECEBO2_Sanity_K1_lam1'
    ]
    
    classifiers = ['CART', 'KNN', 'GaussianNB', 'SVC']
    seeds = list(range(42, 52)) # 42..51 (10 seeds)
    
    raw_csv = 'results_stream/P2_raw.csv'
    blocks_csv = 'results_stream/P2_blocks.csv'
    os.makedirs('results_stream', exist_ok=True)
    
    if not os.path.exists(raw_csv):
        pd.DataFrame(columns=[
            'dataset', 'dataset_type', 'classifier', 'method', 'seed',
            'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC',
            'N_train', 'N_min_train', 'N_test', 'N_min_test',
            'K_used', 'lambda_used', 'delta_base_used'
        ]).to_csv(raw_csv, index=False)
        
    if not os.path.exists(blocks_csv):
        pd.DataFrame(columns=[
            'dataset', 'dataset_type', 'block_id', 'block_range', 'classifier', 'method', 'seed',
            'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC',
            'N_block', 'N_min_block'
        ]).to_csv(blocks_csv, index=False)
        
    for dname in datasets:
        X, y, t, cat_desc = load_dataset(dname)
        n_total = len(X)
        n_tr = int(0.75 * n_total)
        
        X_tr, y_tr, t_tr = X[:n_tr], y[:n_tr], t[:n_tr]
        X_te, y_te, t_te = X[n_tr:], y[n_tr:], t[n_tr:]
        
        n_min_tr = int(np.sum(y_tr == 1))
        n_min_te = int(np.sum(y_te == 1))
        
        # Define 5 non-overlapping 10% blocks from 50% to 100% of stream (for stream sets)
        is_stream = cat_desc != 'row order'
        blocks = []
        if is_stream:
            # 5 blocks of 10% each: [0.50, 0.60], [0.60, 0.70], [0.70, 0.80], [0.80, 0.90], [0.90, 1.00]
            b_size = int(n_total * 0.10)
            for b_i in range(5):
                b_start = int(n_total * (0.50 + b_i * 0.10))
                b_end = int(n_total * (0.50 + (b_i + 1) * 0.10)) if b_i < 4 else n_total
                blocks.append((b_i + 1, f"{int(50 + b_i*10)}%-{int(50 + (b_i+1)*10)}%", X[b_start:b_end], y[b_start:b_end]))
                
        log_print(f"\n=======================================================")
        log_print(f"BENCHMARKING: {dname} ({cat_desc})")
        log_print(f"Train: N={n_tr} (Min={n_min_tr}) | Test: N={len(X_te)} (Min={n_min_te})")
        log_print(f"=======================================================")
        
        for seed in seeds:
            for mname in methods:
                # Check if all 4 classifiers already logged for this (dataset, method, seed)
                df_curr = pd.read_csv(raw_csv, keep_default_na=False)
                done_clfs = df_curr[(df_curr['dataset'] == dname) &
                                    (df_curr['method'] == mname) &
                                    (df_curr['seed'] == seed)]['classifier'].tolist()
                if set(classifiers).issubset(set(done_clfs)):
                    continue
                    
                # Resample once per seed/method
                k_u, lam_u, db_u = np.nan, np.nan, 0.0
                if mname in ['DAECEBO2', 'DAECEBO2_PSI0']:
                    k_u, lam_u = get_best_hyperparams(dname, 'DAECEBO2_Literal' if mname == 'DAECEBO2' else 'DAECEBO2_PSI0')
                elif mname == 'DAECEBO2_Sanity_K1_lam1':
                    k_u, lam_u = 1, 1.0
                    
                try:
                    t_s0 = time.time()
                    X_res, y_res = sample_dataset(mname, X_tr, y_tr, t_tr, seed, dname)
                    t_samp = time.time() - t_s0
                except Exception as e:
                    print(f"Sampling failed {mname} on {dname} seed {seed}: {e}")
                    X_res, y_res = X_tr, y_tr
                    t_samp = 0.0
                    
                # Evaluate each classifier
                for cname in classifiers:
                    if cname in done_clfs:
                        continue
                        
                    # Cap training set to 20,000 for creditcard KNN and SVC as authorized
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
                        print(f"Classify fail {cname} {mname} on {dname} seed {seed}: {e}")
                        m = {'Macro-F1': 0.0, 'Macro-Gmean': 0.0, 'Minority-F1': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'AUC': 0.5}
                        
                    row = {
                        'dataset': dname,
                        'dataset_type': cat_desc,
                        'classifier': cname,
                        'method': mname,
                        'seed': seed,
                        'Macro-F1': round(m['Macro-F1'], 4),
                        'Macro-Gmean': round(m['Macro-Gmean'], 4),
                        'Minority-F1': round(m['Minority-F1'], 4),
                        'Precision': round(m['Precision'], 4),
                        'Recall': round(m['Recall'], 4),
                        'AUC': round(m['AUC'], 4),
                        'N_train': n_tr,
                        'N_min_train': n_min_tr,
                        'N_test': len(X_te),
                        'N_min_test': n_min_te,
                        'K_used': k_u,
                        'lambda_used': lam_u,
                        'delta_base_used': db_u
                    }
                    pd.DataFrame([row]).to_csv(raw_csv, mode='a', header=False, index=False)
                    log_print(f"[{dname} | {cname} | {mname} | seed {seed}] G-mean: {m['Macro-Gmean']:.4f}, Min-F1: {m['Minority-F1']:.4f}, AUC: {m['AUC']:.4f}")
                    
                    # Evaluate on the 5 stream blocks if stream dataset
                    if is_stream:
                        for b_id, b_range, X_b, y_b in blocks:
                            try:
                                y_b_pred = clf.predict(X_b)
                                if hasattr(clf, 'predict_proba'):
                                    y_b_prob = clf.predict_proba(X_b)[:, 1]
                                elif hasattr(clf, 'decision_function'):
                                    y_b_prob = clf.decision_function(X_b)
                                else:
                                    y_b_prob = y_b_pred.astype(float)
                                b_m = compute_all_metrics(y_b, y_b_pred, y_b_prob)
                            except:
                                b_m = {'Macro-F1': 0.0, 'Macro-Gmean': 0.0, 'Minority-F1': 0.0, 'Precision': 0.0, 'Recall': 0.0, 'AUC': 0.5}
                                
                            b_row = {
                                'dataset': dname,
                                'dataset_type': cat_desc,
                                'block_id': b_id,
                                'block_range': b_range,
                                'classifier': cname,
                                'method': mname,
                                'seed': seed,
                                'Macro-F1': round(b_m['Macro-F1'], 4),
                                'Macro-Gmean': round(b_m['Macro-Gmean'], 4),
                                'Minority-F1': round(b_m['Minority-F1'], 4),
                                'Precision': round(b_m['Precision'], 4),
                                'Recall': round(b_m['Recall'], 4),
                                'AUC': round(b_m['AUC'], 4),
                                'N_block': len(X_b),
                                'N_min_block': int(np.sum(y_b == 1))
                            }
                            pd.DataFrame([b_row]).to_csv(blocks_csv, mode='a', header=False, index=False)

    log_print(f"\nFinal Phase 2 benchmark evaluation complete! Saved to {raw_csv} and {blocks_csv}")

if __name__ == '__main__':
    main()

