"""
Phase 3 Part 3: Fair Hyperparameter Search & Selection Protocol
===============================================================
Performs fair hyperparameter search strictly inside the training split (using 2 expanding folds and 3 seeds):
  - DAECEBO2_AUTO: K in {1, 2, 4}, lambda in {0.0, 0.25, 0.5, 1.0}, rule in {'literal', 'psi0'},
                  rho in {0.1, 0.25, 0.5, 1.0}, step_scale in {0.5, 1.0, 2.0} via Successive Halving.
  - Baseline Competitors (SMOTE, BorderlineSMOTE, ADASYN, G-SMOTE, CCR, A-SUWO, MLOS_Fixed, ECEBO, ECEBO_PSI0):
                  rho in {0.1, 0.25, 0.5, 1.0}.

Saves:
  - results_stream/P3_validation_grid.csv
  - results_stream/P3_selected_params.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import smote_variants as sv

sys.path.insert(0, os.path.abspath('.'))
from scripts_stream.mlos_fixed import MLOS_Fixed
from da_ecebo_and_ecebo_implementation import ECEBO
from ecebo_psi0_variant import ECEBO_PSI0
from scripts_stream.da_ecebo_v3_auto import DAECEBO2_AUTO

def score_clf(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    gmean = np.sqrt(rec_maj * rec_min)
    mf1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    return 0.5 * gmean + 0.5 * mf1

def sample_method(mname, X, y, t, seed, rho=1.0, k_w=4, lam=0.25, rule='psi0', step_s=1.0):
    classes, counts = np.unique(y, return_counts=True)
    maj_c, min_c = (classes[0], classes[1]) if counts[0] > counts[1] else (classes[1], classes[0])
    n_maj, n_min = np.sum(y == maj_c), np.sum(y == min_c)
    k_nn = max(1, min(5, n_min - 1))
    
    target_min = n_min + rho * (n_maj - n_min)
    samp_strat = min(1.0, max(0.01, target_min / n_maj))

    if mname == 'None':
        return X, y
    elif mname == 'SMOTE':
        return SMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn, random_state=seed).fit_resample(X, y)
    elif mname == 'BorderlineSMOTE':
        return BorderlineSMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn, random_state=seed).fit_resample(X, y)
    elif mname == 'ADASYN':
        try:
            return ADASYN(sampling_strategy=samp_strat, n_neighbors=k_nn, random_state=seed).fit_resample(X, y)
        except Exception:
            return SMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn, random_state=seed).fit_resample(X, y)
    elif mname == 'G-SMOTE':
        try:
            return sv.G_SMOTE(proportion=rho, random_state=seed).sample(X, y)
        except Exception:
            return X, y
    elif mname == 'CCR':
        try:
            return sv.CCR(proportion=rho, random_state=seed).sample(X, y)
        except Exception:
            return X, y
    elif mname == 'A-SUWO':
        try:
            if n_maj > 10000:
                rng = np.random.RandomState(seed)
                maj_idx = np.where(y == 0)[0]
                min_idx = np.where(y == 1)[0]
                sub_maj = rng.choice(maj_idx, size=10000, replace=False)
                keep = np.concatenate([sub_maj, min_idx])
                return sv.A_SUWO(proportion=rho, random_state=seed).sample(X[keep], y[keep])
            return sv.A_SUWO(proportion=rho, random_state=seed).sample(X, y)
        except Exception:
            return X, y
    elif mname == 'MLOS_Fixed':
        X_in, y_in = X, y
        if n_maj > 20000:
            rng_m = np.random.RandomState(seed)
            maj_idx = np.where(y == maj_c)[0]
            min_idx = np.where(y == min_c)[0]
            sub_maj = rng_m.choice(maj_idx, size=20000, replace=False)
            keep = np.sort(np.concatenate([sub_maj, min_idx]))
            X_in, y_in = X[keep], y[keep]
            delta_n = max(0, int(np.round(rho * (20000 - n_min))))
        else:
            delta_n = max(0, int(np.round(rho * (n_maj - n_min))))
        sw = MLOS_Fixed(minClass=1, len=1, random_state=seed)
        return sw.fit_Sample(X_in, y_in, delta_n)
    elif mname == 'ECEBO':
        m = DAECEBO2_AUTO(k_windows=1, lambd=1.0, rule='literal', rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y)
    elif mname == 'ECEBO_PSI0':
        m = DAECEBO2_AUTO(k_windows=1, lambd=1.0, rule='psi0', rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y)
    elif mname == 'DAECEBO2_AUTO':
        m = DAECEBO2_AUTO(k_windows=k_w, lambd=lam, rule=rule, rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y, timestamps=t)
    else:
        raise ValueError(f"Unknown method {mname}")

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
    elif name == 'weather':
        df = pd.read_csv('datasets_stream/weather.csv')
        y = df.iloc[:, -1].values.astype(int)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'spam_katakis':
        df = pd.read_csv('datasets_stream/spam_katakis.csv')
        y = df.iloc[:, -1].values.astype(int)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'phishing':
        df = pd.read_csv('datasets_stream/phishing.csv')
        y = np.where(df.iloc[:, -1].values == 1, 0, 1) # minor class is 1
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'ludata':
        df = pd.read_csv('datasets_stream/ludata.csv')
        y = df.iloc[:, -1].values.astype(int)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'chessweka':
        df = pd.read_csv('datasets_stream/chessweka.csv')
        y = df.iloc[:, -1].values.astype(int)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")
    return X, y, t

def main():
    val_csv = 'results_stream/P3_validation_grid.csv'
    sel_csv = 'results_stream/P3_selected_params.csv'
    
    val_cols = ['dataset', 'method', 'K_windows', 'lambd', 'rule', 'rho', 'step_scale', 'fold', 'seed', 'val_score']
    sel_cols = ['dataset', 'method', 'best_rho', 'best_K', 'best_lambda', 'best_rule', 'best_step_scale', 'best_val_score']

    if os.path.exists(val_csv):
        df_val = pd.read_csv(val_csv)
    else:
        df_val = pd.DataFrame(columns=val_cols)
        df_val.to_csv(val_csv, index=False)

    if os.path.exists(sel_csv):
        df_sel = pd.read_csv(sel_csv)
    else:
        df_sel = pd.DataFrame(columns=sel_cols)
        df_sel.to_csv(sel_csv, index=False)

    datasets = [
        'ecoli2', 'yeast3', 'synth_sea_p5', 'synth_hyperplane_p5', 'synth_random_rbf_p5',
        'electricity_ir100', 'electricity_ir20', 'creditcard',
        'weather', 'spam_katakis', 'phishing', 'ludata', 'chessweka'
    ]

    competitors = ['SMOTE', 'BorderlineSMOTE', 'ADASYN', 'G-SMOTE', 'CCR', 'A-SUWO', 'MLOS_Fixed', 'ECEBO', 'ECEBO_PSI0']
    rho_grid = [0.1, 0.25, 0.5, 1.0]
    seeds = [42, 43, 44]

    for dname in datasets:
        print(f"\n==========================================")
        print(f"RUNNING VALIDATION PROTOCOL: {dname}")
        print(f"==========================================")
        
        X, y, t = load_data(dname)
        n_total = len(X)
        n_tr = int(0.75 * n_total)
        X_tr, y_tr, t_tr = X[:n_tr], y[:n_tr], t[:n_tr]

        # 2 Expanding-window folds inside the training split
        f1_tr_end = int(0.50 * n_tr)
        f1_val_end = int(0.75 * n_tr)
        fold1 = (X_tr[:f1_tr_end], y_tr[:f1_tr_end], t_tr[:f1_tr_end],
                 X_tr[f1_tr_end:f1_val_end], y_tr[f1_tr_end:f1_val_end])

        f2_tr_end = int(0.667 * n_tr)
        fold2 = (X_tr[:f2_tr_end], y_tr[:f2_tr_end], t_tr[:f2_tr_end],
                 X_tr[f2_tr_end:], y_tr[f2_tr_end:])
        
        folds = [('fold1', fold1), ('fold2', fold2)]

        # 1. Competitor rho grid
        for comp in competitors:
            # Check if already selected
            df_s_curr = pd.read_csv(sel_csv) if os.path.exists(sel_csv) else pd.DataFrame()
            if not df_s_curr.empty and len(df_s_curr[(df_s_curr['dataset'] == dname) & (df_s_curr['method'] == comp)]) > 0:
                continue

            best_sc = -1.0
            best_rho = 1.0

            for rho in rho_grid:
                scores = []
                for fname, (X_f_tr, y_f_tr, t_f_tr, X_f_val, y_f_val) in folds:
                    for s in seeds:
                        try:
                            X_res, y_res = sample_method(comp, X_f_tr, y_f_tr, t_f_tr, s, rho=rho)
                            clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
                            sc = score_clf(y_f_val, clf.predict(X_f_val))
                        except Exception:
                            sc = 0.0
                        scores.append(sc)
                        row = {
                            'dataset': dname, 'method': comp, 'K_windows': 1, 'lambd': 1.0,
                            'rule': 'none', 'rho': rho, 'step_scale': 1.0, 'fold': fname, 'seed': s,
                            'val_score': round(sc, 4)
                        }
                        pd.DataFrame([row]).to_csv(val_csv, mode='a', header=False, index=False)

                mean_sc = np.mean(scores)
                if mean_sc > best_sc:
                    best_sc = mean_sc
                    best_rho = rho

            sel_row = {
                'dataset': dname, 'method': comp, 'best_rho': best_rho,
                'best_K': 1, 'best_lambda': 1.0, 'best_rule': 'default',
                'best_step_scale': 1.0, 'best_val_score': round(best_sc, 4)
            }
            pd.DataFrame([sel_row]).to_csv(sel_csv, mode='a', header=False, index=False)
            print(f"[{dname} | {comp}] Best rho: {best_rho} (val score: {best_sc:.4f})")

        # 2. DAECEBO2_AUTO Successive Halving
        df_s_curr = pd.read_csv(sel_csv) if os.path.exists(sel_csv) else pd.DataFrame()
        if df_s_curr.empty or len(df_s_curr[(df_s_curr['dataset'] == dname) & (df_s_curr['method'] == 'DAECEBO2_AUTO')]) == 0:
            print(f"Running Successive Halving for DAECEBO2_AUTO on {dname}...")
            
            # Stage 1: All 288 candidate cells on Fold 1 with Seed 42
            K_grid = [1, 2, 4]
            lam_grid = [0.0, 0.25, 0.5, 1.0]
            rule_grid = ['literal', 'psi0']
            step_grid = [0.5, 1.0, 2.0]

            stage1_candidates = []
            X_f1_tr, y_f1_tr, t_f1_tr, X_f1_val, y_f1_val = fold1

            for kw in K_grid:
                for lm in lam_grid:
                    for rl in rule_grid:
                        for rh in rho_grid:
                            for st in step_grid:
                                try:
                                    X_res, y_res = sample_method('DAECEBO2_AUTO', X_f1_tr, y_f1_tr, t_f1_tr, 42,
                                                                 rho=rh, k_w=kw, lam=lm, rule=rl, step_s=st)
                                    clf = DecisionTreeClassifier(random_state=42).fit(X_res, y_res)
                                    sc = score_clf(y_f1_val, clf.predict(X_f1_val))
                                except Exception:
                                    sc = 0.0

                                row = {
                                    'dataset': dname, 'method': 'DAECEBO2_AUTO_stage1',
                                    'K_windows': kw, 'lambd': lm, 'rule': rl, 'rho': rh,
                                    'step_scale': st, 'fold': 'fold1', 'seed': 42, 'val_score': round(sc, 4)
                                }
                                pd.DataFrame([row]).to_csv(val_csv, mode='a', header=False, index=False)
                                stage1_candidates.append((sc, kw, lm, rl, rh, st))

            # Select Top 8 candidates
            stage1_candidates.sort(key=lambda x: x[0], reverse=True)
            top8 = stage1_candidates[:8]

            # Stage 2: Evaluate Top 8 on Both Folds across 3 Seeds (48 evaluations)
            best_auto_sc = -1.0
            best_params = top8[0][1:]

            for _, kw, lm, rl, rh, st in top8:
                scores = []
                for fname, (X_f_tr, y_f_tr, t_f_tr, X_f_val, y_f_val) in folds:
                    for s in seeds:
                        try:
                            X_res, y_res = sample_method('DAECEBO2_AUTO', X_f_tr, y_f_tr, t_f_tr, s,
                                                         rho=rh, k_w=kw, lam=lm, rule=rl, step_s=st)
                            clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
                            sc = score_clf(y_f_val, clf.predict(X_f_val))
                        except Exception:
                            sc = 0.0
                        scores.append(sc)
                        row = {
                            'dataset': dname, 'method': 'DAECEBO2_AUTO',
                            'K_windows': kw, 'lambd': lm, 'rule': rl, 'rho': rh,
                            'step_scale': st, 'fold': fname, 'seed': s, 'val_score': round(sc, 4)
                        }
                        pd.DataFrame([row]).to_csv(val_csv, mode='a', header=False, index=False)

                mean_sc = np.mean(scores)
                if mean_sc > best_auto_sc:
                    best_auto_sc = mean_sc
                    best_params = (kw, lm, rl, rh, st)

            sel_row = {
                'dataset': dname, 'method': 'DAECEBO2_AUTO', 'best_rho': best_params[3],
                'best_K': best_params[0], 'best_lambda': best_params[1], 'best_rule': best_params[2],
                'best_step_scale': best_params[4], 'best_val_score': round(best_auto_sc, 4)
            }
            pd.DataFrame([sel_row]).to_csv(sel_csv, mode='a', header=False, index=False)
            print(f"[{dname} | DAECEBO2_AUTO] Best params: K={best_params[0]}, lam={best_params[1]}, rule={best_params[2]}, rho={best_params[3]}, step={best_params[4]} (val score: {best_auto_sc:.4f})")

    print("\nValidation protocol complete!")

if __name__ == '__main__':
    main()

