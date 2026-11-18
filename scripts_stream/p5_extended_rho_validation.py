"""
Phase 5: Extended rho Validation & Test Evaluation Protocol
===========================================================
Performs legitimate hyperparameter search strictly inside the 75% training split:
  - 2 expanding-window folds inside the training split (never touching test partition)
  - Extended rho grid: {0.02, 0.05, 0.1, 0.25, 0.5, 1.0} evaluated for ALL oversamplers
  - For DAECEBO2_AUTO: evaluates K in {1, 2, 4}, lambda in {0.0, 0.25, 0.5, 1.0},
    rule in {'literal', 'psi0'}, step_scale in {0.5, 1.0}, and rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}
  - Selects the best configuration per method on validation score (0.5 * Gmean + 0.5 * Macro-F1)
  - Evaluates the chosen configuration on the 25% chronological test stream over 10 seeds (42 to 51)

100% Append-Only and Resumable.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import smote_variants as sv

sys.path.insert(0, os.path.abspath('.'))
from scripts_stream.p3_part3_validation import load_data, score_clf
from scripts_stream.mlos_fixed import MLOS_Fixed
from scripts_stream.da_ecebo_v3_auto import DAECEBO2_AUTO

VAL_GRID_CSV = 'results_stream/P5_validation_grid.csv'
SELECTED_CSV = 'results_stream/P5_selected_params.csv'
TEST_RES_CSV = 'results_stream/P5_test_results.csv'

os.makedirs('results_stream', exist_ok=True)

STREAM_DATASETS = [
    'synth_sea_p5',
    'synth_hyperplane_p5',
    'synth_random_rbf_p5',
    'electricity_ir20',
    'electricity_ir100'
]

ALL_OVERSAMPLERS = [
    'SMOTE',
    'BorderlineSMOTE',
    'ADASYN',
    'G-SMOTE',
    'CCR',
    'ECEBO',
    'ECEBO_PSI0',
    'DAECEBO2_AUTO'
]

RHO_GRID = [0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
VAL_SEEDS = [42, 43, 44]
TEST_SEEDS = list(range(42, 52))

def sample_method_p5(mname, X, y, t, seed, rho=1.0, k_w=4, lam=0.25, rule='psi0', step_s=1.0):
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
        delta_n = max(0, int(np.round(rho * (n_maj - n_min))))
        sw = sv.Geometric_SMOTE(n_to_sample=delta_n, random_state=seed)
        return sw.fit_resample(X, y)
    elif mname == 'CCR':
        delta_n = max(0, int(np.round(rho * (n_maj - n_min))))
        sw = sv.CCR(n_to_sample=delta_n, random_state=seed)
        return sw.fit_resample(X, y)
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

def load_done_val():
    if not os.path.exists(VAL_GRID_CSV):
        return set()
    try:
        df = pd.read_csv(VAL_GRID_CSV)
        done = set()
        for _, r in df.iterrows():
            key = (str(r['dataset']), str(r['method']), float(r['rho']), int(r['K_windows']),
                   float(r['lambd']), str(r['rule']), float(r['step_scale']), str(r['fold']), int(r['seed']))
            done.add(key)
        return done
    except Exception:
        return set()

def load_done_test():
    if not os.path.exists(TEST_RES_CSV):
        return set()
    try:
        df = pd.read_csv(TEST_RES_CSV)
        done = set()
        for _, r in df.iterrows():
            key = (str(r['dataset']), str(r['method']), int(r['seed']))
            done.add(key)
        return done
    except Exception:
        return set()

def evaluate_config(X_tr, y_tr, t_tr, X_val, y_val, mname, rho, k_w, lam, rule, step_s, seed):
    try:
        X_res, y_res = sample_method_p5(mname, X_tr, y_tr, t_tr, seed, rho=rho, k_w=k_w, lam=lam, rule=rule, step_s=step_s)
        clf = DecisionTreeClassifier(random_state=seed)
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_val)
        return score_clf(y_val, y_pred)
    except Exception as e:
        return 0.0

def run_validation():
    done_val = load_done_val()
    first_write = not os.path.exists(VAL_GRID_CSV)
    
    print(f"--- Loaded {len(done_val)} existing validation evaluations ---")
    
    for dname in STREAM_DATASETS:
        print(f"\n==========================================")
        print(f"P5 VALIDATION RUN: {dname}")
        print(f"==========================================")
        X, y, t = load_data(dname)
        n_tr = int(0.75 * len(X))
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
        
        # 1. Standard competitors
        for comp in ALL_OVERSAMPLERS:
            if comp == 'DAECEBO2_AUTO':
                continue
            
            for rho in RHO_GRID:
                for fold_name, (f_xtr, f_ytr, f_ttr, f_xval, f_yval) in folds:
                    for seed in VAL_SEEDS:
                        key = (str(dname), str(comp), float(rho), 1, 1.0, 'literal', 1.0, str(fold_name), int(seed))
                        if key in done_val:
                            continue
                        
                        score = evaluate_config(f_xtr, f_ytr, f_ttr, f_xval, f_yval, comp, rho, 1, 1.0, 'literal', 1.0, seed)
                        
                        rec = {
                            'dataset': dname, 'method': comp, 'K_windows': 1, 'lambd': 1.0, 'rule': 'literal',
                            'rho': rho, 'step_scale': 1.0, 'fold': fold_name, 'seed': seed, 'val_score': round(score, 5)
                        }
                        df_row = pd.DataFrame([rec])
                        df_row.to_csv(VAL_GRID_CSV, mode='a', header=first_write, index=False)
                        first_write = False
                        done_val.add(key)
        
        # 2. DAECEBO2_AUTO grid
        comp = 'DAECEBO2_AUTO'
        da_configs = []
        for rho in RHO_GRID:
            for k_w in [1, 2, 4]:
                for lam in [0.0, 0.25, 0.5, 1.0]:
                    for rule in ['literal', 'psi0']:
                        for step_s in [0.5, 1.0]:
                            da_configs.append((rho, k_w, lam, rule, step_s))
                            
        for (rho, k_w, lam, rule, step_s) in da_configs:
            for fold_name, (f_xtr, f_ytr, f_ttr, f_xval, f_yval) in folds:
                for seed in VAL_SEEDS:
                    key = (str(dname), str(comp), float(rho), int(k_w), float(lam), str(rule), float(step_s), str(fold_name), int(seed))
                    if key in done_val:
                        continue
                    
                    score = evaluate_config(f_xtr, f_ytr, f_ttr, f_xval, f_yval, comp, rho, k_w, lam, rule, step_s, seed)
                    
                    rec = {
                        'dataset': dname, 'method': comp, 'K_windows': k_w, 'lambd': lam, 'rule': rule,
                        'rho': rho, 'step_scale': step_s, 'fold': fold_name, 'seed': seed, 'val_score': round(score, 5)
                    }
                    df_row = pd.DataFrame([rec])
                    df_row.to_csv(VAL_GRID_CSV, mode='a', header=first_write, index=False)
                    first_write = False
                    done_val.add(key)
        
        print(f"Completed validation grid for {dname}")

def select_best_params():
    df_val = pd.read_csv(VAL_GRID_CSV)
    selected = []
    
    for (dname, comp), g in df_val.groupby(['dataset', 'method']):
        # Mean across folds and seeds
        agg = g.groupby(['rho', 'K_windows', 'lambd', 'rule', 'step_scale'])['val_score'].mean().reset_index()
        best_row = agg.sort_values(by='val_score', ascending=False).iloc[0]
        
        selected.append({
            'dataset': dname,
            'method': comp,
            'best_rho': best_row['rho'],
            'best_K': int(best_row['K_windows']),
            'best_lambda': best_row['lambd'],
            'best_rule': best_row['rule'],
            'best_step_scale': best_row['step_scale'],
            'best_val_score': round(best_row['val_score'], 5),
            'n_configs_tried': len(agg)
        })
        
    df_sel = pd.DataFrame(selected)
    df_sel.to_csv(SELECTED_CSV, index=False)
    print(f"\nSaved best selected parameters to {SELECTED_CSV}:")
    print(df_sel.to_string())
    return df_sel

def run_test_evaluation(df_sel):
    done_test = load_done_test()
    first_write = not os.path.exists(TEST_RES_CSV)
    
    print(f"\n--- Running Test Evaluation Across 10 Seeds (Strictly on Selected Params) ---")
    
    for dname in STREAM_DATASETS:
        X, y, t = load_data(dname)
        n_tr = int(0.75 * len(X))
        X_tr, y_tr, t_tr = X[:n_tr], y[:n_tr], t[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]
        
        d_sel = df_sel[df_sel['dataset'] == dname]
        
        for _, srow in d_sel.iterrows():
            comp = srow['method']
            rho = srow['best_rho']
            k_w = int(srow['best_K'])
            lam = srow['best_lambda']
            rule = srow['best_rule']
            step_s = srow['best_step_scale']
            
            for seed in TEST_SEEDS:
                key = (str(dname), str(comp), int(seed))
                if key in done_test:
                    continue
                
                try:
                    X_res, y_res = sample_method_p5(comp, X_tr, y_tr, t_tr, seed, rho=rho, k_w=k_w, lam=lam, rule=rule, step_s=step_s)
                    clf = DecisionTreeClassifier(random_state=seed)
                    clf.fit(X_res, y_res)
                    y_pred = clf.predict(X_te)
                    
                    try:
                        y_prob = clf.predict_proba(X_te)[:, 1]
                        auc = roc_auc_score(y_te, y_prob)
                    except Exception:
                        auc = 0.5
                        
                    mf1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
                    min_f1 = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
                    prec = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
                    rec = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
                    
                    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
                    tn, fp, fn, tp = cm.ravel()
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    gmean = np.sqrt(spec * sens)
                    
                    rec_out = {
                        'dataset': dname, 'method': comp, 'seed': seed,
                        'best_rho': rho, 'best_K': k_w, 'best_lambda': lam, 'best_rule': rule, 'best_step_scale': step_s,
                        'Macro-F1': round(mf1, 5), 'Macro-Gmean': round(gmean, 5), 'Minority-F1': round(min_f1, 5),
                        'Precision': round(prec, 5), 'Recall': round(rec, 5), 'AUC': round(auc, 5)
                    }
                    df_out = pd.DataFrame([rec_out])
                    df_out.to_csv(TEST_RES_CSV, mode='a', header=first_write, index=False)
                    first_write = False
                    done_test.add(key)
                except Exception as e:
                    print(f"Error on {dname} {comp} s={seed}: {e}")

if __name__ == '__main__':
    t0 = time.time()
    run_validation()
    df_sel = select_best_params()
    run_test_evaluation(df_sel)
    print(f"\nAll Phase 5 extended validation & test evaluation completed in {time.time()-t0:.1f}s!")
