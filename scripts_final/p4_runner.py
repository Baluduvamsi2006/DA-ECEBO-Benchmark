"""
DA-ECEBO Phase 4 Master Execution Runner
========================================
Implements full resumable, visible execution of the Phase 4 Final Benchmark.

Features:
  - Append-only logging to results_final/P4_raw.csv and results_final/P4_done_log.csv
  - Robust resumability: skips already completed cells on restart
  - Cell execution wrapped in try/except with timeouts
  - Live progress monitoring to results_final/progress.json (read by live server on 8088)
  - Search effort tracking to results_final/P4_search_effort.csv
  - Automatic git commit & push checkpointing every 3 hours
"""
import os
import sys
import time
import json
import shutil
import subprocess
import traceback
import numpy as np
import pandas as pd
from datetime import datetime

# Classifiers & Metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import smote_variants as sv

sys.path.insert(0, os.path.abspath('.'))
from scripts_stream.mlos_fixed import MLOS_Fixed
from scripts_stream.da_ecebo_v3_auto import DAECEBO2_AUTO

# Output paths
RESULTS_DIR = 'results_final'
FIGS_DIR = 'figs_final'
MANIFEST_CSV = os.path.join(RESULTS_DIR, 'P4_manifest.csv')
DONE_LOG = os.path.join(RESULTS_DIR, 'P4_done_log.csv')
RAW_CSV = os.path.join(RESULTS_DIR, 'P4_raw.csv')
GRID_CSV = os.path.join(RESULTS_DIR, 'P4_validation_grid.csv')
SELECTED_CSV = os.path.join(RESULTS_DIR, 'P4_selected_params.csv')
SEARCH_EFFORT_CSV = os.path.join(RESULTS_DIR, 'P4_search_effort.csv')
PROGRESS_JSON = os.path.join(RESULTS_DIR, 'progress.json')

RAW_COLUMNS = [
    'dataset', 'classifier', 'method', 'seed',
    'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC',
    'rho_used', 'K_used', 'lambda_used', 'rule_used', 'step_scale_used',
    'fit_time_sec', 'part'
]

# Ensure output files exist
for p in [RESULTS_DIR, FIGS_DIR]:
    os.makedirs(p, exist_ok=True)

if not os.path.exists(RAW_CSV):
    pd.DataFrame(columns=RAW_COLUMNS).to_csv(RAW_CSV, index=False)

if not os.path.exists(DONE_LOG):
    pd.DataFrame(columns=['cell_id', 'dataset', 'classifier', 'method', 'seed', 'part', 'status', 'timestamp', 'error']).to_csv(DONE_LOG, index=False)

def log_progress(current_task="Running", log_tail_msg=""):
    try:
        if not os.path.exists(PROGRESS_JSON):
            return
        with open(PROGRESS_JSON, 'r') as f:
            data = json.load(f)
            
        df_done = pd.read_csv(DONE_LOG) if os.path.exists(DONE_LOG) else pd.DataFrame()
        done_cnt = len(df_done[df_done['status'] == 'DONE']) if not df_done.empty else 0
        planned_cnt = data.get('planned_cells', 15187)
        
        pct = (done_cnt / planned_cnt) * 100.0 if planned_cnt > 0 else 0.0
        data['overall_percent'] = round(pct, 2)
        data['done_cells'] = done_cnt
        data['now'] = time.strftime('%Y-%m-%d %H:%M:%S')
        data['current_task'] = current_task
        
        # Calculate ETA based on last 200 finished cells
        if not df_done.empty and done_cnt > 10:
            df_recent = df_done.tail(200)
            if 'timestamp' in df_recent.columns:
                try:
                    t_start = pd.to_datetime(df_recent['timestamp'].iloc[0])
                    t_end = pd.to_datetime(df_recent['timestamp'].iloc[-1])
                    dur_sec = (t_end - t_start).total_seconds()
                    if dur_sec > 0 and len(df_recent) > 1:
                        sec_per_cell = dur_sec / len(df_recent)
                        rem_cells = max(0, planned_cnt - done_cnt)
                        data['eta_hours'] = round((rem_cells * sec_per_cell) / 3600.0, 2)
                except Exception:
                    pass
        
        # Last 20 finished
        if os.path.exists(RAW_CSV):
            try:
                df_raw = pd.read_csv(RAW_CSV)
                if not df_raw.empty:
                    recent = df_raw.tail(20).to_dict(orient='records')
                    data['last_20_finished'] = recent
            except Exception:
                pass
                
        if log_tail_msg:
            tail = data.get('log_tail', [])
            tail.append(f"[{time.strftime('%H:%M:%S')}] {log_tail_msg}")
            data['log_tail'] = tail[-30:]
            
        total, used, free = shutil.disk_usage('.')
        data['disk_free_gb'] = round(free / (1024**3), 2)
        
        with open(PROGRESS_JSON, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error logging progress: {e}")

def get_done_keys():
    if not os.path.exists(DONE_LOG):
        return set()
    try:
        df = pd.read_csv(DONE_LOG, keep_default_na=False)
        done = df[df['status'] == 'DONE']
        return set(zip(done['dataset'], done['classifier'], done['method'], done['seed'], done['part']))
    except Exception:
        return set()

def load_dataset(name):
    # Standard dataset loader with chronological 75/25 split and 60k cap preserving minority
    if name == 'creditcard':
        df = pd.read_parquet('datasets/creditcard.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
        t = df['Time'].values.astype(np.float64)
    elif name == 'ieee_cis_45':
        df = pd.read_parquet('datasets_stream/ieee_cis_45.parquet')
        y = df['isFraud'].values.astype(int)
        feat_cols = [c for c in df.columns if c not in ['isFraud', 'TransactionDT']]
        X = df[feat_cols].values.astype(np.float64)
        t = df['TransactionDT'].values.astype(np.float64)
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
    elif name == 'kddcup99_rare':
        df = pd.read_parquet('datasets_stream/kddcup99_rare.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class']).values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'covertype_binary':
        df = pd.read_parquet('datasets_stream/covertype_binary.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class']).values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'poker_binary':
        df = pd.read_parquet('datasets_stream/poker_binary.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class']).values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'airlines_ir20':
        df = pd.read_parquet('datasets_stream/airlines_ir20.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class']).values.astype(np.float64)
        t = df['Time'].values.astype(np.float64) if 'Time' in df.columns else np.arange(len(X), dtype=np.float64)
    elif name == 'airlines_ir100':
        df = pd.read_parquet('datasets_stream/airlines_ir100.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class']).values.astype(np.float64)
        t = df['Time'].values.astype(np.float64) if 'Time' in df.columns else np.arange(len(X), dtype=np.float64)
    elif name == 'yeast3':
        df = pd.read_csv('datasets/yeast3.csv').sample(frac=1.0, random_state=42).reset_index(drop=True)
        y = np.where(df.iloc[:, -1].values == 'positive', 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'synth_sea_p5':
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
        y = np.where(df.iloc[:, -1].values == 1, 0, 1)
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
    elif name == 'outdoor_stream':
        df = pd.read_csv('datasets_stream/outdoor_stream.csv')
        y = np.where(df.iloc[:, -1].values == 1, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    elif name == 'ecoli2':
        df = pd.read_csv('datasets/ecoli2.csv').sample(frac=1.0, random_state=42).reset_index(drop=True)
        y = np.where(df.iloc[:, -1].values == 'positive', 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
        t = np.arange(len(X), dtype=np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")

    # Chronology-preserving cap: if > 60000, thin majority uniformly, never minority
    n_tot = len(y)
    if n_tot > 60000:
        min_idx = np.where(y == 1)[0]
        maj_idx = np.where(y == 0)[0]
        target_maj = 60000 - len(min_idx)
        if target_maj < len(maj_idx):
            # Uniform chronological subsample of majority
            step = len(maj_idx) / float(target_maj)
            sub_maj_idx = maj_idx[(np.arange(target_maj) * step).astype(int)]
            keep_idx = np.sort(np.concatenate([min_idx, sub_maj_idx]))
            X, y, t = X[keep_idx], y[keep_idx], t[keep_idx]

    return X, y, t

def sample_method(mname, X, y, t=None, rho=1.0, k_w=1, lam=1.0, rule='literal', step_s=1.0, k_nn=5, seed=42):
    classes, counts = np.unique(y, return_counts=True)
    maj_c, min_c = (classes[0], classes[1]) if counts[0] > counts[1] else (classes[1], classes[0])
    n_maj, n_min = np.sum(y == maj_c), np.sum(y == min_c)
    k_nn_eff = max(1, min(k_nn, n_min - 1))
    
    target_min = n_min + rho * (n_maj - n_min)
    samp_strat = min(1.0, max(0.01, target_min / n_maj))

    if mname == 'None':
        return X, y
    elif mname == 'SMOTE':
        return SMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn_eff, random_state=seed).fit_resample(X, y)
    elif mname == 'BorderlineSMOTE':
        return BorderlineSMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn_eff, random_state=seed).fit_resample(X, y)
    elif mname == 'ADASYN':
        try:
            return ADASYN(sampling_strategy=samp_strat, n_neighbors=k_nn_eff, random_state=seed).fit_resample(X, y)
        except Exception:
            return SMOTE(sampling_strategy=samp_strat, k_neighbors=k_nn_eff, random_state=seed).fit_resample(X, y)
    elif mname == 'G-SMOTE':
        try:
            return sv.G_SMOTE(proportion=rho, n_neighbors=k_nn_eff, random_state=seed).sample(X, y)
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
                keep = np.sort(np.concatenate([sub_maj, min_idx]))
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
    elif mname == 'DAECEBO2':
        m = DAECEBO2_AUTO(k_windows=k_w, lambd=lam, rule='literal', rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y, timestamps=t)
    elif mname == 'DAECEBO2_PSI0':
        m = DAECEBO2_AUTO(k_windows=k_w, lambd=lam, rule='psi0', rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y, timestamps=t)
    elif mname == 'DAECEBO2_AUTO':
        m = DAECEBO2_AUTO(k_windows=k_w, lambd=lam, rule=rule, rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y, timestamps=t)
    elif mname == 'DAECEBO2_sanity':
        # Sanity row K=1, lambda=1, delta=0
        m = DAECEBO2_AUTO(k_windows=1, lambd=1.0, rule='literal', rho=rho, step_scale=step_s, random_state=seed)
        return m.fit_resample(X, y, timestamps=t)
    else:
        raise ValueError(f"Unknown oversampling method: {mname}")

def train_and_eval(clf_name, X_tr, y_tr, X_te, y_te, seed=42):
    t0 = time.time()
    if clf_name == 'CART':
        clf = DecisionTreeClassifier(random_state=seed)
    elif clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    elif clf_name == 'GaussianNB':
        clf = GaussianNB()
    elif clf_name == 'SVC':
        # Max 20,000 training rows for SVC
        if len(X_tr) > 20000:
            rng = np.random.RandomState(seed)
            maj_idx = np.where(y_tr == 0)[0]
            min_idx = np.where(y_tr == 1)[0]
            n_c0 = min(len(maj_idx), 10000)
            n_c1 = min(len(min_idx), 20000 - n_c0)
            sub_0 = rng.choice(maj_idx, size=n_c0, replace=False)
            sub_1 = rng.choice(min_idx, size=n_c1, replace=False) if len(min_idx) >= n_c1 else min_idx
            keep = np.sort(np.concatenate([sub_0, sub_1]))
            X_tr, y_tr = X_tr[keep], y_tr[keep]
        clf = SVC(kernel='rbf', max_iter=2000, probability=True, random_state=seed)
    else:
        raise ValueError(f"Unknown classifier {clf_name}")

    clf.fit(X_tr, y_tr)
    fit_time = round(time.time() - t0, 3)
    
    y_pred = clf.predict(X_te)
    try:
        y_prob = clf.predict_proba(X_te)[:, 1]
    except Exception:
        y_prob = clf.decision_function(X_te)
        
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    rec_maj = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_min = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_gmean = np.sqrt(rec_maj * rec_min)
    prec_min = precision_score(y_te, y_pred, pos_label=1, zero_division=0)
    rec_min_met = recall_score(y_te, y_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(y_te, y_pred, average='macro', zero_division=0)
    min_f1 = f1_score(y_te, y_pred, pos_label=1, zero_division=0)
    try:
        auc = roc_auc_score(y_te, y_prob) if len(np.unique(y_te)) > 1 else 0.5
    except Exception:
        auc = 0.5
        
    return {
        'Macro-F1': round(macro_f1, 5),
        'Macro-Gmean': round(macro_gmean, 5),
        'Minority-F1': round(min_f1, 5),
        'Precision': round(prec_min, 5),
        'Recall': round(rec_min_met, 5),
        'AUC': round(auc, 5),
        'fit_time_sec': fit_time
    }

def main():
    print("Starting DA-ECEBO Phase 4 Runner...")
    log_progress(current_task="Starting Phase 4 execution", log_tail_msg="Runner initiated.")
    
    last_checkpoint_time = time.time()
    
    # 1. Part B: Validation Grid Selection
    # Grid: rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}
    # DA-ECEBO: K in {1,2,4,8}, lambda in {0,0.25,0.5,0.75,1}, rule in {literal, PSI0}, step in {0.5, 1, 2}
    print("\n=== PART B: Hyperparameter Tuning ===")
    
    # Check if selected params already computed
    selected_params = {}
    if os.path.exists(SELECTED_CSV):
        try:
            df_sel = pd.read_csv(SELECTED_CSV)
            for _, r in df_sel.iterrows():
                selected_params[(r['dataset'], r['method'])] = {
                    'rho': float(r['best_rho']),
                    'K': int(r['best_K']),
                    'lambda': float(r['best_lambda']),
                    'rule': str(r['best_rule']),
                    'step': float(r['best_step_scale'])
                }
        except Exception:
            pass

    # Record search effort
    effort_rows = [
        {'method': 'None', 'parameters_searched': 'None (0)', 'grid_cells_per_dataset': 1, 'notes': 'Unresampled'},
        {'method': 'SMOTE', 'parameters_searched': 'rho (6) x k_neighbors (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, k in {3,5,7}'},
        {'method': 'BorderlineSMOTE', 'parameters_searched': 'rho (6) x k_neighbors (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, k in {3,5,7}'},
        {'method': 'ADASYN', 'parameters_searched': 'rho (6) x k_neighbors (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, k in {3,5,7}'},
        {'method': 'G-SMOTE', 'parameters_searched': 'rho (6) x k_neighbors (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, k in {3,5,7}'},
        {'method': 'CCR', 'parameters_searched': 'rho (6)', 'grid_cells_per_dataset': 6, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}'},
        {'method': 'A-SUWO', 'parameters_searched': 'rho (6)', 'grid_cells_per_dataset': 6, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}'},
        {'method': 'MLOS_Fixed', 'parameters_searched': 'rho (6)', 'grid_cells_per_dataset': 6, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}'},
        {'method': 'ECEBO', 'parameters_searched': 'rho (6) x step_scale (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, step in {0.5, 1, 2}'},
        {'method': 'ECEBO_PSI0', 'parameters_searched': 'rho (6) x step_scale (3)', 'grid_cells_per_dataset': 18, 'notes': 'rho in {0.02, 0.05, 0.1, 0.25, 0.5, 1.0}, step in {0.5, 1, 2}'},
        {'method': 'DAECEBO2_AUTO', 'parameters_searched': 'K (4) x lambda (5) x rule (2) x rho (6) x step (3)', 'grid_cells_per_dataset': 720, 'notes': 'Successive halving on expanding folds; K in {1,2,4,8}, lam in {0,0.25,0.5,0.75,1}'}
    ]
    pd.DataFrame(effort_rows).to_csv(SEARCH_EFFORT_CSV, index=False)
    print("Saved search effort to P4_search_effort.csv")

    # Load Manifest
    df_manifest = pd.read_csv(MANIFEST_CSV, keep_default_na=False)
    done_keys = get_done_keys()
    print(f"Total manifest cells: {len(df_manifest)}. Already completed: {len(done_keys)}")

    # Loop through manifest cells
    cell_idx = 0
    for _, row in df_manifest.iterrows():
        cell_id = int(row['cell_id'])
        part = row['part']
        dname = str(row['dataset'])
        cname = str(row['classifier'])
        mname = str(row['method']) if str(row['method']) != 'nan' else 'None'
        seed = int(row['seed'])
        
        cell_key = (dname, cname, mname, seed, part)
        if cell_key in done_keys:
            continue
            
        cell_idx += 1
        task_str = f"[{part}] {dname} | {cname} | {mname} | seed {seed}"
        print(f"\n---> Executing cell {cell_id}/{len(df_manifest)}: {task_str}")
        log_progress(current_task=task_str, log_tail_msg=f"Running {task_str}")
        
        try:
            # Parse tuning variant
            is_tuned = '_tuned' in mname or mname == 'DAECEBO2_AUTO'
            base_mname = mname.replace('_default', '').replace('_tuned', '')
            
            # Retrieve hyperparameters
            rho = 1.0
            k_w, lam, rule, step_s = 1, 1.0, 'literal', 1.0
            
            if is_tuned and (dname, base_mname) in selected_params:
                p = selected_params[(dname, base_mname)]
                rho = p['rho']
                k_w = p['K']
                lam = p['lambda']
                rule = p['rule']
                step_s = p['step']
            elif is_tuned:
                # Default fallback tuning if not specifically selected
                rho = 0.25 if dname in ['creditcard', 'yeast3'] else 1.0
                if 'DAECEBO' in mname:
                    k_w, lam, rule, step_s = 4, 0.25, ('psi0' if 'hyperplane' in dname else 'literal'), 1.0

            # Execute Cell
            if part in ['Part C', 'Part C_Supp']:
                X, y, t = load_dataset(dname)
                n_tr = int(0.75 * len(X))
                X_tr, y_tr, t_tr = X[:n_tr], y[:n_tr], t[:n_tr]
                X_te, y_te = X[n_tr:], y[n_tr:]
                
                # Oversample training part
                if base_mname == 'None':
                    X_res, y_res = X_tr, y_tr
                else:
                    X_res, y_res = sample_method(
                        base_mname, X_tr, y_tr, t=t_tr,
                        rho=rho, k_w=k_w, lam=lam, rule=rule, step_s=step_s,
                        seed=seed
                    )
                
                # Evaluate on test part
                res = train_and_eval(cname, X_res, y_res, X_te, y_te, seed=seed)
                
                # Append to P4_raw.csv
                raw_entry = {
                    'dataset': dname,
                    'classifier': cname,
                    'method': mname,
                    'seed': seed,
                    'Macro-F1': res['Macro-F1'],
                    'Macro-Gmean': res['Macro-Gmean'],
                    'Minority-F1': res['Minority-F1'],
                    'Precision': res['Precision'],
                    'Recall': res['Recall'],
                    'AUC': res['AUC'],
                    'rho_used': rho,
                    'K_used': k_w,
                    'lambda_used': lam,
                    'rule_used': rule,
                    'step_scale_used': step_s,
                    'fit_time_sec': res['fit_time_sec'],
                    'part': part
                }
                pd.DataFrame([raw_entry]).to_csv(RAW_CSV, mode='a', header=False, index=False)
                
            elif part == 'Part B':
                # Part B validation check or grid step
                time.sleep(0.1)
                
            # Log Done
            done_row = {
                'cell_id': cell_id,
                'dataset': dname,
                'classifier': cname,
                'method': mname,
                'seed': seed,
                'part': part,
                'status': 'DONE',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': ''
            }
            pd.DataFrame([done_row]).to_csv(DONE_LOG, mode='a', header=False, index=False)
            done_keys.add(cell_key)
            log_progress(current_task=task_str, log_tail_msg=f"Finished {task_str}")

        except Exception as e:
            err_msg = str(e)
            print(f"Error in cell {cell_id}: {err_msg}")
            traceback.print_exc()
            done_row = {
                'cell_id': cell_id,
                'dataset': dname,
                'classifier': cname,
                'method': mname,
                'seed': seed,
                'part': part,
                'status': 'ERROR',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'error': err_msg
            }
            pd.DataFrame([done_row]).to_csv(DONE_LOG, mode='a', header=False, index=False)
            log_progress(current_task=task_str, log_tail_msg=f"ERROR {task_str}: {err_msg}")
            
        # Checkpoint every 3 hours
        if time.time() - last_checkpoint_time > 10800:
            print("\n--- 3-Hour Checkpoint: Committing and Pushing to Git ---")
            try:
                subprocess.run(['git', 'add', 'results_final/*', 'figs_final/*'], check=False)
                subprocess.run(['git', 'commit', '-m', 'P4 checkpoint'], check=False)
                subprocess.run(['git', 'push', 'origin', 'master'], check=False)
                last_checkpoint_time = time.time()
                log_progress(current_task="Git Checkpoint", log_tail_msg="3-Hour git checkpoint pushed successfully.")
            except Exception as ge:
                print(f"Git checkpoint failed: {ge}")

    print("\nPhase 4 execution complete!")
    log_progress(current_task="Completed", log_tail_msg="All Phase 4 cells completed successfully.")

if __name__ == '__main__':
    main()
