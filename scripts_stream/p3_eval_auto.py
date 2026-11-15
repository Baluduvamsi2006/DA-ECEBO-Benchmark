"""
Phase 3 Final Evaluation Protocol: P3_raw.csv and P3_summary.csv
================================================================
Fair benchmark comparing:
  1. None (baseline)
  2. Fixed default rho=1.0 competitors (SMOTE, BorderlineSMOTE, ADASYN, G-SMOTE, CCR, A-SUWO, MLOS_Fixed, ECEBO, ECEBO_PSI0)
  3. Validated rho competitors (using optimal rho* from results_stream/P3_selected_params.csv)
  4. DAECEBO2_AUTO (using optimal K*, lambda*, rule*, rho*, step* from results_stream/P3_selected_params.csv)

Classifiers: CART, KNN, GaussianNB, SVC (max_iter=2000, 20k cap on creditcard)
Seeds: 42 to 51 (10 seeds)
Test: Future 25% chronological test partition + 5 later 10% blocks for stream datasets
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
from scripts_stream.p3_part3_validation import load_data, sample_method

ALL_DATASETS = [
    'ecoli2',
    'yeast3',
    'synth_sea_p5',
    'synth_hyperplane_p5',
    'synth_random_rbf_p5',
    'electricity_ir100',
    'electricity_ir20',
    'creditcard'
]

STREAM_DATASETS = [
    'synth_sea_p5',
    'synth_hyperplane_p5',
    'synth_random_rbf_p5',
    'electricity_ir100',
    'electricity_ir20',
    'creditcard'
]

SEEDS = list(range(42, 52))

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

def main():
    os.makedirs('results_stream', exist_ok=True)
    raw_csv = 'results_stream/P3_raw.csv'
    blocks_csv = 'results_stream/P3_blocks.csv'
    summary_csv = 'results_stream/P3_summary.csv'
    selected_csv = 'results_stream/P3_selected_params.csv'

    cols = ['dataset', 'classifier', 'method', 'seed', 'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC', 'rho_used', 'K_used', 'lambda_used', 'rule_used', 'step_scale_used']

    if not os.path.exists(raw_csv):
        pd.DataFrame(columns=cols).to_csv(raw_csv, index=False)

    if not os.path.exists(blocks_csv):
        bcols = ['dataset', 'block', 'classifier', 'method', 'seed', 'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']
        pd.DataFrame(columns=bcols).to_csv(blocks_csv, index=False)

    # Load selected parameters
    if not os.path.exists(selected_csv):
        print(f"Error: {selected_csv} does not exist yet. Please wait for validation to complete.")
        return

    df_sel = pd.read_csv(selected_csv)

    # 1. Populate P3_raw.csv with existing default rho=1.0 runs from P2_raw and P3_mlos_fixed
    df_raw = pd.read_csv(raw_csv)
    existing_keys = set(zip(df_raw['dataset'], df_raw['classifier'], df_raw['method'], df_raw['seed']))

    # Import P2_raw default methods
    if os.path.exists('results_stream/P2_raw.csv'):
        df_p2 = pd.read_csv('results_stream/P2_raw.csv')
        # Skip buggy MLOS, DAECEBO2_Sanity, and rename DAECEBO2 to DAECEBO2_p2_legacy
        for _, r in df_p2.iterrows():
            m = str(r['method'])
            if m in ['MLOS', 'DAECEBO2_Sanity_K1_lam1']:
                continue
            method_name = m if m in ['None', 'ECEBO', 'ECEBO_PSI0', 'DAECEBO2', 'DAECEBO2_PSI0'] else f"{m}_default"
            key = (r['dataset'], r['classifier'], method_name, int(r['seed']))
            if key not in existing_keys:
                row_dict = {
                    'dataset': r['dataset'],
                    'classifier': r['classifier'],
                    'method': method_name,
                    'seed': int(r['seed']),
                    'Macro-F1': float(r['Macro-F1']),
                    'Macro-Gmean': float(r['Macro-Gmean']),
                    'Minority-F1': float(r['Minority-F1']),
                    'Precision': float(r['Precision']),
                    'Recall': float(r['Recall']),
                    'AUC': float(r['AUC']),
                    'rho_used': 1.0,
                    'K_used': r.get('K_used', 1),
                    'lambda_used': r.get('lambda_used', 1.0),
                    'rule_used': r.get('rule_used', 'default'),
                    'step_scale_used': 1.0
                }
                pd.DataFrame([row_dict]).to_csv(raw_csv, mode='a', header=False, index=False)
                existing_keys.add(key)

    # Import P3_mlos_fixed
    if os.path.exists('results_stream/P3_mlos_fixed.csv'):
        df_mf = pd.read_csv('results_stream/P3_mlos_fixed.csv')
        for _, r in df_mf.iterrows():
            key = (r['dataset'], r['classifier'], 'MLOS_Fixed_default', int(r['seed']))
            if key not in existing_keys:
                row_dict = {
                    'dataset': r['dataset'],
                    'classifier': r['classifier'],
                    'method': 'MLOS_Fixed_default',
                    'seed': int(r['seed']),
                    'Macro-F1': float(r['Macro-F1']),
                    'Macro-Gmean': float(r['Macro-Gmean']),
                    'Minority-F1': float(r['Minority-F1']),
                    'Precision': float(r['Precision']),
                    'Recall': float(r['Recall']),
                    'AUC': float(r['AUC']),
                    'rho_used': 1.0,
                    'K_used': 1,
                    'lambda_used': 1.0,
                    'rule_used': 'default',
                    'step_scale_used': 1.0
                }
                pd.DataFrame([row_dict]).to_csv(raw_csv, mode='a', header=False, index=False)
                existing_keys.add(key)

    print("Initial P3_raw.csv populated with baseline & default runs.")

    # 2. Benchmark Tuned Competitors (where rho* < 1.0) and DAECEBO2_AUTO
    classifiers = ['CART', 'KNN', 'GaussianNB', 'SVC']

    for dname in ALL_DATASETS:
        print(f"\n==========================================")
        print(f"BENCHMARKING PHASE 3 AUTO / TUNED: {dname}")
        print(f"==========================================")

        X, y, t = load_data(dname)
        n_total = len(X)
        n_tr = int(0.75 * n_total)
        X_tr, y_tr, t_tr = X[:n_tr], y[:n_tr], t[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]

        # Check which methods to run for this dataset
        dset_params = df_sel[df_sel['dataset'] == dname].set_index('method').to_dict('index')

        # List methods to run:
        methods_to_run = []
        for mname, pinfo in dset_params.items():
            if mname == 'DAECEBO2_AUTO':
                methods_to_run.append(('DAECEBO2_AUTO', pinfo))
            else:
                rho_star = float(pinfo['best_rho'])
                tuned_name = f"{mname}_tuned"
                if rho_star < 1.0:
                    methods_to_run.append((tuned_name, pinfo))
                else:
                    # If rho* == 1.0, tuned is identical to default; duplicate existing rows with label tuned
                    for cname in classifiers:
                        for seed in SEEDS:
                            t_key = (dname, cname, tuned_name, seed)
                            if t_key not in existing_keys:
                                base_method = mname if mname in ['None', 'ECEBO', 'ECEBO_PSI0'] else f"{mname}_default"
                                df_curr = pd.read_csv(raw_csv)
                                m_match = df_curr[(df_curr['dataset'] == dname) & (df_curr['classifier'] == cname) & (df_curr['method'] == base_method) & (df_curr['seed'] == seed)]
                                if not m_match.empty:
                                    dup_row = m_match.iloc[0].to_dict()
                                    dup_row['method'] = tuned_name
                                    pd.DataFrame([dup_row]).to_csv(raw_csv, mode='a', header=False, index=False)
                                    existing_keys.add(t_key)

        for seed in SEEDS:
            # Prepare resampled datasets
            for m_label, pinfo in methods_to_run:
                # Check if all 4 classifiers already done
                df_curr = pd.read_csv(raw_csv)
                done_clfs = df_curr[(df_curr['dataset'] == dname) & (df_curr['method'] == m_label) & (df_curr['seed'] == seed)]['classifier'].tolist()
                rem_clfs = [c for c in classifiers if c not in done_clfs]
                if not rem_clfs:
                    continue

                actual_method = 'DAECEBO2_AUTO' if m_label == 'DAECEBO2_AUTO' else m_label.replace('_tuned', '')
                rho_val = float(pinfo['best_rho'])
                kw_val = int(pinfo.get('best_K', 1))
                lam_val = float(pinfo.get('best_lambda', 1.0))
                rule_val = str(pinfo.get('best_rule', 'default'))
                step_val = float(pinfo.get('best_step_scale', 1.0))

                try:
                    X_res, y_res = sample_method(actual_method, X_tr, y_tr, t_tr, seed,
                                                 rho=rho_val, k_w=kw_val, lam=lam_val, rule=rule_val, step_s=step_val)
                except Exception as e:
                    print(f"Sampling error on {dname} | {m_label} | seed {seed}: {e}")
                    continue

                for cname in rem_clfs:
                    # Apply majority subsampling for SVC/KNN on creditcard if resampled > 20,000
                    X_fit, y_fit = X_res, y_res
                    if dname == 'creditcard' and cname in ['KNN', 'SVC'] and len(X_fit) > 20000:
                        rng_sub = np.random.RandomState(seed)
                        maj_idx = np.where(y_fit == 0)[0]
                        min_idx = np.where(y_fit == 1)[0]
                        n_c0 = min(len(maj_idx), 10000)
                        n_c1 = min(len(min_idx), 20000 - n_c0)
                        sub_maj = rng_sub.choice(maj_idx, size=n_c0, replace=False)
                        sub_min = rng_sub.choice(min_idx, size=n_c1, replace=(n_c1 > len(min_idx)))
                        keep_idx = np.sort(np.concatenate([sub_maj, sub_min]))
                        X_fit, y_fit = X_fit[keep_idx], y_fit[keep_idx]

                    clf = get_classifier(cname, seed)
                    clf.fit(X_fit, y_fit)
                    y_pred = clf.predict(X_te)
                    if hasattr(clf, "predict_proba"):
                        y_prob = clf.predict_proba(X_te)[:, 1]
                    else:
                        y_prob = y_pred

                    m = compute_all_metrics(y_te, y_pred, y_prob)
                    row = {
                        'dataset': dname,
                        'classifier': cname,
                        'method': m_label,
                        'seed': seed,
                        'Macro-F1': round(m['Macro-F1'], 4),
                        'Macro-Gmean': round(m['Macro-Gmean'], 4),
                        'Minority-F1': round(m['Minority-F1'], 4),
                        'Precision': round(m['Precision'], 4),
                        'Recall': round(m['Recall'], 4),
                        'AUC': round(m['AUC'], 4),
                        'rho_used': rho_val,
                        'K_used': kw_val,
                        'lambda_used': lam_val,
                        'rule_used': rule_val,
                        'step_scale_used': step_val
                    }
                    pd.DataFrame([row]).to_csv(raw_csv, mode='a', header=False, index=False)
                    print(f"[{dname} | {cname} | {m_label} | seed {seed}] G-mean: {m['Macro-Gmean']:.4f}, Min-F1: {m['Minority-F1']:.4f}, AUC: {m['AUC']:.4f}")

    # Generate P3_summary.csv
    df_final_raw = pd.read_csv(raw_csv)
    metrics_list = ['Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']
    summary = df_final_raw.groupby(['dataset', 'classifier', 'method'])[metrics_list].agg(['mean', 'std']).reset_index()
    summary.to_csv(summary_csv, index=False)
    print("Saved results_stream/P3_summary.csv successfully!")

if __name__ == '__main__':
    main()

