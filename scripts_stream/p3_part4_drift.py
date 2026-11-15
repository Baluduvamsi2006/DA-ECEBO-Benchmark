"""
Phase 3 Part 4: Concept-Drift Baselines Benchmark
=================================================
Evaluates standard streaming concept-drift baselines:
  - ARF (Adaptive Random Forest, Gomes 2017)
  - SRP (Streaming Random Patches, Gomes 2019)
  - HAT (Hoeffding Adaptive Tree + ADWIN, Bifet 2009)
  - HT (Hoeffding Tree, Domingos & Hulten 2000)
  - OOB (Online Over-sampling Bagging, Wang, Minku, Yao 2015)
  - UOB (Online Under-sampling Bagging, Wang, Minku, Yao 2015)
  - Hoeffding Tree variants of None, SMOTE, ECEBO, DAECEBO2

Protocols:
  - Protocol A: Static Test (Warmup on 75% train, evaluate on 25% test without model update)
  - Protocol B: Prequential (Warmup on 75% train, test-then-train on 25% test with model updates)

Outputs:
  - results_stream/P3_drift_raw.csv
  - results_stream/P3_drift_summary.csv
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, precision_score, recall_score
import river.forest as rf
import river.ensemble as re
import river.tree as rt

sys.path.insert(0, os.path.abspath('.'))
from scripts_stream.p3_part3_validation import load_data, sample_method

class OnlineBaggingBase:
    def __init__(self, n_models=10, seed=42):
        self.n_models = n_models
        self.rng = np.random.RandomState(seed)
        self.models = [rt.HoeffdingTreeClassifier() for _ in range(n_models)]
        self.n_min = 0
        self.n_maj = 0
        self.min_class = 1

    def predict_one(self, x):
        votes = [m.predict_one(x) for m in self.models]
        return 1 if votes.count(1) > votes.count(0) else 0

    def predict_proba_one(self, x):
        p1s = [m.predict_proba_one(x).get(1, 0.0) for m in self.models]
        p_mean = np.mean(p1s) if p1s else 0.0
        return {0: 1.0 - p_mean, 1: p_mean}

class OOBClassifier(OnlineBaggingBase):
    def learn_one(self, x, y):
        if y == self.min_class:
            self.n_min += 1
            lam = max(1.0, self.n_maj / (self.n_min + 1e-5))
        else:
            self.n_maj += 1
            lam = 1.0
        for m in self.models:
            k = self.rng.poisson(lam)
            for _ in range(k):
                m.learn_one(x, y)

class UOBClassifier(OnlineBaggingBase):
    def learn_one(self, x, y):
        if y != self.min_class:
            self.n_maj += 1
            lam = min(1.0, (self.n_min + 1e-5) / self.n_maj)
        else:
            self.n_min += 1
            lam = 1.0
        for m in self.models:
            k = self.rng.poisson(lam)
            for _ in range(k):
                m.learn_one(x, y)

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

def get_river_model(mname, seed):
    if mname == 'ARF':
        return rf.ARFClassifier(seed=seed)
    elif mname == 'SRP':
        return re.SRPClassifier(seed=seed)
    elif mname == 'HAT':
        return rt.HoeffdingAdaptiveTreeClassifier(seed=seed)
    elif mname == 'HT':
        return rt.HoeffdingTreeClassifier()
    elif mname == 'OOB':
        return OOBClassifier(seed=seed)
    elif mname == 'UOB':
        return UOBClassifier(seed=seed)
    raise ValueError(f"Unknown river model {mname}")

def main():
    out_csv = 'results_stream/P3_drift_raw.csv'
    cols = ['dataset', 'protocol', 'model', 'seed', 'Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']

    if os.path.exists(out_csv):
        df_done = pd.read_csv(out_csv)
    else:
        df_done = pd.DataFrame(columns=cols)
        df_done.to_csv(out_csv, index=False)

    stream_datasets = [
        'synth_sea_p5', 'synth_hyperplane_p5', 'synth_random_rbf_p5',
        'electricity_ir100', 'electricity_ir20', 'creditcard',
        'weather', 'spam_katakis', 'phishing', 'ludata'
    ]

    models = ['ARF', 'SRP', 'HAT', 'HT', 'OOB', 'UOB']
    protocols = ['Protocol_A_Static', 'Protocol_B_Prequential']
    seeds = list(range(42, 52))

    for dname in stream_datasets:
        print(f"\n==========================================")
        print(f"EVALUATING DRIFT BASELINES: {dname}")
        print(f"==========================================")

        X, y, t = load_data(dname)
        n_total = len(X)
        n_tr = int(0.75 * n_total)
        
        # Chronological cap for massive sets if > 60000
        if n_total > 60000:
            # Preserve chronological 75/25 with cap
            n_sub = 60000
            n_tr = int(0.75 * n_sub)
            X = X[:n_sub]
            y = y[:n_sub]
            t = t[:n_sub]

        X_tr, y_tr = X[:n_tr], y[:n_tr]
        X_te, y_te = X[n_tr:], y[n_tr:]
        n_features = X.shape[1]

        for seed in seeds:
            for proto in protocols:
                for mname in models:
                    df_curr = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
                    if not df_curr.empty:
                        done = df_curr[(df_curr['dataset'] == dname) &
                                       (df_curr['protocol'] == proto) &
                                       (df_curr['model'] == mname) &
                                       (df_curr['seed'] == seed)]
                        if len(done) > 0:
                            continue

                    model = get_river_model(mname, seed)

                    # 1. Warm-up on first 75%
                    for i in range(len(X_tr)):
                        xi = {f'f{j}': X_tr[i, j] for j in range(n_features)}
                        yi = int(y_tr[i])
                        model.learn_one(xi, yi)

                    # 2. Test phase on last 25%
                    preds = []
                    probs = []
                    for i in range(len(X_te)):
                        xi = {f'f{j}': X_te[i, j] for j in range(n_features)}
                        yi = int(y_te[i])

                        pred = model.predict_one(xi)
                        prob_dict = model.predict_proba_one(xi)
                        prob_val = prob_dict.get(1, 0.0) if isinstance(prob_dict, dict) else float(prob_dict)

                        preds.append(1 if pred == 1 else 0)
                        probs.append(prob_val)

                        # In Protocol B (Prequential), model updates after predicting
                        if proto == 'Protocol_B_Prequential':
                            model.learn_one(xi, yi)

                    m = compute_all_metrics(y_te, np.array(preds), np.array(probs))
                    row = {
                        'dataset': dname,
                        'protocol': proto,
                        'model': mname,
                        'seed': seed,
                        'Macro-F1': round(m['Macro-F1'], 4),
                        'Macro-Gmean': round(m['Macro-Gmean'], 4),
                        'Minority-F1': round(m['Minority-F1'], 4),
                        'Precision': round(m['Precision'], 4),
                        'Recall': round(m['Recall'], 4),
                        'AUC': round(m['AUC'], 4)
                    }
                    pd.DataFrame([row]).to_csv(out_csv, mode='a', header=False, index=False)
                    print(f"[{dname} | {proto} | {mname} | seed {seed}] G-mean: {m['Macro-Gmean']:.4f}, Min-F1: {m['Minority-F1']:.4f}, AUC: {m['AUC']:.4f}")

    # Generate Summary Table
    df_raw = pd.read_csv(out_csv)
    metrics_list = ['Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']
    summary = df_raw.groupby(['dataset', 'protocol', 'model'])[metrics_list].agg(['mean', 'std']).reset_index()
    summary.to_csv('results_stream/P3_drift_summary.csv', index=False)
    print("Saved results_stream/P3_drift_summary.csv")

if __name__ == '__main__':
    main()

