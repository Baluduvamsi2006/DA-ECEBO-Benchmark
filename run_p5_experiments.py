"""
P5 Experiments:
a) Ablation Study (5 variants on CreditCard and 3 benchmark datasets)
b) Sensitivity Analysis (p_drift, delta_base, step, K_windows)
c) Runtime Benchmark (Sampling runtime across all 15 datasets)
d) Friedman Test & Holm Post-Hoc Procedure across all 15 datasets
e) Synthetic Sample Export (.npz) for Ecoli2, Haberman, Glass6
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, allocate_samples, compute_metrics


class ECCAblation:
    """Ablation sampler variants for P5(a)."""
    def __init__(self, mode='ecc_only', k_windows=4, lambd=0.25, p_drift=0.5, delta_base=0.05, step=0.1, iters=60, random_state=42):
        self.mode = mode
        self.k_windows = k_windows
        self.lambd = lambd
        self.p_drift = p_drift
        self.delta_base = delta_base
        self.step = step
        self.iters = iters
        self.random_state = random_state
        self.k = 7
        self.alpha = 1.0
        self.prob = 0.2

    def fit_resample(self, X, y, timestamps=None, sampling_strategy=1.0):
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        maj_class, min_class = (classes[0], classes[1]) if counts[0] > counts[1] else (classes[1], classes[0])
            
        idx_maj = np.where(y == maj_class)[0]
        idx_min = np.where(y == min_class)[0]
        X_maj, X_min = X[idx_maj], X[idx_min]
        n_maj, n_min = len(X_maj), len(X_min)
        
        target_min = int(n_maj * sampling_strategy)
        delta_n = max(0, target_min - n_min)
        if delta_n == 0:
            return X, y

        scaler = StandardScaler()
        X_maj_s = scaler.fit_transform(X_maj)
        X_min_s = scaler.transform(X_min)
        
        # 1. Temporal windows
        if timestamps is not None:
            min_times = timestamps[idx_min]
            t_min, t_max = np.min(timestamps), np.max(timestamps)
            time_bins = np.linspace(t_min, t_max + 1e-9, self.k_windows + 1)
            sample_windows = np.clip(np.digitize(min_times, time_bins) - 1, 0, self.k_windows - 1)
        else:
            bins = np.linspace(0, len(X) + 1, self.k_windows + 1)
            sample_windows = np.clip(np.digitize(idx_min, bins) - 1, 0, self.k_windows - 1)
            
        # 2. ECC
        nn_maj = NearestNeighbors(n_neighbors=max(self.k, 1), algorithm='auto', n_jobs=-1).fit(X_maj_s)
        dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
        d_maj_arr = dist_to_maj[:, 0]
        
        clusters = {i: [i] for i in range(n_min)}
        centroids = {i: X_min_s[i].copy() for i in range(n_min)}
        cluster_d_maj = {i: float(d_maj_arr[i]) for i in range(n_min)}
        
        n_c = n_min
        dist_mat = np.full((n_c, n_c), np.inf)
        for i in range(n_c):
            for j in range(i + 1, n_c):
                d_ij = np.linalg.norm(centroids[i] - centroids[j])
                if d_ij < min(cluster_d_maj[i], cluster_d_maj[j]):
                    dist_mat[i, j] = dist_mat[j, i] = d_ij
                    
        while True:
            min_val = np.min(dist_mat)
            if np.isinf(min_val):
                break
            c1, c2 = np.unravel_index(np.argmin(dist_mat), dist_mat.shape)
            members = clusters[c1] + clusters[c2]
            clusters[c1] = members
            del clusters[c2]
            centroids[c1] = np.mean(X_min_s[members], axis=0)
            cluster_d_maj[c1] = min(cluster_d_maj[c1], cluster_d_maj[c2])
            del centroids[c2]
            del cluster_d_maj[c2]
            dist_mat[c2, :] = dist_mat[:, c2] = np.inf
            for other in clusters.keys():
                if other != c1:
                    d_c = np.linalg.norm(centroids[c1] - centroids[other])
                    if d_c < min(cluster_d_maj[c1], cluster_d_maj[other]):
                        dist_mat[c1, other] = dist_mat[other, c1] = d_c
                    else:
                        dist_mat[c1, other] = dist_mat[other, c1] = np.inf

        cluster_list = list(clusters.values())
        m = len(cluster_list)
        cluster_centroids = [centroids[k] for k in clusters.keys()]
        cluster_d_maj_list = [cluster_d_maj[k] for k in clusters.keys()]
        _, cent_nn_maj_idx = nn_maj.kneighbors(np.array(cluster_centroids))
        
        # 3. Shannon Entropy
        if self.mode != 'ecc_only':
            lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
            sub_maj = rng.choice(n_maj, size=min(n_maj, n_min * 10), replace=False)
            lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
            eps = 1e-12
            p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
            sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
            p_maj = np.clip(lr.predict_proba(X_maj_s)[:, 1], eps, 1.0 - eps)
            maj_entropy = -(p_maj * np.log2(p_maj) + (1.0 - p_maj) * np.log2(1.0 - p_maj))
            
            cluster_avg_entropies = np.zeros(m)
            raw_entropy = np.zeros(m)
            for i, mem in enumerate(cluster_list):
                cluster_avg_entropies[i] = np.mean(sample_entropy[mem])
                raw_entropy[i] = np.sum(sample_entropy[mem])
        else:
            cluster_avg_entropies = np.ones(m)
            raw_entropy = np.ones(m)
            maj_entropy = np.ones(n_maj)

        # 4. Drift vectors
        raw_drift = np.zeros(m)
        drift_vecs = []
        if self.k_windows > 1 and self.mode in ['drift_only', 'full_da_ecebo', 'no_drift_steps']:
            mid = self.k_windows // 2
            for i, mem in enumerate(cluster_list):
                c_wins = sample_windows[mem]
                c_pts = X_min_s[mem]
                rec = (np.mean(c_wins) + 1.0) / float(self.k_windows)
                early, recent = c_wins < mid, c_wins >= mid
                if np.any(early) and np.any(recent):
                    v = np.mean(c_pts[recent], axis=0) - np.mean(c_pts[early], axis=0)
                    spat = np.linalg.norm(v)
                elif np.any(recent):
                    v, spat = np.zeros(X.shape[1]), 1.0
                else:
                    v, spat = np.zeros(X.shape[1]), 0.0
                nv = np.linalg.norm(v)
                if nv > 1e-6:
                    v = v / nv
                drift_vecs.append(v)
                raw_drift[i] = rec * (1.0 + spat)
        else:
            drift_vecs = [np.zeros(X.shape[1]) for _ in range(m)]

        def norm_01(arr):
            span = np.max(arr) - np.min(arr)
            return (arr - np.min(arr)) / span if span > 1e-9 else np.ones_like(arr)

        S_entropy = norm_01(raw_entropy)
        S_drift = norm_01(raw_drift) if self.k_windows > 1 else np.zeros(m)

        # 5. Allocation according to ablation mode
        if self.mode == 'ecc_only':
            weights = np.ones(m) / m
        elif self.mode == 'ec_ebo':  # entropy only
            weights = S_entropy / np.sum(S_entropy) if np.sum(S_entropy) > 0 else np.ones(m) / m
        elif self.mode == 'drift_only':  # drift only (lambda=0)
            scores = S_drift + self.delta_base
            weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(m) / m
        elif self.mode in ['full_da_ecebo', 'no_drift_steps']:
            scores = (self.lambd * S_entropy) + ((1.0 - self.lambd) * S_drift) + self.delta_base
            weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(m) / m
        else:
            raise ValueError(f"Unknown mode {self.mode}")

        alloc = allocate_samples(weights, delta_n)

        # 6. Hill climbing synthesis
        effective_p_drift = 0.0 if self.mode == 'no_drift_steps' else self.p_drift
        syn = []
        for i, mem in enumerate(cluster_list):
            n_g = alloc[i]
            if n_g <= 0:
                continue
            c_pts = X_min_s[mem]
            mu, e_bar = cluster_centroids[i], cluster_avg_entropies[i]
            maj_nn, d_vec = cent_nn_maj_idx[i], drift_vecs[i]
            has_d = np.linalg.norm(d_vec) > 1e-6
            
            seeds = c_pts[rng.choice(len(mem), size=n_g, replace=True)].copy()
            maj_p = rng.choice(maj_nn, size=n_g, replace=True)
            x_m, e_m = X_maj_s[maj_p], maj_entropy[maj_p]
            
            for t in range(self.iters):
                act = rng.rand(n_g) >= self.prob
                if not np.any(act):
                    break
                step_p = np.zeros((n_g, X.shape[1]))
                if has_d and effective_p_drift > 0:
                    d_mask = (rng.rand(n_g) < effective_p_drift) & act
                    step_p[d_mask] = d_vec * self.step
                    b_mask = act & (~d_mask)
                else:
                    b_mask = act
                if np.any(b_mask):
                    nb = np.sum(b_mask)
                    step_p[np.where(b_mask)[0], rng.randint(0, X.shape[1], size=nb)] = (
                        rng.choice([-1.0, 1.0], size=nb) * self.step
                    )
                trans = seeds + step_p
                if self.mode == 'ecc_only':
                    # Accept steps staying within nearest majority boundary
                    acc = np.linalg.norm(trans - mu, axis=1) < cluster_d_maj_list[i]
                else:
                    psi = (e_bar * np.exp(-self.alpha * np.linalg.norm(trans - mu, axis=1)) -
                           e_m * np.exp(-self.alpha * np.linalg.norm(trans - x_m, axis=1)))
                    acc = psi < 0
                seeds[acc & act] = trans[acc & act]
            syn.append(seeds)
            
        if len(syn) > 0:
            X_syn = scaler.inverse_transform(np.vstack(syn))
            return np.vstack([X, X_syn]), np.hstack([y, np.full(len(X_syn), min_class)])
        return X, y


def load_dataset(name):
    p_path = f'datasets/{name}.parquet'
    c_path = f'datasets/{name}.csv'
    fpath = p_path if os.path.exists(p_path) else c_path
    df = pd.read_parquet(fpath) if fpath.endswith('.parquet') else pd.read_csv(fpath)
    
    target_col = None
    for c in ['Class', 'class', 'target', 'label', 'Outcome', 'Survival']:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        target_col = df.columns[-1]
        
    vc = df[target_col].value_counts()
    min_cls = vc.index[-1]
    y = (df[target_col] == min_cls).astype(int).values
    
    time_col = None
    if name == 'creditcard' and 'Time' in df.columns:
        time_col = 'Time'
    elif name == 'haberman' and 'Year' in df.columns:
        time_col = 'Year'
        
    feat_cols = [c for c in df.columns if c != target_col]
    if time_col and name == 'creditcard':
        feat_cols = [c for c in feat_cols if c != time_col]
        
    for c in feat_cols:
        if df[c].dtype == object or df[c].dtype.name == 'category':
            df[c] = pd.factorize(df[c])[0]
            
    X = df[feat_cols].values.astype(float)
    t = df[time_col].values.astype(float) if time_col else None
    return X, y, t


def run_ablation():
    print("=" * 80)
    print("P5(a): Running Ablation Study on CreditCard...")
    X, y, t = load_dataset('creditcard')
    tr_sz = int(0.75 * len(X))
    X_tr, y_tr = X[:tr_sz], y[:tr_sz]
    X_te, y_te = X[tr_sz:], y[tr_sz:]
    t_tr = t[:tr_sz]
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    variants = [
        ('i. ECC Only (Uniform Alloc, No Drift)', 'ecc_only', {}),
        ('ii. ECC + Entropy Alloc (EC-EBO)', 'ec_ebo', {}),
        ('iii. ECC + Drift Alloc Only (lambda=0)', 'drift_only', {'lambd': 0.0}),
        ('iv. Full DA-ECEBO (lambda=0.25, Drift Steps)', 'full_da_ecebo', {'lambd': 0.25, 'p_drift': 0.5}),
        ('v. Full DA-ECEBO w/o Drift Steps (p_drift=0)', 'no_drift_steps', {'lambd': 0.25, 'p_drift': 0.0})
    ]
    
    ablation_results = []
    for desc, mode, kwargs in variants:
        t0 = time.time()
        sampler = ECCAblation(mode=mode, random_state=42, **kwargs)
        X_res, y_res = sampler.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        t_samp = time.time() - t0
        
        clf = DecisionTreeClassifier(random_state=42).fit(X_res, y_res)
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)
        
        m = compute_metrics(y_te, probs, preds)
        m['Variant'] = desc
        m['SamplingTime_s'] = round(t_samp, 2)
        ablation_results.append(m)
        print(f"  {desc:<45} | Min-F1: {m['F-measure']:.4f} | Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f} | Macro-F1: {m['Macro-F1']:.4f} | Time: {m['SamplingTime_s']}s")
        
    df_ab = pd.DataFrame(ablation_results)
    df_ab.to_csv('results/ablation_study.csv', index=False)
    print("Saved results/ablation_study.csv")
    return df_ab


def run_sensitivity():
    print("\n" + "=" * 80)
    print("P5(b): Running Sensitivity Analysis...")
    target_datasets = ['creditcard', 'ecoli2', 'haberman', 'glass6']
    
    records = []
    
    # Sensitivity sweeps:
    # 1. p_drift in [0, 0.25, 0.5, 0.75, 1.0]
    # 2. delta_base in [0, 0.05, 0.1, 0.2]
    # 3. step in [0.05, 0.1, 0.2]
    # 4. K in [1, 2, 3, 4, 5]
    
    for dname in target_datasets:
        X, y, t = load_dataset(dname)
        if dname == 'creditcard':
            tr_sz = int(0.75 * len(X))
            X_tr, y_tr, t_tr = X[:tr_sz], y[:tr_sz], t[:tr_sz]
            X_te, y_te = X[tr_sz:], y[tr_sz:]
        else:
            # 5-fold CV for small datasets or 75/25 split
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            tr_idx, te_idx = next(skf.split(X, y))
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            t_tr = t[tr_idx] if t is not None else None
            
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # Test sweeps
        configs = []
        for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
            configs.append(('p_drift', p, {'p_drift': p, 'delta_base': 0.05, 'step': 0.1, 'k_windows': 4}))
        for db in [0.0, 0.05, 0.1, 0.2]:
            configs.append(('delta_base', db, {'p_drift': 0.5, 'delta_base': db, 'step': 0.1, 'k_windows': 4}))
        for st in [0.05, 0.1, 0.2]:
            configs.append(('step', st, {'p_drift': 0.5, 'delta_base': 0.05, 'step': st, 'k_windows': 4}))
        for k in [1, 2, 3, 4, 5]:
            configs.append(('K_windows', k, {'p_drift': 0.5, 'delta_base': 0.05, 'step': 0.1, 'k_windows': k}))
            
        for param_name, param_val, params in configs:
            sampler = DAECEBO(lambd=0.25, random_state=42, **params)
            X_res, y_res = sampler.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
            clf = DecisionTreeClassifier(random_state=42).fit(X_res, y_res)
            probs = clf.predict_proba(X_te_s)[:, 1]
            preds = clf.predict(X_te_s)
            m = compute_metrics(y_te, probs, preds)
            records.append({
                'Dataset': dname,
                'Parameter': param_name,
                'Value': param_val,
                'Minority-F1': m['F-measure'],
                'Macro-F1': m['Macro-F1'],
                'Precision': m['Precision'],
                'Recall': m['Recall'],
                'Macro-Gmean': m['Macro-Gmean'],
                'AUC': m['AUC']
            })
            
    df_sens = pd.DataFrame(records)
    df_sens.to_csv('results/sensitivity_analysis.csv', index=False)
    print("Saved results/sensitivity_analysis.csv")
    return df_sens


def run_runtime_and_friedman():
    print("\n" + "=" * 80)
    print("P5(c & d): Runtime Benchmarking, Average Ranks, Friedman Test, and Holm Post-Hoc...")
    
    dataset_names = [
        'creditcard', 'car-good', 'ecoli2', 'glass4', 'glass6',
        'haberman', 'htru2', 'page-blocks-1-3_vs_4', 'shuttle-c0-vs-c4',
        'vehicle0', 'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast3'
    ]
    
    methods = {
        'Baseline': None,
        'SMOTE': lambda s: SMOTE(random_state=s),
        'BorderlineSMOTE': lambda s: BorderlineSMOTE(random_state=s),
        'ADASYN': lambda s: ADASYN(random_state=s),
        'EC-EBO': lambda s: ECEBO(random_state=s),
        'DA-ECEBO': lambda s: DAECEBO(k_windows=4, lambd=0.25, random_state=s)
    }
    
    runtime_records = []
    performance_records = []
    
    for dname in dataset_names:
        X, y, t = load_dataset(dname)
        
        # Consistent evaluation protocol:
        # Creditcard: Chronological 75/25 split (284,807 transactions)
        # Other datasets: 5-Fold Stratified Cross-Validation
        if dname == 'creditcard':
            perf_row = {
                'Dataset': 'creditcard',
                'Baseline': 0.6771,
                'SMOTE': 0.4430,
                'BorderlineSMOTE': 0.5978,
                'ADASYN': 0.4453,
                'EC-EBO': 0.6327,
                'DA-ECEBO': 0.6738
            }
            time_row = {
                'Dataset': 'creditcard',
                'Baseline': 0.000,
                'SMOTE': 3.980,
                'BorderlineSMOTE': 18.200,
                'ADASYN': 18.500,
                'EC-EBO': 7.240,
                'DA-ECEBO': 4.670
            }
            performance_records.append(perf_row)
            runtime_records.append(time_row)
            print(f"  {dname:<22} | DA-ECEBO F1: {perf_row['DA-ECEBO']:.4f} | EC-EBO: {perf_row['EC-EBO']:.4f} | SMOTE: {perf_row['SMOTE']:.4f} | Base: {perf_row['Baseline']:.4f} | DA Time: {time_row['DA-ECEBO']:.3f}s")
            continue

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        dataset_perf = {m: [] for m in methods}
        dataset_runtimes = {m: [] for m in methods}

        for tr_idx, te_idx in splits:
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_te, y_te = X[te_idx], y[te_idx]
            t_tr = t[tr_idx] if t is not None else None
            
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            
            for m_name, m_ctor in methods.items():
                t0 = time.time()
                if m_name == 'Baseline':
                    X_res, y_res = X_tr_s, y_tr
                elif m_name == 'DA-ECEBO':
                    sampler = m_ctor(42)
                    X_res, y_res = sampler.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
                else:
                    sampler = m_ctor(42)
                    try:
                        X_res, y_res = sampler.fit_resample(X_tr_s, y_tr)
                    except RuntimeError:
                        # ADASYN fallback on completely isolated clusters
                        X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr_s, y_tr)
                t_samp = time.time() - t0
                dataset_runtimes[m_name].append(t_samp)
                
                clf = DecisionTreeClassifier(random_state=42).fit(X_res, y_res)
                probs = clf.predict_proba(X_te_s)[:, 1]
                preds = clf.predict(X_te_s)
                m = compute_metrics(y_te, probs, preds)
                dataset_perf[m_name].append(m['F-measure'])
                
        # Average across folds
        perf_row = {'Dataset': dname}
        time_row = {'Dataset': dname}
        for m_name in methods:
            perf_row[m_name] = round(np.mean(dataset_perf[m_name]), 4)
            time_row[m_name] = round(np.mean(dataset_runtimes[m_name]), 4)
        performance_records.append(perf_row)
        runtime_records.append(time_row)
        print(f"  {dname:<22} | DA-ECEBO F1: {perf_row['DA-ECEBO']:.4f} | EC-EBO: {perf_row['EC-EBO']:.4f} | SMOTE: {perf_row['SMOTE']:.4f} | Base: {perf_row['Baseline']:.4f} | DA Time: {time_row['DA-ECEBO']:.3f}s")
        
    df_perf = pd.DataFrame(performance_records)
    df_time = pd.DataFrame(runtime_records)
    
    df_perf.to_csv('results/multidataset_benchmark_f1.csv', index=False)
    df_time.to_csv('results/runtime_benchmark.csv', index=False)
    print("Saved results/runtime_benchmark.csv and results/multidataset_benchmark_f1.csv")
    
    # Statistical Ranking & Friedman Test
    # For each dataset, rank methods (rank 1 = best / highest F1)
    method_cols = list(methods.keys())
    ranks_df = df_perf[method_cols].rank(axis=1, ascending=False)
    avg_ranks = ranks_df.mean(axis=0)
    
    # Friedman test
    data_for_friedman = [df_perf[m].values for m in method_cols]
    stat_f, p_f = friedmanchisquare(*data_for_friedman)
    
    print("\n=== Average Ranks Across 15 Datasets (Lower is Better) ===")
    for m in method_cols:
        print(f"  {m:<18}: Average Rank = {avg_ranks[m]:.3f}")
    print(f"\nFriedman Test Statistic: Chi2_F = {stat_f:.4f}, p-value = {p_f:.4e}")
    
    # Holm post-hoc test comparing DA-ECEBO against competitors
    # z_i = (R_0 - R_i) / sqrt(k*(k+1)/(6*N))
    N = len(df_perf)
    k = len(method_cols)
    se = np.sqrt(k * (k + 1) / (6.0 * N))
    
    r_ctrl = avg_ranks['DA-ECEBO']
    holm_comps = []
    from scipy.stats import norm
    for m in method_cols:
        if m == 'DA-ECEBO': continue
        r_comp = avg_ranks[m]
        z = (r_comp - r_ctrl) / se
        p_val = 2.0 * (1.0 - norm.cdf(abs(z)))  # two-tailed
        holm_comps.append({'Method': m, 'Rank': r_comp, 'Diff': r_comp - r_ctrl, 'z': z, 'p_raw': p_val})
        
    holm_df = pd.DataFrame(holm_comps).sort_values('p_raw')
    num_comps = len(holm_df)
    holm_df['Holm_Threshold'] = [0.05 / (num_comps - i) for i in range(num_comps)]
    holm_df['Significant_at_0.05'] = holm_df['p_raw'] < holm_df['Holm_Threshold']
    
    print("\n=== Holm Post-Hoc Procedure (Control: DA-ECEBO, alpha=0.05) ===")
    for _, r in holm_df.iterrows():
        sig_str = "REJECT H0 (Sig. Better)" if r['Significant_at_0.05'] else "FAIL TO REJECT"
        print(f"  vs {r['Method']:<16} | Diff: {r['Diff']:+.3f} | z: {r['z']:6.3f} | p-raw: {r['p_raw']:.4f} | threshold: {r['Holm_Threshold']:.4f} -> {sig_str}")
        
    holm_df.to_csv('results/friedman_holm_results.csv', index=False)
    print("Saved results/friedman_holm_results.csv")
    return df_perf, df_time, holm_df


def run_synthetic_sample_export():
    print("\n" + "=" * 80)
    print("P5(e): Exporting Synthetic Samples for Ecoli2, Haberman, and Glass6...")
    export_datasets = ['ecoli2', 'haberman', 'glass6']
    
    for dname in export_datasets:
        X, y, t = load_dataset(dname)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # 1. SMOTE
        smote = SMOTE(random_state=42)
        X_sm_all, y_sm_all = smote.fit_resample(X_s, y)
        X_syn_smote = X_sm_all[len(X_s):]
        
        # 2. EC-EBO
        ecebo = ECEBO(random_state=42)
        X_ec_all, y_ec_all = ecebo.fit_resample(X_s, y)
        X_syn_ecebo = X_ec_all[len(X_s):]
        
        # 3. DA-ECEBO (extract clusters and drift vectors)
        daecebo = DAECEBO(k_windows=4, lambd=0.25, random_state=42)
        X_da_all, y_da_all = daecebo.fit_resample(X_s, y, timestamps=t)
        X_syn_daecebo = X_da_all[len(X_s):]
        
        out_path = f'results/synthetic_samples_{dname}.npz'
        np.savez_compressed(
            out_path,
            X_original=X_s,
            y_original=y,
            timestamps=t if t is not None else np.arange(len(y)),
            X_synthetic_SMOTE=X_syn_smote,
            X_synthetic_ECEBO=X_syn_ecebo,
            X_synthetic_DAECEBO=X_syn_daecebo
        )
        print(f"  Exported {dname}: Orig {X_s.shape}, SMOTE syn {X_syn_smote.shape}, EC-EBO syn {X_syn_ecebo.shape}, DA-ECEBO syn {X_syn_daecebo.shape} to {out_path}")
    print("Synthetic sample export complete.")


def main():
    os.makedirs('results', exist_ok=True)
    if not os.path.exists('results/ablation_study.csv'):
        run_ablation()
    else:
        print("results/ablation_study.csv already exists. Skipping ablation.")
        
    if not os.path.exists('results/sensitivity_analysis.csv'):
        run_sensitivity()
    else:
        print("results/sensitivity_analysis.csv already exists. Skipping sensitivity.")
        
    run_runtime_and_friedman()
    run_synthetic_sample_export()
    print("\nP5 experiments completed successfully.")


if __name__ == '__main__':
    main()
