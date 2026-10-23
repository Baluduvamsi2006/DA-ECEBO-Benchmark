"""
T2: Random-Score Control and Quota Allocation Diagnostics (10 Seeds, CART, CreditCard 75/25):
Variants:
1. Random Uniform[0,1] + delta_base 0.05 per cluster (no time, no entropy, same hill-climbing)
2. Allocation Proportional to Cluster Size
3. Top-10 Highest Recency (quota split equally among 10 most recent clusters)
4. Top-30 Highest Recency (quota split equally among 30 most recent clusters)
5. Top-10 Highest Entropy (quota split equally among 10 highest entropy clusters)
6. Top-30 Highest Entropy (quota split equally among 30 highest entropy clusters)
Comparators:
7. EC-EBO
8. (a) Real-Time DA-ECEBO (headline K=2, lambda=0.0)

Report: Minority-F1, Precision, Recall, 10 seeds, with paired Wilcoxon vs EC-EBO and vs (a).
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import (
    ECEBO, DAECEBO, compute_metrics, allocate_samples
)


def _eval_t2_seed(args):
    s, X_tr_s, y_tr, t_tr, X_te_s, y_te, cluster_list, cluster_centroids, cent_nn_maj_idx, d_maj_arr, delta_n = args
    records = []
    rng = np.random.RandomState(s)
    
    idx_maj = np.where(y_tr == 0)[0]
    idx_min = np.where(y_tr == 1)[0]
    X_maj_s = X_tr_s[idx_maj]
    X_min_s = X_tr_s[idx_min]
    n_maj, n_min = len(X_maj_s), len(X_min_s)
    m = len(cluster_list)
    cluster_sizes = np.array([len(c) for c in cluster_list])
    
    # Boundary Entropy
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=s)
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
        
    def norm_01(arr):
        span = np.max(arr) - np.min(arr)
        return (arr - np.min(arr)) / span if span > 1e-9 else np.ones_like(arr)
        
    S_entropy = norm_01(raw_entropy)
    
    # Recency & Drift under K=2
    t_min_val, t_max_val = np.min(t_tr), np.max(t_tr)
    time_bins = np.linspace(t_min_val, t_max_val + 1e-9, 2 + 1)
    sample_windows = np.clip(np.digitize(t_tr[idx_min], time_bins) - 1, 0, 1)
    
    raw_drift = np.zeros(m)
    drift_vecs = []
    mid = 1
    for i, mem in enumerate(cluster_list):
        c_wins = sample_windows[mem]
        c_pts = X_min_s[mem]
        rec = (np.mean(c_wins) + 1.0) / 2.0
        early, recent = c_wins < mid, c_wins >= mid
        if np.any(early) and np.any(recent):
            v = np.mean(c_pts[recent], axis=0) - np.mean(c_pts[early], axis=0)
            spat = np.linalg.norm(v)
        elif np.any(recent):
            v, spat = np.zeros(X_tr_s.shape[1]), 1.0
        else:
            v, spat = np.zeros(X_tr_s.shape[1]), 0.0
        nv = np.linalg.norm(v)
        if nv > 1e-6:
            v = v / nv
        drift_vecs.append(v)
        raw_drift[i] = rec * (1.0 + spat)
        
    S_drift = norm_01(raw_drift)
    
    # Generic Hill-Climbing Synthesis function with custom quotas
    def synthesize_with_quotas(alloc, d_vecs=None, p_drift_val=0.0):
        syn = []
        for i, mem in enumerate(cluster_list):
            n_g = alloc[i]
            if n_g <= 0:
                continue
            c_pts = X_min_s[mem]
            mu, e_bar = cluster_centroids[i], cluster_avg_entropies[i]
            maj_nn = cent_nn_maj_idx[i]
            d_vec = d_vecs[i] if d_vecs is not None else np.zeros(X_tr_s.shape[1])
            has_d = np.linalg.norm(d_vec) > 1e-6
            
            seeds = c_pts[rng.choice(len(mem), size=n_g, replace=True)].copy()
            maj_p = rng.choice(maj_nn, size=n_g, replace=True)
            x_m, e_m = X_maj_s[maj_p], maj_entropy[maj_p]
            
            for t in range(60):
                act = rng.rand(n_g) >= 0.2
                if not np.any(act):
                    break
                step_p = np.zeros_like(seeds)
                if has_d and p_drift_val > 0.0:
                    d_mask = (rng.rand(n_g) < p_drift_val) & act
                    step_p[d_mask] = d_vec * 0.1
                    b_mask = (~d_mask) & act
                else:
                    b_mask = act.copy()
                    
                if np.any(b_mask):
                    nb = np.sum(b_mask)
                    step_p[np.where(b_mask)[0], rng.randint(0, X_tr_s.shape[1], size=nb)] = (
                        rng.choice([-1.0, 1.0], size=nb) * 0.1
                    )
                trans = seeds + step_p
                psi = (e_bar * np.exp(-1.0 * np.linalg.norm(trans - mu, axis=1)) -
                       e_m * np.exp(-1.0 * np.linalg.norm(trans - x_m, axis=1)))
                acc = psi < 0
                seeds[acc & act] = trans[acc & act]
            syn.append(seeds)
            
        if len(syn) > 0:
            X_syn = scaler_inverse(np.vstack(syn))
            return np.vstack([scaler_inverse(X_tr_s), X_syn]), np.hstack([y_tr, np.ones(len(X_syn), dtype=int)])
        return scaler_inverse(X_tr_s), y_tr

    def scaler_inverse(arr):
        # We can work directly in scaled space for tree training
        return arr

    variants = {}
    
    # 1. Random uniform[0,1] + delta_base 0.05
    r_scores = rng.uniform(0.0, 1.0, size=m) + 0.05
    w_rnd = r_scores / np.sum(r_scores)
    alloc_rnd = allocate_samples(w_rnd, delta_n)
    X_rnd, y_rnd = synthesize_with_quotas(alloc_rnd, d_vecs=None, p_drift_val=0.0)
    clf = DecisionTreeClassifier(random_state=s).fit(X_rnd, y_rnd)
    variants['1. Random Uniform[0,1] + 0.05'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 2. Proportional to cluster size
    w_sz = cluster_sizes.astype(float) / np.sum(cluster_sizes)
    alloc_sz = allocate_samples(w_sz, delta_n)
    X_sz, y_sz = synthesize_with_quotas(alloc_sz, d_vecs=None, p_drift_val=0.0)
    clf = DecisionTreeClassifier(random_state=s).fit(X_sz, y_sz)
    variants['2. Proportional to Cluster Size'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 3. Top-10 Highest Recency
    top10_rec_idx = np.argsort(-S_drift)[:10]
    w_top10_rec = np.zeros(m)
    w_top10_rec[top10_rec_idx] = 1.0 / 10.0
    alloc_top10_rec = allocate_samples(w_top10_rec, delta_n)
    X_t10r, y_t10r = synthesize_with_quotas(alloc_top10_rec, d_vecs=drift_vecs, p_drift_val=0.5)
    clf = DecisionTreeClassifier(random_state=s).fit(X_t10r, y_t10r)
    variants['3. Top-10 Highest Recency'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 4. Top-30 Highest Recency
    top30_rec_idx = np.argsort(-S_drift)[:30]
    w_top30_rec = np.zeros(m)
    w_top30_rec[top30_rec_idx] = 1.0 / 30.0
    alloc_top30_rec = allocate_samples(w_top30_rec, delta_n)
    X_t30r, y_t30r = synthesize_with_quotas(alloc_top30_rec, d_vecs=drift_vecs, p_drift_val=0.5)
    clf = DecisionTreeClassifier(random_state=s).fit(X_t30r, y_t30r)
    variants['4. Top-30 Highest Recency'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 5. Top-10 Highest Entropy
    top10_ent_idx = np.argsort(-S_entropy)[:10]
    w_top10_ent = np.zeros(m)
    w_top10_ent[top10_ent_idx] = 1.0 / 10.0
    alloc_top10_ent = allocate_samples(w_top10_ent, delta_n)
    X_t10e, y_t10e = synthesize_with_quotas(alloc_top10_ent, d_vecs=None, p_drift_val=0.0)
    clf = DecisionTreeClassifier(random_state=s).fit(X_t10e, y_t10e)
    variants['5. Top-10 Highest Entropy'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 6. Top-30 Highest Entropy
    top30_ent_idx = np.argsort(-S_entropy)[:30]
    w_top30_ent = np.zeros(m)
    w_top30_ent[top30_ent_idx] = 1.0 / 30.0
    alloc_top30_ent = allocate_samples(w_top30_ent, delta_n)
    X_t30e, y_t30e = synthesize_with_quotas(alloc_top30_ent, d_vecs=None, p_drift_val=0.0)
    clf = DecisionTreeClassifier(random_state=s).fit(X_t30e, y_t30e)
    variants['6. Top-30 Highest Entropy'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 7. EC-EBO comparator
    s_ec = ECEBO(random_state=s)
    X_res, y_res = s_ec.fit_resample(X_tr_s, y_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    variants['7. EC-EBO'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    # 8. (a) Real-Time DA-ECEBO headline comparator (K=2, lambda=0.0)
    s_da = DAECEBO(k_windows=2, lambd=0.0, random_state=s)
    X_res, y_res = s_da.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    variants['8. (a) Real-Time DA-ECEBO'] = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    
    for vname, m_dict in variants.items():
        m_dict['Variant'] = vname
        m_dict['Seed'] = s
        records.append(m_dict)
        
    return records


def run_t2():
    print("=" * 80)
    print("T2: RANDOM-SCORE CONTROL & QUOTA SCHEMES (10 SEEDS, CART)...")
    print("=" * 80)
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    tr_sz = int(0.75 * len(df))
    train_df, test_df = df.iloc[:tr_sz], df.iloc[tr_sz:]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = train_df['Class'].values.astype(int)
    t_tr = train_df['Time'].values.astype(float)
    X_te = test_df[feat_cols].values
    y_te = test_df['Class'].values.astype(int)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    idx_maj = np.where(y_tr == 0)[0]
    idx_min = np.where(y_tr == 1)[0]
    X_maj_s = X_tr_s[idx_maj]
    X_min_s = X_tr_s[idx_min]
    n_min = len(X_min_s)
    delta_n = len(X_maj_s) - n_min
    
    from sklearn.neighbors import NearestNeighbors
    nn_maj = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=1).fit(X_maj_s)
    dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
    d_maj_arr = dist_to_maj[:, 0]
    
    clusters = {i: [i] for i in range(n_min)}
    centroids = {i: X_min_s[i].copy() for i in range(n_min)}
    cluster_d_maj = {i: float(d_maj_arr[i]) for i in range(n_min)}
    dist_mat = np.full((n_min, n_min), np.inf)
    for i in range(n_min):
        for j in range(i + 1, n_min):
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
    cluster_centroids = [centroids[k] for k in clusters.keys()]
    _, cent_nn_maj_idx = nn_maj.kneighbors(np.array(cluster_centroids))
    
    seeds = list(range(42, 52))
    tasks = [
        (s, X_tr_s, y_tr, t_tr, X_te_s, y_te, cluster_list, cluster_centroids, cent_nn_maj_idx, d_maj_arr, delta_n)
        for s in seeds
    ]
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        results_lists = list(executor.map(_eval_t2_seed, tasks))
    print(f"T2 completed in {time.time() - t0:.2f}s")
    
    records = [r for sublist in results_lists for r in sublist]
    df_raw = pd.DataFrame(records)
    df_raw.to_csv('results/t2_random_controls_raw.csv', index=False)
    
    variant_names = [
        '1. Random Uniform[0,1] + 0.05',
        '2. Proportional to Cluster Size',
        '3. Top-10 Highest Recency',
        '4. Top-30 Highest Recency',
        '5. Top-10 Highest Entropy',
        '6. Top-30 Highest Entropy',
        '7. EC-EBO',
        '8. (a) Real-Time DA-ECEBO'
    ]
    
    summary = []
    for v in variant_names:
        sub = df_raw[df_raw['Variant'] == v]
        summary.append({
            'Variant': v,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}",
            'F1_mean': sub['F-measure'].mean()
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv('results/t2_random_controls_summary.csv', index=False)
    
    # Paired Wilcoxon Tests vs EC-EBO and vs (a) Real-Time DA-ECEBO
    f1_ec = df_raw[df_raw['Variant'] == '7. EC-EBO'].sort_values('Seed')['F-measure'].values
    f1_da = df_raw[df_raw['Variant'] == '8. (a) Real-Time DA-ECEBO'].sort_values('Seed')['F-measure'].values
    
    pvals = []
    for v in variant_names:
        if v in ['7. EC-EBO', '8. (a) Real-Time DA-ECEBO']:
            continue
        f1_v = df_raw[df_raw['Variant'] == v].sort_values('Seed')['F-measure'].values
        w_ec = wilcoxon(f1_v, f1_ec)
        w_da = wilcoxon(f1_v, f1_da)
        pvals.append({
            'Variant': v,
            'F1_Mean': np.mean(f1_v),
            'Diff_vs_ECEBO': np.mean(f1_v - f1_ec),
            'p_vs_ECEBO': w_ec.pvalue,
            'Sig_vs_ECEBO': w_ec.pvalue < 0.05,
            'Diff_vs_DA': np.mean(f1_v - f1_da),
            'p_vs_DA': w_da.pvalue,
            'Sig_vs_DA': w_da.pvalue < 0.05
        })
    df_pvals = pd.DataFrame(pvals)
    df_pvals.to_csv('results/t2_wilcoxon_tests.csv', index=False)
    
    print("\n=== T2: Performance Summary (10 Seeds, CART) ===")
    print(df_sum[['Variant', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC']].to_string(index=False))
    print("\n=== T2: Paired Wilcoxon Tests vs EC-EBO and vs (a) DA-ECEBO ===")
    print(df_pvals.to_string(index=False))


if __name__ == '__main__':
    run_t2()
