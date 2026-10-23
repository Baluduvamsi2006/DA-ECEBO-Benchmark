"""
T1: Allocation Concentration Diagnostics across 5 Variants:
1. EC-EBO (raw entropy sum allocation)
2. (c) K=1-style smoothed entropy (min-max entropy + delta_base 0.05)
3. (a) Real-Time DA-ECEBO (headline K=2, lambda=0.0)
4. (b) Permuted-Time DA-ECEBO (K=2, lambda=0.0 with permuted timestamps)
5. ECC-Uniform (equal quota to all clusters)

Metrics recorded per seed (10 seeds, 42..51):
- Gini coefficient of quota vector
- Effective number of clusters (1 / sum p_i^2)
- Share of total quota going to top 10 clusters
- Minority-F1 (CART classifier on CreditCard test set)
Spearman rank correlation between concentration metrics and F1 across all 50 (variant x seed) pairs.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import (
    ECEBO, DAECEBO, compute_metrics, allocate_samples
)


def gini_coeff(arr):
    arr = np.array(arr, dtype=float)
    if np.all(arr == 0):
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * arr)) / (n * np.sum(arr)) - (n + 1.0) / n)


def effective_n(arr):
    arr = np.array(arr, dtype=float)
    tot = np.sum(arr)
    if tot == 0:
        return 0.0
    p = arr / tot
    return float(1.0 / np.sum(p ** 2))


def top10_share(arr):
    arr = np.array(arr, dtype=float)
    tot = np.sum(arr)
    if tot == 0:
        return 0.0
    p = arr / tot
    return float(np.sum(np.sort(p)[-10:]))


def _eval_t1_seed(args):
    s, X_tr_s, y_tr, t_tr, X_te_s, y_te, cluster_list, cluster_centroids, cent_nn_maj_idx, d_maj_arr, delta_n = args
    records = []
    rng = np.random.RandomState(s)
    
    idx_maj = np.where(y_tr == 0)[0]
    idx_min = np.where(y_tr == 1)[0]
    X_maj_s = X_tr_s[idx_maj]
    X_min_s = X_tr_s[idx_min]
    n_maj, n_min = len(X_maj_s), len(X_min_s)
    m = len(cluster_list)
    
    # Logistic regression for boundary entropy
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=s)
    sub_maj = rng.choice(n_maj, size=min(n_maj, n_min * 10), replace=False)
    lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
    eps = 1e-12
    p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
    sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
    
    cluster_avg_entropies = np.zeros(m)
    raw_entropy = np.zeros(m)
    for i, mem in enumerate(cluster_list):
        cluster_avg_entropies[i] = np.mean(sample_entropy[mem])
        raw_entropy[i] = np.sum(sample_entropy[mem])
        
    def norm_01(arr):
        span = np.max(arr) - np.min(arr)
        return (arr - np.min(arr)) / span if span > 1e-9 else np.ones_like(arr)
        
    S_entropy = norm_01(raw_entropy)
    
    # Timestamps & Drift
    t_min_val, t_max_val = np.min(t_tr), np.max(t_tr)
    time_bins = np.linspace(t_min_val, t_max_val + 1e-9, 2 + 1)
    sample_windows_real = np.clip(np.digitize(t_tr[idx_min], time_bins) - 1, 0, 1)
    
    t_tr_perm = np.random.RandomState(s).permutation(t_tr)
    time_bins_p = np.linspace(np.min(t_tr_perm), np.max(t_tr_perm) + 1e-9, 2 + 1)
    sample_windows_perm = np.clip(np.digitize(t_tr_perm[idx_min], time_bins_p) - 1, 0, 1)
    
    def get_drift(windows):
        raw_drift = np.zeros(m)
        mid = 1
        for i, mem in enumerate(cluster_list):
            c_wins = windows[mem]
            c_pts = X_min_s[mem]
            rec = (np.mean(c_wins) + 1.0) / 2.0
            early, recent = c_wins < mid, c_wins >= mid
            if np.any(early) and np.any(recent):
                v = np.mean(c_pts[recent], axis=0) - np.mean(c_pts[early], axis=0)
                spat = np.linalg.norm(v)
            elif np.any(recent):
                spat = 1.0
            else:
                spat = 0.0
            raw_drift[i] = rec * (1.0 + spat)
        return norm_01(raw_drift)
        
    S_drift_real = get_drift(sample_windows_real)
    S_drift_perm = get_drift(sample_windows_perm)
    
    variants = {}
    
    # 1. EC-EBO (raw entropy sum allocation)
    w_ec = raw_entropy / np.sum(raw_entropy) if np.sum(raw_entropy) > 0 else np.ones(m) / m
    q_ec = allocate_samples(w_ec, delta_n)
    s_ec = ECEBO(random_state=s)
    X_res, y_res = s_ec.fit_resample(X_tr_s, y_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    m_ec = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    variants['EC-EBO'] = (q_ec, m_ec['F-measure'])
    
    # 2. (c) K=1-style smoothed entropy
    scores_c = S_entropy + 0.05
    w_c = scores_c / np.sum(scores_c)
    q_c = allocate_samples(w_c, delta_n)
    s_c = DAECEBO(k_windows=1, lambd=0.0, random_state=s)
    X_res, y_res = s_c.fit_resample(X_tr_s, y_tr, timestamps=None)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    m_c = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    variants['(c) K=1-style Smoothed Entropy'] = (q_c, m_c['F-measure'])
    
    # 3. (a) Real-Time DA-ECEBO (headline K=2, lambda=0)
    scores_a = S_drift_real + 0.05
    w_a = scores_a / np.sum(scores_a)
    q_a = allocate_samples(w_a, delta_n)
    s_a = DAECEBO(k_windows=2, lambd=0.0, random_state=s)
    X_res, y_res = s_a.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    m_a = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    variants['(a) Real-Time DA-ECEBO'] = (q_a, m_a['F-measure'])
    
    # 4. (b) Permuted-Time DA-ECEBO
    scores_b = S_drift_perm + 0.05
    w_b = scores_b / np.sum(scores_b)
    q_b = allocate_samples(w_b, delta_n)
    s_b = DAECEBO(k_windows=2, lambd=0.0, random_state=s)
    X_res, y_res = s_b.fit_resample(X_tr_s, y_tr, timestamps=t_tr_perm)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    m_b = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    variants['(b) Permuted-Time'] = (q_b, m_b['F-measure'])
    
    # 5. ECC-Uniform
    w_u = np.ones(m) / m
    q_u = allocate_samples(w_u, delta_n)
    # Uniform quota synthesis: using same hill climbing with uniform quota
    # We can use DAECEBO with delta_base=1e9 (or custom synthesis)
    s_u = DAECEBO(k_windows=1, lambd=1.0, delta_base=1e8, random_state=s)
    X_res, y_res = s_u.fit_resample(X_tr_s, y_tr, timestamps=None)
    clf = DecisionTreeClassifier(random_state=s).fit(X_res, y_res)
    m_u = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    variants['ECC-Uniform'] = (q_u, m_u['F-measure'])
    
    for vname, (q_vec, f1_val) in variants.items():
        records.append({
            'Seed': s,
            'Variant': vname,
            'Gini': gini_coeff(q_vec),
            'Effective_N': effective_n(q_vec),
            'Top10_Share': top10_share(q_vec),
            'Minority-F1': f1_val,
            'Quota_Vec': q_vec
        })
        
    return records


def run_t1():
    print("=" * 80)
    print("T1: ALLOCATION CONCENTRATION DIAGNOSTICS (10 SEEDS)...")
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
        results_lists = list(executor.map(_eval_t1_seed, tasks))
    print(f"T1 completed across 10 seeds in {time.time() - t0:.2f}s")
    
    records = [r for sublist in results_lists for r in sublist]
    df_raw = pd.DataFrame(records)
    df_raw.drop(columns=['Quota_Vec']).to_csv('results/t1_concentration_raw.csv', index=False)
    
    variant_order = [
        'EC-EBO',
        '(c) K=1-style Smoothed Entropy',
        '(a) Real-Time DA-ECEBO',
        '(b) Permuted-Time',
        'ECC-Uniform'
    ]
    
    summary = []
    for v in variant_order:
        sub = df_raw[df_raw['Variant'] == v]
        summary.append({
            'Variant': v,
            'Gini_Quota': f"{sub['Gini'].mean():.4f} +/- {sub['Gini'].std():.4f}",
            'Effective_N': f"{sub['Effective_N'].mean():.2f} +/- {sub['Effective_N'].std():.2f}",
            'Top10_Share': f"{sub['Top10_Share'].mean():.4f} +/- {sub['Top10_Share'].std():.4f}",
            'Minority-F1': f"{sub['Minority-F1'].mean():.4f} +/- {sub['Minority-F1'].std():.4f}",
            'Gini_mean': sub['Gini'].mean(),
            'EffN_mean': sub['Effective_N'].mean(),
            'Top10_mean': sub['Top10_Share'].mean(),
            'F1_mean': sub['Minority-F1'].mean()
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv('results/t1_concentration_summary.csv', index=False)
    
    # Spearman Correlations across all 50 observations
    corr_gini, p_gini = spearmanr(df_raw['Gini'], df_raw['Minority-F1'])
    corr_effn, p_effn = spearmanr(df_raw['Effective_N'], df_raw['Minority-F1'])
    corr_top10, p_top10 = spearmanr(df_raw['Top10_Share'], df_raw['Minority-F1'])
    
    corrs = [
        {'Metric': 'Gini Coefficient vs Minority-F1', 'Spearman_rho': corr_gini, 'p_value': p_gini},
        {'Metric': 'Effective N Clusters vs Minority-F1', 'Spearman_rho': corr_effn, 'p_value': p_effn},
        {'Metric': 'Top-10 Share vs Minority-F1', 'Spearman_rho': corr_top10, 'p_value': p_top10}
    ]
    df_corr = pd.DataFrame(corrs)
    df_corr.to_csv('results/t1_spearman_correlations.csv', index=False)
    
    print("\n=== T1: Allocation Concentration Diagnostics (Mean over 10 Seeds) ===")
    print(df_sum[['Variant', 'Gini_Quota', 'Effective_N', 'Top10_Share', 'Minority-F1']].to_string(index=False))
    print("\n=== Spearman Rank Correlations with Minority-F1 (N=50 observations) ===")
    print(df_corr.to_string(index=False))


if __name__ == '__main__':
    run_t1()
