"""
Task A - Equivalence / Reduction Unit Tests between ECEBO and DAECEBO
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

import da_ecebo_and_ecebo_implementation as impl
from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, allocate_samples

def get_datasets():
    datasets = {}
    
    # 1. Ecoli2
    df = pd.read_csv('datasets/ecoli2.csv')
    X = df.drop(columns=['Class']).values.astype(float)
    y = df['Class'].values
    datasets['Ecoli2'] = (X, y)
    
    # 2. Glass6
    df = pd.read_csv('datasets/glass6.csv')
    X = df.drop(columns=['Class']).values.astype(float)
    y = df['Class'].values
    datasets['Glass6'] = (X, y)
    
    # 3. Haberman
    df = pd.read_csv('datasets/haberman.csv')
    X = df.drop(columns=['Class']).values.astype(float)
    y = df['Class'].values
    datasets['Haberman'] = (X, y)
    
    # 4. Creditcard 5000-row subsample (keep all frauds)
    df_cc = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    frauds = df_cc[df_cc['Class'] == 1]
    non_frauds = df_cc[df_cc['Class'] == 0].iloc[:5000 - len(frauds)]
    sub_df = pd.concat([non_frauds, frauds]).sort_values('Time').reset_index(drop=True)
    feature_cols = [c for c in sub_df.columns if c not in ['Time', 'Class']]
    X_cc = sub_df[feature_cols].values.astype(float)
    y_cc = sub_df['Class'].values
    datasets['Creditcard_5k'] = (X_cc, y_cc)
    
    return datasets

def extract_quotas_and_internal_steps(X, y, seed, is_daecebo=False, k_windows=1, lambd=1.0, delta_base=0.0):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    maj_class, min_class = (classes[0], classes[1]) if counts[0] > counts[1] else (classes[1], classes[0])
    idx_maj = np.where(y == maj_class)[0]
    idx_min = np.where(y == min_class)[0]
    X_maj, X_min = X[idx_maj], X[idx_min]
    n_maj, n_min = len(X_maj), len(X_min)
    delta_n = max(0, n_maj - n_min)
    
    scaler = StandardScaler()
    X_maj_s = scaler.fit_transform(X_maj)
    X_min_s = scaler.transform(X_min)
    
    if is_daecebo:
        bins = np.linspace(0, len(X) + 1, k_windows + 1)
        sample_windows = np.clip(np.digitize(idx_min, bins) - 1, 0, k_windows - 1)
        
    nn_maj = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=-1).fit(X_maj_s)
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

    lr = LogisticRegression(max_iter=1000, random_state=seed)
    sub_maj = rng.choice(n_maj, size=min(n_maj, n_min * 10), replace=False)
    lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
    
    eps = 1e-12
    p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
    sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
    
    cluster_list = list(clusters.values())
    m = len(cluster_list)
    raw_entropy = np.zeros(m)
    for i, mem in enumerate(cluster_list):
        raw_entropy[i] = np.sum(sample_entropy[mem])
        
    if not is_daecebo:
        tot_e = np.sum(raw_entropy)
        weights = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
        alloc = allocate_samples(weights, delta_n)
    else:
        def norm_01(arr):
            span = np.max(arr) - np.min(arr)
            return (arr - np.min(arr)) / span if span > 1e-9 else np.ones_like(arr)
        S_entropy = norm_01(raw_entropy)
        scores = (S_entropy + delta_base) if k_windows == 1 else (
            (lambd * S_entropy) + delta_base
        )
        weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(m) / m
        alloc = allocate_samples(weights, delta_n)
        
    return {
        'clusters': cluster_list,
        'centroids': [centroids[k] for k in clusters.keys()],
        'sub_maj': sub_maj,
        'raw_entropy': raw_entropy,
        'weights': weights,
        'alloc': alloc
    }

