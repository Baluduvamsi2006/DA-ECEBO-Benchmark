"""
========================================================================================
DA-ECEBO and EC-EBO Research Implementation
========================================================================================
Reference Papers:
1. Tao, L., Yu, F., Wang, F., & Wang, Q. (2026).
   EC-EBO: Entropy-balanced oversampling via externally constrained clustering for 
   imbalanced data. Knowledge-Based Systems, 338, 115544.
2. Novel Extension:
   DA-ECEBO: Drift-Aware Entropy-Balanced Oversampling via Externally Constrained 
   Clustering for Non-Stationary Imbalanced Streams.
========================================================================================
"""

import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler


def compute_metrics(y_true, y_prob, y_pred):
    """Compute comprehensive classification performance metrics including Macro versions."""
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = 0.5
    f1 = f1_score(y_true, y_pred, zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    gmean = np.sqrt(rec * spec)
    return {
        'AUC': float(auc),
        'Macro-F1': float(macro_f1),
        'Macro-Gmean': float(gmean),
        'F-measure': float(f1),
        'Precision': float(prec),
        'Recall': float(rec),
        'Specificity': float(spec)
    }


def allocate_samples(weights, delta_n):
    """
    Exact, non-negative integer quota allocation using the Largest Remainder (Hare-Niemeyer) method.
    Guarantees sum(alloc) == delta_n exactly and alloc[i] >= 0 for all i, preventing negative counts.
    """
    if delta_n <= 0:
        return np.zeros(len(weights), dtype=int)
    w_sum = np.sum(weights)
    if w_sum <= 0:
        w = np.ones(len(weights)) / len(weights)
    else:
        w = np.asarray(weights, dtype=float) / w_sum
    quotas = w * delta_n
    floor_alloc = np.floor(quotas).astype(int)
    remainder = quotas - floor_alloc
    shortfall = int(delta_n - np.sum(floor_alloc))
    if shortfall > 0:
        top_indices = np.argsort(-remainder)[:shortfall]
        floor_alloc[top_indices] += 1
    return floor_alloc


class ECEBO:
    """
    Standard EC-EBO as defined in Tao et al., Knowledge-Based Systems 338 (2026) 115544.
    
    Parameters
    ----------
    iters : int, default=60
        Maximum hill-climbing search iterations for each synthetic seed.
    prob : float, default=0.2
        Stochastic perturbation probability during hill climbing.
    alpha : float, default=1.0
        Boundary decay parameter controlling entropy gradient sensitivity.
    k : int, default=7
        Number of majority neighbors for external constraint boundary estimation.
    step : float, default=0.1
        Step size along candidate synthesis search directions.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=42):
        self.iters = iters
        self.prob = prob
        self.alpha = alpha
        self.k = k
        self.step = step
        self.random_state = random_state

    def fit_resample(self, X, y, sampling_strategy=1.0, **kwargs):
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
        
        # 1. External Constrained Clustering (ECC)
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

        # 2. Shannon Entropy Balancing
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        sub_maj = rng.choice(n_maj, size=min(n_maj, n_min * 10), replace=False)
        lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
        
        eps = 1e-12
        p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
        sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
        p_maj = np.clip(lr.predict_proba(X_maj_s)[:, 1], eps, 1.0 - eps)
        maj_entropy = -(p_maj * np.log2(p_maj) + (1.0 - p_maj) * np.log2(1.0 - p_maj))
        
        cluster_list = list(clusters.values())
        m = len(cluster_list)
        cluster_centroids = [centroids[k] for k in clusters.keys()]
        _, cent_nn_maj_idx = nn_maj.kneighbors(np.array(cluster_centroids))
        
        cluster_avg_entropies = np.zeros(m)
        raw_entropy = np.zeros(m)
        for i, mem in enumerate(cluster_list):
            cluster_avg_entropies[i] = np.mean(sample_entropy[mem])
            raw_entropy[i] = np.sum(sample_entropy[mem])
            
        tot_e = np.sum(raw_entropy)
        weights = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
        alloc = allocate_samples(weights, delta_n)
        
        # 3. Constrained Hill-Climbing Synthesis (psi < 0)
        syn = []
        for i, mem in enumerate(cluster_list):
            n_g = alloc[i]
            if n_g <= 0:
                continue
            c_pts = X_min_s[mem]
            mu, e_bar = cluster_centroids[i], cluster_avg_entropies[i]
            maj_nn = cent_nn_maj_idx[i]
            seeds = c_pts[rng.choice(len(mem), size=n_g, replace=True)].copy()
            maj_p = rng.choice(maj_nn, size=n_g, replace=True)
            x_m, e_m = X_maj_s[maj_p], maj_entropy[maj_p]
            
            for t in range(self.iters):
                act = rng.rand(n_g) >= self.prob
                if not np.any(act):
                    break
                step_p = np.zeros((n_g, X.shape[1]))
                nb = np.sum(act)
                step_p[np.where(act)[0], rng.randint(0, X.shape[1], size=nb)] = (
                    rng.choice([-1.0, 1.0], size=nb) * self.step
                )
                trans = seeds + step_p
                psi = (e_bar * np.exp(-self.alpha * np.linalg.norm(trans - mu, axis=1)) -
                       e_m * np.exp(-self.alpha * np.linalg.norm(trans - x_m, axis=1)))
                acc = psi < 0
                seeds[acc & act] = trans[acc & act]
            syn.append(seeds)
            
        if len(syn) > 0:
            X_syn = scaler.inverse_transform(np.vstack(syn))
            return np.vstack([X, X_syn]), np.hstack([y, np.full(len(X_syn), min_class)])
        return X, y


class DAECEBO:
    """
    Drift-Aware Entropy-Balanced Oversampling via Externally Constrained Clustering (DA-ECEBO).
    
    Formalizes temporal drift into:
    1. Recency velocity D_recency (quantifying active vs. obsolete historical sub-clusters)
    2. Spatial centroid displacement D_spatial (quantifying temporal shift in feature space)
    
    Balances drift and boundary entropy via:
        Score(C_i) = lambda * S_entropy(C_i) + (1 - lambda) * S_drift(C_i) + delta_base
    
    Parameters
    ----------
    k_windows : int, default=4
        Number of chronological time windows for temporal drift partitioning.
    lambd : float, default=0.25
        Tradeoff balance parameter (lambda=1.0 recovers static EC-EBO; lambda < 1.0 biases toward drift).
    iters : int, default=60
        Maximum hill-climbing search iterations for synthetic generation.
    prob : float, default=0.2
        Stochastic exploration probability.
    alpha : float, default=1.0
        Entropy boundary decay parameter.
    k : int, default=7
        Number of majority neighbors for external constraint boundary.
    step : float, default=0.1
        Step size along synthesis perturbation directions.
    p_drift : float, default=0.5
        Probability of stepping along the active cluster drift trajectory vector.
    delta_base : float, default=0.05
        Baseline smoothing allocation constant across all sub-clusters.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    def __init__(self, k_windows=4, lambd=0.25, iters=60, prob=0.2, 
                 alpha=1.0, k=7, step=0.1, p_drift=0.5, delta_base=0.05, random_state=42):
        self.k_windows = max(1, int(k_windows))
        self.lambd = float(lambd)
        self.iters = iters
        self.prob = prob
        self.alpha = alpha
        self.k = k
        self.step = step
        self.p_drift = p_drift
        self.delta_base = delta_base
        self.random_state = random_state

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
        
        # 1. Temporal Window Indexing
        if timestamps is not None:
            min_times = timestamps[idx_min]
            t_min, t_max = np.min(timestamps), np.max(timestamps)
            time_bins = np.linspace(t_min, t_max + 1e-9, self.k_windows + 1)
            sample_windows = np.clip(np.digitize(min_times, time_bins) - 1, 0, self.k_windows - 1)
        else:
            bins = np.linspace(0, len(X) + 1, self.k_windows + 1)
            sample_windows = np.clip(np.digitize(idx_min, bins) - 1, 0, self.k_windows - 1)
            
        # 2. Externally Constrained Clustering
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

        # 3. Boundary Entropy
        lr = LogisticRegression(max_iter=1000, random_state=self.random_state)
        sub_maj = rng.choice(n_maj, size=min(n_maj, n_min * 10), replace=False)
        lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
        
        eps = 1e-12
        p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
        sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
        p_maj = np.clip(lr.predict_proba(X_maj_s)[:, 1], eps, 1.0 - eps)
        maj_entropy = -(p_maj * np.log2(p_maj) + (1.0 - p_maj) * np.log2(1.0 - p_maj))
        
        cluster_list = list(clusters.values())
        m = len(cluster_list)
        cluster_centroids = [centroids[k] for k in clusters.keys()]
        _, cent_nn_maj_idx = nn_maj.kneighbors(np.array(cluster_centroids))
        
        cluster_avg_entropies = np.zeros(m)
        raw_entropy = np.zeros(m)
        for i, mem in enumerate(cluster_list):
            cluster_avg_entropies[i] = np.mean(sample_entropy[mem])
            raw_entropy[i] = np.sum(sample_entropy[mem])
            
        # 4. Temporal Drift Dynamics (Recency & Spatial Displacement)
        raw_drift = np.zeros(m)
        drift_vecs = []
        if self.k_windows > 1:
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
        
        # 5. Tradeoff Allocation Score
        scores = (S_entropy + self.delta_base) if self.k_windows == 1 else (
            (self.lambd * S_entropy) + ((1.0 - self.lambd) * S_drift) + self.delta_base
        )
        weights = scores / np.sum(scores) if np.sum(scores) > 0 else np.ones(m) / m
        alloc = allocate_samples(weights, delta_n)
        
        # 6. Drift-Biased Hill Climbing Synthesis
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
                if has_d and self.p_drift > 0:
                    d_mask = (rng.rand(n_g) < self.p_drift) & act
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
                psi = (e_bar * np.exp(-self.alpha * np.linalg.norm(trans - mu, axis=1)) -
                       e_m * np.exp(-self.alpha * np.linalg.norm(trans - x_m, axis=1)))
                acc = psi < 0
                seeds[acc & act] = trans[acc & act]
            syn.append(seeds)
            
        if len(syn) > 0:
            X_syn = scaler.inverse_transform(np.vstack(syn))
            return np.vstack([X, X_syn]), np.hstack([y, np.full(len(X_syn), min_class)])
        return X, y


def run_benchmark():
    """Main routine executing expanding-window research on creditcard benchmark."""
    print("=" * 80)
    print("DA-ECEBO & EC-EBO RESEARCH BENCHMARK")
    print("=" * 80)
    
    data_path = 'datasets/creditcard.parquet'
    df = pd.read_parquet(data_path).sort_values('Time').reset_index(drop=True)
    n_total = len(df)
    train_size = int(0.75 * n_total)
    
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    feature_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    X_tr = train_df[feature_cols].values
    y_tr = train_df['Class'].values
    t_tr = train_df['Time'].values
    
    X_te = test_df[feature_cols].values
    y_te = test_df['Class'].values
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    print(f"Total Transactions: {n_total:,} | Historical Train: {len(train_df):,} | Future Test: {len(test_df):,}")
    print(f"Historical Frauds: {y_tr.sum():,} | Future Frauds: {y_te.sum():,}")
    
    samplers = {
        'Baseline': None,
        'RandomOver': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42),
        'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'EC-EBO': ECEBO(random_state=42),
        'DA-ECEBO (K=4, lam=0.25)': DAECEBO(k_windows=4, lambd=0.25, random_state=42)
    }
    
    results = []
    for s_name, sampler in samplers.items():
        t0 = time.time()
        if sampler is None:
            X_res, y_res = X_tr_s, y_tr
        elif 'DA-ECEBO' in s_name:
            X_res, y_res = sampler.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        else:
            X_res, y_res = sampler.fit_resample(X_tr_s, y_tr)
        t_samp = time.time() - t0
        
        clf = DecisionTreeClassifier(random_state=42)
        clf.fit(X_res, y_res)
        probs = clf.predict_proba(X_te_s)[:, 1]
        preds = clf.predict(X_te_s)
        
        m = compute_metrics(y_te, probs, preds)
        m['Method'] = s_name
        m['SampTime_s'] = round(t_samp, 2)
        results.append(m)
        print(f"  {s_name:<28} | F1: {m['F-measure']:.4f} | Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f} | AUC: {m['AUC']:.4f}")
        
    return pd.DataFrame(results)


if __name__ == '__main__':
    run_benchmark()
