"""
Candidate Fix for DA-ECEBO (DAECEBO2)
=====================================
Modifies only the cluster-scoring normalization step:
  - S_entropy = raw_entropy / sum(raw_entropy)
  - S_drift   = raw_drift   / sum(raw_drift) (if sum is 0, uniform 1/m)
  - with k_windows == 1, S_drift = uniform 1/m
  - score = lambd * S_entropy + (1 - lambd) * S_drift + delta_base
  - delta_base default = 0.0
  - weights = score / sum(score)
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from da_ecebo_and_ecebo_implementation import allocate_samples


class DAECEBO2:
    """
    Candidate Fix: Drift-Aware Entropy-Balanced Oversampling (DA-ECEBO v2).
    
    Guarantees strict reduction to EC-EBO when k_windows=1, lambd=1.0, delta_base=0.0.
    """
    def __init__(self, k_windows=4, lambd=0.25, iters=60, prob=0.2, 
                 alpha=1.0, k=7, step=0.1, p_drift=0.5, delta_base=0.0, random_state=42):
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

        # --- REVISED STEP 5: PROPER L1 CLUSTER-SCORE NORMALIZATION ---
        tot_e = np.sum(raw_entropy)
        S_entropy = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
        
        if self.k_windows > 1:
            tot_d = np.sum(raw_drift)
            S_drift = raw_drift / tot_d if tot_d > 0 else np.ones(m) / m
        else:
            S_drift = np.ones(m) / m
            
        scores = (self.lambd * S_entropy) + ((1.0 - self.lambd) * S_drift) + self.delta_base
        tot_scores = np.sum(scores)
        weights = scores / tot_scores if tot_scores > 0 else np.ones(m) / m
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

