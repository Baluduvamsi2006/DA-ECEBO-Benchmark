"""
DAECEBO2_PSI0: Candidate Fix with |psi| -> 0 Hill-Climbing Synthesis
=====================================================================
Combines:
  1. DAECEBO2 L1-normalized cluster scoring (da_ecebo_v2_candidate.py)
  2. ECEBO_PSI0 hill-climbing synthesis condition (|psi(g')| < |psi(g)|)
     (ecebo_psi0_variant.py)

LABEL: interpretation, not the authors' code.
Guarantees exact mathematical equivalence to ECEBO_PSI0 when:
  k_windows=1, lambd=1.0, delta_base=0.0
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from da_ecebo_and_ecebo_implementation import allocate_samples


def compute_sample_entropy(lr, X_s):
    eps = 1e-12
    p = np.clip(lr.predict_proba(X_s)[:, 1], eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


class DAECEBO2_PSI0:
    """
    Drift-Aware Entropy-Balanced Oversampling with |psi| -> 0 hill climbing.
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

    def fit_resample(self, X, y, sampling_strategy=1.0):
        rng = np.random.RandomState(self.random_state)
        classes, counts = np.unique(y, return_counts=True)
        maj_class = classes[np.argmax(counts)]
        min_class = classes[np.argmin(counts)]
        
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
        X_sub = np.vstack([X_maj_s[sub_maj], X_min_s])
        y_sub = np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)])
        lr.fit(X_sub, y_sub)
        
        sample_entropy = compute_sample_entropy(lr, X_min_s)
        maj_entropy = compute_sample_entropy(lr, X_maj_s)
        
        cluster_list = list(clusters.values())
        m = len(cluster_list)
        cluster_centroids = [centroids[k] for k in clusters.keys()]
        _, cent_nn_maj_idx = nn_maj.kneighbors(np.array(cluster_centroids))
        
        cluster_avg_entropies = np.zeros(m)
        raw_entropy = np.zeros(m)
        for i, mem in enumerate(cluster_list):
            cluster_avg_entropies[i] = np.mean(sample_entropy[mem])
            raw_entropy[i] = np.sum(sample_entropy[mem])
            
        # 3 & 4. Temporal Drift Decomposition
        drift_vecs = []
        raw_drift = np.zeros(m)
        
        if self.k_windows > 1:
            w_size = int(np.ceil(n_min / self.k_windows))
            time_labels = np.zeros(n_min, dtype=int)
            for w in range(self.k_windows):
                time_labels[w * w_size : min((w + 1) * w_size, n_min)] = w
                
            for i, mem in enumerate(cluster_list):
                mem_arr = np.array(mem)
                w_indices = time_labels[mem_arr]
                weights_rec = (w_indices + 1) / self.k_windows
                rec = np.mean(weights_rec)
                
                recent = mem_arr[w_indices >= (self.k_windows // 2)]
                older = mem_arr[w_indices < (self.k_windows // 2)]
                
                if len(recent) > 0 and len(older) > 0:
                    c_rec = np.mean(X_min_s[recent], axis=0)
                    c_old = np.mean(X_min_s[older], axis=0)
                    v = c_rec - c_old
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

        # 5. L1 Normalized Cluster Scores
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
        
        # 6. Constrained Synthesis with |psi| -> 0 Hill-Climbing
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
            
            # Initial psi value at seeds
            psi_curr = (e_bar * np.exp(-self.alpha * np.linalg.norm(seeds - mu, axis=1)) -
                        e_m * np.exp(-self.alpha * np.linalg.norm(seeds - x_m, axis=1)))
            
            for t in range(self.iters):
                act = rng.rand(n_g) >= self.prob
                if not np.any(act):
                    break
                step_p = np.zeros((n_g, X.shape[1]))
                if has_d and self.p_drift > 0:
                    d_mask = (rng.rand(n_g) < self.p_drift) & act
                    step_p[d_mask] = d_vec * self.step
                    rnd_mask = (~d_mask) & act
                    nb = np.sum(rnd_mask)
                    if nb > 0:
                        step_p[np.where(rnd_mask)[0], rng.randint(0, X.shape[1], size=nb)] = (
                            rng.choice([-1.0, 1.0], size=nb) * self.step
                        )
                else:
                    nb = np.sum(act)
                    step_p[np.where(act)[0], rng.randint(0, X.shape[1], size=nb)] = (
                        rng.choice([-1.0, 1.0], size=nb) * self.step
                    )
                trans = seeds + step_p
                psi_prop = (e_bar * np.exp(-self.alpha * np.linalg.norm(trans - mu, axis=1)) -
                            e_m * np.exp(-self.alpha * np.linalg.norm(trans - x_m, axis=1)))
                
                # INTERPRETATION: Accept if candidate approaches zero (|psi(trans)| < |psi(seeds)|)
                acc = np.abs(psi_prop) < np.abs(psi_curr)
                seeds[acc & act] = trans[acc & act]
                psi_curr[acc & act] = psi_prop[acc & act]
                
            syn.append(seeds)
            
        if len(syn) > 0:
            X_syn = scaler.inverse_transform(np.vstack(syn))
            return np.vstack([X, X_syn]), np.hstack([y, np.full(len(X_syn), min_class)])
        return X, y

