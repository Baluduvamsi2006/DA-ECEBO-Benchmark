"""
Part 1: R1. Acceptance and Duplicates Audit
===========================================
Datasets: Ecoli2 and Haberman, seed 42.
Models:
  1. ECEBO literal (da_ecebo_and_ecebo_implementation.py)
  2. DAECEBO original min-max version (da_ecebo_and_ecebo_implementation.py)
  3. DAECEBO2 literal (da_ecebo_v2_candidate.py, k_windows=1, lambd=1.0, delta_base=0.0)
  4. DAECEBO2_PSI0 (da_ecebo_v2_psi0.py, k_windows=1, lambd=1.0, delta_base=0.0)
Also test DAECEBO and DAECEBO2 with default (k_windows=4, lambd=0.25).

Definitions:
- "accepted": Number of hill-climbing candidate proposals accepted by the test condition.
  For literal models: psi_prop < 0.
  For PSI0 models: |psi_prop| < |psi_curr|.
- Numerator for acceptance rate: total accepted proposal steps across all iterations.
- Denominator for acceptance rate: total active proposed perturbation steps across all iterations (iters=60).
- "duplicate": Number of synthetic points in X_syn that match a real minority sample in X_min
  with max absolute feature difference < 1e-9 (or np.array_equal on raw or standardized space).

Output: results_stream/R1_acceptance.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath('.'))
from da_ecebo_and_ecebo_implementation import allocate_samples

def compute_sample_entropy(lr, X_s):
    eps = 1e-12
    p = np.clip(lr.predict_proba(X_s)[:, 1], eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

def load_data(name):
    df = pd.read_csv(f'datasets/{name}.csv')
    X = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values
    classes, counts = np.unique(y_raw, return_counts=True)
    min_class = classes[np.argmin(counts)]
    y = np.where(y_raw == min_class, 1, 0)
    return X, y

def run_instrumented_sampler(X, y, model_type='ecebo_literal', seed=42, k_windows=1, lambd=1.0, delta_base=0.0):
    rng = np.random.RandomState(seed)
    n_maj = int(np.sum(y == 0))
    n_min = int(np.sum(y == 1))
    delta_n = max(0, n_maj - n_min)
    
    X_maj = X[y == 0]
    X_min = X[y == 1]
    
    scaler = StandardScaler()
    X_maj_s = scaler.fit_transform(X_maj)
    X_min_s = scaler.transform(X_min)
    
    # 1. ECC Clustering
    nn_maj = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=-1).fit(X_maj_s)
    dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
    d_maj_arr = dist_to_maj[:, 0]
    
    clusters = {i: [i] for i in range(n_min)}
    centroids = {i: X_min_s[i].copy() for i in range(n_min)}
    cluster_d_maj = {i: float(d_maj_arr[i]) for i in range(n_min)}
    
    dist_mat = np.full((n_min, n_min), np.inf)
    for i in range(n_min):
        d_i = np.linalg.norm(centroids[i] - X_min_s[i+1:], axis=1)
        thresh = np.minimum(cluster_d_maj[i], d_maj_arr[i+1:])
        valid = np.where(d_i < thresh)[0]
        for j_idx in valid:
            j = i + 1 + j_idx
            dist_mat[i, j] = dist_mat[j, i] = d_i[j_idx]
            
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
        for other in list(clusters.keys()):
            if other != c1:
                d_c = np.linalg.norm(centroids[c1] - centroids[other])
                if d_c < min(cluster_d_maj[c1], cluster_d_maj[other]):
                    dist_mat[c1, other] = dist_mat[other, c1] = d_c
                else:
                    dist_mat[c1, other] = dist_mat[other, c1] = np.inf
                    
    # 2. Entropy / Drift Allocation
    lr = LogisticRegression(max_iter=1000, random_state=seed)
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
        
    if model_type == 'daecebo_orig':
        # Original min-max scaling from da_ecebo_and_ecebo_implementation.py lines 380-391
        e_min, e_max = np.min(raw_entropy), np.max(raw_entropy)
        s_entropy = (raw_entropy - e_min) / (e_max - e_min + 1e-9) if e_max > e_min else np.zeros(m)
        s_drift = np.zeros(m) # no time attribute passed
        score = lambd * s_entropy + (1.0 - lambd) * s_drift + 0.05
        weights = score / np.sum(score)
    elif model_type in ['ecebo_literal', 'ecebo_psi0']:
        tot_e = np.sum(raw_entropy)
        weights = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
    elif model_type in ['daecebo2_literal', 'daecebo2_psi0']:
        tot_e = np.sum(raw_entropy)
        s_entropy = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
        s_drift = np.ones(m) / m
        score = lambd * s_entropy + (1.0 - lambd) * s_drift + delta_base
        weights = score / np.sum(score)
    else:
        raise ValueError(f"Unknown model_type {model_type}")
        
    alloc = allocate_samples(weights, delta_n)
    
    # 3. Hill Climbing Synthesis
    syn = []
    total_proposals = 0
    total_accepted = 0
    alpha = 1.0
    step = 0.1
    iters = 60
    prob = 0.2
    
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
        
        psi_curr = (e_bar * np.exp(-alpha * np.linalg.norm(seeds - mu, axis=1)) -
                    e_m * np.exp(-alpha * np.linalg.norm(seeds - x_m, axis=1)))
                    
        for t in range(iters):
            act = rng.rand(n_g) >= prob
            if not np.any(act):
                break
            nb = int(np.sum(act))
            total_proposals += nb
            
            step_p = np.zeros((n_g, X.shape[1]))
            step_p[np.where(act)[0], rng.randint(0, X.shape[1], size=nb)] = (
                rng.choice([-1.0, 1.0], size=nb) * step
            )
            trans = seeds + step_p
            psi_prop = (e_bar * np.exp(-alpha * np.linalg.norm(trans - mu, axis=1)) -
                        e_m * np.exp(-alpha * np.linalg.norm(trans - x_m, axis=1)))
                        
            if model_type in ['ecebo_literal', 'daecebo_orig', 'daecebo2_literal']:
                acc = psi_prop < 0
                n_acc = int(np.sum(acc & act))
                total_accepted += n_acc
                seeds[acc & act] = trans[acc & act]
            else: # psi0 models
                acc = np.abs(psi_prop) < np.abs(psi_curr)
                n_acc = int(np.sum(acc & act))
                total_accepted += n_acc
                seeds[acc & act] = trans[acc & act]
                psi_curr[acc & act] = psi_prop[acc & act]
                
        syn.append(seeds)
        
    X_syn_s = np.vstack(syn)
    X_syn = scaler.inverse_transform(X_syn_s)
    
    # Exact duplicate check against X_min
    # In standardized space:
    nn_std = NearestNeighbors(n_neighbors=1).fit(X_min_s)
    dists_std, _ = nn_std.kneighbors(X_syn_s)
    n_dups_std = int(np.sum(dists_std.ravel() < 1e-9))
    
    # In raw space:
    nn_raw = NearestNeighbors(n_neighbors=1).fit(X_min)
    dists_raw, _ = nn_raw.kneighbors(X_syn)
    n_dups_raw = int(np.sum(dists_raw.ravel() < 1e-9))
    
    return {
        'total_proposals': total_proposals,
        'total_accepted': total_accepted,
        'acceptance_rate': total_accepted / max(1, total_proposals),
        'n_synthetic': len(X_syn),
        'n_duplicates_std': n_dups_std,
        'duplicate_share_std': n_dups_std / len(X_syn),
        'n_duplicates_raw': n_dups_raw,
        'duplicate_share_raw': n_dups_raw / len(X_syn),
    }

def main():
    datasets = ['Ecoli2', 'Haberman']
    models = [
        ('ECEBO literal', 'ecebo_literal', 1, 1.0, 0.0,
         "da_ecebo_and_ecebo_implementation.py:133-136", "psi = (e_bar*exp(-alpha*||trans-mu||) - e_m*exp(-alpha*||trans-x_m||)); acc = psi < 0; seeds[acc & act] = trans[acc & act]",
         "dists_to_real < 1e-9 in standardized X_min_s space"),
        ('DAECEBO (original min-max)', 'daecebo_orig', 4, 0.25, 0.05,
         "da_ecebo_and_ecebo_implementation.py:408-411", "psi = (e_bar*exp(-alpha*||trans-mu||) - e_m*exp(-alpha*||trans-x_m||)); acc = psi < 0; seeds[acc & act] = trans[acc & act]",
         "dists_to_real < 1e-9 in standardized X_min_s space"),
        ('DAECEBO2 literal', 'daecebo2_literal', 1, 1.0, 0.0,
         "da_ecebo_v2_candidate.py:133-136", "psi = (e_bar*exp(-alpha*||trans-mu||) - e_m*exp(-alpha*||trans-x_m||)); acc = psi < 0; seeds[acc & act] = trans[acc & act]",
         "dists_to_real < 1e-9 in standardized X_min_s space"),
        ('DAECEBO2_PSI0', 'daecebo2_psi0', 1, 1.0, 0.0,
         "da_ecebo_v2_psi0.py:141-144", "acc = np.abs(psi_prop) < np.abs(psi_curr); seeds[acc & act] = trans[acc & act]; psi_curr[acc & act] = psi_prop[acc & act]",
         "dists_to_real < 1e-9 in standardized X_min_s space")
    ]
    
    rows = []
    print("=== PART 1: R1. RECONCILIATION AUDIT (Seed 42) ===")
    for dname in datasets:
        X, y = load_data(dname)
        print(f"\n--- Dataset: {dname} (N={len(X)}, N_maj={np.sum(y==0)}, N_min={np.sum(y==1)}, delta_n={np.sum(y==0)-np.sum(y==1)}) ---")
        for mname, mtype, kw, lam, db, code_lines, code_snip, dup_def in models:
            res = run_instrumented_sampler(X, y, model_type=mtype, seed=42, k_windows=kw, lambd=lam, delta_base=db)
            print(f"[{mname}] Proposals: {res['total_proposals']}, Accepted: {res['total_accepted']} "
                  f"(AccRate: {res['acceptance_rate']:.4f}), Syn: {res['n_synthetic']}, "
                  f"ExactDups: {res['n_duplicates_std']} (DupShare: {res['duplicate_share_std']:.4f})")
            rows.append({
                'dataset': dname,
                'seed': 42,
                'model_name': mname,
                'model_type': mtype,
                'code_file_and_lines': code_lines,
                'acceptance_rule_code': code_snip,
                'duplicate_definition': dup_def,
                'numerator_accepted': res['total_accepted'],
                'denominator_proposals': res['total_proposals'],
                'acceptance_rate': round(res['acceptance_rate'], 4),
                'synthetic_requested': res['n_synthetic'],
                'synthetic_generated': res['n_synthetic'],
                'exact_copies_count': res['n_duplicates_std'],
                'exact_copies_share': round(res['duplicate_share_std'], 4),
                'exact_copies_raw_count': res['n_duplicates_raw'],
                'exact_copies_raw_share': round(res['duplicate_share_raw'], 4)
            })
            
    df_out = pd.DataFrame(rows)
    os.makedirs('results_stream', exist_ok=True)
    out_path = 'results_stream/R1_acceptance.csv'
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved R1 audit to {out_path}")

if __name__ == '__main__':
    main()

