import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.abspath('.'))
from da_ecebo_and_ecebo_implementation import allocate_samples

def compute_sample_entropy(lr, X_s):
    eps = 1e-12
    p = np.clip(lr.predict_proba(X_s)[:, 1], eps, 1.0 - eps)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def load_data(name):
    if name == 'creditcard':
        df = pd.read_parquet('datasets/creditcard.parquet')
        n_tr = int(len(df) * 0.75)
        df_tr = df.iloc[:n_tr]
        X = df_tr.drop(columns=['Class']).values.astype(np.float64)
        y = df_tr['Class'].values.astype(int)
        return X, y
    elif name == 'Pima':
        path = 'datasets_keel/pima.dat'
        lines = []
        with open(path, 'r') as f:
            for line in f:
                if not line.startswith('@') and line.strip():
                    lines.append(line.strip().split(','))
        df = pd.DataFrame(lines)
        X = df.iloc[:, :-1].values.astype(np.float64)
        y_raw = df.iloc[:, -1].values
        # Minority is positive
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        return X, y
    else:
        path = os.path.join('datasets', f'{name}.csv')
        df = pd.read_csv(path)
        X = df.iloc[:, :-1].values.astype(np.float64)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        return X, y

def run_instrumented_sampling(X, y, mode='ECEBO', random_state=42, iters=60, prob=0.2, alpha=1.0, k=7, step=0.1):
    rng = np.random.RandomState(random_state)
    classes, counts = np.unique(y, return_counts=True)
    maj_class = classes[np.argmax(counts)]
    min_class = classes[np.argmin(counts)]
    
    idx_maj = np.where(y == maj_class)[0]
    idx_min = np.where(y == min_class)[0]
    X_maj, X_min = X[idx_maj], X[idx_min]
    n_maj, n_min = len(X_maj), len(X_min)
    
    delta_n = max(0, n_maj - n_min)
    if delta_n == 0:
        return None

    scaler = StandardScaler()
    X_maj_s = scaler.fit_transform(X_maj)
    X_min_s = scaler.transform(X_min)
    
    # 1. External Constrained Clustering (ECC)
    nn_maj = NearestNeighbors(n_neighbors=max(k, 1), algorithm='auto', n_jobs=-1).fit(X_maj_s)
    dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
    d_maj_arr = dist_to_maj[:, 0]
    
    clusters = {i: [i] for i in range(n_min)}
    centroids = {i: X_min_s[i].copy() for i in range(n_min)}
    cluster_d_maj = {i: float(d_maj_arr[i]) for i in range(n_min)}
    
    n_c = n_min
    dist_mat = np.full((n_c, n_c), np.inf)
    for i in range(n_c):
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

    # 2. Shannon Entropy Balancing
    lr = LogisticRegression(max_iter=1000, random_state=random_state)
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
        
    tot_e = np.sum(raw_entropy)
    weights = raw_entropy / tot_e if tot_e > 0 else np.ones(m) / m
    alloc = allocate_samples(weights, delta_n)
    
    # 3. Instrumented Synthesis
    syn = []
    total_proposals = 0
    total_accepted = 0
    all_starting_psi = []
    
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
        
        # Calculate psi at starting point
        psi_init = (e_bar * np.exp(-alpha * np.linalg.norm(seeds - mu, axis=1)) -
                    e_m * np.exp(-alpha * np.linalg.norm(seeds - x_m, axis=1)))
        all_starting_psi.extend(psi_init.tolist())
        
        psi_curr = psi_init.copy()
        
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
            
            if mode == 'ECEBO':
                # Original test: psi < 0
                acc = psi_prop < 0
                n_acc = int(np.sum(acc & act))
                total_accepted += n_acc
                seeds[acc & act] = trans[acc & act]
            else: # ECEBO_PSI0
                # |psi| approaches zero test
                acc = np.abs(psi_prop) < np.abs(psi_curr)
                n_acc = int(np.sum(acc & act))
                total_accepted += n_acc
                seeds[acc & act] = trans[acc & act]
                psi_curr[acc & act] = psi_prop[acc & act]
                
        syn.append(seeds)
        
    X_syn_s = np.vstack(syn)
    X_syn = scaler.inverse_transform(X_syn_s)
    
    # Analyze synthetic samples in scaled feature space X_min_s
    # Share of synthetic rows that EXACTLY equal a real minority row
    # nn query of synthetic points to real minority points
    nn_real = NearestNeighbors(n_neighbors=1).fit(X_min_s)
    dists_to_real, idx_real = nn_real.kneighbors(X_syn_s)
    dists_to_real = dists_to_real.ravel()
    
    is_exact_dup = dists_to_real < 1e-9
    exact_dup_share = float(np.mean(is_exact_dup))
    
    # Internal baseline distance: median 1-NN distance among real minority samples
    nn_self = NearestNeighbors(n_neighbors=2).fit(X_min_s)
    dists_self, _ = nn_self.kneighbors(X_min_s)
    median_nn_real = float(np.median(dists_self[:, 1]))
    if median_nn_real < 1e-9:
        median_nn_real = 1e-9
        
    non_dup_dists = dists_to_real[~is_exact_dup]
    if len(non_dup_dists) > 0:
        ratio_dist = float(np.median(non_dup_dists) / median_nn_real)
    else:
        ratio_dist = np.nan
        
    acceptance_rate = float(total_accepted / total_proposals) if total_proposals > 0 else 0.0
    
    all_starting_psi = np.array(all_starting_psi)
    psi_min = float(np.min(all_starting_psi))
    psi_median = float(np.median(all_starting_psi))
    psi_max = float(np.max(all_starting_psi))
    share_psi_neg = float(np.mean(all_starting_psi < 0))
    
    return {
        "Mode": mode,
        "Dataset": None, # set by caller
        "Seed": random_state,
        "Requested": delta_n,
        "Generated": len(X_syn),
        "Exact_Duplicate_Share": round(exact_dup_share, 4),
        "Median_Dist_Ratio_NonDup": round(ratio_dist, 4) if not np.isnan(ratio_dist) else np.nan,
        "Proposals": total_proposals,
        "Accepted": total_accepted,
        "Acceptance_Rate": round(acceptance_rate, 4),
        "Starting_Psi_Min": round(psi_min, 6),
        "Starting_Psi_Median": round(psi_median, 6),
        "Starting_Psi_Max": round(psi_max, 6),
        "Share_Starting_Psi_LT_0": round(share_psi_neg, 4)
    }

print("Task F1/F2 runner functions defined.")

if __name__ == '__main__':
    datasets = ['Ecoli2', 'Glass6', 'Haberman', 'Pima', 'Yeast3', 'Vehicle0', 'creditcard']
    seeds = [42, 43, 44, 45, 46]
    
    rows_f1 = []
    rows_f2 = []
    
    for dname in datasets:
        print(f'Loading {dname}...')
        X, y = load_data(dname)
        print(f'  {dname}: {X.shape[0]} rows, {X.shape[1]} features, maj={np.sum(y==0)}, min={np.sum(y==1)}')
        
        for s in seeds:
            print(f'  Running {dname} seed {s} ECEBO...')
            res_f1 = run_instrumented_sampling(X, y, mode='ECEBO', random_state=s)
            res_f1['Dataset'] = dname
            rows_f1.append(res_f1)
            
            print(f'  Running {dname} seed {s} ECEBO_PSI0...')
            res_f2 = run_instrumented_sampling(X, y, mode='ECEBO_PSI0', random_state=s)
            res_f2['Dataset'] = dname
            rows_f2.append(res_f2)
            
    df_f1 = pd.DataFrame(rows_f1)
    df_f2 = pd.DataFrame(rows_f2)
    
    os.makedirs('results_stream', exist_ok=True)
    df_f1.to_csv('results_stream/F1_ecebo_mechanism.csv', index=False)
    df_f2.to_csv('results_stream/F2_ecebo_psi0_mechanism.csv', index=False)
    print('F1 and F2 finished successfully!')
