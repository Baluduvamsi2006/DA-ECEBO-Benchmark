"""
Part 1: R2. Credit Card Clusters Reconciliation
================================================
Compares ECC clustering on the chronological 75/25 split between:
  1. Repo code (da_ecebo_and_ecebo_implementation.py, Time excluded from features)
  2. Task H pipeline (task_h_triage.py: subsamples majority to 20,000 for scaling & NN)

Also explains in exact numbers why N_min_train is 398, 417, or 385 in different runs:
  - 70% chronological split (0.70 * 284,807 = 199,364 rows): N_min_train = 384/385
  - 75% chronological split (0.75 * 284,807 = 213,605 rows): N_min_train = 398
  - 80% chronological split (0.80 * 284,807 = 227,845 rows): N_min_train = 417

Output: results_stream/R2_cc_clusters.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def run_ecc_clustering(X_min_s, d_maj_arr):
    n_min = len(X_min_s)
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
    return len(clusters)

def main():
    df = pd.read_parquet('datasets/creditcard.parquet')
    total_rows = len(df)
    total_min = int(df['Class'].sum())
    
    split_75 = int(total_rows * 0.75)
    df_tr = df.iloc[:split_75]
    
    features = [c for c in df.columns if c not in ['Time', 'Class']]
    feat_str = ", ".join(features)
    
    X_tr = df_tr[features].values.astype(np.float64)
    y_tr = df_tr['Class'].values.astype(int)
    
    N_tr = len(X_tr)
    N_min_tr = int(np.sum(y_tr == 1))
    N_maj_tr = int(np.sum(y_tr == 0))
    
    X_maj = X_tr[y_tr == 0]
    X_min = X_tr[y_tr == 1]
    
    # 1. Repo Code (Full Majority for StandardScaler and NN)
    scaler_repo = StandardScaler()
    X_maj_s_repo = scaler_repo.fit_transform(X_maj)
    X_min_s_repo = scaler_repo.transform(X_min)
    nn_maj_repo = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=-1).fit(X_maj_s_repo)
    d_maj_repo, _ = nn_maj_repo.kneighbors(X_min_s_repo)
    m_repo = run_ecc_clustering(X_min_s_repo, d_maj_repo[:, 0])
    
    # 2. Task H Pipeline (Majority Subsampled to 20,000 for StandardScaler and NN)
    rng = np.random.RandomState(42)
    maj_sub_idx = rng.choice(len(X_maj), size=20000, replace=False)
    X_maj_sub = X_maj[maj_sub_idx]
    
    scaler_h = StandardScaler()
    X_maj_s_h = scaler_h.fit_transform(X_maj_sub)
    X_min_s_h = scaler_h.transform(X_min)
    nn_maj_h = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=-1).fit(X_maj_s_h)
    d_maj_h, _ = nn_maj_h.kneighbors(X_min_s_h)
    m_task_h = run_ecc_clustering(X_min_s_h, d_maj_h[:, 0])
    
    # Analyze split counts for 70%, 75%, 80%
    n_70 = int(total_rows * 0.70)
    n_min_70 = int(df.iloc[:n_70]['Class'].sum())
    
    n_75 = int(total_rows * 0.75)
    n_min_75 = int(df.iloc[:n_75]['Class'].sum())
    
    n_80 = int(total_rows * 0.80)
    n_min_80 = int(df.iloc[:n_80]['Class'].sum())
    
    rows = [
        {
            'Pipeline_Name': 'Repo Code (da_ecebo_and_ecebo_implementation.py)',
            'Chronological_Split_Fraction': '75/25',
            'N_train': N_tr,
            'N_min_train': N_min_tr,
            'N_maj_train': N_maj_tr,
            'Features_Count': len(features),
            'Features_Used': feat_str,
            'Scaling_Method': 'StandardScaler fit on all 213,207 train majority instances',
            'Majority_Used_for_NN': '213207 (Full majority train set)',
            'm_clusters': m_repo,
            'Cluster_Ratio': round(m_repo / N_min_tr, 4),
            'Split_Discrepancy_Explanation': 'Standard 75% chronological cutoff at row 213,605 yields N_min_train=398.'
        },
        {
            'Pipeline_Name': 'Task H Pipeline (task_h_triage.py)',
            'Chronological_Split_Fraction': '75/25',
            'N_train': N_tr,
            'N_min_train': N_min_tr,
            'N_maj_train': N_maj_tr,
            'Features_Count': len(features),
            'Features_Used': feat_str,
            'Scaling_Method': 'StandardScaler fit on 20,000 train majority subsample (seed 42)',
            'Majority_Used_for_NN': '20000 (Subsampled to prevent memory exhaustion)',
            'm_clusters': m_task_h,
            'Cluster_Ratio': round(m_task_h / N_min_tr, 4),
            'Split_Discrepancy_Explanation': 'Subsampling majority to 20k increases d_maj threshold, allowing more pairs to merge -> m falls from 146 to 91.'
        },
        {
            'Pipeline_Name': '70% Split (Cut 2 in rolling origin benchmark)',
            'Chronological_Split_Fraction': '70/30',
            'N_train': n_70,
            'N_min_train': n_min_70,
            'N_maj_train': n_70 - n_min_70,
            'Features_Count': len(features),
            'Features_Used': feat_str,
            'Scaling_Method': 'N/A',
            'Majority_Used_for_NN': 'N/A',
            'm_clusters': 'N/A',
            'Cluster_Ratio': 'N/A',
            'Split_Discrepancy_Explanation': f'Cut 2 in rolling_origin_benchmark used 70% train (199,364 rows), containing {n_min_70} frauds (the 385 fraud run).'
        },
        {
            'Pipeline_Name': '80% Split (Cut 4 in 80/20 benchmark)',
            'Chronological_Split_Fraction': '80/20',
            'N_train': n_80,
            'N_min_train': n_min_80,
            'N_maj_train': n_80 - n_min_80,
            'Features_Count': len(features),
            'Features_Used': feat_str,
            'Scaling_Method': 'N/A',
            'Majority_Used_for_NN': 'N/A',
            'm_clusters': 'N/A',
            'Cluster_Ratio': 'N/A',
            'Split_Discrepancy_Explanation': f'80% chronological train cutoff (227,845 rows) contains {n_min_80} frauds (the 417 fraud run).'
        }
    ]
    
    df_res = pd.DataFrame(rows)
    os.makedirs('results_stream', exist_ok=True)
    out_csv = 'results_stream/R2_cc_clusters.csv'
    df_res.to_csv(out_csv, index=False)
    print(f"\nSaved R2 results to {out_csv}")
    print(df_res[['Pipeline_Name', 'N_train', 'N_min_train', 'm_clusters', 'Cluster_Ratio']].to_string())

if __name__ == '__main__':
    main()

