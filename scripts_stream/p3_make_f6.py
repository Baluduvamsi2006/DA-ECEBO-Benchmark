"""
Phase 3 Part 5: Figure P3_F6 Generator
======================================
Illustrative figure for Review Comment 4:
  - 2-D t-SNE of credit card TRAINING minority fraud samples only (with majority context in light grey).
  - Colored by temporal arrival window (K=4).
  - Cluster outlines / convex hulls.
  - Arrows showing the mean drift vector per cluster.
  - Number of synthetic samples (quota) written at each cluster.
  - Saves underlying data to results_stream/P3_F6_data.csv.
  - Saves vector PDF to figs_stream/P3_F6_creditcard_clusters.pdf (and .png).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull

sys.path.insert(0, os.path.abspath('.'))
from da_ecebo_v2_candidate import DAECEBO2

def main():
    os.makedirs('figs_stream', exist_ok=True)
    os.makedirs('results_stream', exist_ok=True)

    print("Loading credit card dataset...")
    df = pd.read_parquet('datasets/creditcard.parquet')
    y_full = df['Class'].values.astype(int)
    X_full = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
    t_full = df['Time'].values.astype(np.float64)

    # 75% chronological train split
    n_tr = int(0.75 * len(X_full))
    X_tr, y_tr, t_tr = X_full[:n_tr], y_full[:n_tr], t_full[:n_tr]

    min_idx = np.where(y_tr == 1)[0]
    maj_idx = np.where(y_tr == 0)[0]
    X_min, t_min = X_tr[min_idx], t_tr[min_idx]

    # Subsample 1,000 majority instances for background context
    rng = np.random.RandomState(42)
    bg_maj_idx = rng.choice(maj_idx, size=1500, replace=False)
    X_bg_maj = X_tr[bg_maj_idx]

    # Assign K=4 time windows
    t_start, t_end = np.min(t_tr), np.max(t_tr)
    time_bins = np.linspace(t_start, t_end + 1e-9, 5)
    windows = np.clip(np.digitize(t_min, time_bins) - 1, 0, 3)

    # Run DAECEBO2 clustering and quota calculation
    model = DAECEBO2(k_windows=4, lambd=0.25, delta_base=0.0, random_state=42)
    # We inspect the internal cluster assignments
    scaler = StandardScaler()
    X_maj_s = scaler.fit_transform(X_tr[maj_idx[:20000]])
    X_min_s = scaler.transform(X_min)

    # Run clustering using DAECEBO2 logic
    from sklearn.neighbors import NearestNeighbors
    nn_maj = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=-1).fit(X_maj_s)
    dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
    d_maj = np.min(dist_to_maj, axis=1)

    clusters = []
    cluster_windows = []
    unassigned = set(range(len(X_min)))

    while unassigned:
        seed_i = unassigned.pop()
        curr_clust = [seed_i]
        curr_wins = [windows[seed_i]]
        r_c = d_maj[seed_i]
        to_rem = set()
        for o_i in unassigned:
            dist = np.linalg.norm(X_min_s[seed_i] - X_min_s[o_i])
            if dist < r_c:
                curr_clust.append(o_i)
                curr_wins.append(windows[o_i])
                to_rem.add(o_i)
                r_c = max(r_c, d_maj[o_i])
        unassigned -= to_rem
        clusters.append(curr_clust)
        cluster_windows.append(curr_wins)

    m = len(clusters)
    print(f"Total minority clusters identified: {m}")

    # Compute quotas for Delta N = len(maj_idx) - len(min_idx)
    delta_n = len(maj_idx) - len(min_idx)
    raw_entropy = np.zeros(m)
    raw_drift = np.zeros(m)
    drift_vectors = []
    decay_weights = np.array([np.exp(k_w - 4) for k_w in range(4)])

    for i, (clust, c_wins) in enumerate(zip(clusters, cluster_windows)):
        c_pts = X_min_s[clust]
        n_c = len(clust)
        if n_c <= 1:
            raw_entropy[i] = 1.0
            drift_vectors.append(np.zeros(X_min.shape[1]))
        else:
            dists = np.sqrt(np.sum((c_pts[:, np.newaxis, :] - c_pts[np.newaxis, :, :])**2, axis=-1))
            k_val = min(7, n_c)
            d_c = np.sum(np.partition(dists, k_val - 1, axis=1)[:, :k_val]) / (n_c * k_val)
            raw_entropy[i] = -np.sum(np.exp(-1.0 * (dists / (d_c + 1e-8))**2))
            
            # Mean drift vector: direction from older samples to newer samples
            w_arr = np.array(c_wins)
            if np.any(w_arr >= 2) and np.any(w_arr < 2):
                v = np.mean(c_pts[w_arr >= 2], axis=0) - np.mean(c_pts[w_arr < 2], axis=0)
                nv = np.linalg.norm(v)
                v = v / nv if nv > 1e-6 else np.zeros(X_min.shape[1])
            else:
                v = np.zeros(X_min.shape[1])
            drift_vectors.append(v)

        w_counts = np.bincount(c_wins, minlength=4)
        raw_drift[i] = np.sum((w_counts / n_c) * decay_weights)

    tot_e = np.sum(raw_entropy)
    S_entropy = raw_entropy / tot_e if tot_e > 0 else np.full(m, 1.0/m)
    tot_d = np.sum(raw_drift)
    S_drift = raw_drift / tot_d if tot_d > 0 else np.full(m, 1.0/m)

    scores = 0.25 * S_entropy + 0.75 * S_drift
    weights = scores / np.sum(scores)

    exact_quotas = delta_n * weights
    quotas = np.floor(exact_quotas).astype(int)
    rem = delta_n - np.sum(quotas)
    if rem > 0:
        frac = exact_quotas - quotas
        ranks = np.argsort(-frac)
        for r_i in range(rem):
            quotas[ranks[r_i % m]] += 1

    # Compute 2-D t-SNE on combined background majority + minority
    print("Computing 2-D t-SNE embedding...")
    X_tsne_input = np.vstack([scaler.transform(X_bg_maj), X_min_s])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_embedded = tsne.fit_transform(X_tsne_input)

    emb_bg_maj = X_embedded[:len(X_bg_maj)]
    emb_min = X_embedded[len(X_bg_maj):]

    # Build P3_F6_data.csv table
    f6_data_rows = []
    cluster_centroids_tsne = []
    cluster_drift_tsne = []

    for i, clust in enumerate(clusters):
        pts_emb = emb_min[clust]
        centroid = np.mean(pts_emb, axis=0)
        cluster_centroids_tsne.append(centroid)
        
        # Estimate drift in 2-D embedding space (new - old)
        w_arr = np.array(cluster_windows[i])
        if np.any(w_arr >= 2) and np.any(w_arr < 2):
            v_2d = np.mean(pts_emb[w_arr >= 2], axis=0) - np.mean(pts_emb[w_arr < 2], axis=0)
            nv = np.linalg.norm(v_2d)
            v_2d = (v_2d / nv) * 3.5 if nv > 1e-4 else np.zeros(2)
        else:
            v_2d = np.zeros(2)
        cluster_drift_tsne.append(v_2d)

        f6_data_rows.append({
            'cluster_id': i + 1,
            'n_samples_cluster': len(clust),
            'quota_allocated': int(quotas[i]),
            'quota_share': round(float(weights[i]), 5),
            'mean_window': round(float(np.mean(cluster_windows[i])), 2),
            'tsne_centroid_x': round(float(centroid[0]), 3),
            'tsne_centroid_y': round(float(centroid[1]), 3),
            'tsne_drift_dx': round(float(v_2d[0]), 3),
            'tsne_drift_dy': round(float(v_2d[1]), 3)
        })

    df_f6 = pd.DataFrame(f6_data_rows)
    df_f6.to_csv('results_stream/P3_F6_data.csv', index=False)
    print("Saved results_stream/P3_F6_data.csv")

    # Plotting Figure P3_F6
    print("Plotting vector PDF figs_stream/P3_F6_creditcard_clusters.pdf...")
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # 1. Background Majority in light grey
    ax.scatter(emb_bg_maj[:, 0], emb_bg_maj[:, 1], c='#cbd5e1', s=12, alpha=0.35, label='Majority Non-Fraud (Context)', edgecolors='none')

    # 2. Minority points colored by time window (K=4)
    colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444'] # W1 (oldest) to W4 (newest)
    window_labels = ['Window 1 (Oldest, Days 0-0.5)', 'Window 2 (Days 0.5-1.0)', 'Window 3 (Days 1.0-1.5)', 'Window 4 (Newest, Days 1.5-2.0)']
    
    for w_i in range(4):
        mask = (windows == w_i)
        ax.scatter(emb_min[mask, 0], emb_min[mask, 1], c=colors[w_i], s=38, alpha=0.9,
                   label=f'Minority Fraud: {window_labels[w_i]}', edgecolors='#0f172a', linewidths=0.6, zorder=4)

    # 3. Cluster outlines and drift arrows for top clusters by quota
    top_clusters = np.argsort(-quotas)[:15] # top 15 clusters
    for c_i in top_clusters:
        pts = emb_min[clusters[c_i]]
        centroid = cluster_centroids_tsne[c_i]
        q_val = quotas[c_i]
        
        # Convex hull if >= 3 points
        if len(pts) >= 3:
            try:
                hull = ConvexHull(pts)
                hull_pts = np.vstack([pts[hull.vertices], pts[hull.vertices[0]]])
                ax.plot(hull_pts[:, 0], hull_pts[:, 1], color='#475569', linestyle='--', linewidth=1.1, alpha=0.7, zorder=3)
            except Exception:
                pass

        # Quota annotation tag
        ax.annotate(f"Quota: {q_val:,}", xy=centroid, xytext=(centroid[0] + 1.2, centroid[1] + 1.2),
                    fontsize=7.5, fontweight='bold', color='#0f172a',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#fef08a', edgecolor='#b45309', alpha=0.85, linewidth=0.8),
                    arrowprops=dict(arrowstyle='->', color='#b45309', lw=0.7), zorder=6)

        # Drift arrow
        v = cluster_drift_tsne[c_i]
        if np.linalg.norm(v) > 0.5:
            ax.annotate('', xy=(centroid[0] + v[0], centroid[1] + v[1]), xytext=(centroid[0], centroid[1]),
                        arrowprops=dict(arrowstyle='->,head_width=0.4,head_length=0.6', color='#dc2626', lw=1.8), zorder=5)

    ax.set_title("Credit Card Fraud Minority Clusters: Temporal Windows, Drift Vectors, and Quota Allocations",
                 fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=10)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=10)
    ax.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    pdf_path = 'figs_stream/P3_F6_creditcard_clusters.pdf'
    png_path = 'figs_stream/P3_F6_creditcard_clusters.png'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    fig.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path} successfully!")

if __name__ == '__main__':
    main()

