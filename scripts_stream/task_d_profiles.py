import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

os.makedirs("results_stream", exist_ok=True)
os.makedirs("figs_stream", exist_ok=True)

# Datasets definition
datasets_info = [
    {
        "name": "creditcard",
        "file": "datasets/creditcard.parquet",
        "source": "ULB Machine Learning Group (Kaggle)",
        "category": "A",
        "type": "Natural",
        "time_col": "Time",
        "time_unit": "seconds",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "Chronological credit card transactions"
    },
    {
        "name": "electricity_ir20",
        "file": "datasets_stream/electricity_ir20.parquet",
        "source": "OpenML (ID 151, Harries 1999)",
        "category": "B",
        "type": "Derived",
        "time_col": "period",
        "time_unit": "half-hour intervals",
        "target_col": "class",
        "minority_val": "UP",
        "majority_val": "DOWN",
        "note": "Subsampled UP class to achieve IR ~ 20"
    },
    {
        "name": "electricity_ir100",
        "file": "datasets_stream/electricity_ir100.parquet",
        "source": "OpenML (ID 151, Harries 1999)",
        "category": "B",
        "type": "Derived",
        "time_col": "period",
        "time_unit": "half-hour intervals",
        "target_col": "class",
        "minority_val": "UP",
        "majority_val": "DOWN",
        "note": "Subsampled UP class to achieve IR ~ 100"
    },
    {
        "name": "airlines_ir20",
        "file": "datasets_stream/airlines_ir20.parquet",
        "source": "OpenML (ID 1167, Ikonomovska)",
        "category": "B",
        "type": "Derived",
        "time_col": "Time",
        "time_unit": "flight departure minutes",
        "target_col": "Delay",
        "minority_val": "1",
        "majority_val": "0",
        "note": "Subsampled Delay=1 to achieve IR ~ 20"
    },
    {
        "name": "airlines_ir100",
        "file": "datasets_stream/airlines_ir100.parquet",
        "source": "OpenML (ID 1167, Ikonomovska)",
        "category": "B",
        "type": "Derived",
        "time_col": "Time",
        "time_unit": "flight departure minutes",
        "target_col": "Delay",
        "minority_val": "1",
        "majority_val": "0",
        "note": "Subsampled Delay=1 to achieve IR ~ 100"
    },
    {
        "name": "covertype_binary",
        "file": "datasets_stream/covertype_binary.parquet",
        "source": "OpenML (ID 159, Blackard & Denis)",
        "category": "C",
        "type": "Natural",
        "time_col": "row_index",
        "time_unit": "spatial cell sequence",
        "target_col": "Binary_Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "Class 5 (Aspen, 9493 rows, 1.63%) as minority"
    },
    {
        "name": "poker_binary",
        "file": "datasets_stream/poker_binary.parquet",
        "source": "OpenML (pokerhand)",
        "category": "C",
        "type": "Natural",
        "time_col": "row_index",
        "time_unit": "deal order sequence",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "Class 3 (Flush, 17541 rows, 2.11%) as minority"
    },
    {
        "name": "kddcup99_binary",
        "file": "datasets_stream/kddcup99_binary.parquet",
        "source": "scikit-learn / UCI KDDCup99 10%",
        "category": "B",
        "type": "Natural",
        "time_col": "row_index",
        "time_unit": "connection event sequence",
        "target_col": "Binary_Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "neptune attack (107201 rows) as minority"
    },
    {
        "name": "synth_sea_p5",
        "file": "datasets_stream/synth_sea_p5.parquet",
        "source": "river.datasets.synth.SEA",
        "category": "D",
        "type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "SEA concept drift at 5k, 10k, 15k steps, minority ~5%"
    },
    {
        "name": "synth_sea_p1",
        "file": "datasets_stream/synth_sea_p1.parquet",
        "source": "river.datasets.synth.SEA",
        "category": "D",
        "type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "SEA concept drift at 5k, 10k, 15k steps, minority ~1%"
    },
    {
        "name": "synth_hyperplane_p5",
        "file": "datasets_stream/synth_hyperplane_p5.parquet",
        "source": "river.datasets.synth.Hyperplane",
        "category": "D",
        "type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "Hyperplane rotating concept drift, minority ~5%"
    },
    {
        "name": "synth_random_rbf_p5",
        "file": "datasets_stream/synth_random_rbf_p5.parquet",
        "source": "river.datasets.synth.RandomRBFDrift",
        "category": "D",
        "type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "majority_val": 0,
        "note": "RandomRBF drifting centroids, minority ~5%"
    }
]

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

table_d1_rows = []
table_d2_rows = []

for meta in datasets_info:
    name = meta["name"]
    print(f"\nProcessing {name}...")
    df = pd.read_parquet(meta["file"])
    
    # Standardize target
    y_raw = df[meta["target_col"]].values
    y = np.where(y_raw == meta["minority_val"], 1, 0)
    
    # Feature columns (exclude target and auxiliary row_index)
    feat_cols = [c for c in df.columns if c != meta["target_col"] and c != "row_index"]
    X_df = df[feat_cols].copy()
    
    # Convert any categorical features to numeric factorized
    for c in X_df.columns:
        if isinstance(X_df[c].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(X_df[c]):
            X_df[c] = pd.factorize(X_df[c])[0]

            
    X = X_df.values.astype(np.float64)
    
    n_rows, n_feats = X.shape
    n_maj = int(np.sum(y == 0))
    n_min = int(np.sum(y == 1))
    ir = round(n_maj / max(n_min, 1), 2)
    
    # Time attribute stats
    time_name = meta["time_col"]
    time_unit = meta["time_unit"]
    if time_name in df.columns:
        t_vals = pd.to_numeric(df[time_name], errors='coerce').fillna(0).values
        t_min, t_max = float(np.min(t_vals)), float(np.max(t_vals))
        t_span = t_max - t_min
    else:
        t_min, t_max = 0.0, float(n_rows - 1)
        t_span = float(n_rows - 1)
        
    # Chronological 75/25 split
    split_idx = int(0.75 * n_rows)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_te, y_te = X[split_idx:], y[split_idx:]
    
    tr_maj, tr_min = int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))
    te_maj, te_min = int(np.sum(y_te == 0)), int(np.sum(y_te == 1))
    tr_ir = round(tr_maj / max(tr_min, 1), 2)
    te_ir = round(te_maj / max(te_min, 1), 2)
    flag_lt_30 = bool(te_min < 30)
    
    # Napierala k=5 neighbour typology on train fold minority
    # k=6 to ignore self
    nn_all = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(X_tr)
    min_indices_tr = np.where(y_tr == 1)[0]
    
    if len(min_indices_tr) > 0:
        _, neighbor_indices = nn_all.kneighbors(X_tr[min_indices_tr])
        # ignore column 0 (self)
        n_maj_neighbors = np.sum(y_tr[neighbor_indices[:, 1:]] == 0, axis=1)
        
        safe_pct = round(100.0 * np.sum(n_maj_neighbors <= 1) / len(min_indices_tr), 2)
        border_pct = round(100.0 * np.sum((n_maj_neighbors >= 2) & (n_maj_neighbors <= 3)) / len(min_indices_tr), 2)
        rare_pct = round(100.0 * np.sum(n_maj_neighbors == 4) / len(min_indices_tr), 2)
        outlier_pct = round(100.0 * np.sum(n_maj_neighbors == 5) / len(min_indices_tr), 2)
    else:
        safe_pct = border_pct = rare_pct = outlier_pct = 0.0
        
    # EC-EBO cluster ratio on train minority
    # If minority > 2000, take random sample of 2000
    rng = np.random.RandomState(42)
    X_maj_tr = X_tr[y_tr == 0]
    X_min_tr = X_tr[y_tr == 1]
    
    if len(X_min_tr) > 2000:
        sample_idx = rng.choice(len(X_min_tr), size=2000, replace=False)
        X_min_sample = X_min_tr[sample_idx]
    else:
        X_min_sample = X_min_tr
        
    n_sample_min = len(X_min_sample)
    if n_sample_min > 0 and len(X_maj_tr) > 0:
        scaler = StandardScaler()
        X_maj_s = scaler.fit_transform(X_maj_tr)
        X_min_s = scaler.transform(X_min_sample)
        
        nn_maj = NearestNeighbors(n_neighbors=7, n_jobs=-1).fit(X_maj_s)
        dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
        d_maj_arr = dist_to_maj[:, 0]
        
        m_clusters = run_ecc_clustering(X_min_s, d_maj_arr)
        cluster_ratio = round(m_clusters / n_sample_min, 4)
    else:
        m_clusters = 0
        cluster_ratio = 0.0
        
    table_d1_rows.append({
        "Dataset": name,
        "Source": meta["source"],
        "Category": meta["category"],
        "Imbalance_Type": meta["type"],
        "Total_Rows": n_rows,
        "Total_Features": n_feats,
        "Majority_Class": f"{meta['majority_val']} ({n_maj})",
        "Minority_Class": f"{meta['minority_val']} ({n_min})",
        "Imbalance_Ratio": ir,
        "Time_Attribute": time_name,
        "Time_Unit": time_unit,
        "Time_Min": t_min,
        "Time_Max": t_max,
        "Time_Span": t_span,
        "Train_Maj": tr_maj,
        "Train_Min": tr_min,
        "Train_IR": tr_ir,
        "Test_Maj": te_maj,
        "Test_Min": te_min,
        "Test_IR": te_ir,
        "Flag_Test_LT_30": flag_lt_30,
        "Safe_Pct": safe_pct,
        "Borderline_Pct": border_pct,
        "Rare_Pct": rare_pct,
        "Outlier_Pct": outlier_pct,
        "EC_EBO_m": m_clusters,
        "EC_EBO_N_min": n_sample_min,
        "Cluster_Ratio": cluster_ratio
    })
    
    # ----------------------------------------------------
    # D2: Drift profile
    # Split full stream into 10 equal blocks
    # ----------------------------------------------------
    block_size = int(np.ceil(n_rows / 10))
    blocks_X = []
    blocks_y = []
    for b in range(10):
        start = b * block_size
        end = min((b + 1) * block_size, n_rows)
        blocks_X.append(X[start:end])
        blocks_y.append(y[start:end])
        
    # (a) Minority class rate per block
    min_rates = [np.mean(by == 1) for by in blocks_y]
    
    # (b) Standardized mean shift relative to block 1
    # mu_1, std_1
    mu_1 = np.mean(blocks_X[0], axis=0)
    std_1 = np.std(blocks_X[0], axis=0) + 1e-8
    
    shift_matrix = np.zeros((10, n_feats))
    for b in range(10):
        mu_b = np.mean(blocks_X[b], axis=0)
        shift_matrix[b] = (mu_b - mu_1) / std_1
        
    # Max shift per feature across all blocks
    max_shift_per_feat = np.max(np.abs(shift_matrix), axis=0)
    top_feat_indices = np.argsort(max_shift_per_feat)[::-1]
    
    max_feature_shift = round(float(np.max(max_shift_per_feat)), 4)
    max_shift_feat_name = feat_cols[top_feat_indices[0]] if n_feats > 0 else "N/A"
    top_3_feat_names = [feat_cols[idx] for idx in top_feat_indices[:min(3, n_feats)]]
    top_5_feat_indices = top_feat_indices[:min(5, n_feats)]
    
    # (c) Classifier degradation
    # Train DecisionTreeClassifier on block 1, test on blocks 1..10
    clf = DecisionTreeClassifier(random_state=42)
    # Check if block 1 has both classes
    if len(np.unique(blocks_y[0])) >= 2:
        clf.fit(blocks_X[0], blocks_y[0])
        b_aucs = []
        for b in range(10):
            if len(np.unique(blocks_y[b])) >= 2:
                # predict proba for class 1
                proba = clf.predict_proba(blocks_X[b])
                if proba.shape[1] == 2:
                    auc = roc_auc_score(blocks_y[b], proba[:, 1])
                else:
                    auc = 0.5
            else:
                auc = 0.5
            b_aucs.append(auc)
    else:
        b_aucs = [0.5] * 10
        
    b1_auc = round(b_aucs[0], 4)
    b10_auc = round(b_aucs[-1], 4)
    delta_auc = round(b10_auc - b1_auc, 4)
    
    # Drift assessment logic
    prior_drift = (max(min_rates) - min(min_rates)) > (0.5 * np.mean(min_rates))
    covariate_drift = max_feature_shift > 0.5
    concept_drift = abs(delta_auc) > 0.15
    
    drift_types = []
    if prior_drift:
        drift_types.append("P(y) prior shift")
    if covariate_drift:
        drift_types.append("P(x) covariate shift")
    if concept_drift:
        drift_types.append("P(y|x) concept drift")
        
    if not drift_types:
        assessment = "Minimal drift: stable prior, covariates, and classifier AUC across 10 blocks."
    else:
        assessment = f"Visible drift detected: {', '.join(drift_types)} (max feat shift={max_feature_shift:.2f}, delta AUC={delta_auc:+.2f})."
        
    table_d2_rows.append({
        "Dataset": name,
        "Min_Minority_Rate": round(float(min(min_rates)), 5),
        "Max_Minority_Rate": round(float(max(min_rates)), 5),
        "Mean_Minority_Rate": round(float(np.mean(min_rates)), 5),
        "Max_Feature_Shift": max_feature_shift,
        "Max_Shift_Feature": max_shift_feat_name,
        "Top_3_Shifting_Features": "; ".join(top_3_feat_names),
        "Block1_AUC": b1_auc,
        "Block10_AUC": b10_auc,
        "Delta_AUC": delta_auc,
        "Visible_Drift_Assessment": assessment
    })
    
    # ----------------------------------------------------
    # Plot 3-panel figure
    # ----------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    blocks_idx = np.arange(1, 11)
    
    # Panel 1: Minority Rate
    axes[0].plot(blocks_idx, min_rates, marker='o', color='#1f77b4', lw=2)
    axes[0].set_title(f"{name}: Minority Rate P(y=1)", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[0].set_ylabel("Minority Proportion", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(blocks_idx)
    
    # Panel 2: Standardized Mean Shift (Top 5 features)
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b']
    for idx_f, feat_idx in enumerate(top_5_feat_indices):
        feat_name = feat_cols[feat_idx]
        axes[1].plot(blocks_idx, shift_matrix[:, feat_idx], marker='s', label=feat_name,
                     color=colors[idx_f % len(colors)], lw=1.8, alpha=0.85)
    axes[1].set_title(f"{name}: P(x) Standardized Feature Shift", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[1].set_ylabel(r"Shift $(\mu_b - \mu_1) / \sigma_1$", fontsize=10)
    axes[1].axhline(0, color='gray', linestyle=':', lw=1)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(blocks_idx)
    axes[1].legend(fontsize=7, loc='best')
    
    # Panel 3: Decision Tree AUC degradation
    axes[2].plot(blocks_idx, b_aucs, marker='^', color='#e377c2', lw=2)
    axes[2].axhline(b1_auc, color='gray', linestyle=':', label=f'Block 1 baseline ({b1_auc:.2f})')
    axes[2].set_title(f"{name}: Classifier AUC P(y|x) Degradation", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[2].set_ylabel("AUC (Block-1 CART)", fontsize=10)
    axes[2].set_ylim(0.4, 1.05)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xticks(blocks_idx)
    axes[2].legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    png_path = os.path.join("figs_stream", f"D_drift_{name}.png")
    pdf_path = os.path.join("figs_stream", f"D_drift_{name}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved drift figures: {png_path} and {pdf_path}")

# Save CSVs
df_d1 = pd.DataFrame(table_d1_rows)
out_d1_path = os.path.join("results_stream", "D_datasets.csv")
df_d1.to_csv(out_d1_path, index=False)
print(f"\nSaved Table D1 to {out_d1_path}")

df_d2 = pd.DataFrame(table_d2_rows)
out_d2_path = os.path.join("results_stream", "D_drift.csv")
df_d2.to_csv(out_d2_path, index=False)
print(f"Saved Table D2 to {out_d2_path}")
print("Task D execution completed.")
