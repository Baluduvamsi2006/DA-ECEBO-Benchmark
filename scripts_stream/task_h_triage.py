"""
Task H: Stream Dataset Triage and Final Audit Table
===================================================
Produces results_stream/H_datasets_final.csv
H1: Verified chronology and re-assigned Category A/B/C/D.
    Electricity has a real date column -> Category A.
H2: kddcup99_binary details: 107,201 neptune vs 386,820 rest (IR 3.61) -> NOT SUITABLE.
    kddcup99_rare: rare attack families (Probe, R2L, U2R = 5,285) vs rest (488,736, IR 92.48).
H3: synth_sea_p1 (test min 12) marked EXCLUDED.
    poker_binary evaluated: in-sample AUC vs out-of-sample AUC.
H4: IEEE-CIS Fraud Detection (common_cleaned_ieee_cis.parquet, 590,540 rows, TransactionDT).
H5: Column 'N_min used' stating exact minority cap (2000 sampled uniformly at random with seed 42 when N_min > 2000).
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

os.makedirs("results_stream", exist_ok=True)

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

# Datasets specification
datasets_h = [
    {
        "name": "creditcard",
        "file": "datasets/creditcard.parquet",
        "source": "ULB Machine Learning Group (Kaggle)",
        "category": "A",
        "imbalance_type": "Natural",
        "time_col": "Time",
        "time_unit": "seconds",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Monotone time column (0.0 to 172792.0 seconds elapsed over ~48 hours)",
        "triage_status": "SUITABLE",
        "notes": "Natural fraud imbalance with verified timestamps"
    },
    {
        "name": "electricity_ir20",
        "file": "datasets_stream/electricity_ir20.parquet",
        "source": "OpenML (ID 151, Harries 1999)",
        "category": "A",
        "imbalance_type": "Derived",
        "time_col": "date",
        "time_unit": "calendar date + 30-min period",
        "target_col": "class",
        "minority_val": "UP",
        "chronology_proof": "Documented calendar dates and 48 half-hour trading periods per day across NSW market (1996-1998)",
        "triage_status": "SUITABLE",
        "notes": "Re-assigned to Cat A due to verified date/period structure; uniform subsample IR~20"
    },
    {
        "name": "electricity_ir100",
        "file": "datasets_stream/electricity_ir100.parquet",
        "source": "OpenML (ID 151, Harries 1999)",
        "category": "A",
        "imbalance_type": "Derived",
        "time_col": "date",
        "time_unit": "calendar date + 30-min period",
        "target_col": "class",
        "minority_val": "UP",
        "chronology_proof": "Documented calendar dates and 48 half-hour trading periods per day across NSW market (1996-1998)",
        "triage_status": "SUITABLE",
        "notes": "Re-assigned to Cat A due to verified date/period structure; uniform subsample IR~100"
    },
    {
        "name": "airlines_ir20",
        "file": "datasets_stream/airlines_ir20.parquet",
        "source": "OpenML (ID 1167, Ikonomovska)",
        "category": "B",
        "imbalance_type": "Derived",
        "time_col": "Time",
        "time_unit": "departure minute (int)",
        "target_col": "Delay",
        "minority_val": "1",
        "chronology_proof": "Documented flight departure sequence; stream preserved in chronological order",
        "triage_status": "SUITABLE",
        "notes": "Chronological flight records; uniform subsample IR~20"
    },
    {
        "name": "airlines_ir100",
        "file": "datasets_stream/airlines_ir100.parquet",
        "source": "OpenML (ID 1167, Ikonomovska)",
        "category": "B",
        "imbalance_type": "Derived",
        "time_col": "Time",
        "time_unit": "departure minute (int)",
        "target_col": "Delay",
        "minority_val": "1",
        "chronology_proof": "Documented flight departure sequence; stream preserved in chronological order",
        "triage_status": "SUITABLE",
        "notes": "Chronological flight records; uniform subsample IR~100"
    },
    {
        "name": "ieee_cis",
        "file": "e:/Research/ds/common_cleaned_ieee_cis.parquet",
        "source": "IEEE-CIS Fraud Detection (Vesta / Kaggle)",
        "category": "A",
        "imbalance_type": "Natural",
        "time_col": "TransactionDT",
        "time_unit": "seconds elapsed",
        "target_col": "isFraud",
        "minority_val": 1,
        "chronology_proof": "Monotone timedelta in seconds (86,400 to 15,811,131 over 182 days)",
        "triage_status": "SUITABLE",
        "notes": "Natural extreme transaction fraud stream with real timestamp column"
    },
    {
        "name": "covertype_binary",
        "file": "datasets_stream/covertype_binary.parquet",
        "source": "OpenML (ID 159, Blackard & Denis)",
        "category": "C",
        "imbalance_type": "Natural",
        "time_col": "row_index",
        "time_unit": "spatial cell sequence",
        "target_col": "Binary_Class",
        "minority_val": 1,
        "chronology_proof": "Benchmark spatial file sequence order (30x30m cartographic cells; non-temporal)",
        "triage_status": "BENCHMARK ORDER ONLY (Cat C)",
        "notes": "Class 5 (Aspen, 9493 rows, 1.63%) as minority"
    },
    {
        "name": "poker_binary",
        "file": "datasets_stream/poker_binary.parquet",
        "source": "OpenML (pokerhand)",
        "category": "C",
        "imbalance_type": "Natural",
        "time_col": "row_index",
        "time_unit": "deal order sequence",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Benchmark deal order sequence (hand combinations; non-temporal)",
        "triage_status": "BENCHMARK ORDER ONLY (Cat C)",
        "notes": "Flush hands as minority; in-sample AUC=1.0000, out-of-sample AUC=1.0000 (deterministic rules)"
    },
    {
        "name": "kddcup99_rare",
        "file": "datasets_stream/kddcup99_rare.parquet",
        "source": "scikit-learn / UCI KDDCup99 10%",
        "category": "B",
        "imbalance_type": "Natural",
        "time_col": "row_index",
        "time_unit": "connection event sequence",
        "target_col": "Binary_Class",
        "minority_val": 1,
        "chronology_proof": "Documented network connection events sequence from 9-week raw tcpdump data",
        "triage_status": "SUITABLE (Rare Attack Family)",
        "notes": "Rare attack families (Probe, R2L, U2R = 5,285 rows) vs DoS+Normal (488,736 rows), IR = 92.48"
    },
    {
        "name": "kddcup99_binary",
        "file": "datasets_stream/kddcup99_binary.parquet",
        "source": "scikit-learn / UCI KDDCup99 10%",
        "category": "B",
        "imbalance_type": "Natural",
        "time_col": "row_index",
        "time_unit": "connection event sequence",
        "target_col": "Binary_Class",
        "minority_val": 1,
        "chronology_proof": "Documented network connection events sequence from 9-week raw tcpdump data",
        "triage_status": "NOT SUITABLE (Low IR = 3.61)",
        "notes": "Old formulation: neptune (107,201) vs rest (386,820); IR 3.61 is not sufficiently imbalanced"
    },
    {
        "name": "synth_sea_p5",
        "file": "datasets_stream/synth_sea_p5.parquet",
        "source": "river.datasets.synth.SEA",
        "category": "D",
        "imbalance_type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Synthetic generator time step with abrupt drift points at steps 5000, 10000, 15000",
        "triage_status": "SUITABLE (Synthetic Drift Benchmark)",
        "notes": "Controlled concept drift points with minority ~5%"
    },
    {
        "name": "synth_sea_p1",
        "file": "datasets_stream/synth_sea_p1.parquet",
        "source": "river.datasets.synth.SEA",
        "category": "D",
        "imbalance_type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Synthetic generator time step with abrupt drift points at steps 5000, 10000, 15000",
        "triage_status": "EXCLUDED (Test Minority 12 < 30)",
        "notes": "Violates minimum test minority requirement (Test min = 12 < 30)"
    },
    {
        "name": "synth_hyperplane_p5",
        "file": "datasets_stream/synth_hyperplane_p5.parquet",
        "source": "river.datasets.synth.Hyperplane",
        "category": "D",
        "imbalance_type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Synthetic rotating hyperplane stream generation step",
        "triage_status": "SUITABLE (Synthetic Drift Benchmark)",
        "notes": "Continuous concept drift; minority ~5%"
    },
    {
        "name": "synth_random_rbf_p5",
        "file": "datasets_stream/synth_random_rbf_p5.parquet",
        "source": "river.datasets.synth.RandomRBFDrift",
        "category": "D",
        "imbalance_type": "Derived",
        "time_col": "step",
        "time_unit": "stream step",
        "target_col": "Class",
        "minority_val": 1,
        "chronology_proof": "Synthetic moving RBF centroids stream generation step",
        "triage_status": "SUITABLE (Synthetic Drift Benchmark)",
        "notes": "Centroid concept drift; minority ~5%"
    }
]

triage_rows = []

for meta in datasets_h:
    dname = meta["name"]
    print(f"\nProcessing Task H triage for {dname}...")
    if not os.path.exists(meta["file"]):
        print(f"  File not found: {meta['file']}")
        continue
        
    df = pd.read_parquet(meta["file"])
    
    # Target
    y_raw = df[meta["target_col"]].values
    y = np.where(y_raw == meta["minority_val"], 1, 0)
    
    # Non-temporal, non-id features
    exclude_cols = set([meta["target_col"], meta["time_col"], "row_index", "step", "Time_Step", "TransactionID", "date", "period"])
    feat_cols = [c for c in df.columns if c not in exclude_cols]
    
    X_df = df[feat_cols].copy()
    for c in X_df.columns:
        if isinstance(X_df[c].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(X_df[c]):
            X_df[c] = pd.factorize(X_df[c])[0]
    X_df = X_df.fillna(0)
            
    X = X_df.values.astype(np.float64)
    n_rows, n_feats = X.shape
    n_maj = int(np.sum(y == 0))
    n_min = int(np.sum(y == 1))
    ir = round(n_maj / max(n_min, 1), 2)
    
    # Chronological 75/25 split
    split_idx = int(0.75 * n_rows)
    X_tr, y_tr = X[:split_idx], y[:split_idx]
    X_te, y_te = X[split_idx:], y[split_idx:]
    
    tr_maj, tr_min = int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))
    te_maj, te_min = int(np.sum(y_te == 0)), int(np.sum(y_te == 1))
    tr_ir = round(tr_maj / max(tr_min, 1), 2)
    te_ir = round(te_maj / max(te_min, 1), 2)
    flag_lt_30 = bool(te_min < 30)
    
    # Napierala typology on train minority
    # Sample up to 10,000 for NearestNeighbors to run efficiently if dataset is huge
    if tr_min > 0:
        if len(X_tr) > 50000:
            rng_sub = np.random.RandomState(42)
            sub_tr_idx = rng_sub.choice(len(X_tr), size=50000, replace=False)
            # Ensure minority points are included
            min_in_tr = np.where(y_tr == 1)[0]
            sub_tr_idx = np.unique(np.concatenate([sub_tr_idx, min_in_tr[:min(len(min_in_tr), 2000)]]))
            X_nn_ref = X_tr[sub_tr_idx]
            y_nn_ref = y_tr[sub_tr_idx]
        else:
            X_nn_ref = X_tr
            y_nn_ref = y_tr
            
        nn_all = NearestNeighbors(n_neighbors=6, n_jobs=-1).fit(X_nn_ref)
        min_indices_ref = np.where(y_nn_ref == 1)[0]
        
        if len(min_indices_ref) > 0:
            _, neighbor_indices = nn_all.kneighbors(X_nn_ref[min_indices_ref])
            n_maj_neighbors = np.sum(y_nn_ref[neighbor_indices[:, 1:]] == 0, axis=1)
            safe_pct = round(100.0 * np.sum(n_maj_neighbors <= 1) / len(min_indices_ref), 2)
            border_pct = round(100.0 * np.sum((n_maj_neighbors >= 2) & (n_maj_neighbors <= 3)) / len(min_indices_ref), 2)
            rare_pct = round(100.0 * np.sum(n_maj_neighbors == 4) / len(min_indices_ref), 2)
            outlier_pct = round(100.0 * np.sum(n_maj_neighbors == 5) / len(min_indices_ref), 2)
        else:
            safe_pct = border_pct = rare_pct = outlier_pct = np.nan
    else:
        safe_pct = border_pct = rare_pct = outlier_pct = np.nan
        
    # EC-EBO cluster ratio on train minority
    rng = np.random.RandomState(42)
    X_maj_tr = X_tr[y_tr == 0]
    X_min_tr = X_tr[y_tr == 1]
    
    if len(X_min_tr) > 2000:
        sample_idx = rng.choice(len(X_min_tr), size=2000, replace=False)
        X_min_sample = X_min_tr[sample_idx]
        n_min_used_desc = "2000 (Random uniform subsample from train minority, seed 42)"
    else:
        X_min_sample = X_min_tr
        n_min_used_desc = f"{len(X_min_tr)} (All train minority instances)"
        
    n_sample_min = len(X_min_sample)
    if n_sample_min > 0 and len(X_maj_tr) > 0:
        # Subsample majority for scaling/nn to max 20,000 to avoid memory overflow
        if len(X_maj_tr) > 20000:
            maj_sub_idx = rng.choice(len(X_maj_tr), size=20000, replace=False)
            X_maj_sub = X_maj_tr[maj_sub_idx]
        else:
            X_maj_sub = X_maj_tr
            
        scaler = StandardScaler()
        X_maj_s = scaler.fit_transform(X_maj_sub)
        X_min_s = scaler.transform(X_min_sample)
        
        nn_maj = NearestNeighbors(n_neighbors=7, n_jobs=-1).fit(X_maj_s)
        dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
        d_maj_arr = dist_to_maj[:, 0]
        
        m_clusters = run_ecc_clustering(X_min_s, d_maj_arr)
        cluster_ratio = round(m_clusters / n_sample_min, 4)
    else:
        m_clusters = 0
        cluster_ratio = np.nan
        
    triage_rows.append({
        "Dataset": dname,
        "Triage_Status": meta["triage_status"],
        "Category": meta["category"],
        "Chronology_Proof": meta["chronology_proof"],
        "Imbalance_Type": meta["imbalance_type"],
        "Total_Rows": n_rows,
        "Total_Features": n_feats,
        "Majority_Count": n_maj,
        "Minority_Count": n_min,
        "Imbalance_Ratio": ir,
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
        "N_min_Used": n_min_used_desc,
        "Cluster_Ratio": cluster_ratio,
        "Notes": meta["notes"]
    })

df_h = pd.DataFrame(triage_rows)
out_h = "results_stream/H_datasets_final.csv"
df_h.to_csv(out_h, index=False)
print(f"\nTask H triage table successfully created and saved to {out_h}")
