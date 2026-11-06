"""
Task G: Fixed Drift Profiles (D2b)
==================================
G1: Remove timestamp columns from feature ranking and classifier input.
    Document input features and earlier usage.
G2: Out-of-sample block-1 reference model (first half train, second half test)
    evaluated on blocks 2..10 for DecisionTree and LogisticRegression. Report N_pos, N_neg, NaN for 1-class blocks.
G3: Pooled standard deviation with 1e-6 floor. Flag features with std < 1e-6 in block 1.
    Top 5 shifting features and their shifts.
G4: Fact-based drift verdict based strictly on G2/G3 metrics.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

os.makedirs("results_stream", exist_ok=True)
os.makedirs("figs_stream", exist_ok=True)

# Datasets specifications for Task G
datasets_g = [
    {
        "name": "creditcard",
        "file": "datasets/creditcard.parquet",
        "time_col": "Time",
        "timestamp_used_earlier": "YES (Included in D2 feature ranking and classifier)",
        "used_in_task_b": "NO (Task B evaluated KEEL datasets only)",
        "chronology_proof": "Monotone numeric timestamp (seconds elapsed from start of dataset: 0 to 172792)",
        "category": "A",
        "target_col": "Class",
        "minority_val": 1
    },
    {
        "name": "electricity_ir20",
        "file": "datasets_stream/electricity_ir20.parquet",
        "time_col": "date",
        "aux_time_cols": ["period"],
        "timestamp_used_earlier": "YES (period was included as numeric feature)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented 30-minute operational intervals across NSW electricity market (1996-1998)",
        "category": "A",
        "target_col": "class",
        "minority_val": "UP"
    },
    {
        "name": "electricity_ir100",
        "file": "datasets_stream/electricity_ir100.parquet",
        "time_col": "date",
        "aux_time_cols": ["period"],
        "timestamp_used_earlier": "YES (period was included as numeric feature)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented 30-minute operational intervals across NSW electricity market (1996-1998)",
        "category": "A",
        "target_col": "class",
        "minority_val": "UP"
    },
    {
        "name": "airlines_ir20",
        "file": "datasets_stream/airlines_ir20.parquet",
        "time_col": "Time",
        "timestamp_used_earlier": "YES (Time was included as departure minute feature)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented commercial airline flights ordered chronologically by scheduled departure",
        "category": "B",
        "target_col": "Delay",
        "minority_val": "1"
    },
    {
        "name": "airlines_ir100",
        "file": "datasets_stream/airlines_ir100.parquet",
        "time_col": "Time",
        "timestamp_used_earlier": "YES (Time was included as departure minute feature)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented commercial airline flights ordered chronologically by scheduled departure",
        "category": "B",
        "target_col": "Delay",
        "minority_val": "1"
    },
    {
        "name": "covertype_binary",
        "file": "datasets_stream/covertype_binary.parquet",
        "time_col": "row_index",
        "timestamp_used_earlier": "NO (row_index was excluded)",
        "used_in_task_b": "NO",
        "chronology_proof": "Benchmark spatial file sequence order (30x30m cartographic cells; not strictly temporal)",
        "category": "C",
        "target_col": "Binary_Class",
        "minority_val": 1
    },
    {
        "name": "poker_binary",
        "file": "datasets_stream/poker_binary.parquet",
        "time_col": "row_index",
        "timestamp_used_earlier": "NO (row_index was excluded)",
        "used_in_task_b": "NO",
        "chronology_proof": "Benchmark deal order sequence (hand combinations; non-temporal)",
        "category": "C",
        "target_col": "Class",
        "minority_val": 1
    },
    {
        "name": "kddcup99_binary",
        "file": "datasets_stream/kddcup99_binary.parquet",
        "time_col": "row_index",
        "timestamp_used_earlier": "NO (row_index was excluded)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented network connection events sequence from 9-week raw tcpdump data",
        "category": "B",
        "target_col": "Binary_Class",
        "minority_val": 1
    },
    {
        "name": "kddcup99_rare",
        "file": "datasets_stream/kddcup99_rare.parquet",
        "time_col": "row_index",
        "timestamp_used_earlier": "NEW DATASET (Built for Task H)",
        "used_in_task_b": "NO",
        "chronology_proof": "Documented network connection events sequence from 9-week raw tcpdump data",
        "category": "B",
        "target_col": "Binary_Class",
        "minority_val": 1
    },
    {
        "name": "synth_sea_p5",
        "file": "datasets_stream/synth_sea_p5.parquet",
        "time_col": "step",
        "aux_time_cols": ["Time_Step"],
        "timestamp_used_earlier": "YES (Time_Step was created and included in D2 features)",
        "used_in_task_b": "NO",
        "chronology_proof": "Synthetic generator time step with abrupt drift points at steps 5000, 10000, 15000",
        "category": "D",
        "target_col": "Class",
        "minority_val": 1
    },
    {
        "name": "synth_hyperplane_p5",
        "file": "datasets_stream/synth_hyperplane_p5.parquet",
        "time_col": "step",
        "aux_time_cols": ["Time_Step"],
        "timestamp_used_earlier": "YES (Time_Step was created and included in D2 features)",
        "used_in_task_b": "NO",
        "chronology_proof": "Synthetic rotating hyperplane stream generation step",
        "category": "D",
        "target_col": "Class",
        "minority_val": 1
    },
    {
        "name": "synth_random_rbf_p5",
        "file": "datasets_stream/synth_random_rbf_p5.parquet",
        "time_col": "step",
        "aux_time_cols": ["Time_Step"],
        "timestamp_used_earlier": "YES (Time_Step was created and included in D2 features)",
        "used_in_task_b": "NO",
        "chronology_proof": "Synthetic moving RBF centroids stream generation step",
        "category": "D",
        "target_col": "Class",
        "minority_val": 1
    },
    {
        "name": "ieee_cis",
        "file": "e:/Research/ds/common_cleaned_ieee_cis.parquet",
        "time_col": "TransactionDT",
        "timestamp_used_earlier": "NEW DATASET (Acquired for Task H)",
        "used_in_task_b": "NO",
        "chronology_proof": "Monotone timedelta in seconds (86,400 to 15,811,131) from reference date",
        "category": "A",
        "target_col": "isFraud",
        "minority_val": 1
    }
]

feature_audit_rows = []
drift_rows = []

for meta in datasets_g:
    dname = meta["name"]
    print(f"\nProcessing Task G for {dname}...")
    
    if not os.path.exists(meta["file"]):
        print(f"  File not found: {meta['file']}")
        continue
        
    df = pd.read_parquet(meta["file"])
    
    # Identify target
    y_raw = df[meta["target_col"]].values
    y = np.where(y_raw == meta["minority_val"], 1, 0)
    
    # Identify all temporal columns to remove
    temporal_cols = set([meta["time_col"]])
    if "aux_time_cols" in meta:
        temporal_cols.update(meta["aux_time_cols"])
    temporal_cols.update(["row_index", "Time_Step", "step", "TransactionDT", "TransactionID", "date"])
    
    # Candidate feature columns
    feat_cols = [c for c in df.columns if c != meta["target_col"] and c not in temporal_cols]
    
    # Audit row
    feature_audit_rows.append({
        "Dataset": dname,
        "Timestamp_Column": meta["time_col"],
        "Was_Timestamp_Input_Earlier": meta["timestamp_used_earlier"],
        "Used_in_Task_B": meta["used_in_task_b"],
        "Chronology_Verification": meta["chronology_proof"],
        "Category": meta["category"],
        "Total_Input_Features": len(feat_cols),
        "Input_Features_List": ", ".join(feat_cols[:15]) + ("..." if len(feat_cols) > 15 else "")
    })
    
    # Prepare X matrix
    X_df = df[feat_cols].copy()
    for c in X_df.columns:
        if isinstance(X_df[c].dtype, pd.CategoricalDtype) or not pd.api.types.is_numeric_dtype(X_df[c]):
            X_df[c] = pd.factorize(X_df[c])[0]
            
    X = X_df.values.astype(np.float64)
    n_rows, n_feats = X.shape
    
    # Split into 10 equal blocks
    block_size = int(np.ceil(n_rows / 10))
    blocks_X = []
    blocks_y = []
    for b in range(10):
        start = b * block_size
        end = min((b + 1) * block_size, n_rows)
        blocks_X.append(X[start:end])
        blocks_y.append(y[start:end])
        
    # G2: Model training on FIRST HALF of block 1
    b1_len = len(blocks_X[0])
    b1_half = b1_len // 2
    X_tr_b1 = blocks_X[0][:b1_half]
    y_tr_b1 = blocks_y[0][:b1_half]
    
    # Reference test sets: second half of block 1, then blocks 2..10
    test_blocks_X = [blocks_X[0][b1_half:]] + blocks_X[1:]
    test_blocks_y = [blocks_y[0][b1_half:]] + blocks_y[1:]
    
    # Check if first half of block 1 has both classes
    classes_tr_b1 = np.unique(y_tr_b1)
    can_train = len(classes_tr_b1) >= 2
    
    dt_clf = DecisionTreeClassifier(random_state=42)
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    
    if can_train:
        dt_clf.fit(X_tr_b1, y_tr_b1)
        try:
            lr_clf.fit(X_tr_b1, y_tr_b1)
            lr_fitted = True
        except Exception:
            lr_fitted = False
    else:
        lr_fitted = False
        
    block_n_pos = []
    block_n_neg = []
    dt_aucs = []
    lr_aucs = []
    
    for b in range(10):
        by = test_blocks_y[b]
        bX = test_blocks_X[b]
        n_pos = int(np.sum(by == 1))
        n_neg = int(np.sum(by == 0))
        block_n_pos.append(n_pos)
        block_n_neg.append(n_neg)
        
        if n_pos == 0 or n_neg == 0 or not can_train:
            dt_aucs.append(np.nan)
            lr_aucs.append(np.nan)
        else:
            # DT AUC
            try:
                p_dt = dt_clf.predict_proba(bX)[:, 1]
                dt_auc = roc_auc_score(by, p_dt)
            except Exception:
                dt_auc = np.nan
            dt_aucs.append(dt_auc)
            
            # LR AUC
            if lr_fitted:
                try:
                    p_lr = lr_clf.predict_proba(bX)[:, 1]
                    lr_auc = roc_auc_score(by, p_lr)
                except Exception:
                    lr_auc = np.nan
            else:
                lr_auc = np.nan
            lr_aucs.append(lr_auc)
            
    # G3: Standardized Mean Shift with pooled std (floor 1e-6)
    # Pooled std over all 10 blocks:
    std_pooled = np.std(X, axis=0)
    std_pooled = np.maximum(std_pooled, 1e-6)
    
    # Flag features with std < 1e-6 in block 1
    std_b1 = np.std(blocks_X[0], axis=0)
    near_zero_b1_idx = np.where(std_b1 < 1e-6)[0]
    near_zero_b1_names = [feat_cols[idx] for idx in near_zero_b1_idx]
    
    mu_1 = np.mean(blocks_X[0], axis=0)
    shift_matrix = np.zeros((10, n_feats))
    for b in range(10):
        mu_b = np.mean(blocks_X[b], axis=0)
        shift_matrix[b] = (mu_b - mu_1) / std_pooled
        
    max_shift_per_feat = np.max(np.abs(shift_matrix[1:]), axis=0) if len(shift_matrix) > 1 else np.zeros(n_feats)
    top_indices = np.argsort(max_shift_per_feat)[::-1]
    
    top5_names = [feat_cols[idx] for idx in top_indices[:min(5, n_feats)]]
    top5_shifts = [round(float(max_shift_per_feat[idx]), 4) for idx in top_indices[:min(5, n_feats)]]
    top5_str = "; ".join([f"{name}: {shift:.2f}" for name, shift in zip(top5_names, top5_shifts)])
    
    max_overall_shift = round(float(np.max(max_shift_per_feat)), 4) if len(max_shift_per_feat) > 0 else 0.0
    
    # Metrics for summary
    b1_dt_auc = round(float(dt_aucs[0]), 4) if not np.isnan(dt_aucs[0]) else np.nan
    b10_dt_auc = round(float(dt_aucs[-1]), 4) if not np.isnan(dt_aucs[-1]) else np.nan
    delta_dt_auc = round(float(b10_dt_auc - b1_dt_auc), 4) if (not np.isnan(b1_dt_auc) and not np.isnan(b10_dt_auc)) else np.nan
    
    b1_lr_auc = round(float(lr_aucs[0]), 4) if not np.isnan(lr_aucs[0]) else np.nan
    b10_lr_auc = round(float(lr_aucs[-1]), 4) if not np.isnan(lr_aucs[-1]) else np.nan
    delta_lr_auc = round(float(b10_lr_auc - b1_lr_auc), 4) if (not np.isnan(b1_lr_auc) and not np.isnan(b10_lr_auc)) else np.nan
    
    # G4: Fact-based verdict derived strictly from G2 and G3 numbers
    verdict_parts = []
    if max_overall_shift > 0.5:
        verdict_parts.append(f"P(x) covariate shift detected (max pooled shift {max_overall_shift:.2f} on {top5_names[0]})")
    else:
        verdict_parts.append(f"Stable covariates (max pooled shift {max_overall_shift:.2f})")
        
    if not np.isnan(delta_dt_auc):
        if delta_dt_auc < -0.15:
            verdict_parts.append(f"Severe concept drift / P(y|x) degradation (DT AUC {b1_dt_auc:.2f} -> {b10_dt_auc:.2f}, delta {delta_dt_auc:+.2f})")
        elif delta_dt_auc > 0.15:
            verdict_parts.append(f"Expanding separation / concept shift (DT AUC {b1_dt_auc:.2f} -> {b10_dt_auc:.2f}, delta {delta_dt_auc:+.2f})")
        else:
            verdict_parts.append(f"Moderate concept stability (DT AUC {b1_dt_auc:.2f} -> {b10_dt_auc:.2f}, delta {delta_dt_auc:+.2f})")
    else:
        verdict_parts.append("One-class block encountered; AUC undefined")
        
    verdict = "; ".join(verdict_parts) + "."
    
    drift_rows.append({
        "Dataset": dname,
        "N_Features_Used": n_feats,
        "Timestamp_Excluded": meta["time_col"],
        "Block1_OutSample_Npos": block_n_pos[0],
        "Block1_OutSample_Nneg": block_n_neg[0],
        "Block10_Npos": block_n_pos[-1],
        "Block10_Nneg": block_n_neg[-1],
        "DT_AUC_Block1": b1_dt_auc,
        "DT_AUC_Block10": b10_dt_auc,
        "Delta_DT_AUC": delta_dt_auc,
        "LR_AUC_Block1": b1_lr_auc,
        "LR_AUC_Block10": b10_lr_auc,
        "Delta_LR_AUC": delta_lr_auc,
        "Max_Pooled_Feature_Shift": max_overall_shift,
        "Top_Shifting_Feature": top5_names[0] if len(top5_names) > 0 else "None",
        "Top5_Features_And_Shifts": top5_str,
        "NearZero_Std_Block1_Features": ", ".join(near_zero_b1_names[:5]) + ("..." if len(near_zero_b1_names) > 5 else ""),
        "Visible_Drift_Verdict_G4": verdict
    })
    
    # Plot updated 3-panel figure D2b
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=300)
    blocks_idx = np.arange(1, 11)
    
    # Panel 1: Minority Rate across blocks
    min_rates = [np.mean(by == 1) for by in blocks_y]
    axes[0].plot(blocks_idx, min_rates, marker='o', color='#2b6cb0', lw=2)
    axes[0].set_title(f"{dname}: P(y=1) Minority Rate", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[0].set_ylabel("Minority Proportion", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].set_xticks(blocks_idx)
    
    # Panel 2: Standardized Feature Shift (Top 5 Non-Temporal Features)
    colors = ['#c53030', '#2f855a', '#dd6b20', '#6b46c1', '#805ad5']
    for idx_f, feat_idx in enumerate(top_indices[:min(5, n_feats)]):
        feat_name = feat_cols[feat_idx]
        axes[1].plot(blocks_idx, shift_matrix[:, feat_idx], marker='s', label=feat_name,
                     color=colors[idx_f % len(colors)], lw=1.8, alpha=0.85)
    axes[1].set_title(f"{dname}: Non-Temporal P(x) Feature Shift", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[1].set_ylabel(r"Shift $(\mu_b - \mu_1) / \sigma_{pooled}$", fontsize=10)
    axes[1].axhline(0, color='gray', linestyle=':', lw=1)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].set_xticks(blocks_idx)
    axes[1].legend(fontsize=7, loc='best')
    
    # Panel 3: Out-of-Sample Classifier Degradation (DT vs LR)
    axes[2].plot(blocks_idx, dt_aucs, marker='^', color='#d53f8c', lw=2, label='DecisionTree')
    axes[2].plot(blocks_idx, lr_aucs, marker='v', color='#3182ce', lw=2, linestyle='--', label='LogisticRegression')
    axes[2].set_title(f"{dname}: Out-of-Sample AUC Degradation", fontsize=11, fontweight='bold')
    axes[2].set_xlabel("Stream Block (1..10)", fontsize=10)
    axes[2].set_ylabel("AUC (Ref Model on B1.1)", fontsize=10)
    axes[2].set_ylim(0.4, 1.05)
    axes[2].grid(True, linestyle='--', alpha=0.5)
    axes[2].set_xticks(blocks_idx)
    axes[2].legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    png_path = os.path.join("figs_stream", f"D2b_drift_{dname}.png")
    pdf_path = os.path.join("figs_stream", f"D2b_drift_{dname}.pdf")
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.close()
    print(f"  Saved {png_path} and {pdf_path}")

df_audit = pd.DataFrame(feature_audit_rows)
out_audit = "results_stream/D2b_feature_audit.csv"
df_audit.to_csv(out_audit, index=False)
print(f"\nSaved {out_audit}")

df_d2b = pd.DataFrame(drift_rows)
out_d2b = "results_stream/D2b_drift.csv"
df_d2b.to_csv(out_d2b, index=False)
print(f"Saved {out_d2b}")
print("Task G execution completed successfully!")

