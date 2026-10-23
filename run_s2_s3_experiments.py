"""
S2 & S3 Experiments:
S2: Permuted-Timestamp Control (Real vs Permuted vs No-Time/K=1 vs EC-EBO) + Cluster Analysis.
S3: Better Rolling-Origin Evaluation (5 Non-Overlapping Chronological Test Blocks, 10 seeds).
Parallelized with ProcessPoolExecutor for maximum speed.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import ECEBO, DAECEBO, compute_metrics


def _eval_s2_seed(args):
    s, k_opt, lam_opt, X_tr_s, y_tr, t_tr, X_te_s, y_te = args
    records = []
    
    # (a) Real timestamps
    s_a = DAECEBO(k_windows=k_opt, lambd=lam_opt, random_state=s)
    X_a, y_a = s_a.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf_a = DecisionTreeClassifier(random_state=s).fit(X_a, y_a)
    m_a = compute_metrics(y_te, clf_a.predict_proba(X_te_s)[:, 1], clf_a.predict(X_te_s))
    m_a['Variant'] = '(a) Real Timestamps (DA-ECEBO)'
    m_a['Seed'] = s
    records.append(m_a)
    
    # (b) Permuted timestamps
    rng_p = np.random.RandomState(s)
    t_tr_perm = rng_p.permutation(t_tr)
    s_b = DAECEBO(k_windows=k_opt, lambd=lam_opt, random_state=s)
    X_b, y_b = s_b.fit_resample(X_tr_s, y_tr, timestamps=t_tr_perm)
    clf_b = DecisionTreeClassifier(random_state=s).fit(X_b, y_b)
    m_b = compute_metrics(y_te, clf_b.predict_proba(X_te_s)[:, 1], clf_b.predict(X_te_s))
    m_b['Variant'] = '(b) Permuted Timestamps'
    m_b['Seed'] = s
    records.append(m_b)
    
    # (c) No time info (K=1 style with min-max scoring and delta_base)
    s_c = DAECEBO(k_windows=1, lambd=lam_opt, random_state=s)
    X_c, y_c = s_c.fit_resample(X_tr_s, y_tr, timestamps=None)
    clf_c = DecisionTreeClassifier(random_state=s).fit(X_c, y_c)
    m_c = compute_metrics(y_te, clf_c.predict_proba(X_te_s)[:, 1], clf_c.predict(X_te_s))
    m_c['Variant'] = '(c) No Time Info (K=1)'
    m_c['Seed'] = s
    records.append(m_c)
    
    # (d) EC-EBO
    s_d = ECEBO(random_state=s)
    X_d, y_d = s_d.fit_resample(X_tr_s, y_tr)
    clf_d = DecisionTreeClassifier(random_state=s).fit(X_d, y_d)
    m_d = compute_metrics(y_te, clf_d.predict_proba(X_te_s)[:, 1], clf_d.predict(X_te_s))
    m_d['Variant'] = '(d) EC-EBO'
    m_d['Seed'] = s
    records.append(m_d)
    
    return records


def run_s2_control(k_opt=2, lam_opt=0.0):
    print("=" * 80)
    print(f"S2: Permuted-Timestamp Control Experiment (Headline config: K={k_opt}, lambda={lam_opt})...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    tr_sz = int(0.75 * len(df))
    train_df, test_df = df.iloc[:tr_sz], df.iloc[tr_sz:]
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    X_tr = train_df[feat_cols].values
    y_tr = train_df['Class'].values.astype(int)
    t_tr = train_df['Time'].values.astype(float)
    X_te = test_df[feat_cols].values
    y_te = test_df['Class'].values.astype(int)
    
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    
    seeds = list(range(42, 52))
    tasks = [(s, k_opt, lam_opt, X_tr_s, y_tr, t_tr, X_te_s, y_te) for s in seeds]
    
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
        results_lists = list(executor.map(_eval_s2_seed, tasks))
    
    records = [r for sublist in results_lists for r in sublist]
    print(f"S2 10-seed control completed in {time.time() - t0:.2f}s")
    
    df_raw = pd.DataFrame(records)
    df_raw.to_csv('results/s2_permuted_control_raw.csv', index=False)
    
    # Summary
    variants = [
        '(a) Real Timestamps (DA-ECEBO)',
        '(b) Permuted Timestamps',
        '(c) No Time Info (K=1)',
        '(d) EC-EBO'
    ]
    summary = []
    for v in variants:
        sub = df_raw[df_raw['Variant'] == v]
        summary.append({
            'Variant': v,
            'Minority-F1': f"{sub['F-measure'].mean():.4f} +/- {sub['F-measure'].std():.4f}",
            'Precision': f"{sub['Precision'].mean():.4f} +/- {sub['Precision'].std():.4f}",
            'Recall': f"{sub['Recall'].mean():.4f} +/- {sub['Recall'].std():.4f}",
            'Macro-Gmean': f"{sub['Macro-Gmean'].mean():.4f} +/- {sub['Macro-Gmean'].std():.4f}",
            'AUC': f"{sub['AUC'].mean():.4f} +/- {sub['AUC'].std():.4f}",
            'F1_mean': sub['F-measure'].mean()
        })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv('results/s2_permuted_control_summary.csv', index=False)
    
    # Paired Wilcoxon
    f1_a = df_raw[df_raw['Variant'] == variants[0]].sort_values('Seed')['F-measure'].values
    f1_b = df_raw[df_raw['Variant'] == variants[1]].sort_values('Seed')['F-measure'].values
    f1_c = df_raw[df_raw['Variant'] == variants[2]].sort_values('Seed')['F-measure'].values
    f1_d = df_raw[df_raw['Variant'] == variants[3]].sort_values('Seed')['F-measure'].values
    
    w_ab = wilcoxon(f1_a, f1_b)
    w_ac = wilcoxon(f1_a, f1_c)
    w_ad = wilcoxon(f1_a, f1_d)
    
    pvals = [
        {'Comparison': '(a) Real vs (b) Permuted', 'Mean_Diff': np.mean(f1_a - f1_b), 'Stat': w_ab.statistic, 'p_value': w_ab.pvalue, 'Significant': w_ab.pvalue < 0.05},
        {'Comparison': '(a) Real vs (c) No-Time (K=1)', 'Mean_Diff': np.mean(f1_a - f1_c), 'Stat': w_ac.statistic, 'p_value': w_ac.pvalue, 'Significant': w_ac.pvalue < 0.05},
        {'Comparison': '(a) Real vs (d) EC-EBO', 'Mean_Diff': np.mean(f1_a - f1_d), 'Stat': w_ad.statistic, 'p_value': w_ad.pvalue, 'Significant': w_ad.pvalue < 0.05}
    ]
    df_pvals = pd.DataFrame(pvals)
    df_pvals.to_csv('results/s2_wilcoxon_pvalues.csv', index=False)
    
    # Cluster Analysis:
    print("\n  Analyzing ECC cluster structure on CreditCard train set (398 frauds)...")
    from sklearn.neighbors import NearestNeighbors
    idx_maj = np.where(y_tr == 0)[0]
    idx_min = np.where(y_tr == 1)[0]
    X_maj_s = X_tr_s[idx_maj]
    X_min_s = X_tr_s[idx_min]
    n_min = len(X_min_s)
    
    nn_maj = NearestNeighbors(n_neighbors=7, algorithm='auto', n_jobs=1).fit(X_maj_s)
    dist_to_maj, _ = nn_maj.kneighbors(X_min_s)
    d_maj_arr = dist_to_maj[:, 0]
    
    clusters = {i: [i] for i in range(n_min)}
    centroids = {i: X_min_s[i].copy() for i in range(n_min)}
    cluster_d_maj = {i: float(d_maj_arr[i]) for i in range(n_min)}
    dist_mat = np.full((n_min, n_min), np.inf)
    for i in range(n_min):
        for j in range(i + 1, n_min):
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
                    
    cluster_list = list(clusters.values())
    m = len(cluster_list)
    cluster_sizes = [len(c) for c in cluster_list]
    
    # Calculate quotas under headline config
    min_times = t_tr[idx_min]
    t_min_val, t_max_val = np.min(t_tr), np.max(t_tr)
    time_bins = np.linspace(t_min_val, t_max_val + 1e-9, k_opt + 1)
    sample_windows = np.clip(np.digitize(min_times, time_bins) - 1, 0, k_opt - 1)
    
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    rng = np.random.RandomState(42)
    sub_maj = rng.choice(len(X_maj_s), size=min(len(X_maj_s), n_min * 10), replace=False)
    lr.fit(np.vstack([X_maj_s[sub_maj], X_min_s]), np.hstack([np.zeros(len(sub_maj)), np.ones(n_min)]))
    eps = 1e-12
    p_min = np.clip(lr.predict_proba(X_min_s)[:, 1], eps, 1.0 - eps)
    sample_entropy = -(p_min * np.log2(p_min) + (1.0 - p_min) * np.log2(1.0 - p_min))
    
    c_entropy = np.zeros(m)
    c_drift = np.zeros(m)
    for j, c_idx in enumerate(cluster_list):
        c_entropy[j] = np.mean(sample_entropy[c_idx])
        c_times = sample_windows[c_idx]
        t_early = np.sum(c_times < (k_opt // 2)) if k_opt > 1 else len(c_times)
        t_recent = np.sum(c_times >= (k_opt // 2)) if k_opt > 1 else len(c_times)
        c_drift[j] = t_recent / (t_early + 1e-5)
        
    s_ent = (c_entropy - np.min(c_entropy)) / (np.max(c_entropy) - np.min(c_entropy) + 1e-9)
    s_dr = (c_drift - np.min(c_drift)) / (np.max(c_drift) - np.min(c_drift) + 1e-9)
    s_comb = lam_opt * s_ent + (1.0 - lam_opt) * s_dr + 0.05
    alloc_props = s_comb / np.sum(s_comb)
    
    delta_n = len(X_maj_s) - n_min
    quotas_float = delta_n * alloc_props
    quotas_floor = np.floor(quotas_float).astype(int)
    remainders = quotas_float - quotas_floor
    unallocated = delta_n - np.sum(quotas_floor)
    top_rem_idx = np.argsort(-remainders)[:unallocated]
    quotas = quotas_floor.copy()
    quotas[top_rem_idx] += 1
    
    n_zero_quota = int(np.sum(quotas == 0))
    
    cluster_stats = {
        'm_clusters': m,
        'cluster_size_min': int(np.min(cluster_sizes)),
        'cluster_size_q25': float(np.percentile(cluster_sizes, 25)),
        'cluster_size_median': float(np.median(cluster_sizes)),
        'cluster_size_mean': float(np.mean(cluster_sizes)),
        'cluster_size_q75': float(np.percentile(cluster_sizes, 75)),
        'cluster_size_max': int(np.max(cluster_sizes)),
        'zero_quota_clusters': n_zero_quota,
        'zero_quota_pct': (n_zero_quota / m) * 100.0
    }
    pd.DataFrame([cluster_stats]).to_csv('results/s2_cluster_stats.csv', index=False)
    
    print("\n=== S2 Permuted-Timestamp Control Summary ===")
    print(df_sum.to_string(index=False))
    print("\n=== S2 Paired Wilcoxon Tests ===")
    print(df_pvals.to_string(index=False))
    print("\n=== S2 Cluster Distribution Statistics ===")
    print(f"Total ECC clusters m: {m}")
    print(f"Cluster sizes: Min={cluster_stats['cluster_size_min']}, Q25={cluster_stats['cluster_size_q25']}, Median={cluster_stats['cluster_size_median']}, Mean={cluster_stats['cluster_size_mean']:.2f}, Q75={cluster_stats['cluster_size_q75']}, Max={cluster_stats['cluster_size_max']}")
    print(f"Zero-quota clusters under headline config: {n_zero_quota} / {m} ({cluster_stats['zero_quota_pct']:.2f}%)")


def _eval_s3_block_seed(args):
    bname, b_idx, n_frauds_test, s, X_tr_s, y_tr, t_tr, X_te_s, y_te, k_opt, lam_opt = args
    records = []
    
    # Baseline
    clf = DecisionTreeClassifier(random_state=s).fit(X_tr_s, y_tr)
    m = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    m['Block'] = bname; m['Block_Idx'] = b_idx; m['Test_Frauds'] = n_frauds_test; m['Method'] = 'Baseline'; m['Seed'] = s
    records.append(m)
    
    # SMOTE
    X_sm, y_sm = SMOTE(random_state=s).fit_resample(X_tr_s, y_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_sm, y_sm)
    m = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    m['Block'] = bname; m['Block_Idx'] = b_idx; m['Test_Frauds'] = n_frauds_test; m['Method'] = 'SMOTE'; m['Seed'] = s
    records.append(m)
    
    # EC-EBO
    X_ec, y_ec = ECEBO(random_state=s).fit_resample(X_tr_s, y_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_ec, y_ec)
    m = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    m['Block'] = bname; m['Block_Idx'] = b_idx; m['Test_Frauds'] = n_frauds_test; m['Method'] = 'EC-EBO'; m['Seed'] = s
    records.append(m)
    
    # DA-ECEBO
    X_da, y_da = DAECEBO(k_windows=k_opt, lambd=lam_opt, random_state=s).fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_da, y_da)
    m = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    m['Block'] = bname; m['Block_Idx'] = b_idx; m['Test_Frauds'] = n_frauds_test; m['Method'] = 'DA-ECEBO'; m['Seed'] = s
    records.append(m)
    
    return records


def run_s3_rolling_blocks(k_opt=2, lam_opt=0.0):
    print("\n" + "=" * 80)
    print("S3: Better Rolling-Origin Evaluation (5 Non-Overlapping Test Blocks x 10 seeds)...")
    df = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n = len(df)
    feat_cols = [c for c in df.columns if c not in ['Time', 'Class']]
    
    block_pcts = [
        (0.5, 0.6, 'Block 1 (50%-60%)'),
        (0.6, 0.7, 'Block 2 (60%-70%)'),
        (0.7, 0.8, 'Block 3 (70%-80%)'),
        (0.8, 0.9, 'Block 4 (80%-90%)'),
        (0.9, 1.0, 'Block 5 (90%-100%)')
    ]
    
    methods = ['Baseline', 'SMOTE', 'EC-EBO', 'DA-ECEBO']
    records = []
    seeds = list(range(42, 52))
    
    t0 = time.time()
    for b_idx, (t_start_pct, t_end_pct, bname) in enumerate(block_pcts):
        tr_end = int(t_start_pct * n)
        te_end = int(t_end_pct * n)
        
        train_df = df.iloc[:tr_end]
        test_df = df.iloc[tr_end:te_end]
        
        X_tr = train_df[feat_cols].values
        y_tr = train_df['Class'].values.astype(int)
        t_tr = train_df['Time'].values.astype(float)
        
        X_te = test_df[feat_cols].values
        y_te = test_df['Class'].values.astype(int)
        
        n_frauds_test = int(np.sum(y_te == 1))
        print(f"\n  {bname}: Train 0..{tr_end:,} ({np.sum(y_tr == 1)} frauds) -> Test {tr_end:,}..{te_end:,} ({n_frauds_test} frauds)...", flush=True)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        tasks = [(bname, b_idx + 1, n_frauds_test, s, X_tr_s, y_tr, t_tr, X_te_s, y_te, k_opt, lam_opt) for s in seeds]
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
            block_results = list(executor.map(_eval_s3_block_seed, tasks))
            
        for sublist in block_results:
            records.extend(sublist)
            
    print(f"\nS3 rolling-origin evaluation completed in {time.time() - t0:.2f}s")
    df_raw = pd.DataFrame(records)
    df_raw.to_csv('results/s3_rolling_blocks_raw.csv', index=False)
    
    # Per-block summary table
    summary = []
    da_gt_ec_counts = 0
    da_gt_base_counts = 0
    total_blocks = len(block_pcts)
    
    for b_idx, (_, _, bname) in enumerate(block_pcts):
        sub_b = df_raw[df_raw['Block'] == bname]
        n_f = sub_b['Test_Frauds'].iloc[0]
        row = {'Block': bname, 'Test_Frauds': n_f}
        f1_dict = {}
        for mname in methods:
            sub_m = sub_b[sub_b['Method'] == mname]
            f1_mean = sub_m['F-measure'].mean()
            f1_std = sub_m['F-measure'].std()
            row[f"{mname}_F1"] = f"{f1_mean:.4f} +/- {f1_std:.4f}"
            f1_dict[mname] = f1_mean
            
        if f1_dict['DA-ECEBO'] > f1_dict['EC-EBO']:
            da_gt_ec_counts += 1
        if f1_dict['DA-ECEBO'] > f1_dict['Baseline']:
            da_gt_base_counts += 1
        row['DA_beats_ECEBO'] = f1_dict['DA-ECEBO'] > f1_dict['EC-EBO']
        row['DA_beats_Baseline'] = f1_dict['DA-ECEBO'] > f1_dict['Baseline']
        summary.append(row)
        
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv('results/s3_rolling_blocks_summary.csv', index=False)
    
    # Pooled bootstrap CI of paired difference (DA-ECEBO minus EC-EBO) over all blocks
    da_f1_all = df_raw[df_raw['Method'] == 'DA-ECEBO'].sort_values(['Block_Idx', 'Seed'])['F-measure'].values
    ec_f1_all = df_raw[df_raw['Method'] == 'EC-EBO'].sort_values(['Block_Idx', 'Seed'])['F-measure'].values
    diffs = da_f1_all - ec_f1_all
    
    B = 1000
    rng_b = np.random.RandomState(42)
    b_diff_means = []
    for _ in range(B):
        idx = rng_b.choice(len(diffs), size=len(diffs), replace=True)
        b_diff_means.append(np.mean(diffs[idx]))
    ci_pooled = np.percentile(b_diff_means, [2.5, 97.5])
    
    pooled_stats = {
        'DA_beats_ECEBO_blocks': f"{da_gt_ec_counts} / {total_blocks}",
        'DA_beats_Baseline_blocks': f"{da_gt_base_counts} / {total_blocks}",
        'Pooled_Mean_Diff': np.mean(diffs),
        'Pooled_95_CI': f"[{ci_pooled[0]:.4f}, {ci_pooled[1]:.4f}]"
    }
    pd.DataFrame([pooled_stats]).to_csv('results/s3_pooled_ci.csv', index=False)
    
    print("\n=== S3 Rolling Blocks Summary (10 Seeds per Block) ===")
    print(df_sum.to_string(index=False))
    print("\n=== S3 Pooled Statistics ===")
    print(f"DA-ECEBO > EC-EBO in: {pooled_stats['DA_beats_ECEBO_blocks']} blocks")
    print(f"DA-ECEBO > Baseline in: {pooled_stats['DA_beats_Baseline_blocks']} blocks")
    print(f"Pooled mean difference (DA minus EC-EBO): {pooled_stats['Pooled_Mean_Diff']:.4f} (95% CI: {pooled_stats['Pooled_95_CI']})")


def main():
    os.makedirs('results', exist_ok=True)
    k_opt = 2
    lam_opt = 0.0
    if os.path.exists('results/cv_grid_v3.csv'):
        df_cv = pd.read_csv('results/cv_grid_v3.csv')
        best_r = df_cv.sort_values('Composite_CV', ascending=False).iloc[0]
        k_opt = int(best_r['K_windows'])
        lam_opt = float(best_r['lambda'])
        print(f"Read locked headline config from cv_grid_v3.csv: K={k_opt}, lambda={lam_opt}")
    else:
        print(f"cv_grid_v3.csv not found; defaulting to K={k_opt}, lambda={lam_opt}")
        
    run_s2_control(k_opt=k_opt, lam_opt=lam_opt)
    run_s3_rolling_blocks(k_opt=k_opt, lam_opt=lam_opt)


if __name__ == '__main__':
    main()
