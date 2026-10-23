"""
T3: Correct Statistical Inference for Rolling-Origin Evaluation & Per-Block Dynamic CV Selection.
1. Statistical Inference on 5-Block Rolling-Origin Benchmark:
   - Per-block mean of 10 seeds (DA-ECEBO vs EC-EBO)
   - Sign test and exact two-sided binomial p-value across 5 blocks
   - Block-level bootstrap (B=10,000 resamples of the 5 blocks)
   - Within-block transaction bootstrap (B=1,000 resamples of test rows per block)
2. Per-Block Dynamic Expanding-Window CV Selection of (K, lambda):
   - For each block i: train on strictly prior stream history.
   - Run expanding-window CV (2 folds, 25 grid cells) inside the training split.
   - Select optimal (K*_i, lambda*_i) per block.
   - Train DA-ECEBO(K*_i, lambda*_i) across 10 seeds, evaluate on block i.
   - Report per-block selected configs and resulting F1 vs fixed headline, EC-EBO, Baseline.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from concurrent.futures import ProcessPoolExecutor

from da_ecebo_and_ecebo_implementation import (
    ECEBO, DAECEBO, compute_metrics
)


def run_t3_part1_inference():
    print("=" * 80)
    print("T3 (Part 1): Statistical Inference on 5-Block Rolling Origin...")
    print("=" * 80)
    
    df_raw = pd.read_csv('results/s3_rolling_blocks_raw.csv')
    df_sum = pd.read_csv('results/s3_rolling_blocks_summary.csv')
    
    blocks = df_sum['Block'].tolist()
    n_blocks = len(blocks)
    
    # 1. Per-block means of 10 seeds
    da_means = []
    ec_means = []
    base_means = []
    diff_means = []
    
    for b in blocks:
        sub_da = df_raw[(df_raw['Block'] == b) & (df_raw['Method'] == 'DA-ECEBO')].sort_values('Seed')['F-measure'].values
        sub_ec = df_raw[(df_raw['Block'] == b) & (df_raw['Method'] == 'EC-EBO')].sort_values('Seed')['F-measure'].values
        sub_ba = df_raw[(df_raw['Block'] == b) & (df_raw['Method'] == 'Baseline')].sort_values('Seed')['F-measure'].values
        
        m_da = np.mean(sub_da)
        m_ec = np.mean(sub_ec)
        m_ba = np.mean(sub_ba)
        
        da_means.append(m_da)
        ec_means.append(m_ec)
        base_means.append(m_ba)
        diff_means.append(m_da - m_ec)
        
    diff_means = np.array(diff_means)
    k_wins = int(np.sum(diff_means > 0))
    
    # 2. Sign Test & Exact Binomial p-value
    b_test = binomtest(k_wins, n=n_blocks, p=0.5, alternative='two-sided')
    binom_p = b_test.pvalue
    
    # 3. Block-Level Bootstrap (resampling blocks, B=10,000)
    B_block = 10000
    rng_b = np.random.RandomState(42)
    b_boot_means = []
    for _ in range(B_block):
        draw_idx = rng_b.choice(n_blocks, size=n_blocks, replace=True)
        b_boot_means.append(np.mean(diff_means[draw_idx]))
    block_ci = np.percentile(b_boot_means, [2.5, 97.5])
    
    # 4. Within-Block Test Transaction Bootstrap (B=1,000)
    df_data = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n_tot = len(df_data)
    feat_cols = [c for c in df_data.columns if c not in ['Time', 'Class']]
    
    block_pcts = [
        (0.5, 0.6, 'Block 1 (50%-60%)'),
        (0.6, 0.7, 'Block 2 (60%-70%)'),
        (0.7, 0.8, 'Block 3 (70%-80%)'),
        (0.8, 0.9, 'Block 4 (80%-90%)'),
        (0.9, 1.0, 'Block 5 (90%-100%)')
    ]
    
    within_cis = []
    B_trans = 1000
    rng_t = np.random.RandomState(42)
    
    print("\n  Computing within-block transaction bootstraps (B=1,000 resamples)...")
    for b_idx, (t_start, t_end, bname) in enumerate(block_pcts):
        tr_end = int(t_start * n_tot)
        te_end = int(t_end * n_tot)
        
        train_df = df_data.iloc[:tr_end]
        test_df = df_data.iloc[tr_end:te_end]
        
        X_tr = train_df[feat_cols].values
        y_tr = train_df['Class'].values.astype(int)
        t_tr = train_df['Time'].values.astype(float)
        X_te = test_df[feat_cols].values
        y_te = test_df['Class'].values.astype(int)
        
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        
        # Use Seed 42 models for transaction bootstrap
        s_ec = ECEBO(random_state=42)
        X_ec, y_ec = s_ec.fit_resample(X_tr_s, y_tr)
        clf_ec = DecisionTreeClassifier(random_state=42).fit(X_ec, y_ec)
        pr_ec = clf_ec.predict_proba(X_te_s)[:, 1]
        pd_ec = clf_ec.predict(X_te_s)
        
        s_da = DAECEBO(k_windows=2, lambd=0.0, random_state=42)
        X_da, y_da = s_da.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
        clf_da = DecisionTreeClassifier(random_state=42).fit(X_da, y_da)
        pr_da = clf_da.predict_proba(X_te_s)[:, 1]
        pd_da = clf_da.predict(X_te_s)
        
        n_te = len(y_te)
        boot_diffs = []
        for _ in range(B_trans):
            t_idx = rng_t.choice(n_te, size=n_te, replace=True)
            y_b = y_te[t_idx]
            if len(np.unique(y_b)) < 2:
                continue
            m_ec_b = compute_metrics(y_b, pr_ec[t_idx], pd_ec[t_idx])
            m_da_b = compute_metrics(y_b, pr_da[t_idx], pd_da[t_idx])
            boot_diffs.append(m_da_b['F-measure'] - m_ec_b['F-measure'])
            
        ci_t = np.percentile(boot_diffs, [2.5, 97.5])
        within_cis.append({
            'Block': bname,
            'DA_F1_Mean': da_means[b_idx],
            'ECEBO_F1_Mean': ec_means[b_idx],
            'Mean_Diff': diff_means[b_idx],
            'WithinBlock_Trans_95_CI': f"[{ci_t[0]:.4f}, {ci_t[1]:.4f}]",
            'CI_excludes_zero': (ci_t[0] > 0) or (ci_t[1] < 0)
        })
        
    df_within = pd.DataFrame(within_cis)
    df_within.to_csv('results/t3_within_block_trans_bootstrap.csv', index=False)
    
    inference_summary = {
        'Total_Blocks': n_blocks,
        'DA_beats_ECEBO_Blocks': f"{k_wins} / {n_blocks}",
        'Sign_Test_Exact_Binomial_p': binom_p,
        'Sign_Test_Significant_0.05': binom_p < 0.05,
        'Block_Level_Bootstrap_Mean_Diff': np.mean(diff_means),
        'Block_Level_Bootstrap_95_CI': f"[{block_ci[0]:.4f}, {block_ci[1]:.4f}]",
        'Block_Level_CI_excludes_zero': (block_ci[0] > 0) or (block_ci[1] < 0)
    }
    pd.DataFrame([inference_summary]).to_csv('results/t3_block_inference_summary.csv', index=False)
    
    print("\n=== T3 (Part 1): Block-Level Statistical Inference ===")
    for k, v in inference_summary.items():
        print(f"  {k:35s}: {v}")
    print("\n=== T3 (Part 1): Within-Block Transaction Bootstrap CIs (B=1,000) ===")
    print(df_within[['Block', 'DA_F1_Mean', 'ECEBO_F1_Mean', 'Mean_Diff', 'WithinBlock_Trans_95_CI', 'CI_excludes_zero']].to_string(index=False))


def _eval_cv_task(task):
    k, lam, s, X_tr1, y_tr1, t_tr1, X_val1, y_val1, X_tr2, y_tr2, t_tr2, X_val2, y_val2 = task
    # Fold 1
    s1 = DAECEBO(k_windows=k, lambd=lam, random_state=s)
    X_r1, y_r1 = s1.fit_resample(X_tr1, y_tr1, timestamps=t_tr1)
    clf1 = DecisionTreeClassifier(random_state=s).fit(X_r1, y_r1)
    m1 = compute_metrics(y_val1, clf1.predict_proba(X_val1)[:, 1], clf1.predict(X_val1))
    
    # Fold 2
    s2 = DAECEBO(k_windows=k, lambd=lam, random_state=s)
    X_r2, y_r2 = s2.fit_resample(X_tr2, y_tr2, timestamps=t_tr2)
    clf2 = DecisionTreeClassifier(random_state=s).fit(X_r2, y_r2)
    m2 = compute_metrics(y_val2, clf2.predict_proba(X_val2)[:, 1], clf2.predict(X_val2))
    
    comp = 0.5 * (0.5 * (m1['F-measure'] + m2['F-measure'])) + 0.5 * (0.5 * (m1['Macro-Gmean'] + m2['Macro-Gmean']))
    return (k, lam, s, comp)


def _eval_block_test_seed(args):
    bname, s, k_opt, lam_opt, X_tr_s, y_tr, t_tr, X_te_s, y_te = args
    # DA-ECEBO with dynamic (k_opt, lam_opt)
    s_da = DAECEBO(k_windows=k_opt, lambd=lam_opt, random_state=s)
    X_da, y_da = s_da.fit_resample(X_tr_s, y_tr, timestamps=t_tr)
    clf = DecisionTreeClassifier(random_state=s).fit(X_da, y_da)
    m_da = compute_metrics(y_te, clf.predict_proba(X_te_s)[:, 1], clf.predict(X_te_s))
    m_da['Seed'] = s
    m_da['Block'] = bname
    m_da['K_opt'] = k_opt
    m_da['lambda_opt'] = lam_opt
    return m_da


def run_t3_part2_dynamic_cv():
    print("\n" + "=" * 80)
    print("T3 (Part 2): Per-Block Dynamic Expanding-Window CV Selection of (K, lambda)...")
    print("=" * 80)
    
    df_data = pd.read_parquet('datasets/creditcard.parquet').sort_values('Time').reset_index(drop=True)
    n_tot = len(df_data)
    feat_cols = [c for c in df_data.columns if c not in ['Time', 'Class']]
    
    block_pcts = [
        (0.5, 0.6, 'Block 1 (50%-60%)'),
        (0.6, 0.7, 'Block 2 (60%-70%)'),
        (0.7, 0.8, 'Block 3 (70%-80%)'),
        (0.8, 0.9, 'Block 4 (80%-90%)'),
        (0.9, 1.0, 'Block 5 (90%-100%)')
    ]
    
    k_list = [1, 2, 3, 4, 5]
    lam_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    cv_seeds = [42, 43, 44]  # 3 seeds for fast and robust CV selection
    test_seeds = list(range(42, 52)) # 10 seeds for final test evaluation
    
    selected_configs = []
    
    for b_idx, (t_start, t_end, bname) in enumerate(block_pcts):
        tr_end = int(t_start * n_tot)
        te_end = int(t_end * n_tot)
        
        train_df = df_data.iloc[:tr_end]
        test_df = df_data.iloc[tr_end:te_end]
        
        X_tr = train_df[feat_cols].values
        y_tr = train_df['Class'].values.astype(int)
        t_tr = train_df['Time'].values.astype(float)
        X_te = test_df[feat_cols].values
        y_te = test_df['Class'].values.astype(int)
        
        n_tr = len(train_df)
        n_frauds_tr = int(np.sum(y_tr == 1))
        n_frauds_te = int(np.sum(y_te == 1))
        
        print(f"\n--- {bname}: Training on 0..{tr_end:,} ({n_frauds_tr} frauds) ---")
        print(f"  Running 2-fold expanding-window CV inside training set...")
        
        # 2 folds inside train:
        # Fold 1: train 0..65% -> val 65%..85%
        # Fold 2: train 0..85% -> val 85%..100%
        f1_tr_end = int(0.65 * n_tr)
        f1_val_end = int(0.85 * n_tr)
        f2_tr_end = int(0.85 * n_tr)
        f2_val_end = n_tr
        
        sc1 = StandardScaler()
        X_tr1 = sc1.fit_transform(X_tr[:f1_tr_end])
        X_val1 = sc1.transform(X_tr[f1_tr_end:f1_val_end])
        y_tr1, t_tr1 = y_tr[:f1_tr_end], t_tr[:f1_tr_end]
        y_val1 = y_tr[f1_tr_end:f1_val_end]
        
        sc2 = StandardScaler()
        X_tr2 = sc2.fit_transform(X_tr[:f2_tr_end])
        X_val2 = sc2.transform(X_tr[f2_tr_end:f2_val_end])
        y_tr2, t_tr2 = y_tr[:f2_tr_end], t_tr[:f2_tr_end]
        y_val2 = y_tr[f2_tr_end:f2_val_end]
        
        cv_tasks = []
        for k in k_list:
            for lam in lam_list:
                for s in cv_seeds:
                    cv_tasks.append((k, lam, s, X_tr1, y_tr1, t_tr1, X_val1, y_val1, X_tr2, y_tr2, t_tr2, X_val2, y_val2))
                    
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
            cv_res = list(executor.map(_eval_cv_task, cv_tasks))
            
        # Aggregate CV
        df_cv = pd.DataFrame(cv_res, columns=['K', 'lambda', 'seed', 'composite'])
        grid_mean = df_cv.groupby(['K', 'lambda'])['composite'].mean().reset_index()
        best_row = grid_mean.sort_values('composite', ascending=False).iloc[0]
        k_opt = int(best_row['K'])
        lam_opt = float(best_row['lambda'])
        best_cv_score = float(best_row['composite'])
        print(f"  CV completed in {time.time() - t0:.2f}s -> Selected (K={k_opt}, lambda={lam_opt}) with CV score {best_cv_score:.4f}")
        
        # Test on block with 10 seeds
        scaler_full = StandardScaler()
        X_tr_s = scaler_full.fit_transform(X_tr)
        X_te_s = scaler_full.transform(X_te)
        
        test_tasks = [(bname, s, k_opt, lam_opt, X_tr_s, y_tr, t_tr, X_te_s, y_te) for s in test_seeds]
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), 10)) as executor:
            test_results = list(executor.map(_eval_block_test_seed, test_tasks))
            
        f1_vals = [r['F-measure'] for r in test_results]
        prec_vals = [r['Precision'] for r in test_results]
        rec_vals = [r['Recall'] for r in test_results]
        
        selected_configs.append({
            'Block': bname,
            'Test_Frauds': n_frauds_te,
            'Selected_K': k_opt,
            'Selected_lambda': lam_opt,
            'Train_CV_Score': best_cv_score,
            'Dynamic_DA_F1': f"{np.mean(f1_vals):.4f} +/- {np.std(f1_vals):.4f}",
            'Dynamic_F1_mean': np.mean(f1_vals),
            'Precision': f"{np.mean(prec_vals):.4f} +/- {np.std(prec_vals):.4f}",
            'Recall': f"{np.mean(rec_vals):.4f} +/- {np.std(rec_vals):.4f}"
        })
        
    df_dyn = pd.DataFrame(selected_configs)
    
    # Merge with previous static headline (K=2, lam=0), EC-EBO, and Baseline results from s3_rolling_blocks_summary.csv
    df_static = pd.read_csv('results/s3_rolling_blocks_summary.csv')
    df_merged = df_dyn.merge(df_static[['Block', 'Baseline_F1', 'EC-EBO_F1', 'DA-ECEBO_F1']], on='Block')
    df_merged.rename(columns={'DA-ECEBO_F1': 'Fixed_Headline_DA_F1'}, inplace=True)
    
    # Win comparison
    df_merged['Dyn_beats_ECEBO'] = df_merged['Dynamic_F1_mean'] > df_merged['EC-EBO_F1'].apply(lambda x: float(x.split(' ')[0]))
    df_merged['Dyn_beats_Headline'] = df_merged['Dynamic_F1_mean'] > df_merged['Fixed_Headline_DA_F1'].apply(lambda x: float(x.split(' ')[0]))
    df_merged['Dyn_beats_Baseline'] = df_merged['Dynamic_F1_mean'] > df_merged['Baseline_F1'].apply(lambda x: float(x.split(' ')[0]))
    
    df_merged.to_csv('results/t3_dynamic_cv_rolling_origin.csv', index=False)
    
    print("\n=== T3 (Part 2): Dynamic Per-Block CV Selection Results ===")
    print(df_merged[['Block', 'Selected_K', 'Selected_lambda', 'Train_CV_Score', 'Dynamic_DA_F1', 'Fixed_Headline_DA_F1', 'EC-EBO_F1', 'Baseline_F1', 'Dyn_beats_ECEBO', 'Dyn_beats_Baseline']].to_string(index=False))
    
    dyn_ec_wins = int(df_merged['Dyn_beats_ECEBO'].sum())
    dyn_base_wins = int(df_merged['Dyn_beats_Baseline'].sum())
    print(f"\nDynamic DA-ECEBO > EC-EBO in:   {dyn_ec_wins} / {len(block_pcts)} blocks")
    print(f"Dynamic DA-ECEBO > Baseline in: {dyn_base_wins} / {len(block_pcts)} blocks")


def main():
    os.makedirs('results', exist_ok=True)
    run_t3_part1_inference()
    run_t3_part2_dynamic_cv()


if __name__ == '__main__':
    main()
