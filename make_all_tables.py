"""
make_all_tables.py
Regenerates every single research table programmatically from the raw result CSV files.
Zero hand-copied numbers.
"""

import os
import pandas as pd
import numpy as np


def print_separator(title):
    print("\n" + "=" * 90)
    print(f"=== {title.upper()} ===")
    print("=" * 90)


def table_s1():
    print_separator("S1: 5x5 Expanding-Window Cross-Validation Grid (10 Seeds)")
    fpath = 'results/cv_grid_v3.csv'
    if not os.path.exists(fpath):
        fpath = 'results/cv_grid_v2.csv'
    df = pd.read_csv(fpath)
    pivot_mean = df.pivot(index='K_windows', columns='lambda', values='Composite_CV')
    pivot_std = df.pivot(index='K_windows', columns='lambda', values='Composite_CV_std')
    
    print("Mean Composite CV (0.5*F1 + 0.5*Gmean):")
    print(pivot_mean.round(4).to_string())
    print("\nStandard Deviation:")
    print(pivot_std.round(4).to_string())
    
    best_row = df.sort_values('Composite_CV', ascending=False).iloc[0]
    best_score = best_row['Composite_CV']
    best_se = best_row['Composite_CV_std'] / np.sqrt(10)
    thresh = best_score - best_se
    within_1se = df[df['Composite_CV'] >= thresh].sort_values(['K_windows', 'lambda'], ascending=[True, False])
    one_se_row = within_1se.iloc[0]
    
    print(f"\nArgmax cell: K={int(best_row['K_windows'])}, lambda={best_row['lambda']} (Composite = {best_score:.4f} +/- {best_row['Composite_CV_std']:.4f}, SE = {best_se:.4f})")
    print(f"1-SE Rule cell (Threshold >= {thresh:.4f}): K={int(one_se_row['K_windows'])}, lambda={one_se_row['lambda']} (Composite = {one_se_row['Composite_CV']:.4f})")


def table_s2():
    print_separator("S2: Permuted-Timestamp Control Experiment")
    f_sum = 'results/s2_permuted_control_summary.csv'
    f_pvals = 'results/s2_wilcoxon_pvalues.csv'
    f_clust = 'results/s2_cluster_stats.csv'
    if os.path.exists(f_sum):
        df_sum = pd.read_csv(f_sum)
        print("Performance across Control Variants (10 seeds, CART):")
        print(df_sum[['Variant', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC']].to_string(index=False))
    if os.path.exists(f_pvals):
        df_p = pd.read_csv(f_pvals)
        print("\nPaired Wilcoxon Signed-Rank Tests (vs Real Timestamps):")
        print(df_p.to_string(index=False))
    if os.path.exists(f_clust):
        df_c = pd.read_csv(f_clust)
        print("\nECC Cluster Statistics (CreditCard Train Set, 398 Frauds):")
        print(df_c.to_string(index=False))


def table_s3():
    print_separator("S3: Better Rolling-Origin Evaluation (5 Non-Overlapping Test Blocks)")
    f_sum = 'results/s3_rolling_blocks_summary.csv'
    f_ci = 'results/s3_pooled_ci.csv'
    if os.path.exists(f_sum):
        df_sum = pd.read_csv(f_sum)
        print("Per-Block Minority-F1 (Mean +/- Std over 10 seeds):")
        print(df_sum.to_string(index=False))
    if os.path.exists(f_ci):
        df_ci = pd.read_csv(f_ci)
        print("\nPooled Summary & Bootstrap Confidence Interval:")
        print(df_ci.to_string(index=False))


def table_s4_classifiers():
    print_separator("S4(b): Multi-Model Classifier Benchmark & True Win/Loss Tally")
    fpath = 'results/classifier_benchmark_summary.csv'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print("Summary Across 6 Classifiers x 6 Methods (10 Seeds):")
        print(df[['Classifier', 'Method', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC', 'Is_Best']].to_string(index=False))
        
        # True Win/Loss Tally
        print("\nTrue Win/Loss List for DA-ECEBO [K=4, lam=0.25]:")
        classifiers = df['Classifier'].unique()
        for c in classifiers:
            sub = df[df['Classifier'] == c]
            base_f1 = sub[sub['Method'] == 'Baseline']['Minority-F1_mean'].values[0]
            smote_f1 = sub[sub['Method'] == 'SMOTE']['Minority-F1_mean'].values[0]
            ec_f1 = sub[sub['Method'] == 'EC-EBO']['Minority-F1_mean'].values[0]
            da_f1 = sub[sub['Method'] == 'DA-ECEBO']['Minority-F1_mean'].values[0]
            
            w_ec = "WIN" if da_f1 > ec_f1 else "LOSS"
            w_sm = "WIN" if da_f1 > smote_f1 else "LOSS"
            w_base = "WIN" if da_f1 > base_f1 else "LOSS"
            print(f"  {c:18s}: DA={da_f1:.4f} | vs EC-EBO ({ec_f1:.4f}) -> {w_ec:4s} ({da_f1-ec_f1:+.4f}) | vs SMOTE ({smote_f1:.4f}) -> {w_sm:4s} ({da_f1-smote_f1:+.4f}) | vs Baseline ({base_f1:.4f}) -> {w_base:4s} ({da_f1-base_f1:+.4f})")


def table_s4c_runtime():
    print_separator("S4(c): Runtime Benchmark (5 Runs per Method, Median + IQR)")
    fpath = 'results/runtime_benchmark_5runs.csv'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(df[['Dataset', 'Method', 'Time_str', 'HillClimb_Iters']].to_string(index=False))


def table_s4d_rbf_svm():
    print_separator("S4(d): RBF SVM (sklearn SVC) with Stratified Majority Subsample")
    fpath = 'results/rbf_svm_benchmark_summary.csv'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(df[['Classifier', 'Method', 'Minority-F1', 'Precision', 'Recall', 'Macro-Gmean', 'AUC', 'Is_Best']].to_string(index=False))


def table_s5_haberman():
    print_separator("S5: Haberman Chronological Split & Bootstrap CI (17 Test Minority Samples)")
    fpath = 'results/haberman_bootstrap_ci.csv'
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        print(df[['Method', 'Test_Minority_Samples', 'Minority-F1', 'Minority-F1_95_CI', 'AUC', 'AUC_95_CI', 'AUC_lt_0.5']].to_string(index=False))


def table_ablation():
    print_separator("R2/S1: Multi-Seed Ablation Study & Wilcoxon Tests")
    f_sum = 'results/ablation_multiseed_summary.csv'
    f_p = 'results/ablation_wilcoxon_pvalues.csv'
    if os.path.exists(f_sum):
        df = pd.read_csv(f_sum)
        print(df[['Variant', 'Minority-F1', 'Precision', 'Recall', 'Macro-F1', 'Macro-Gmean', 'AUC']].to_string(index=False))
    if os.path.exists(f_p):
        df_p = pd.read_csv(f_p)
        print("\nPaired Wilcoxon Signed-Rank p-values:")
        print(df_p.to_string(index=False))


def main():
    print("=" * 90)
    print("PROGRAMMATIC REGENERATION OF ALL RESEARCH TABLES FROM CSV DATA")
    print("=" * 90)
    table_s1()
    table_s2()
    table_s3()
    table_s4_classifiers()
    table_s4c_runtime()
    table_s4d_rbf_svm()
    table_s5_haberman()
    table_ablation()


if __name__ == '__main__':
    main()
