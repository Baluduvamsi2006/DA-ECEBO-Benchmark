"""
Phase 3 Part 1: Rigorous Statistical Analysis (Unpooled)
========================================================
Replaces the pooled 80-pair tests with:
  (a) Per-dataset paired Wilcoxon over 10 seeds (Holm-corrected across methods) -> P3_stats_per_dataset.csv
  (b) Across-dataset Friedman and paired Wilcoxon on dataset means (N=8) -> P3_stats_across_datasets.csv & P3_avg_ranks.csv
  (c) Per-dataset CART means +- std over 10 seeds -> P3_per_dataset_means.csv
  (d) Audit of DAECEBO2 vs ECEBO ties and K=1, lambda=1 selections -> P3_tie_check.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

def holm_bonferroni(p_values):
    p_vals = np.array(p_values, dtype=float)
    m = len(p_vals)
    adj_p = np.ones(m)
    # Handle NaNs or 1s
    valid_idx = np.where(~np.isnan(p_vals))[0]
    if len(valid_idx) == 0:
        return adj_p
    sorted_order = np.argsort(p_vals[valid_idx])
    sorted_p = p_vals[valid_idx][sorted_order]
    
    adj_sorted = np.zeros(len(sorted_p))
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        val = p * (len(sorted_p) - i)
        running_max = max(running_max, val)
        adj_sorted[i] = min(1.0, running_max)
        
    for idx_in_sorted, orig_idx in enumerate(valid_idx[sorted_order]):
        adj_p[orig_idx] = adj_sorted[idx_in_sorted]
    return adj_p

def run_paired_wilcoxon(da_vals, comp_vals):
    diffs = da_vals - comp_vals
    n_wins = int(np.sum(diffs > 1e-9))
    n_ties = int(np.sum(np.abs(diffs) <= 1e-9))
    n_losses = int(np.sum(diffs < -1e-9))
    wtl = f"{n_wins}/{n_ties}/{n_losses}"
    
    nonzero = diffs[np.abs(diffs) > 1e-9]
    if len(nonzero) == 0:
        return wtl, 0.0, 1.0
    try:
        res = wilcoxon(nonzero, alternative='two-sided')
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat, p_val = 0.0, 1.0
    return wtl, stat, p_val

def main():
    raw_csv = 'results_stream/P2_raw.csv'
    df = pd.read_csv(raw_csv, keep_default_na=False)

    da_methods = ['DAECEBO2', 'DAECEBO2_PSI0']
    competitors = ['None', 'SMOTE', 'BorderlineSMOTE', 'ADASYN', 'G-SMOTE', 'CCR', 'A-SUWO', 'MLOS', 'ECEBO', 'ECEBO_PSI0']
    classifiers = ['CART', 'KNN', 'GaussianNB', 'SVC']
    metrics = ['Macro-F1', 'Macro-Gmean']
    datasets = sorted(df['dataset'].unique())

    # ==========================================
    # (a) Per-Dataset Paired Wilcoxon (10 Seeds)
    # ==========================================
    print("Computing (a) Per-dataset paired Wilcoxon over 10 seeds...")
    p1a_rows = []
    for dname in datasets:
        df_d = df[df['dataset'] == dname]
        for cname in classifiers:
            df_c = df_d[df_d['classifier'] == cname]
            for da_m in da_methods:
                for metric in metrics:
                    da_sub = df_c[df_c['method'] == da_m].sort_values('seed')
                    da_vals = da_sub[metric].values

                    temp_rows = []
                    for comp in competitors:
                        comp_sub = df_c[df_c['method'] == comp].sort_values('seed')
                        comp_vals = comp_sub[metric].values
                        wtl, stat, p_val = run_paired_wilcoxon(da_vals, comp_vals)
                        temp_rows.append({
                            'dataset': dname,
                            'classifier': cname,
                            'da_method': da_m,
                            'competitor': comp,
                            'metric': metric,
                            'n_seeds': len(da_vals),
                            'WTL': wtl,
                            'wilcoxon_stat': stat,
                            'raw_p_value': round(p_val, 5)
                        })
                    # Holm correction across competitors
                    raw_ps = [r['raw_p_value'] for r in temp_rows]
                    holm_ps = holm_bonferroni(raw_ps)
                    for r, hp in zip(temp_rows, holm_ps):
                        r['holm_p_value'] = round(hp, 5)
                        p1a_rows.append(r)

    df_p1a = pd.DataFrame(p1a_rows)
    df_p1a.to_csv('results_stream/P3_stats_per_dataset.csv', index=False)
    print("Saved results_stream/P3_stats_per_dataset.csv")

    # ========================================================
    # (b) Across-Dataset Analysis (Dataset Means, N=8 Datasets)
    # ========================================================
    print("Computing (b) Across-dataset Friedman & Wilcoxon on dataset means...")
    # Compute mean over 10 seeds per dataset
    all_methods = [m for m in df['method'].unique() if m != 'DAECEBO2_Sanity_K1_lam1']
    means_df = df.groupby(['dataset', 'classifier', 'method'])[metrics + ['Minority-F1', 'Precision', 'Recall', 'AUC']].mean().reset_index()

    p1b_stats = []
    p1b_ranks = []

    for cname in classifiers:
        sub_c = means_df[means_df['classifier'] == cname]
        for metric in metrics:
            piv = sub_c.pivot(index='dataset', columns='method', values=metric)
            piv = piv[all_methods].dropna()
            N_ds = len(piv)

            # Friedman test
            f_stat, f_pval = friedmanchisquare(*[piv[m].values for m in all_methods])

            # Average ranks (rank 1 = highest metric)
            # scipy rankdata ranks ascending, so we rank negative values
            ranks = np.zeros(piv.shape)
            for i in range(N_ds):
                ranks[i, :] = rankdata(-piv.iloc[i, :].values)
            avg_ranks = np.mean(ranks, axis=0)

            for m_name, ar in zip(all_methods, avg_ranks):
                p1b_ranks.append({
                    'classifier': cname,
                    'metric': metric,
                    'method': m_name,
                    'N_datasets': N_ds,
                    'avg_rank': round(ar, 3),
                    'friedman_Q': round(f_stat, 2),
                    'friedman_p_value': f"{f_pval:.4e}"
                })

            # Wilcoxon of DA vs each competitor on the dataset means (N=8 pairs)
            for da_m in da_methods:
                da_vals = piv[da_m].values
                temp_rows = []
                for comp in competitors:
                    comp_vals = piv[comp].values
                    wtl, stat, p_val = run_paired_wilcoxon(da_vals, comp_vals)
                    temp_rows.append({
                        'classifier': cname,
                        'da_method': da_m,
                        'competitor': comp,
                        'metric': metric,
                        'N_datasets': N_ds,
                        'WTL_datasets': wtl,
                        'wilcoxon_stat': stat,
                        'raw_p_value': round(p_val, 5)
                    })
                raw_ps = [r['raw_p_value'] for r in temp_rows]
                holm_ps = holm_bonferroni(raw_ps)
                for r, hp in zip(temp_rows, holm_ps):
                    r['holm_p_value'] = round(hp, 5)
                    p1b_stats.append(r)

    pd.DataFrame(p1b_ranks).to_csv('results_stream/P3_avg_ranks.csv', index=False)
    pd.DataFrame(p1b_stats).to_csv('results_stream/P3_stats_across_datasets.csv', index=False)
    print("Saved results_stream/P3_avg_ranks.csv and results_stream/P3_stats_across_datasets.csv")

    # ========================================================
    # (c) Per-Dataset CART Results (Mean +- Std over 10 Seeds)
    # ========================================================
    print("Computing (c) Per-dataset CART result tables...")
    cart_df = df[df['classifier'] == 'CART']
    cart_metrics = ['Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision']

    c_summary = cart_df.groupby(['dataset', 'method'])[cart_metrics].agg(['mean', 'std']).reset_index()
    c_rows = []
    for idx, row in c_summary.iterrows():
        dname = row['dataset'].values[0] if hasattr(row['dataset'], 'values') else row['dataset']
        mname = row['method'].values[0] if hasattr(row['method'], 'values') else row['method']
        r_dict = {'dataset': dname, 'method': mname}
        for m in cart_metrics:
            mean_val = row[(m, 'mean')]
            std_val = row[(m, 'std')]
            r_dict[f'{m}_mean'] = round(mean_val, 4)
            r_dict[f'{m}_std'] = round(std_val, 4)
            r_dict[f'{m}_formatted'] = f"{mean_val:.4f} +- {std_val:.4f}"
        c_rows.append(r_dict)

    df_p1c = pd.DataFrame(c_rows)
    df_p1c.to_csv('results_stream/P3_per_dataset_means.csv', index=False)
    print("Saved results_stream/P3_per_dataset_means.csv")

    # ========================================================
    # (d) Audit of DAECEBO2 vs ECEBO Ties & K=1, lambda=1
    # ========================================================
    print("Computing (d) Audit of DAECEBO2 vs ECEBO ties...")
    grid_csv = 'results_stream/P2_validation_grid.csv'
    df_grid = pd.read_csv(grid_csv)

    # Find where DAECEBO2_Literal validation selected K=1, lambda=1.0
    best_grid = df_grid[df_grid['rule'] == 'DAECEBO2_Literal'].sort_values(['dataset', 'val_score_mean'], ascending=[True, False])
    best_per_ds = best_grid.groupby('dataset').first().reset_index()
    print("Validation selections for DAECEBO2_Literal:")
    print(best_per_ds[['dataset', 'K_windows', 'lambd', 'delta_base', 'val_score_mean']])

    # Compare DAECEBO2 vs ECEBO across all rows in P2_raw.csv
    da_rows = df[df['method'] == 'DAECEBO2'].sort_values(['dataset', 'classifier', 'seed']).reset_index(drop=True)
    ec_rows = df[df['method'] == 'ECEBO'].sort_values(['dataset', 'classifier', 'seed']).reset_index(drop=True)

    tie_rows = []
    for i in range(len(da_rows)):
        d_r = da_rows.iloc[i]
        e_r = ec_rows.iloc[i]
        assert d_r['dataset'] == e_r['dataset'] and d_r['classifier'] == e_r['classifier'] and d_r['seed'] == e_r['seed']
        
        diff_gmean = abs(d_r['Macro-Gmean'] - e_r['Macro-Gmean'])
        diff_mf1 = abs(d_r['Macro-F1'] - e_r['Macro-F1'])
        diff_auc = abs(d_r['AUC'] - e_r['AUC'])
        is_exact_tie = (diff_gmean < 1e-9 and diff_mf1 < 1e-9 and diff_auc < 1e-9)

        tie_rows.append({
            'dataset': d_r['dataset'],
            'classifier': d_r['classifier'],
            'seed': d_r['seed'],
            'K_used': d_r['K_used'],
            'lambda_used': d_r['lambda_used'],
            'delta_base_used': d_r['delta_base_used'],
            'is_K1_lam1': bool(d_r['K_used'] == 1 and d_r['lambda_used'] == 1.0),
            'exact_tie_all_metrics': is_exact_tie,
            'da_gmean': d_r['Macro-Gmean'],
            'ec_gmean': e_r['Macro-Gmean'],
            'da_macro_f1': d_r['Macro-F1'],
            'ec_macro_f1': e_r['Macro-F1'],
            'da_auc': d_r['AUC'],
            'ec_auc': e_r['AUC']
        })

    df_tie = pd.DataFrame(tie_rows)
    df_tie.to_csv('results_stream/P3_tie_check.csv', index=False)
    print("Saved results_stream/P3_tie_check.csv")
    print(f"Total rows audited: {len(df_tie)}")
    print(f"Total exact ties: {df_tie['exact_tie_all_metrics'].sum()} / {len(df_tie)}")
    print(f"Total K=1, lambda=1 selections: {df_tie['is_K1_lam1'].sum()} / {len(df_tie)}")

if __name__ == '__main__':
    main()

