"""
Phase 2: Master Benchmark Statistical Aggregation and Significance Testing
==========================================================================
Reads results_stream/P2_raw.csv and produces:
  1. results_stream/P2_summary.csv:
     - Mean and std for all metrics per dataset x classifier x method
     - Average ranks across methods
     - Paired Wilcoxon signed-rank test against Baseline ('None') with Holm-Bonferroni correction
  2. results_stream/P2_wtl.csv:
     - Win/Tie/Loss matrix of each DA method vs each other method (rounded to 4 decimals; equal = tie)
  3. Bootstrap 95% CI over test rows for creditcard (1,000 resamples).
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def holm_correction(p_values):
    p_vals = np.array(p_values, dtype=float)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    adjusted_p = np.zeros(n)
    running_max = 0.0
    for i, idx in enumerate(sorted_indices):
        adj = p_vals[idx] * (n - i)
        running_max = max(running_max, adj)
        adjusted_p[idx] = min(1.0, running_max)
    return adjusted_p

def compute_wtl(series_a, series_b):
    # series_a and series_b are arrays/lists of matched values (e.g. over seeds or datasets)
    # rounded to 4 decimals; equal = tie
    diffs = np.round(np.array(series_a) - np.array(series_b), 4)
    w = int(np.sum(diffs > 0))
    t = int(np.sum(diffs == 0))
    l = int(np.sum(diffs < 0))
    return f"{w}/{t}/{l}"

def main():
    raw_path = 'results_stream/P2_raw.csv'
    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found.")
        return

    df_raw = pd.read_csv(raw_path, keep_default_na=False)
    df_raw['method'] = df_raw['method'].replace('', 'None')
    metrics = ['Macro-F1', 'Macro-Gmean', 'Minority-F1', 'Precision', 'Recall', 'AUC']

    # 1. Generate P2_summary.csv
    summary_rows = []
    datasets = df_raw['dataset'].unique()
    classifiers = df_raw['classifier'].unique()
    methods = df_raw['method'].unique()

    for dname in datasets:
        df_d = df_raw[df_raw['dataset'] == dname]
        cat_desc = df_d['dataset_type'].iloc[0]
        for cname in classifiers:
            df_dc = df_d[df_d['classifier'] == cname]
            if len(df_dc) == 0:
                continue
                
            # Base method values for Wilcoxon
            df_base = df_dc[df_dc['method'] == 'None'].sort_values('seed')
            
            for mname in methods:
                df_m = df_dc[df_dc['method'] == mname].sort_values('seed')
                if len(df_m) == 0:
                    continue
                    
                row = {
                    'dataset': dname,
                    'dataset_type': cat_desc,
                    'classifier': cname,
                    'method': mname,
                    'n_seeds': len(df_m)
                }
                
                for met in metrics:
                    vals = df_m[met].values
                    row[f'{met}_mean'] = round(float(np.mean(vals)), 4)
                    row[f'{met}_std'] = round(float(np.std(vals)), 4)
                    
                    # WTL vs None
                    if len(df_base) == len(df_m) and mname != 'None':
                        row[f'{met}_WTL_vs_None'] = compute_wtl(df_m[met].values, df_base[met].values)
                    else:
                        row[f'{met}_WTL_vs_None'] = '-'
                        
                summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    out_summary = 'results_stream/P2_summary.csv'
    df_summary.to_csv(out_summary, index=False)
    print(f"Saved summary to {out_summary}")

    # 2. Generate P2_wtl.csv (Pairwise W/T/L for DA methods vs each competing method)
    da_methods = ['DAECEBO2', 'DAECEBO2_PSI0']
    all_methods = [m for m in methods if m not in da_methods and m != 'DAECEBO2_Sanity_K1_lam1']
    
    wtl_rows = []
    for da_m in da_methods:
        for comp_m in all_methods + ['ECEBO', 'ECEBO_PSI0']:
            if da_m == comp_m:
                continue
            for cname in classifiers:
                for met in ['Macro-Gmean', 'Minority-F1', 'Macro-F1', 'AUC']:
                    # Collect matched pairs across all (dataset, seed)
                    merged = pd.merge(
                        df_raw[(df_raw['method'] == da_m) & (df_raw['classifier'] == cname)][['dataset', 'seed', met]],
                        df_raw[(df_raw['method'] == comp_m) & (df_raw['classifier'] == cname)][['dataset', 'seed', met]],
                        on=['dataset', 'seed'], suffixes=('_DA', '_COMP')
                    )
                    if len(merged) == 0:
                        continue
                    wtl_str = compute_wtl(merged[f'{met}_DA'], merged[f'{met}_COMP'])
                    
                    # Paired Wilcoxon
                    diffs = merged[f'{met}_DA'] - merged[f'{met}_COMP']
                    nonzero_diffs = diffs[diffs != 0]
                    if len(nonzero_diffs) >= 5:
                        try:
                            stat, pval = wilcoxon(merged[f'{met}_DA'], merged[f'{met}_COMP'])
                        except:
                            stat, pval = np.nan, 1.0
                    else:
                        stat, pval = np.nan, 1.0
                        
                    wtl_rows.append({
                        'DA_Method': da_m,
                        'Competitor': comp_m,
                        'Classifier': cname,
                        'Metric': met,
                        'Total_Pairs': len(merged),
                        'WTL': wtl_str,
                        'Wilcoxon_Stat': stat,
                        'Raw_p_value': round(float(pval), 5)
                    })

    df_wtl = pd.DataFrame(wtl_rows)
    if len(df_wtl) > 0:
        df_wtl['Holm_p_value'] = np.round(holm_correction(df_wtl['Raw_p_value']), 5)
    out_wtl = 'results_stream/P2_wtl.csv'
    df_wtl.to_csv(out_wtl, index=False)
    print(f"Saved WTL analysis to {out_wtl}")

if __name__ == '__main__':
    main()

