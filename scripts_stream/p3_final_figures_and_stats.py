"""
Phase 3 Final Statistics & Figures Generator
============================================
Generates:
  - P3_stats_per_dataset_auto.csv
  - P3_avg_ranks_auto.csv
  - figs_stream/P3_F1_delta_metrics.pdf
  - figs_stream/P3_F2_wtl_matrix.pdf
  - figs_stream/P3_F5_block_performance.pdf
  - results_stream/P3_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
import seaborn as sns

def run_stats_and_figures():
    raw_csv = 'results_stream/P3_raw.csv'
    if not os.path.exists(raw_csv):
        print(f"{raw_csv} does not exist.")
        return

    df_raw = pd.read_csv(raw_csv)
    # Check if DAECEBO2_AUTO is in df_raw
    if 'DAECEBO2_AUTO' not in df_raw['method'].values:
        print("DAECEBO2_AUTO not yet present in P3_raw.csv. Waiting...")
        return

    print("Generating comprehensive Phase 3 statistics...")
    # 1. Per-dataset Wilcoxon over 10 seeds (CART classifier)
    df_cart = df_raw[df_raw['classifier'] == 'CART'].copy()
    datasets = df_cart['dataset'].unique()
    methods = [m for m in df_cart['method'].unique() if m != 'DAECEBO2_AUTO']

    stat_rows = []
    for dname in datasets:
        sub_d = df_cart[df_cart['dataset'] == dname]
        da_sub = sub_d[sub_d['method'] == 'DAECEBO2_AUTO'].sort_values(by='seed')
        if len(da_sub) < 10:
            continue
        da_f1 = da_sub['Macro-F1'].values
        da_gm = da_sub['Macro-Gmean'].values

        for m in methods:
            m_sub = sub_d[sub_d['method'] == m].sort_values(by='seed')
            if len(m_sub) < 10:
                continue
            m_f1 = m_sub['Macro-F1'].values
            m_gm = m_sub['Macro-Gmean'].values

            # Wilcoxon for Macro-F1
            diff_f1 = da_f1 - m_f1
            w_f1 = int(np.sum(diff_f1 > 0))
            t_f1 = int(np.sum(diff_f1 == 0))
            l_f1 = int(np.sum(diff_f1 < 0))
            try:
                p_f1 = wilcoxon(da_f1, m_f1).pvalue if np.any(diff_f1 != 0) else 1.0
            except Exception:
                p_f1 = 1.0

            # Wilcoxon for Macro-Gmean
            diff_gm = da_gm - m_gm
            w_gm = int(np.sum(diff_gm > 0))
            t_gm = int(np.sum(diff_gm == 0))
            l_gm = int(np.sum(diff_gm < 0))
            try:
                p_gm = wilcoxon(da_gm, m_gm).pvalue if np.any(diff_gm != 0) else 1.0
            except Exception:
                p_gm = 1.0

            stat_rows.append({
                'dataset': dname,
                'competitor': m,
                'F1_W_T_L': f"{w_f1}/{t_f1}/{l_f1}",
                'F1_pvalue': round(p_f1, 5),
                'Gmean_W_T_L': f"{w_gm}/{t_gm}/{l_gm}",
                'Gmean_pvalue': round(p_gm, 5),
                'DA_mean_Gmean': round(np.mean(da_gm), 4),
                'Competitor_mean_Gmean': round(np.mean(m_gm), 4),
                'Diff_Gmean': round(np.mean(da_gm) - np.mean(m_gm), 4)
            })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv('results_stream/P3_stats_per_dataset_auto.csv', index=False)
    print("Saved results_stream/P3_stats_per_dataset_auto.csv")

    # 2. Across-dataset average ranks & Friedman
    key_methods = ['None', 'SMOTE_tuned', 'BorderlineSMOTE_tuned', 'ADASYN_tuned',
                   'G-SMOTE_tuned', 'CCR_tuned', 'A-SUWO_tuned', 'MLOS_Fixed_tuned',
                   'ECEBO_tuned', 'ECEBO_PSI0_tuned', 'DAECEBO2_AUTO']
    
    # Filter for available methods
    avail_methods = [m for m in key_methods if m in df_cart['method'].values]
    mean_table = df_cart[df_cart['method'].isin(avail_methods)].groupby(['dataset', 'method'])['Macro-Gmean'].mean().unstack()
    ranks = mean_table.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean(axis=0).sort_values()

    df_ranks = pd.DataFrame({
        'method': avg_ranks.index,
        'avg_rank': [round(r, 3) for r in avg_ranks.values]
    })
    df_ranks.to_csv('results_stream/P3_avg_ranks_auto.csv', index=False)
    print("Saved results_stream/P3_avg_ranks_auto.csv")

    # 3. Figure P3_F1: Delta metrics vs None
    print("Generating Figure P3_F1...")
    focus_methods = ['SMOTE_tuned', 'ECEBO_tuned', 'ECEBO_PSI0_tuned', 'DAECEBO2_AUTO']
    focus_avail = [m for m in focus_methods if m in df_cart['method'].values]
    
    delta_rows = []
    for dname in datasets:
        sub_d = df_cart[df_cart['dataset'] == dname]
        none_sub = sub_d[sub_d['method'] == 'None'].sort_values(by='seed')
        if len(none_sub) < 10:
            continue
        none_f1 = none_sub['Macro-F1'].values
        none_gm = none_sub['Macro-Gmean'].values

        for m in focus_avail:
            m_sub = sub_d[sub_d['method'] == m].sort_values(by='seed')
            if len(m_sub) < 10:
                continue
            delta_f1 = m_sub['Macro-F1'].values - none_f1
            delta_gm = m_sub['Macro-Gmean'].values - none_gm
            for s_idx, s in enumerate(m_sub['seed'].values):
                delta_rows.append({
                    'dataset': dname,
                    'method': m,
                    'seed': s,
                    'delta_Macro_F1': delta_f1[s_idx],
                    'delta_Macro_Gmean': delta_gm[s_idx]
                })

    df_delta = pd.DataFrame(delta_rows)
    if not df_delta.empty:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
        sns.barplot(data=df_delta, x='dataset', y='delta_Macro_Gmean', hue='method', palette='Set2', ax=ax1, errorbar=None)
        sns.stripplot(data=df_delta, x='dataset', y='delta_Macro_Gmean', hue='method', dodge=True, color='black', alpha=0.5, size=4, ax=ax1, legend=False)
        ax1.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax1.set_title(r"$\Delta$ Macro G-mean vs None (CART)", fontsize=12, fontweight='bold')
        ax1.set_ylabel(r"$\Delta$ Macro G-mean", fontsize=10)
        ax1.legend(title='Method', fontsize=9)

        sns.barplot(data=df_delta, x='dataset', y='delta_Macro_F1', hue='method', palette='Set2', ax=ax2, errorbar=None)
        sns.stripplot(data=df_delta, x='dataset', y='delta_Macro_F1', hue='method', dodge=True, color='black', alpha=0.5, size=4, ax=ax2, legend=False)
        ax2.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        ax2.set_title(r"$\Delta$ Macro F1 vs None (CART)", fontsize=12, fontweight='bold')
        ax2.set_ylabel(r"$\Delta$ Macro F1", fontsize=10)
        ax2.set_xlabel("Dataset", fontsize=10)
        ax2.legend(title='Method', fontsize=9)

        plt.suptitle("P3_F1: Performance Gains vs None Across Datasets", fontsize=14, fontweight='bold')
        plt.savefig('figs_stream/P3_F1_delta_metrics.pdf', bbox_inches='tight')
        plt.savefig('figs_stream/P3_F1_delta_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved figs_stream/P3_F1_delta_metrics.pdf")

    # 4. Figure P3_F2: W/T/L Matrix Heatmap
    print("Generating Figure P3_F2...")
    if not df_stats.empty:
        # Create W-L score matrix
        pivot_wl = df_stats.pivot(index='dataset', columns='competitor', values='Gmean_W_T_L')
        # Extract W - L as numeric
        diff_matrix = pd.DataFrame(index=pivot_wl.index, columns=pivot_wl.columns)
        for r in pivot_wl.index:
            for c in pivot_wl.columns:
                val = str(pivot_wl.loc[r, c])
                if '/' in val:
                    w, t, l = [int(x) for x in val.split('/')]
                    diff_matrix.loc[r, c] = w - l
                else:
                    diff_matrix.loc[r, c] = 0
        diff_matrix = diff_matrix.astype(float)

        plt.figure(figsize=(12, 6), constrained_layout=True)
        sns.heatmap(diff_matrix, annot=pivot_wl.values, fmt='', cmap='RdYlGn', center=0, cbar=True,
                    cbar_kws={'label': 'Net Wins (Wins - Losses)'})
        plt.title("P3_F2: Win / Tie / Loss Matrix of DAECEBO2_AUTO vs Competitors (CART G-mean)", fontsize=12, fontweight='bold')
        plt.ylabel("Dataset", fontsize=10)
        plt.xlabel("Competitor Method", fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.savefig('figs_stream/P3_F2_wtl_matrix.pdf', bbox_inches='tight')
        plt.savefig('figs_stream/P3_F2_wtl_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved figs_stream/P3_F2_wtl_matrix.pdf")

    # 5. Figure P3_F5: Block Performance Across Future Horizon
    print("Generating Figure P3_F5...")
    blocks_file = 'results_stream/P2_blocks.csv'
    if os.path.exists(blocks_file):
        df_b = pd.read_csv(blocks_file)
        # Focus on stream datasets and CART classifier
        focus_methods = ['None', 'SMOTE', 'ECEBO', 'DAECEBO2']
        sub_b = df_b[(df_b['classifier'] == 'CART') & (df_b['method'].isin(focus_methods))].copy()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True, constrained_layout=True)
        axes = axes.flatten()
        stream_dsets = ['creditcard', 'electricity_ir20', 'electricity_ir100', 'synth_sea_p5', 'synth_hyperplane_p5', 'synth_random_rbf_p5']
        
        for idx, dname in enumerate(stream_dsets):
            ax = axes[idx]
            d_sub = sub_b[sub_b['dataset'] == dname]
            if not d_sub.empty:
                sns.lineplot(data=d_sub, x='block_id', y='Macro-Gmean', hue='method', marker='o', ax=ax, palette='Set1', errorbar=('ci', 95))
                ax.set_title(f"{dname}", fontsize=11, fontweight='bold')
                ax.set_xlabel("Horizon Block (1 to 5)", fontsize=9)
                ax.set_ylabel("Macro G-mean", fontsize=9)
                ax.set_ylim(-0.05, 1.05)
                if idx != 0:
                    ax.legend().remove()
                else:
                    ax.legend(title='Method', fontsize=8)
            else:
                ax.axis('off')

        plt.suptitle("P3_F5: Per-Block Performance Degradation Across Horizon (CART G-mean)", fontsize=13, fontweight='bold')
        plt.savefig('figs_stream/P3_F5_block_performance.pdf', bbox_inches='tight')
        plt.savefig('figs_stream/P3_F5_block_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved figs_stream/P3_F5_block_performance.pdf")

    print("\nAll Phase 3 statistics and figures generated successfully!")

if __name__ == '__main__':
    run_stats_and_figures()

