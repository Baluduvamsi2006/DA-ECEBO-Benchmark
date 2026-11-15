"""
Phase 3 Part 5: Figures P3_F3 and P3_F4 Generator
=================================================
P3_F3: Validation score surface heatmaps (K x lambda) for each dataset from P3_validation_grid.csv.
P3_F4: Step audit bar chart (acceptance rate & copy share) literal vs psi0, ECEBO vs DAECEBO2.

Saves vector PDFs to figs_stream/P3_F3_validation_heatmaps.pdf and figs_stream/P3_F4_step_audit.pdf.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_f3_heatmaps():
    grid_csv = 'results_stream/P3_validation_grid.csv'
    if not os.path.exists(grid_csv):
        print(f"{grid_csv} does not exist yet.")
        return

    df = pd.read_csv(grid_csv)
    # Filter for DAECEBO2 runs with K_windows and lambd
    df_da = df[df['method'].str.startswith('DAECEBO2')].copy()
    if df_da.empty:
        return

    # Convert numeric
    df_da['K_windows'] = pd.to_numeric(df_da['K_windows'], errors='coerce')
    df_da['lambd'] = pd.to_numeric(df_da['lambd'], errors='coerce')
    df_da['val_score'] = pd.to_numeric(df_da['val_score'], errors='coerce')
    df_da = df_da.dropna(subset=['K_windows', 'lambd', 'val_score'])

    datasets = df_da['dataset'].unique()
    n_dsets = len(datasets)
    ncols = 4
    nrows = int(np.ceil(n_dsets / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), constrained_layout=True)
    axes = np.array(axes).flatten()

    for idx, dname in enumerate(datasets):
        ax = axes[idx]
        sub = df_da[df_da['dataset'] == dname]
        pivot = sub.groupby(['K_windows', 'lambd'])['val_score'].mean().unstack()

        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar=True,
                    cbar_kws={'label': 'Validation Score'})
        ax.set_title(f"{dname}", fontsize=11, fontweight='bold')
        ax.set_xlabel(r"Drift Memory Rate $\lambda$", fontsize=9)
        ax.set_ylabel(r"Windows $K$", fontsize=9)

    # Hide extra subplots
    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("P3_F3: Validation Score Surfaces ($K \\times \\lambda$) Across Datasets",
                 fontsize=14, fontweight='bold', y=1.02)
    
    out_pdf = 'figs_stream/P3_F3_validation_heatmaps.pdf'
    out_png = 'figs_stream/P3_F3_validation_heatmaps.png'
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_pdf} and {out_png}")

def generate_f4_step_audit():
    # R1 acceptance data
    r1_csv = 'results_stream/R1_acceptance.csv'
    if not os.path.exists(r1_csv):
        print(f"{r1_csv} does not exist.")
        return

    df_r1 = pd.read_csv(r1_csv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Acceptance Rate
    sns.barplot(data=df_r1, x='dataset', y='acceptance_rate', hue='model_name', palette='Blues_d', ax=ax1)
    ax1.set_title("Proposal Acceptance Rate", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Acceptance Rate (Accepted / Proposals)", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.legend(title='Method', fontsize=8)

    # Duplicate Share
    sns.barplot(data=df_r1, x='dataset', y='exact_copies_share', hue='model_name', palette='Reds_d', ax=ax2)
    ax2.set_title("Duplicate / Copy Share", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Copy Share (Exact Copies / Synthetic)", fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.legend(title='Method', fontsize=8)

    plt.suptitle("P3_F4: Step Audit Comparison (Literal vs PSI0)", fontsize=13, fontweight='bold')
    out_pdf = 'figs_stream/P3_F4_step_audit.pdf'
    out_png = 'figs_stream/P3_F4_step_audit.png'
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_pdf} and {out_png}")

def main():
    os.makedirs('figs_stream', exist_ok=True)
    generate_f3_heatmaps()
    generate_f4_step_audit()

if __name__ == '__main__':
    main()
