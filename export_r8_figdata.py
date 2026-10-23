"""
R8: Comprehensive Figure Data Export (no plotting).
Exports all numerical arrays, coordinate curves, matrices, and cluster metadata to results/figdata/.
"""

import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

from da_ecebo_and_ecebo_implementation import DAECEBO, ECEBO


def export_confusion_matrices():
    print("Exporting confusion matrices for seed 42 and 10-seed mean...")
    df_raw = pd.read_csv('results/multiseed_creditcard_results.csv')
    
    # 1. Seed 42
    df_42 = df_raw[df_raw['Seed'] == 42][['Method', 'TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'Minority-F1']].copy()
    df_42.to_csv('results/figdata/confusion_matrices_seed42.csv', index=False)
    
    # 2. 10-seed Mean
    cm_mean = df_raw.groupby('Method')[['TP', 'FP', 'FN', 'TN', 'Precision', 'Recall', 'Minority-F1']].mean().reset_index()
    cm_mean.to_csv('results/figdata/confusion_matrices_10seed_mean.csv', index=False)
    print("Saved results/figdata/confusion_matrices_seed42.csv and confusion_matrices_10seed_mean.csv")


def export_roc_pr_arrays():
    print("Exporting ROC and PR curve coordinates on full test set (71,202 rows)...")
    data = np.load('results/figdata/creditcard_all_models_seed42.npz')
    y_true = data['y_true']
    
    models = {
        'Baseline': data['probs_base'],
        'SMOTE': data['probs_sm'],
        'EC-EBO': data['probs_ec'],
        'DA-ECEBO (4,0.25)': data['probs_da4'],
        'DA-ECEBO (2,0.0)': data['probs_da2']
    }
    
    roc_dict = {'y_true': y_true}
    pr_dict = {'y_true': y_true}
    
    for name, probs in models.items():
        clean_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '_')
        fpr, tpr, thresh_roc = roc_curve(y_true, probs)
        prec, rec, thresh_pr = precision_recall_curve(y_true, probs)
        roc_dict[f'fpr_{clean_name}'] = fpr
        roc_dict[f'tpr_{clean_name}'] = tpr
        pr_dict[f'precision_{clean_name}'] = prec
        pr_dict[f'recall_{clean_name}'] = rec
        
    np.savez_compressed('results/figdata/roc_curves_full_test.npz', **roc_dict)
    np.savez_compressed('results/figdata/pr_curves_full_test.npz', **pr_dict)
    print("Saved results/figdata/roc_curves_full_test.npz and pr_curves_full_test.npz")


def export_synthetic_clusters_and_drift():
    print("Exporting clusters, drift vectors, and shuffled variants for Ecoli2, Haberman, Glass6...")
    export_datasets = ['ecoli2', 'haberman', 'glass6']
    
    for dname in export_datasets:
        df = pd.read_csv(f'datasets/{dname}.csv')
        target_col = [c for c in df.columns if any(k in c.lower() for k in ['class', 'target', 'survival'])][0]
        vc = df[target_col].value_counts()
        min_cls = vc.index[-1]
        y = (df[target_col] == min_cls).astype(int).values
        feat_cols = [c for c in df.columns if c != target_col]
        X = df[feat_cols].values.astype(float)
        t = df['Year'].values.astype(float) if 'Year' in df.columns else None
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        
        # 1. Standard raw order
        da_orig = DAECEBO(k_windows=4, lambd=0.25, random_state=42)
        X_res_orig, y_res_orig = da_orig.fit_resample(X_s, y, timestamps=t)
        
        np.savez_compressed(
            f'results/figdata/synthetic_export_{dname}_raw_order.npz',
            X_original=X_s,
            y_original=y,
            X_resampled=X_res_orig,
            y_resampled=y_res_orig,
            is_sorted_by_class=(dname in ['ecoli2', 'glass6'])
        )
        print(f"  Saved results/figdata/synthetic_export_{dname}_raw_order.npz")
        
        # 2. For ecoli2 and glass6 (which are strictly sorted by class in CSV), create a shuffled-order variant
        if dname in ['ecoli2', 'glass6']:
            rng = np.random.RandomState(42)
            perm = rng.permutation(len(X_s))
            X_shuf, y_shuf = X_s[perm], y[perm]
            da_shuf = DAECEBO(k_windows=4, lambd=0.25, random_state=42)
            X_res_shuf, y_res_shuf = da_shuf.fit_resample(X_shuf, y_shuf)
            
            np.savez_compressed(
                f'results/figdata/synthetic_export_{dname}_shuffled_variant.npz',
                X_original=X_shuf,
                y_original=y_shuf,
                X_resampled=X_res_shuf,
                y_resampled=y_res_shuf,
                permutation=perm
            )
            print(f"  Saved results/figdata/synthetic_export_{dname}_shuffled_variant.npz")


def main():
    os.makedirs('results/figdata', exist_ok=True)
    export_confusion_matrices()
    export_roc_pr_arrays()
    export_synthetic_clusters_and_drift()
    
    # Copy relevant CSVs to figdata
    copy_files = [
        ('results/multiseed_creditcard_results.csv', 'results/figdata/multiseed_creditcard_results.csv'),
        ('results/multiseed_creditcard_summary.csv', 'results/figdata/multiseed_creditcard_summary.csv'),
        ('results/sensitivity_analysis.csv', 'results/figdata/sensitivity_analysis.csv'),
        ('results/dataset_characteristics.csv', 'results/figdata/dataset_characteristics.csv'),
        ('results/haberman_chronological_results.csv', 'results/figdata/haberman_chronological_results.csv'),
        ('results/friedman_holm_results.csv', 'results/figdata/friedman_holm_results.csv'),
        ('results/multidataset_benchmark_f1.csv', 'results/figdata/multidataset_benchmark_f1.csv')
    ]
    for src, dst in copy_files:
        if os.path.exists(src):
            shutil.copyfile(src, dst)
            print(f"Copied {src} -> {dst}")
            
    print("\nR8 Figure Data Export completed successfully.")


if __name__ == '__main__':
    main()
