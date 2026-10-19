"""
Run P3: Comprehensive dataset analysis across all 15 datasets.
- Name, size, features, #maj, #min, IR, timestamp type (real | pseudo), row ordering findings.
- Analysis of Haberman year of operation attribute.
"""

import os
import glob
import pandas as pd
import numpy as np


def analyze_datasets():
    os.makedirs('results', exist_ok=True)
    
    # List all 15 datasets in canonical order
    dataset_names = [
        'creditcard', 'car-good', 'ecoli2', 'glass4', 'glass6',
        'haberman', 'htru2', 'page-blocks-1-3_vs_4', 'shuttle-c0-vs-c4',
        'vehicle0', 'vehicle1', 'vehicle2', 'vehicle3', 'wisconsin', 'yeast3'
    ]
    
    records = []
    
    for name in dataset_names:
        p_path = f'datasets/{name}.parquet'
        c_path = f'datasets/{name}.csv'
        fpath = p_path if os.path.exists(p_path) else c_path
        
        if fpath.endswith('.parquet'):
            df = pd.read_parquet(fpath)
        else:
            df = pd.read_csv(fpath)
            
        # Target column
        target_col = None
        for c in ['Class', 'class', 'target', 'label', 'Outcome', 'Survival']:
            if c in df.columns:
                target_col = c
                break
        if target_col is None:
            target_col = df.columns[-1]
            
        vc = df[target_col].value_counts()
        n_maj, n_min = int(vc.iloc[0]), int(vc.iloc[-1])
        maj_cls, min_cls = vc.index[0], vc.index[-1]
        ir = float(n_maj / n_min)
        
        n_samples = len(df)
        n_features = df.shape[1] - 1  # excluding target
        
        # Check time column
        time_col = None
        time_type = "pseudo (row order)"
        if name == 'creditcard' and 'Time' in df.columns:
            time_col = 'Time'
            time_type = "real (Time in seconds)"
            n_features -= 1  # excluding Time from features
        elif name == 'haberman':
            # Haberman has 'Year' (year of operation, 1958-1969)
            time_col = 'Year'
            time_type = "real (Year of operation, 1958-1969)"
            
        # Check row ordering: check if sorted by class or arbitrary
        y = (df[target_col] == min_cls).astype(int).values
        is_sorted_class = bool(np.all(np.diff(y) >= 0) or np.all(np.diff(y) <= 0))
        
        # Count transitions
        transitions = int(np.sum(np.diff(y) != 0))
        expected_trans = 2.0 * n_maj * n_min / n_samples
        
        if is_sorted_class:
            order_nature = "SORTED BY CLASS (Majority first, Minority last)"
        elif transitions < 0.2 * expected_trans:
            order_nature = "HIGHLY CLUSTERED / PARTIALLY ORDERED"
        else:
            order_nature = "SHUFFLED / MIXED"
            
        records.append({
            'Dataset': name,
            'Total_Samples': n_samples,
            'Feature_Count': n_features,
            'Majority_Count': n_maj,
            'Minority_Count': n_min,
            'Imbalance_Ratio': round(ir, 2),
            'Timestamp_Type': time_type,
            'Time_Column': time_col if time_col else "None (Index fallback)",
            'Is_Sorted_By_Class': is_sorted_class,
            'Class_Transitions': transitions,
            'Row_Order_Assessment': order_nature
        })
        
    df_res = pd.DataFrame(records)
    df_res.to_csv('results/dataset_characteristics.csv', index=False)
    print("Saved results/dataset_characteristics.csv")
    print(df_res.to_string())
    return df_res


if __name__ == '__main__':
    analyze_datasets()
