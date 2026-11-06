"""
Parse and verify all obtained KEEL datasets with robust categorical handling
"""
import os
import glob
import pandas as pd
import numpy as np

def load_keel_dat(filepath):
    data_lines = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        is_data = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            if line.lower().startswith('@data'):
                is_data = True
                continue
            if is_data:
                data_lines.append(line.split(','))
                
    if not data_lines:
        return None
        
    df = pd.DataFrame(data_lines)
    # Target is last column
    tcol = df.columns[-1]
    y_raw = df[tcol].astype(str).str.strip().values
    
    classes, counts = np.unique(y_raw, return_counts=True)
    if len(classes) != 2:
        return None
    min_class = classes[np.argmin(counts)]
    maj_class = classes[np.argmax(counts)]
    y = np.where(y_raw == min_class, 1, 0)
    
    # Feature columns
    feat_df = df.drop(columns=[tcol])
    for col in feat_df.columns:
        # try numeric conversion
        num_s = pd.to_numeric(feat_df[col], errors='coerce')
        if num_s.notna().all():
            feat_df[col] = num_s
        else:
            feat_df[col] = pd.factorize(feat_df[col])[0].astype(float)
            
    X = feat_df.values.astype(float)
    return X, y, min_class, maj_class

all_datasets = {}

# 1. Local datasets in datasets/
local_files = [
    'car-good.csv', 'ecoli2.csv', 'glass4.csv', 'glass6.csv', 'haberman.csv',
    'page-blocks-1-3_vs_4.csv', 'shuttle-c0-vs-c4.csv', 'vehicle0.csv',
    'vehicle1.csv', 'vehicle2.csv', 'vehicle3.csv', 'wisconsin.csv', 'yeast3.csv'
]
for fname in local_files:
    p = os.path.join('datasets', fname)
    if os.path.exists(p):
        df = pd.read_csv(p)
        tcol = [c for c in df.columns if 'class' in c.lower() or c == df.columns[-1]][0]
        y_raw = df[tcol].astype(str).str.strip().values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        
        feat_df = df.drop(columns=[tcol])
        for col in feat_df.columns:
            num_s = pd.to_numeric(feat_df[col], errors='coerce')
            if num_s.notna().all():
                feat_df[col] = num_s
            else:
                feat_df[col] = pd.factorize(feat_df[col])[0].astype(float)
        X = feat_df.values.astype(float)
        dname = fname.replace('.csv', '')
        all_datasets[dname] = (X, y)

# 2. Datasets from datasets_keel/
for f in glob.glob('datasets_keel/*.dat'):
    dname = os.path.basename(f).replace('.dat', '')
    if dname not in all_datasets:
        res = load_keel_dat(f)
        if res is not None:
            X, y, min_c, maj_c = res
            all_datasets[dname] = (X, y)

print(f"Total verified binary classification datasets: {len(all_datasets)}")
print("\nDataset Summary:")
records = []
for name, (X, y) in sorted(all_datasets.items()):
    n_maj = (y == 0).sum()
    n_min = (y == 1).sum()
    ir = n_maj / n_min if n_min > 0 else 0
    records.append({
        'Dataset': name,
        'Samples': len(X),
        'Features': X.shape[1],
        'Majority': n_maj,
        'Minority': n_min,
        'IR': round(ir, 2)
    })
df_info = pd.DataFrame(records)
print(df_info.to_string())
