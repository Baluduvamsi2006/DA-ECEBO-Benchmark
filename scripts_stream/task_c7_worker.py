import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath('.'))

def load_data(name):
    if name == 'synth_sea_p5':
        df = pd.read_parquet('datasets_stream/synth_sea_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2']].values.astype(np.float64)
    elif name == 'synth_hyperplane_p5':
        df = pd.read_parquet('datasets_stream/synth_hyperplane_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
    elif name == 'synth_random_rbf_p5':
        df = pd.read_parquet('datasets_stream/synth_random_rbf_p5.parquet')
        y = df['Class'].values.astype(int)
        X = df[['f0', 'f1', 'f2', 'f3', 'f4']].values.astype(np.float64)
    elif name == 'electricity_ir20':
        df = pd.read_parquet('datasets_stream/electricity_ir20.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
    elif name == 'electricity_ir100':
        df = pd.read_parquet('datasets_stream/electricity_ir100.parquet')
        y = np.where(df['class'].values == 'UP', 1, 0)
        feat_cols = [c for c in df.columns if c not in ['class', 'date', 'period']]
        X = df[feat_cols].values.astype(np.float64)
    elif name == 'creditcard':
        df = pd.read_parquet('datasets/creditcard.parquet')
        y = df['Class'].values.astype(int)
        X = df.drop(columns=['Class', 'Time']).values.astype(np.float64)
    else:
        raise ValueError(f'Unknown dataset {name}')
        
    n_tr = int(len(X) * 0.75)
    return X[:n_tr], y[:n_tr]

def run_method(name, X, y, seed=42):
    if name == 'None':
        return X, y
    elif name == 'SMOTE':
        from imblearn.over_sampling import SMOTE
        k = min(5, (y == 1).sum() - 1)
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    elif name == 'BorderlineSMOTE':
        from imblearn.over_sampling import BorderlineSMOTE
        k = min(5, (y == 1).sum() - 1)
        return BorderlineSMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    elif name == 'ADASYN':
        from imblearn.over_sampling import ADASYN
        k = min(5, (y == 1).sum() - 1)
        return ADASYN(n_neighbors=k, random_state=seed).fit_resample(X, y)
    elif name == 'G-SMOTE':
        import smote_variants as sv
        return sv.G_SMOTE(random_state=seed).sample(X, y)
    elif name == 'CCR':
        import smote_variants as sv
        return sv.CCR(random_state=seed).sample(X, y)
    elif name == 'A-SUWO':
        import smote_variants as sv
        return sv.A_SUWO(random_state=seed).sample(X, y)
    elif name == 'MLOS':
        from scripts_stream.MLOS import MLOS
        sw = MLOS(minClass=1, len=1)
        n_maj = int(np.sum(y == 0))
        n_min = int(np.sum(y == 1))
        return sw.fit_Sample(X, y, n_maj - n_min)
    elif name == 'ECEBO':
        from da_ecebo_and_ecebo_implementation import ECEBO
        return ECEBO(random_state=seed).fit_resample(X, y)
    elif name == 'ECEBO_PSI0':
        from ecebo_psi0_variant import ECEBO_PSI0
        return ECEBO_PSI0(random_state=seed).fit_resample(X, y)
    elif name == 'DAECEBO2':
        from da_ecebo_v2_candidate import DAECEBO2
        return DAECEBO2(random_state=seed).fit_resample(X, y)
    elif name == 'DAECEBO2_PSI0':
        from da_ecebo_v2_psi0 import DAECEBO2_PSI0
        return DAECEBO2_PSI0(random_state=seed).fit_resample(X, y)
    else:
        raise ValueError(f'Unknown method {name}')

if __name__ == '__main__':
    dname = sys.argv[1]
    mname = sys.argv[2]
    
    t0 = time.time()
    try:
        X_tr, y_tr = load_data(dname)
        X_res, y_res = run_method(mname, X_tr, y_tr, seed=42)
        elapsed = time.time() - t0
        print(f"RESULT:{elapsed:.4f}:SUCCESS")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"RESULT:{elapsed:.4f}:ERROR:{str(e)[:100]}")

