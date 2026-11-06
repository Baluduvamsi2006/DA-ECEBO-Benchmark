"""
Worker process for R4 Timing Benchmark:
Measures import_seconds and sampler_seconds around fit_resample only.
"""
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
    elif name == 'ecoli2':
        df = pd.read_csv('datasets/Ecoli2.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
    elif name == 'yeast3':
        df = pd.read_csv('datasets/yeast3.csv')
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        y_raw = df.iloc[:, -1].values
        classes, counts = np.unique(y_raw, return_counts=True)
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        X = df.iloc[:, :-1].values.astype(np.float64)
    else:
        raise ValueError(f"Unknown dataset {name}")
        
    n_tr = int(len(X) * 0.75)
    return X[:n_tr], y[:n_tr]

def main():
    dname = sys.argv[1]
    mname = sys.argv[2]
    seed = 42

    # Measure import time
    t_imp_0 = time.perf_counter()
    sampler_class = None
    if mname == 'None':
        sampler_class = None
    elif mname == 'SMOTE':
        from imblearn.over_sampling import SMOTE
        sampler_class = SMOTE
    elif mname == 'BorderlineSMOTE':
        from imblearn.over_sampling import BorderlineSMOTE
        sampler_class = BorderlineSMOTE
    elif mname == 'ADASYN':
        from imblearn.over_sampling import ADASYN
        sampler_class = ADASYN
    elif mname == 'G-SMOTE':
        import smote_variants as sv
        sampler_class = sv.G_SMOTE
    elif mname == 'CCR':
        import smote_variants as sv
        sampler_class = sv.CCR
    elif mname == 'A-SUWO':
        import smote_variants as sv
        sampler_class = sv.A_SUWO
    elif mname == 'MLOS':
        from scripts_stream.MLOS import MLOS
        sampler_class = MLOS
    elif mname == 'ECEBO':
        from da_ecebo_and_ecebo_implementation import ECEBO
        sampler_class = ECEBO
    elif mname == 'ECEBO_PSI0':
        from ecebo_psi0_variant import ECEBO_PSI0
        sampler_class = ECEBO_PSI0
    elif mname == 'DAECEBO2':
        from da_ecebo_v2_candidate import DAECEBO2
        sampler_class = DAECEBO2
    elif mname == 'DAECEBO2_PSI0':
        from da_ecebo_v2_psi0 import DAECEBO2_PSI0
        sampler_class = DAECEBO2_PSI0
    else:
        raise ValueError(f"Unknown method {mname}")
    t_imp_end = time.perf_counter()
    import_seconds = t_imp_end - t_imp_0

    # Load data
    X_tr, y_tr = load_data(dname)
    n_min = int(np.sum(y_tr == 1))
    n_maj = int(np.sum(y_tr == 0))

    # Cap majority at 20,000 for MLOS to prevent memory explosion
    if mname == 'MLOS' and n_maj > 20000:
        rng = np.random.RandomState(seed)
        maj_idx = np.where(y_tr == 0)[0]
        min_idx = np.where(y_tr == 1)[0]
        sub_maj = rng.choice(maj_idx, size=20000, replace=False)
        keep_idx = np.concatenate([sub_maj, min_idx])
        X_tr, y_tr = X_tr[keep_idx], y_tr[keep_idx]
        n_maj = 20000

    # Instantiation
    k_nn = min(5, n_min - 1)
    if mname == 'None':
        inst = None
    elif mname in ['SMOTE', 'BorderlineSMOTE']:
        inst = sampler_class(k_neighbors=k_nn, random_state=seed)
    elif mname == 'ADASYN':
        inst = sampler_class(n_neighbors=k_nn, random_state=seed)
    elif mname in ['G-SMOTE', 'CCR', 'A-SUWO']:
        inst = sampler_class(random_state=seed)
    elif mname == 'MLOS':
        inst = sampler_class(minClass=1, len=1)
    elif mname in ['ECEBO', 'ECEBO_PSI0']:
        inst = sampler_class(random_state=seed)
    elif mname in ['DAECEBO2', 'DAECEBO2_PSI0']:
        inst = sampler_class(random_state=seed, delta_base=0.0)

    # Time sampler ONLY around fit_resample / sample
    t_samp_0 = time.perf_counter()
    if mname == 'None':
        X_res, y_res = X_tr, y_tr
    elif mname == 'MLOS':
        delta_n = max(0, n_maj - n_min)
        X_res, y_res = inst.fit_Sample(X_tr, y_tr, delta_n)
    elif hasattr(inst, 'fit_resample'):
        X_res, y_res = inst.fit_resample(X_tr, y_tr)
    elif hasattr(inst, 'sample'):
        X_res, y_res = inst.sample(X_tr, y_tr)
    t_samp_end = time.perf_counter()
    sampler_seconds = t_samp_end - t_samp_0

    print(f"RESULT:{sampler_seconds:.4f}:{import_seconds:.4f}:SUCCESS")

if __name__ == '__main__':
    main()

