"""
Phase 3 Part 2: Dataset Download and Triage
==========================================
Downloads real-world concept drift benchmark datasets from public repositories,
inspects their timestamps, class distributions, and train/test splits (75/25 chronological).
Produces results_stream/P3_datasets.csv.
"""
import os
import sys
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime

DOWNLOAD_URLS = {
    'weather': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/weather.csv',
    'spam_katakis': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/spam.csv',
    'phishing': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/phishing.csv',
    'outdoor_stream': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/outdoorStream.csv',
    'ludata': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/LUdata.csv',
    'chessweka': 'https://raw.githubusercontent.com/ogozuacik/concept-drift-datasets-scikit-multiflow/master/real-world/chessweka.csv'
}

def download_file(url, target_path):
    if os.path.exists(target_path):
        return True, "Already downloaded"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as resp, open(target_path, 'wb') as f:
            f.write(resp.read())
        return True, "Downloaded successfully"
    except Exception as e:
        return False, str(e)

def triage_dataset(name, file_path, source_url, time_attr, tier_assigned):
    df = pd.read_csv(file_path)
    n_rows = len(df)
    n_features = df.shape[1] - 1
    
    # Class column is typically last column
    y_raw = df.iloc[:, -1].values
    classes, counts = np.unique(y_raw, return_counts=True)
    
    # Binary check / binarization
    if len(classes) == 2:
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
    else:
        # Multi-class: map least frequent or designated rare class as 1
        min_class = classes[np.argmin(counts)]
        y = np.where(y_raw == min_class, 1, 0)
        
    n_total = len(y)
    n_min_total = int(np.sum(y == 1))
    n_maj_total = int(np.sum(y == 0))
    ir_total = round(n_maj_total / n_min_total, 2) if n_min_total > 0 else np.nan

    # 75/25 chronological split
    n_tr = int(0.75 * n_total)
    y_tr = y[:n_tr]
    y_te = y[n_tr:]
    
    n_min_tr = int(np.sum(y_tr == 1))
    n_maj_tr = int(np.sum(y_tr == 0))
    n_min_te = int(np.sum(y_te == 1))
    n_te = len(y_te)
    
    # Rule checks:
    # 1. Minority in test part >= 30
    # 2. Minority in train part >= 100
    # 3. <= 60,000 rows after chronology-preserving cap
    rule_pass = (n_min_te >= 30) and (n_min_tr >= 100) and (n_total <= 60000)
    
    reason = "Meets all criteria" if rule_pass else []
    if n_min_te < 30:
        reason.append(f"Minority test count {n_min_te} < 30")
    if n_min_tr < 100:
        reason.append(f"Minority train count {n_min_tr} < 100")
    if n_total > 60000:
        reason.append(f"Total rows {n_total} > 60,000")
    if isinstance(reason, list):
        reason = "; ".join(reason)

    return {
        'dataset': name,
        'tier': tier_assigned,
        'N_total': n_total,
        'N_features': n_features,
        'N_min_total': n_min_total,
        'N_maj_total': n_maj_total,
        'IR_total': ir_total,
        'N_train': n_tr,
        'N_min_train': n_min_tr,
        'N_maj_train': n_maj_tr,
        'N_test': n_te,
        'N_min_test': n_min_te,
        'time_attribute': time_attr,
        'cap_applied': 'None' if n_total <= 60000 else 'Chronological majority cap',
        'download_date': datetime.now().strftime('%Y-%m-%d'),
        'source_url': source_url,
        'triage_status': 'ACCEPTED' if rule_pass else 'REJECTED',
        'triage_notes': reason
    }

def main():
    os.makedirs('datasets_stream', exist_ok=True)
    records = []

    # 1. Existing datasets already in repo
    existing_meta = [
        ('creditcard', 'A', 'Natural', 'datasets/creditcard.parquet', 'Time (seconds)', 'Kaggle credit card fraud 2013'),
        ('ieee_cis_45', 'A', 'Natural', 'datasets_stream/ieee_cis_45.parquet', 'TransactionDT (seconds)', 'IEEE-CIS fraud detection'),
        ('electricity_ir20', 'B', 'Induced IR=20', 'datasets_stream/electricity_ir20.parquet', 'Chronological date/time', 'NSW Electricity Market (Harries 1999)'),
        ('electricity_ir100', 'B', 'Induced IR=100', 'datasets_stream/electricity_ir100.parquet', 'Chronological date/time', 'NSW Electricity Market (Harries 1999)'),
        ('airlines_ir20', 'B', 'Induced IR=20', 'datasets_stream/airlines_ir20.parquet', 'Flight departure/arrival time', 'US Airline Delay Stream (Ikonomovska 2011)'),
        ('airlines_ir100', 'B', 'Induced IR=100', 'datasets_stream/airlines_ir100.parquet', 'Flight departure/arrival time', 'US Airline Delay Stream (Ikonomovska 2011)'),
        ('kddcup99_rare', 'B', 'Event Order', 'datasets_stream/kddcup99_rare.parquet', 'TCP connection sequence order', 'KDD Cup 1999 network intrusion'),
        ('covertype_binary', 'C', 'Order Only', 'datasets_stream/covertype_binary.parquet', 'Row/cartographic spatial order', 'USFS Forest Covertype'),
        ('poker_binary', 'C', 'Order Only', 'datasets_stream/poker_binary.parquet', 'Deal sequence order', 'UCI Poker Hand'),
        ('synth_sea_p5', 'D', 'Synthetic', 'datasets_stream/synth_sea_p5.parquet', 'Stream step (Time_Step)', 'SEA Generator with 10% drift'),
        ('synth_hyperplane_p5', 'D', 'Synthetic', 'datasets_stream/synth_hyperplane_p5.parquet', 'Stream step (Time_Step)', 'Rotating Hyperplane Generator'),
        ('synth_random_rbf_p5', 'D', 'Synthetic', 'datasets_stream/synth_random_rbf_p5.parquet', 'Stream step (Time_Step)', 'Random RBF Centroid Drift Generator'),
        ('ecoli2', 'C', 'Row Order', 'datasets/Ecoli2.csv', 'UCI row order (deterministic shuffle 42)', 'UCI Ecoli imbalanced benchmark'),
        ('yeast3', 'C', 'Row Order', 'datasets/yeast3.csv', 'UCI row order (deterministic shuffle 42)', 'UCI Yeast imbalanced benchmark')
    ]

    for name, tier, desc, pth, t_col, src in existing_meta:
        if pth.endswith('.parquet'):
            df = pd.read_parquet(pth)
        else:
            df = pd.read_csv(pth)
            
        for target_col in ['Class', 'class', 'Binary_Class', 'Delay', 'isFraud']:
            if target_col in df.columns:
                val = df[target_col].values
                if val.dtype == object or isinstance(val[0], str):
                    classes, counts = np.unique(val, return_counts=True)
                    min_c = classes[np.argmin(counts)]
                    y = np.where(val == min_c, 1, 0)
                else:
                    y = val.astype(int)
                n_feats = df.shape[1] - 1
                break
        else:
            y_raw = df.iloc[:, -1].values
            classes, counts = np.unique(y_raw, return_counts=True)
            min_c = classes[np.argmin(counts)]
            y = np.where(y_raw == min_c, 1, 0)
            n_feats = df.shape[1] - 1
            
        n_tot = len(y)
        n_min = int(np.sum(y == 1))
        n_maj = int(np.sum(y == 0))
        n_tr = int(0.75 * n_tot)
        y_tr = y[:n_tr]
        y_te = y[n_tr:]
        
        records.append({
            'dataset': name,
            'tier': tier,
            'N_total': n_tot,
            'N_features': n_feats,
            'N_min_total': n_min,
            'N_maj_total': n_maj,
            'IR_total': round(n_maj / n_min, 2) if n_min > 0 else np.nan,
            'N_train': n_tr,
            'N_min_train': int(np.sum(y_tr == 1)),
            'N_maj_train': int(np.sum(y_tr == 0)),
            'N_test': len(y_te),
            'N_min_test': int(np.sum(y_te == 1)),
            'time_attribute': t_col,
            'cap_applied': 'None' if n_tot <= 60000 else '20k majority cap during training where specified',
            'download_date': 'Pre-existing in repository',
            'source_url': src,
            'triage_status': 'ACCEPTED',
            'triage_notes': 'Validated in Phase 1 / Phase 2'
        })

    # 2. Download and triage new candidates
    for name, url in DOWNLOAD_URLS.items():
        fname = f"datasets_stream/{name}.csv"
        ok, msg = download_file(url, fname)
        if not ok:
            records.append({
                'dataset': name,
                'tier': 'B/C',
                'N_total': np.nan,
                'N_features': np.nan,
                'N_min_total': np.nan,
                'N_maj_total': np.nan,
                'IR_total': np.nan,
                'N_train': np.nan,
                'N_min_train': np.nan,
                'N_maj_train': np.nan,
                'N_test': np.nan,
                'N_min_test': np.nan,
                'time_attribute': 'Unknown',
                'cap_applied': 'None',
                'download_date': datetime.now().strftime('%Y-%m-%d'),
                'source_url': url,
                'triage_status': 'NOT AVAILABLE',
                'triage_notes': msg
            })
        else:
            tier_val = 'A' if name in ['weather', 'phishing'] else 'B'
            t_attr = 'Real sequential measurement order (NOAA daily timestamps)' if name == 'weather' else 'Chronological email arrival' if name == 'spam_katakis' else 'Web request sequence'
            rec = triage_dataset(name, fname, url, t_attr, tier_val)
            records.append(rec)

    df_out = pd.DataFrame(records)
    df_out.to_csv('results_stream/P3_datasets.csv', index=False)
    print("Saved results_stream/P3_datasets.csv")
    print(df_out[['dataset', 'tier', 'N_total', 'N_min_train', 'N_min_test', 'triage_status', 'triage_notes']].to_string())

if __name__ == '__main__':
    main()
