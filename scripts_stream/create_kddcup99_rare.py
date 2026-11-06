import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_kddcup99

print("Fetching KDDCup99 10% data...")
data = fetch_kddcup99(percent10=True, as_frame=True)
df = data.frame

# Decode byte columns
for col in df.columns:
    if df[col].dtype == object and len(df) > 0 and isinstance(df[col].iloc[0], bytes):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else str(x))

target_col = 'labels' if 'labels' in df.columns else df.columns[-1]
y_raw = df[target_col].astype(str).str.strip().str.rstrip('.')

# Rare attack definition
dos_attacks = {'smurf', 'neptune', 'back', 'teardrop', 'pod', 'land'}
normal_class = {'normal'}
majority_classes = dos_attacks | normal_class

# Minority = all attacks that are NOT in DoS (i.e. Probe, R2L, U2R)
is_minority = ~y_raw.isin(majority_classes)

df_rare = df.drop(columns=[target_col]).copy()
df_rare['Binary_Class'] = is_minority.astype(int)

maj_count = int(np.sum(df_rare['Binary_Class'] == 0))
min_count = int(np.sum(df_rare['Binary_Class'] == 1))
ir = round(maj_count / min_count, 2)

print(f"kddcup99_rare created: Total {len(df_rare)} rows. Majority: {maj_count}, Minority: {min_count}, IR: {ir}")

out_path = "datasets_stream/kddcup99_rare.parquet"
df_rare.to_parquet(out_path, index=False)
print(f"Saved to {out_path}")

