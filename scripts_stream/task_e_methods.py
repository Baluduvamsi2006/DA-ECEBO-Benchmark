import os
import sys
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import smote_variants as sv

# Load Ecoli2 dataset
ecoli2_path = os.path.join("datasets", "Ecoli2.csv")
if not os.path.exists(ecoli2_path):
    # Try alternative locations
    for p in ["datasets_keel/ecoli2.dat", "datasets_keel/ecoli-0_vs_1.dat"]:
        if os.path.exists(p):
            ecoli2_path = p
            break

df = pd.read_csv(ecoli2_path)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
# Ensure binary 0/1 with 1 as minority
classes, counts = np.unique(y, return_counts=True)
min_class = classes[np.argmin(counts)]
y = np.where(y == min_class, 1, 0)

orig_majority = np.sum(y == 0)
orig_minority = np.sum(y == 1)
print(f"Loaded {ecoli2_path}: {X.shape[0]} rows, {X.shape[1]} features. Majority: {orig_majority}, Minority: {orig_minority}")

methods_to_test = [
    ("SMOTE", "imblearn.over_sampling.SMOTE", lambda: SMOTE(random_state=42)),
    ("BorderlineSMOTE", "imblearn.over_sampling.BorderlineSMOTE", lambda: BorderlineSMOTE(random_state=42)),
    ("ADASYN", "imblearn.over_sampling.ADASYN", lambda: ADASYN(random_state=42)),
    ("A-SUWO", "smote_variants.A_SUWO", lambda: sv.A_SUWO(random_state=42)),
    ("CCR", "smote_variants.CCR", lambda: sv.CCR(random_state=42)),
    ("G-SMOTE", "smote_variants.G_SMOTE", lambda: sv.G_SMOTE(random_state=42)),
]

results = []

for name, pkg, inst_fn in methods_to_test:
    try:
        model = inst_fn()
        if hasattr(model, "fit_resample"):
            X_res, y_res = model.fit_resample(X, y)
        elif hasattr(model, "sample"):
            X_res, y_res = model.sample(X, y)
        else:
            raise AttributeError("Neither fit_resample nor sample found")
        
        maj_count = int(np.sum(y_res == 0))
        min_count = int(np.sum(y_res == 1))
        has_nan = bool(np.isnan(X_res).any() or np.isnan(y_res).any())
        balanced = bool(abs(maj_count - min_count) <= max(10, int(0.1 * maj_count)))
        status = "PASSED" if not has_nan and balanced else "WARNING"
        
        results.append({
            "Method": name,
            "Package / Source": pkg,
            "Public Code Found": "YES (Installed)",
            "Smoke Test on Ecoli2": status,
            "Resampled Majority": maj_count,
            "Resampled Minority": min_count,
            "Contains NaN": has_nan,
            "Notes / Details": f"Original ({orig_majority}:{orig_minority}) -> Resampled ({maj_count}:{min_count})"
        })
    except Exception as e:
        results.append({
            "Method": name,
            "Package / Source": pkg,
            "Public Code Found": "YES (Installed)",
            "Smoke Test on Ecoli2": f"FAILED: {str(e)}",
            "Resampled Majority": np.nan,
            "Resampled Minority": np.nan,
            "Contains NaN": np.nan,
            "Notes / Details": f"Execution error: {str(e)}"
        })

# External literature comparison methods
ext_methods = [
    ("MLOS", "https://github.com/ytyancp/MLOS", "YES (GitHub)", "Untested (Standalone script)",
     "Python implementation (MLOS.py) available from paper author. Needs custom integration."),
    ("HS-SMOTE", "https://github.com/gogossss/HS-SMOTE", "YES (GitHub MATLAB)", "Not applicable (MATLAB only)",
     "MATLAB implementation (.m file) only; no native Python package found."),
    ("NCLO-SMOTE", "None", "NO PUBLIC CODE FOUND", "Not applicable",
     "No official or community GitHub repo found matching NCLO-SMOTE."),
    ("DGS", "None", "NO PUBLIC CODE FOUND", "Not applicable",
     "No official or community GitHub repo found for DGS oversampling."),
    ("ENU", "None", "NO PUBLIC CODE FOUND", "Not applicable",
     "No official or community GitHub repo found for ENU oversampling.")
]

for name, src, found, test_stat, notes in ext_methods:
    results.append({
        "Method": name,
        "Package / Source": src,
        "Public Code Found": found,
        "Smoke Test on Ecoli2": test_stat,
        "Resampled Majority": np.nan,
        "Resampled Minority": np.nan,
        "Contains NaN": np.nan,
        "Notes / Details": notes
    })

res_df = pd.DataFrame(results)
os.makedirs("results_stream", exist_ok=True)
out_csv = os.path.join("results_stream", "E_methods.csv")
res_df.to_csv(out_csv, index=False)
print(f"Task E completed successfully. Saved to {out_csv}")
print(res_df.to_string())

