"""
Test one fold of Task B on ecoli2 to verify metric calculation and timing
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

from da_ecebo_and_ecebo_implementation import ECEBO, compute_metrics
from scripts_stream.verify_datasets import all_datasets

print(f"Available datasets: {len(all_datasets)}")
X, y = all_datasets['ecoli2']
print(f"Testing on ecoli2: shape={X.shape}, minority={y.sum()}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, test_idx = next(skf.split(X, y))
X_tr, y_tr = X[train_idx], y[train_idx]
X_te, y_te = X[test_idx], y[test_idx]

# Test ECEBO with step sizes
for step in [0.01, 0.1, 1.0]:
    t0 = time.time()
    ec = ECEBO(iters=60, alpha=1.0, k=7, prob=0.2, step=step, random_state=42)
    X_res, y_res = ec.fit_resample(X_tr, y_tr)
    print(f"ECEBO step={step}: generated {len(X_res)-len(X_tr)} in {time.time()-t0:.3f}s")
    
# Test classifiers
clf = SVC()
clf.fit(X_tr, y_tr)
probs = clf.decision_function(X_te)
preds = clf.predict(X_te)
m = compute_metrics(y_te, probs, preds)
print("SVC baseline metrics:", m)

