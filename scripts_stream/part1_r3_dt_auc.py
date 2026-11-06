"""
Part 1: R3. Depth-6 Tree on Credit Card Out-of-Sample AUC Audit
===============================================================
Evaluates Decision Tree on creditcard chronological 75/25 split:
  Train: first 213,605 rows (398 frauds)
  Test: remaining 71,202 rows (94 frauds)

Configurations tested:
1. Task H Screening Run: max_depth=6, 20k uniform unstratified train subsample (seed 42), Time excluded, Amount included.
2. Full Train: max_depth=6, all 213,605 train instances, Time excluded, Amount included.
3. Full Train: unpruned CART (max_depth=None), all 213,605 train instances, Time excluded, Amount included.
4. Full Train: max_depth=6, Time included, Amount included.
5. Full Train: unpruned CART, Time included, Amount included.

Output: results_stream/R3_dt_auc.csv
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

def main():
    df = pd.read_parquet('datasets/creditcard.parquet')
    n_tr = int(len(df) * 0.75)
    tr = df.iloc[:n_tr]
    te = df.iloc[n_tr:]
    
    feats_no_time = [c for c in df.columns if c not in ['Time', 'Class']]
    feats_with_time = [c for c in df.columns if c != 'Class']
    
    configs = [
        {
            'Config_Name': 'Task H Screening Subsample (20k unstratified)',
            'Features_Used_Description': 'V1 to V28, Amount (29 features)',
            'Time_Included': False,
            'Amount_Included': True,
            'Subsample_Description': 'Random uniform 20,000 instances from train (seed 42, 33 frauds sampled)',
            'Tree_Type': 'DecisionTreeClassifier(max_depth=6, random_state=42)',
            'max_depth': 6,
            'feats': feats_no_time,
            'subsample': True
        },
        {
            'Config_Name': 'Full Train (max_depth=6, Time excluded)',
            'Features_Used_Description': 'V1 to V28, Amount (29 features)',
            'Time_Included': False,
            'Amount_Included': True,
            'Subsample_Description': 'Full 213,605 train instances (398 frauds)',
            'Tree_Type': 'DecisionTreeClassifier(max_depth=6, random_state=42)',
            'max_depth': 6,
            'feats': feats_no_time,
            'subsample': False
        },
        {
            'Config_Name': 'Full Train (Unpruned CART, Time excluded)',
            'Features_Used_Description': 'V1 to V28, Amount (29 features)',
            'Time_Included': False,
            'Amount_Included': True,
            'Subsample_Description': 'Full 213,605 train instances (398 frauds)',
            'Tree_Type': 'DecisionTreeClassifier(max_depth=None, random_state=42)',
            'max_depth': None,
            'feats': feats_no_time,
            'subsample': False
        },
        {
            'Config_Name': 'Full Train (max_depth=6, Time included)',
            'Features_Used_Description': 'Time, V1 to V28, Amount (30 features)',
            'Time_Included': True,
            'Amount_Included': True,
            'Subsample_Description': 'Full 213,605 train instances (398 frauds)',
            'Tree_Type': 'DecisionTreeClassifier(max_depth=6, random_state=42)',
            'max_depth': 6,
            'feats': feats_with_time,
            'subsample': False
        },
        {
            'Config_Name': 'Full Train (Unpruned CART, Time included)',
            'Features_Used_Description': 'Time, V1 to V28, Amount (30 features)',
            'Time_Included': True,
            'Amount_Included': True,
            'Subsample_Description': 'Full 213,605 train instances (398 frauds)',
            'Tree_Type': 'DecisionTreeClassifier(max_depth=None, random_state=42)',
            'max_depth': None,
            'feats': feats_with_time,
            'subsample': False
        }
    ]
    
    rows = []
    print("=== PART 1: R3. DECISION TREE AUC AUDIT ON CREDITCARD ===")
    for c in configs:
        feats = c['feats']
        X_tr_full = tr[feats].values.astype(np.float64)
        y_tr_full = tr['Class'].values.astype(int)
        X_te = te[feats].values.astype(np.float64)
        y_te = te['Class'].values.astype(int)
        
        if c['subsample']:
            rng = np.random.RandomState(42)
            idx_sub = rng.choice(len(X_tr_full), size=20000, replace=False)
            X_tr, y_tr = X_tr_full[idx_sub], y_tr_full[idx_sub]
        else:
            X_tr, y_tr = X_tr_full, y_tr_full
            
        dt = DecisionTreeClassifier(max_depth=c['max_depth'], random_state=42)
        dt.fit(X_tr, y_tr)
        
        p_tr = dt.predict_proba(X_tr)[:, 1]
        p_te = dt.predict_proba(X_te)[:, 1]
        
        auc_tr = roc_auc_score(y_tr, p_tr)
        auc_te = roc_auc_score(y_te, p_te)
        
        u_vals_te = np.unique(np.round(p_te, 6))
        u_vals_str = f"{len(u_vals_te)} unique values: " + ", ".join([str(v) for v in u_vals_te[:8]])
        if len(u_vals_te) > 8:
            u_vals_str += "..."
            
        print(f"\n[{c['Config_Name']}]")
        print(f"  Train AUC: {auc_tr:.4f} | Test AUC: {auc_te:.4f}")
        print(f"  Unique test probas ({len(u_vals_te)}): {u_vals_te}")
        
        rows.append({
            'Config_Name': c['Config_Name'],
            'Features_Used': c['Features_Used_Description'],
            'Time_Included': c['Time_Included'],
            'Amount_Included': c['Amount_Included'],
            'Subsample_Description': c['Subsample_Description'],
            'Tree_Classifier': c['Tree_Type'],
            'N_train_used': len(X_tr),
            'N_min_train_used': int(np.sum(y_tr == 1)),
            'N_test': len(X_te),
            'N_min_test': int(np.sum(y_te == 1)),
            'Train_AUC': round(auc_tr, 4),
            'Test_AUC': round(auc_te, 4),
            'Unique_Probabilities_Count_Test': len(u_vals_te),
            'Unique_Probabilities_List_Test': u_vals_str
        })
        
    df_out = pd.DataFrame(rows)
    os.makedirs('results_stream', exist_ok=True)
    out_csv = 'results_stream/R3_dt_auc.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved R3 audit to {out_csv}")

if __name__ == '__main__':
    main()

