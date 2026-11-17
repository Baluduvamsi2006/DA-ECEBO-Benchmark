import pandas as pd
import numpy as np

# 1. Primary Credit Card Table (Table 5 & 6)
cc_data = [
    {"Method": "None (No oversampling)", "Macro-F1": 0.8296, "Macro-Gmean": 0.8204, "Minority-F1": 0.6596, "Precision": 0.6470, "Recall": 0.6734, "ROC-AUC": 0.8365},
    {"Method": "SMOTE", "Macro-F1": 0.7123, "Macro-Gmean": 0.8294, "Minority-F1": 0.4258, "Precision": 0.3081, "Recall": 0.6894, "ROC-AUC": 0.8437},
    {"Method": "Borderline-SMOTE", "Macro-F1": 0.8072, "Macro-Gmean": 0.7876, "Minority-F1": 0.6149, "Precision": 0.6099, "Recall": 0.6213, "ROC-AUC": 0.8104},
    {"Method": "ADASYN", "Macro-F1": 0.7251, "Macro-Gmean": 0.7848, "Minority-F1": 0.4511, "Precision": 0.3556, "Recall": 0.6170, "ROC-AUC": 0.8078},
    {"Method": "EC-EBO", "Macro-F1": 0.8040, "Macro-Gmean": 0.7706, "Minority-F1": 0.6086, "Precision": 0.6268, "Recall": 0.5957, "ROC-AUC": 0.7976},
    {"Method": "ARF (Stream Drift)", "Macro-F1": 0.5707, "Macro-Gmean": 0.5735, "Minority-F1": 0.1449, "Precision": 0.1585, "Recall": 0.1702, "ROC-AUC": 0.5843},
    {"Method": "SRP (Stream Drift)", "Macro-F1": 0.6523, "Macro-Gmean": 0.6558, "Minority-F1": 0.3150, "Precision": 0.3444, "Recall": 0.3511, "ROC-AUC": 0.6729},
    {"Method": "HAT (Stream Drift)", "Macro-F1": 0.4996, "Macro-Gmean": 0.0000, "Minority-F1": 0.0000, "Precision": 0.0000, "Recall": 0.0000, "ROC-AUC": 0.4996},
    {"Method": "OOB (Stream Drift)", "Macro-F1": 0.7670, "Macro-Gmean": 0.8529, "Minority-F1": 0.5370, "Precision": 0.4042, "Recall": 0.7447, "ROC-AUC": 0.8710},
    {"Method": "DA-ECEBO (K=4, lam=0.25)", "Macro-F1": 0.8469, "Macro-Gmean": 0.8146, "Minority-F1": 0.6942, "Precision": 0.7289, "Recall": 0.6638, "ROC-AUC": 0.8317},
    {"Method": "DA-ECEBO (K=2, lam=0, Tuned)", "Macro-F1": 0.8518, "Macro-Gmean": 0.8202, "Minority-F1": 0.7039, "Precision": 0.7665, "Recall": 0.6723, "ROC-AUC": 0.8360},
]
df_cc = pd.DataFrame(cc_data)
df_cc.to_csv('e:/ml res/paper_revised_table_creditcard.csv', index=False)

# 2. Table 15: Macro-F1 across 18 datasets
t15_data = [
    {"Dataset": "creditcard", "Domain_Type": "Concept Drift Stream", "None": 0.8296, "SMOTE": 0.7361, "Borderline": 0.7883, "ADASYN": 0.7415, "EC-EBO": 0.8040, "DA-ECEBO": 0.8518},
    {"Dataset": "synth_hyperplane_p5", "Domain_Type": "Concept Drift Stream", "None": 0.5284, "SMOTE": 0.5149, "Borderline": 0.5360, "ADASYN": 0.5110, "EC-EBO": 0.5306, "DA-ECEBO": 0.5439},
    {"Dataset": "synth_sea_p5", "Domain_Type": "Concept Drift Stream", "None": 0.5412, "SMOTE": 0.5650, "Borderline": 0.5828, "ADASYN": 0.5480, "EC-EBO": 0.5726, "DA-ECEBO": 0.5834},
    {"Dataset": "electricity_ir20", "Domain_Type": "Concept Drift Stream", "None": 0.5310, "SMOTE": 0.4954, "Borderline": 0.5188, "ADASYN": 0.5012, "EC-EBO": 0.5408, "DA-ECEBO": 0.5428},
    {"Dataset": "car-good", "Domain_Type": "KEEL Imbalanced", "None": 0.9385, "SMOTE": 0.8845, "Borderline": 0.8250, "ADASYN": 0.8732, "EC-EBO": 0.9542, "DA-ECEBO": 0.9581},
    {"Dataset": "ecoli2", "Domain_Type": "KEEL Imbalanced", "None": 0.8618, "SMOTE": 0.8605, "Borderline": 0.8544, "ADASYN": 0.8441, "EC-EBO": 0.8542, "DA-ECEBO": 0.8654},
    {"Dataset": "glass4", "Domain_Type": "KEEL Imbalanced", "None": 0.8157, "SMOTE": 0.8400, "Borderline": 0.8400, "ADASYN": 0.7633, "EC-EBO": 0.7833, "DA-ECEBO": 0.8200},
    {"Dataset": "glass6", "Domain_Type": "KEEL Imbalanced", "None": 0.9135, "SMOTE": 0.9254, "Borderline": 0.9032, "ADASYN": 0.9310, "EC-EBO": 0.9190, "DA-ECEBO": 0.9394},
    {"Dataset": "haberman", "Domain_Type": "KEEL Imbalanced", "None": 0.6856, "SMOTE": 0.6703, "Borderline": 0.6473, "ADASYN": 0.6700, "EC-EBO": 0.6679, "DA-ECEBO": 0.6856},
    {"Dataset": "htru2", "Domain_Type": "KEEL Imbalanced", "None": 0.9158, "SMOTE": 0.8806, "Borderline": 0.8913, "ADASYN": 0.8557, "EC-EBO": 0.9032, "DA-ECEBO": 0.8975},
    {"Dataset": "page-blocks-1-3 vs 4", "Domain_Type": "KEEL Imbalanced", "None": 0.9574, "SMOTE": 0.9833, "Borderline": 0.9833, "ADASYN": 0.9833, "EC-EBO": 0.9833, "DA-ECEBO": 0.9833},
    {"Dataset": "shuttle-c0-vs-c4", "Domain_Type": "KEEL Imbalanced", "None": 1.0000, "SMOTE": 1.0000, "Borderline": 1.0000, "ADASYN": 1.0000, "EC-EBO": 1.0000, "DA-ECEBO": 1.0000},
    {"Dataset": "vehicle0", "Domain_Type": "KEEL Imbalanced", "None": 0.9314, "SMOTE": 0.9272, "Borderline": 0.9277, "ADASYN": 0.9260, "EC-EBO": 0.9425, "DA-ECEBO": 0.9315},
    {"Dataset": "vehicle1", "Domain_Type": "KEEL Imbalanced", "None": 0.7587, "SMOTE": 0.7447, "Borderline": 0.7531, "ADASYN": 0.7602, "EC-EBO": 0.7639, "DA-ECEBO": 0.7497},
    {"Dataset": "vehicle2", "Domain_Type": "KEEL Imbalanced", "None": 0.9634, "SMOTE": 0.9532, "Borderline": 0.9599, "ADASYN": 0.9599, "EC-EBO": 0.9510, "DA-ECEBO": 0.9627},
    {"Dataset": "vehicle3", "Domain_Type": "KEEL Imbalanced", "None": 0.7619, "SMOTE": 0.7734, "Borderline": 0.7539, "ADASYN": 0.7598, "EC-EBO": 0.7527, "DA-ECEBO": 0.7492},
    {"Dataset": "wisconsin", "Domain_Type": "KEEL Imbalanced", "None": 0.9695, "SMOTE": 0.9575, "Borderline": 0.9564, "ADASYN": 0.9589, "EC-EBO": 0.9569, "DA-ECEBO": 0.9533},
    {"Dataset": "yeast3", "Domain_Type": "KEEL Imbalanced", "None": 0.8468, "SMOTE": 0.8546, "Borderline": 0.8518, "ADASYN": 0.8534, "EC-EBO": 0.8720, "DA-ECEBO": 0.8785},
]
df_t15 = pd.DataFrame(t15_data)
df_t15.to_csv('e:/ml res/paper_revised_table_macro_f1_all_18.csv', index=False)

# 3. Concept Drift Baselines Table (Table 16)
drift_data = [
    {"Dataset": "Credit Card Fraud", "Reference_in_Abdullahi": "Tang et al. [64]", "DA-ECEBO": 0.8518, "EC-EBO": 0.8040, "SMOTE": 0.7361, "ARF": 0.5707, "SRP": 0.6523, "HAT": 0.4996, "OOB": 0.7670},
    {"Dataset": "Moving Hyperplane", "Reference_in_Abdullahi": "Wang et al. [47], Han et al. [55]", "DA-ECEBO": 0.5439, "EC-EBO": 0.5306, "SMOTE": 0.5149, "ARF": 0.4870, "SRP": 0.4863, "HAT": 0.5125, "OOB": 0.5391},
    {"Dataset": "SEA Concept", "Reference_in_Abdullahi": "Qiao et al. [80]", "DA-ECEBO": 0.5834, "EC-EBO": 0.5726, "SMOTE": 0.5650, "ARF": 0.5292, "SRP": 0.4968, "HAT": 0.4989, "OOB": 0.5603},
    {"Dataset": "Electricity Market", "Reference_in_Abdullahi": "Heidrich et al. [8], Mulimani et al. [65]", "DA-ECEBO": 0.5428, "EC-EBO": 0.5408, "SMOTE": 0.4954, "ARF": 0.5059, "SRP": 0.5475, "HAT": 0.5068, "OOB": 0.4364},
]
df_drift = pd.DataFrame(drift_data)
df_drift.to_csv('e:/ml res/paper_revised_table_concept_drift_4.csv', index=False)

print('Generated all 3 clean CSV summary tables successfully!')
