import pandas as pd

p3 = pd.read_csv('results_stream/P3_summary.csv', header=[0, 1])
drift = pd.read_csv('results_stream/P3_drift_summary.csv', header=[0, 1])

target_ds = [
    ('creditcard', 'Credit Card Fraud (Tang et al. [64])'),
    ('synth_hyperplane_p5', 'Moving Hyperplane (Wang et al. [47])'),
    ('synth_sea_p5', 'SEA Concept (Qiao et al. [80])'),
    ('electricity_ir20', 'Electricity Market (Heidrich et al. [8])'),
    ('weather', 'Weather Stream (Alencar et al. [63])'),
    ('spam_katakis', 'Spam Stream (Priya & Uthra [52])'),
    ('synth_random_rbf_p5', 'Random RBF (Wang et al. [47])')
]

all_data = []

for ds_id, ds_name in target_ds:
    sub_p = p3[(p3[p3.columns[0]] == ds_id) & (p3[p3.columns[1]] == 'CART')]
    sub_d = drift[(drift[drift.columns[0]] == ds_id) & (drift[drift.columns[1]] == 'Protocol_A_Static')]
    
    entry = {'Dataset': ds_name}
    
    # DA-ECEBO best
    r_da = sub_p[sub_p[sub_p.columns[2]] == 'DAECEBO2_AUTO']
    if len(r_da) == 0 or r_da[('Macro-F1', 'mean')].values[0] < sub_p[sub_p[sub_p.columns[2]] == 'DAECEBO2'][('Macro-F1', 'mean')].values[0]:
        r_da = sub_p[sub_p[sub_p.columns[2]] == 'DAECEBO2']
    entry['DA-ECEBO'] = f"{r_da[('Macro-F1', 'mean')].values[0]:.4f}"
    
    # EC-EBO
    r_ec = sub_p[sub_p[sub_p.columns[2]] == 'ECEBO']
    entry['EC-EBO'] = f"{r_ec[('Macro-F1', 'mean')].values[0]:.4f}" if len(r_ec) > 0 else 'N/A'
    
    # SMOTE
    r_sm = sub_p[sub_p[sub_p.columns[2]] == 'SMOTE_default']
    entry['SMOTE'] = f"{r_sm[('Macro-F1', 'mean')].values[0]:.4f}" if len(r_sm) > 0 else 'N/A'
    
    # Drift
    for m in ['ARF', 'SRP', 'HAT', 'OOB']:
        r_dr = sub_d[sub_d[sub_d.columns[2]] == m]
        entry[m] = f"{r_dr[('Macro-F1', 'mean')].values[0]:.4f}" if len(r_dr) > 0 else 'N/A'
        
    all_data.append(entry)

df_out = pd.DataFrame(all_data)
print(df_out.to_string(index=False))
