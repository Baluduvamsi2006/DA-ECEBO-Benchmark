# DA-ECEBO Phase 2 Master Empirical Benchmark Report

**Project**: Empirical Evaluation and Validation of DA-ECEBO (Time-Aware EC-EBO)  
**Reference Paper**: Tao et al., *Knowledge-Based Systems* 338 (2026) 115544  
**Scope**: 8 Datasets $\times$ 13 Sampling Methods $\times$ 4 Classifiers $\times$ 10 Seeds = **4,160 Raw Evaluation Cells**  
**Stream Horizon**: 6 Drift Datasets $\times$ 5 Later Non-Overlapping 10% Blocks = **15,600 Stream Block Rows**  
**Saved Artifacts**: All figures in [`figs_stream/`](file:///e:/ml%20res/figs_stream/) (Vector PDFs + PNGs); all metrics in [`results_stream/`](file:///e:/ml%20res/results_stream/).

---

## 1. Executive Summary & Core Statistical Verification

1. **Experimental Completeness**:
   - Total primary test cells logged: **4,160 / 4,160 (100.0%)** in [`results_stream/P2_raw.csv`](file:///e:/ml%20res/results_stream/P2_raw.csv).
   - Total stream block evaluations logged: **15,600 / 15,600 (100.0%)** in [`results_stream/P2_blocks.csv`](file:///e:/ml%20res/results_stream/P2_blocks.csv).
   - Zero missing values, zero unhandled errors, and 100% reproducible deterministic random seeds (42–51).

2. **Mathematical Sanity Check (A1/A4 Equivalence Proof across Phase 2)**:
   - Comparing `DAECEBO2_Sanity_K1_lam1` ($K=1, \lambda=1.0, \delta_{\text{base}}=0.0$) against literal `ECEBO` across all 320 experimental combinations:
     - **Max Absolute Difference in Macro-F1**: **0.000000**
     - **Max Absolute Difference in Macro-Gmean**: **0.000000**
     - **Max Absolute Difference in Minority-F1**: **0.000000**
     - **Max Absolute Difference in Precision**: **0.000000**
     - **Max Absolute Difference in Recall**: **0.000000**
     - **Max Absolute Difference in ROC-AUC**: **0.000000**
   - **Conclusion**: Strict mathematical and empirical identity holds across all datasets, classifiers, and seeds.

3. **Global Multi-Method Significance (Friedman Test)**:
   - Friedman $\chi^2$ test across all 12 distinct sampling methods across 80 test splits each:
     - **CART**: Macro-Gmean $Q = 402.79$ ($p = 1.56 \times 10^{-79}$); AUC $Q = 251.55$ ($p = 1.32 \times 10^{-47}$)
     - **KNN**: Macro-Gmean $Q = 503.41$ ($p = 6.00 \times 10^{-101}$); AUC $Q = 379.79$ ($p = 1.19 \times 10^{-74}$)
     - **GaussianNB**: Macro-Gmean $Q = 412.55$ ($p = 1.32 \times 10^{-81}$); AUC $Q = 270.57$ ($p = 1.36 \times 10^{-51}$)
     - **SVC**: Macro-Gmean $Q = 341.29$ ($p = 1.69 \times 10^{-66}$); AUC $Q = 227.60$ ($p = 1.34 \times 10^{-42}$)

---

## 2. Benchmark Performance Summary Across All 8 Datasets

The table below reports empirical means across all 8 datasets and 10 random seeds (80 runs per method) for the primary classifier (**CART**), derived strictly from [`results_stream/P2_raw.csv`](file:///e:/ml%20res/results_stream/P2_raw.csv):

| Sampling Method | Macro-Gmean | Minority-F1 | Macro-F1 | ROC-AUC | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`None` (Baseline)** | 0.6114 | 0.3470 | 0.6467 | 0.6865 | 0.5489 | 0.4496 |
| **`SMOTE`** | 0.6750 | 0.3273 | 0.6220 | 0.7102 | 0.3541 | 0.5186 |
| **`BorderlineSMOTE`** | 0.6562 | 0.3392 | 0.6365 | 0.7056 | 0.4132 | 0.4851 |
| **`ADASYN`** | 0.6723 | 0.3217 | 0.6162 | 0.7041 | 0.3400 | 0.5230 |
| **`G-SMOTE`** | 0.6368 | 0.3336 | 0.6338 | 0.6953 | 0.4357 | 0.4632 |
| **`CCR`** | 0.7190 | 0.2781 | 0.5699 | 0.7308 | 0.2312 | 0.6375 |
| **`A-SUWO`** | 0.5526 | 0.3576 | 0.6632 | 0.6847 | 0.5522 | 0.3708 |
| **`MLOS`** | 0.1231 | 0.0585 | 0.0588 | 0.3731 | 0.0768 | 0.0792 |
| **`ECEBO` (Literal)** | 0.5989 | 0.3495 | 0.6496 | 0.6819 | 0.5365 | 0.4277 |
| **`ECEBO_PSI0`** | 0.7052 | 0.2883 | 0.5783 | 0.7133 | 0.2520 | 0.6094 |
| **`DAECEBO2` (Validated $K^*, \lambda^*$)** | 0.6093 | 0.3474 | 0.6453 | 0.6864 | 0.5229 | 0.4422 |
| **`DAECEBO2_PSI0`** | **0.7264** | 0.3028 | 0.5850 | **0.7311** | 0.2709 | **0.6384** |
| **`DAECEBO2_Sanity_K1_lam1`** | 0.5989 | 0.3495 | 0.6496 | 0.6819 | 0.5365 | 0.4277 |

---

## 3. Streaming Drift Performance (6 Drift Datasets)

Evaluating exclusively on the 6 concept drift stream datasets (`synth_sea_p5`, `synth_hyperplane_p5`, `synth_random_rbf_p5`, `electricity_ir100`, `electricity_ir20`, `creditcard`) across 60 runs per method with **CART**:

| Sampling Method | Macro-Gmean | Minority-F1 | Macro-F1 | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: |
| **`None`** | 0.5315 | 0.2249 | 0.5853 | 0.6304 |
| **`SMOTE`** | 0.6013 | 0.1802 | 0.5412 | 0.6475 |
| **`BorderlineSMOTE`** | 0.5878 | 0.2109 | 0.5696 | 0.6527 |
| **`ADASYN`** | 0.6024 | 0.1827 | 0.5399 | 0.6442 |
| **`G-SMOTE`** | 0.5574 | 0.1965 | 0.5615 | 0.6343 |
| **`CCR`** | 0.6724 | 0.1366 | 0.4846 | 0.6870 |
| **`A-SUWO`** | 0.4379 | 0.2106 | 0.5896 | 0.6131 |
| **`MLOS`** | 0.1326 | 0.0540 | 0.0567 | 0.4460 |
| **`ECEBO`** | 0.5050 | 0.1987 | 0.5705 | 0.6140 |
| **`ECEBO_PSI0`** | 0.6650 | 0.1526 | 0.4966 | 0.6729 |
| **`DAECEBO2`** | 0.5178 | 0.2106 | 0.5743 | 0.6197 |
| **`DAECEBO2_PSI0`** | **0.6796** | 0.1593 | 0.4985 | **0.6845** |
| **`DAECEBO2_Sanity_K1_lam1`** | 0.5050 | 0.1987 | 0.5705 | 0.6140 |

---

## 4. Credit Card Benchmark (Chronological 75/25 Split, 71,202 Future Test Rows)

Evaluated under the chronological split (213,605 train rows $\to$ 426,414 oversampled rows $\to$ 71,202 test rows) with **CART**:

| Sampling Method | Macro-Gmean | Minority-F1 | Macro-F1 | ROC-AUC | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`None`** | 0.8204 | 0.6596 | 0.8296 | 0.8365 | 0.6480 | 0.6734 |
| **`SMOTE`** | 0.8422 | 0.4733 | 0.7361 | 0.8545 | 0.3540 | 0.7138 |
| **`BorderlineSMOTE`** | 0.8228 | 0.5772 | 0.7883 | 0.8384 | 0.5052 | 0.6787 |
| **`ADASYN`** | 0.7989 | 0.4839 | 0.7415 | 0.8190 | 0.3929 | 0.6404 |
| **`G-SMOTE`** | 0.8188 | 0.5510 | 0.7751 | 0.8352 | 0.4727 | 0.6723 |
| **`CCR`** | **0.8937** | 0.1350 | 0.5640 | **0.8980** | 0.0734 | **0.8011** |
| **`A-SUWO`** | 0.8469 | 0.5984 | 0.7989 | 0.8586 | 0.5186 | 0.7202 |
| **`MLOS`** | 0.0355 | 0.0020 | 0.0026 | 0.3881 | 0.0019 | 0.0021 |
| **`ECEBO`** | 0.7706 | 0.6086 | 0.8040 | 0.7976 | 0.6139 | 0.6043 |
| **`ECEBO_PSI0`** | 0.7960 | 0.2692 | 0.6335 | 0.8166 | 0.1706 | 0.6383 |
| **`DAECEBO2` ($K^*=5, \lambda^*=0.75$)** | 0.8054 | **0.6852** | **0.8424** | 0.8243 | **0.6974** | 0.6745 |
| **`DAECEBO2_PSI0`** | 0.8733 | 0.3106 | 0.6542 | 0.8809 | 0.1983 | 0.7670 |
| **`DAECEBO2_Sanity_K1_lam1`** | 0.7706 | 0.6086 | 0.8040 | 0.7976 | 0.6139 | 0.6043 |

---

## 5. Pairwise Wilcoxon Signed-Rank Tests (80 Matched Pairs per Comparison)

From [`results_stream/P2_wtl.csv`](file:///e:/ml%20res/results_stream/P2_wtl.csv), comparing `DAECEBO2` and `DAECEBO2_PSI0` against standard literature methods on **CART**:

### DAECEBO2 vs Competitors (CART)
- **vs `None`**: Macro-Gmean W/T/L = **42/0/38** ($W = 1488.0, p = 0.52666$); Minority-F1 = **36/0/44** ($W = 1536.0, p = 0.68703$); AUC = **40/0/40** ($W = 1589.0, p = 0.88180$).
- **vs `SMOTE`**: Macro-Gmean W/T/L = **9/0/71** ($p = 0.00000$); Macro-F1 = **58/0/22** ($W = 642.5, p = 0.00000$); Minority-F1 = **47/0/33** ($W = 1426.0, p = 0.35212$).
- **vs `ECEBO`**: Macro-Gmean W/T/L = **37/40/3** ($W = 217.5, p = 0.00000$); Minority-F1 = **19/40/21** ($W = 398.5, p = 0.90098$); Macro-F1 = **28/40/12** ($W = 205.5, p = 0.00446$).

### DAECEBO2_PSI0 vs Competitors (CART)
- **vs `ECEBO`**: Macro-Gmean W/T/L = **66/0/14** ($W = 164.0, p = 0.00000$); AUC W/T/L = **61/0/19** ($W = 345.5, p = 0.00000$).
- **vs `ECEBO_PSI0`**: Macro-Gmean W/T/L = **30/40/10** ($W = 127.0, p = 0.00014$, Holm $p = 0.02954$); AUC W/T/L = **30/40/10** ($W = 134.0, p = 0.00021$, Holm $p = 0.04305$); Minority-F1 W/T/L = **29/40/11** ($W = 165.0, p = 0.00099$).
- **vs `CCR`**: Macro-Gmean W/T/L = **44/0/36** ($W = 1432.0, p = 0.36868$); AUC W/T/L = **41/0/39** ($W = 1563.0, p = 0.79603$); Minority-F1 W/T/L = **55/0/25** ($W = 952.5, p = 0.00078$).

---

## 6. Publication Figures Catalogue (Part 3)

All 7 publication figures are generated as clean vector PDFs and PNG previews in [`figs_stream/`](file:///e:/ml%20res/figs_stream/):

1. **[`figs_stream/F1_delta_metrics.pdf`](file:///e:/ml%20res/figs_stream/F1_delta_metrics.pdf)** (and `.png`):  
   Change in Macro-F1 and Macro-Gmean versus `None` baseline per dataset (bars = mean across seeds, dots = individual seeds).
2. **[`figs_stream/F2_quota_concentration.pdf`](file:///e:/ml%20res/figs_stream/F2_quota_concentration.pdf)** (and `.png`):  
   Cumulative share of quota against clusters ranked by quota on `creditcard`, contrasting EC-EBO's static uniform allocation against DA-ECEBO's dynamic drift-aware allocation.
3. **[`figs_stream/F3_cc_tsne_windows.pdf`](file:///e:/ml%20res/figs_stream/F3_cc_tsne_windows.pdf)** (and `.png`):  
   t-SNE projection of credit card training minority samples colored by temporal arrival window ($K=2$ and $K=4$).
4. **[`figs_stream/F4_step_audit.pdf`](file:///e:/ml%20res/figs_stream/F4_step_audit.pdf)** (and `.png`):  
   Audit of proposal acceptance rates and exact duplicate copy shares across datasets, contrasting literal $\psi$ hill-climbing against PSI0.
5. **[`figs_stream/F5_cluster_recency_quota.pdf`](file:///e:/ml%20res/figs_stream/F5_cluster_recency_quota.pdf)** (and `.png`):  
   Scatter of cluster recency against allocated sampling quota on `creditcard` for EC-EBO vs DA-ECEBO.
6. **[`figs_stream/F6_validation_heatmaps.pdf`](file:///e:/ml%20res/figs_stream/F6_validation_heatmaps.pdf)** (and `.png`):  
   Validation grid heatmaps ($K \times \lambda$) for all datasets from [`results_stream/P2_validation_grid.csv`](file:///e:/ml%20res/results_stream/P2_validation_grid.csv).
7. **[`figs_stream/F7_stream_blocks.pdf`](file:///e:/ml%20res/figs_stream/F7_stream_blocks.pdf)** (and `.png`):  
   Tracking metric trajectories across the 5 stream blocks (50% to 100% of the stream horizon) from [`results_stream/P2_blocks.csv`](file:///e:/ml%20res/results_stream/P2_blocks.csv).

---

## 7. Artifact Index

- **Master Benchmark Outputs**:
  - Raw Cell Data: [`results_stream/P2_raw.csv`](file:///e:/ml%20res/results_stream/P2_raw.csv) (4,160 rows)
  - Stream Block Data: [`results_stream/P2_blocks.csv`](file:///e:/ml%20res/results_stream/P2_blocks.csv) (15,600 rows)
  - Summary Statistics: [`results_stream/P2_summary.csv`](file:///e:/ml%20res/results_stream/P2_summary.csv)
  - Pairwise W/T/L & Tests: [`results_stream/P2_wtl.csv`](file:///e:/ml%20res/results_stream/P2_wtl.csv)
  - Hyperparameter Grid: [`results_stream/P2_validation_grid.csv`](file:///e:/ml%20res/results_stream/P2_validation_grid.csv)
  - Bootstrap 95% CIs: [`results_stream/P2_creditcard_bootstrap_ci.csv`](file:///e:/ml%20res/results_stream/P2_creditcard_bootstrap_ci.csv)
- **Reconciliation Audit CSVs**:
  - R1 Acceptance & Duplicates: [`results_stream/R1_acceptance.csv`](file:///e:/ml%20res/results_stream/R1_acceptance.csv)
  - R2 Credit Card Clusters: [`results_stream/R2_cc_clusters.csv`](file:///e:/ml%20res/results_stream/R2_cc_clusters.csv)
  - R3 Decision Tree AUC: [`results_stream/R3_dt_auc.csv`](file:///e:/ml%20res/results_stream/R3_dt_auc.csv)
  - R4 Precise Timing: [`results_stream/C7_timing_v2.csv`](file:///e:/ml%20res/results_stream/C7_timing_v2.csv)
  - R5 Paper Methods Audit: [`results_stream/R5_comparison_methods.csv`](file:///e:/ml%20res/results_stream/R5_comparison_methods.csv)
