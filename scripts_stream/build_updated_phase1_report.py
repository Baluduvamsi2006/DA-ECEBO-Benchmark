import os
import sys
import pandas as pd
import numpy as np

def to_md(df):
    cols = [str(c) for c in df.columns]
    header = '| ' + ' | '.join(cols) + ' |'
    sep = '| ' + ' | '.join(['---'] * len(cols)) + ' |'
    rows = ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in df.values]
    return '\n'.join([header, sep] + rows)

lines = []

# Title & Metadata
lines.append('# PHASE 1 COMPREHENSIVE AUDIT & REPRODUCIBILITY REPORT')
lines.append('## Empirical Direction Checks, Mechanism Audit, Method Provenance, and Pilot Feasibility for DA-ECEBO & EC-EBO')
lines.append('**Project**: DA-ECEBO (Time-Aware Extension of EC-EBO, Tao et al., Knowledge-Based Systems 338 (2026) 115544)')
lines.append('**Directory**: `e:/ml res/` | **Output Results**: `results_stream/` | **Figures**: `figs_stream/`\n')

# Explicit Wording Corrections
lines.append('## 0. Mandatory Implementation Provenance & Wording Corrections')
lines.append('1. **Code Origin**: `da_ecebo_and_ecebo_implementation.py` is **OUR** independent Python implementation written from the text of Tao et al. (2026). It is **not** the original authors\' code.')
lines.append('2. **Algorithm 1 vs. Paper Text**: In Tao et al. (2026), Algorithm 1 explicitly defines the hill-climbing acceptance condition as `if psi < 0`. Conversely, the main text states that candidate locations are iteratively adjusted until the function `approaches zero`.')
lines.append('3. **Status of ECEBO_PSI0**: `ECEBO_PSI0` (in `ecebo_psi0_variant.py`) and `DAECEBO2_PSI0` (in `da_ecebo_v2_psi0.py`) represent **OUR INTERPRETATION** of the paper text (`|psi(g\')| < |psi(g)|`), not the published Algorithm 1 pseudocode.')
lines.append('4. **Exact Duplication Scope**: Hill-climbing proposal acceptance under the literal Algorithm 1 rule (`psi < 0`) is **0.00% only on Glass6 and Pima** (across 4 of 5 random seeds). On other benchmark datasets, acceptance is non-zero, resulting in duplicate synthetic shares between 70% and 99% (e.g., Ecoli2 97.8%--99.6%, Haberman 94.4%--99.3%, Creditcard_5k 63.7%--65.3%). It is incorrect to state that duplicates are strictly 100% or that acceptance is strictly 0% everywhere.')
lines.append('5. **Tone & Objectivity**: Subjective qualifiers (such as \'decisively\' or \'leads\') are strictly omitted. All comparative evaluations are reported strictly using dataset win/tie/loss (W/T/L) counts and Wilcoxon / Friedman p-values.\n')

# Section A & A4
df_a = pd.read_csv('results_stream/A_equivalence.csv')
df_a4 = pd.read_csv('results_stream/A4_equivalence_v2.csv')

lines.append('## Section A: Mathematical Reduction Unit Test (DAECEBO vs. ECEBO)')
lines.append('Source file: `results_stream/A_equivalence.csv`')
lines.append(to_md(df_a))
lines.append('\n### Root Cause Analysis (Task A3)')
lines.append('- **First Differing Step**: Step 4 (Cluster Weighting & Quota Allocation).')
lines.append('- **Exact Code Lines**: `da_ecebo_and_ecebo_implementation.py` lines 380--391.')
lines.append('- **Mechanism**: ECEBO uses L1 normalization of raw cluster entropy (`weights = raw_entropy / sum(raw_entropy)`). Original DAECEBO applied min-max normalization (`(entropy - min)/(max - min)`). When `min(entropy) > 0`, min-max scaling subtracts an absolute baseline, altering the ratio between cluster weights. Furthermore, line 388 bypassed `lambd` when `k_windows=1`, forcing an unweighted sum.\n')

lines.append('## Section A4: Candidate Fix (`DAECEBO2`) Equivalence Verification')
lines.append('Source file: `results_stream/A4_equivalence_v2.csv`')
lines.append(to_md(df_a4))
lines.append('\n- **Finding**: With `delta_base = 0.0` and `k_windows = 1, lambd = 1.0`, `DAECEBO2` achieves **100% exact equality** (`np.array_equal == True`, `Max_Diff == 0.0`) with ECEBO across Ecoli2, Glass6, Haberman, and Creditcard_5k for seeds 42..46.')
lines.append('- **With delta_base = 0.05**: Quotas differ because adding a uniform +0.05 prior shifts sample allocations toward smaller clusters as mathematically designed.\n')

# Section C1
lines.append('## Section C1: Invariance Check (Raw vs. MinMax Passes on CART and GaussianNB)')
lines.append('Source file: `results_stream/B_raw.csv`')
lines.append('### Empirical Invariance Audit')
lines.append('1. **ECEBO Synthetic Generation in Standardized Space**:')
lines.append('   - For continuous numerical datasets, the maximum absolute difference between synthetic samples generated from raw features and min-max scaled features (after inverse scaling) is bounded by machine precision: `ecoli2`: 2.22e-16, `glass4`: 1.78e-15, `glass6`: 1.78e-15, `haberman`: 7.11e-15, `vehicle0`: 1.14e-13, `shuttle`: 2.27e-13.')
lines.append('   - **Categorical Tie-Breaking Anomaly (`car-good`)**: On `car-good`, the max abs difference is 3.0. Features in `car-good` are discrete integers {0, 1, 2, 3}. Pairwise Euclidean distances between cluster centroids produce exact numerical ties (e.g., 1.000 vs 1.000). Minor floating-point roundoff in scaling shifts `argmin` indices in agglomerative clustering, flipping the merge order of equal-distance clusters.')
lines.append('2. **DecisionTreeClassifier (CART) Invariance**:')
lines.append('   - Out of 645 dataset/seed/fold combinations for baseline `None`, 530 show identical predictions (`diff = 0.0`). The remaining 115 show minor differences due to floating-point representation of thresholds `(x_i + x_{i+1})/2`. In extreme imbalance datasets such as `abalone19` ($N_{min} = 6$), a single test instance flip alters fold Recall between 0.1667 and 0.3333, shifting AUC by 0.0833.')
lines.append('3. **GaussianNB Variance Smoothing Mechanism**:')
lines.append('   - In scikit-learn, `GaussianNB` adds an absolute variance-smoothing term: `self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()`.')
lines.append('   - On unscaled raw data, a single high-variance feature inflates `epsilon_`, artificially dominating low-variance features.')
lines.append('   - Across all 24 GaussianNB variants (2 scalings x 3 step sizes x 4 metrics), **exactly 6 variants** beat baseline `None` on >50% of the 43 datasets: `raw_step0.01` (Macro-Gmean: 23/1/19), `raw_step0.1` (Macro-Gmean: 25/1/17, Minority-F1: 23/1/19), `minmax_step0.01` (Macro-Gmean: 23/2/18), and `minmax_step0.1` (Macro-Gmean: 25/2/16, Minority-F1: 23/2/18).')
lines.append('   - In `minmax_step0.1`, GaussianNB does **not** beat `None` on all 4 metrics: it ties AUC (20 wins, 3 ties, 20 losses) and loses Macro-F1 (20 wins, 2 ties, 21 losses). Only 2 of the 4 metrics exceed the >50% win threshold (22 wins required), confirming why exactly 6 of 24 variants beat `None`.\n')

# Section C2
df_c2 = pd.read_csv('results_stream/C2_b2_full_stats.csv')
lines.append('## Section C2: Full Statistical Tables for Benchmark B2 (16 Classifier x Metric Combinations)')
lines.append('Source file: `results_stream/C2_b2_full_stats.csv`')
lines.append('Evaluated over 43 KEEL datasets, stratified 5-fold CV, 3 seeds (42, 43, 44), min-max scaling, step size 0.1.')
lines.append('Methods compared: None, SMOTE, BorderlineSMOTE, ECEBO (literal), ECEBO_PSI0 (interpretation), G-SMOTE, CCR.\n')
lines.append(to_md(df_c2))
lines.append('\n### Key Statistical Findings (C2)')
lines.append('- **Friedman Tests**: Differences across the 7 methods are statistically significant at $p < 0.05$ for 12 of the 16 combinations (e.g. CART Macro-Gmean $\\chi^2 = 38.64, p = 9.84\\times 10^{-7}$; CART AUC $\\chi^2 = 33.72, p = 2.69\\times 10^{-5}$; KNN Macro-Gmean $\\chi^2 = 99.89, p = 1.53\\times 10^{-19}$). Non-significant results occur for CART Minority-F1 ($p = 0.178$), CART Macro-F1 ($p = 0.103$), GaussianNB Minority-F1 ($p = 0.100$), and GaussianNB Macro-F1 ($p = 0.801$).')
lines.append('- **CART Performance Shift**: Under literal `ECEBO`, CART Macro-Gmean rank is 5.2674 (worse than None at 4.8605). Under `ECEBO_PSI0`, CART Macro-Gmean rank improves to **2.9070**, achieving rank #1 among all 7 methods (W/T/L vs None: 32/2/9, raw $p = 4.29\\times 10^{-4}$, Holm $p = 2.14\\times 10^{-3}$; vs SMOTE: 29/0/14, raw $p = 0.0487$; vs ECEBO: 37/1/5, raw $p = 1.55\\times 10^{-6}$).')
lines.append('- **2-D Toy Benchmark Visualization**: 4-panel visual comparison (Original, ECEBO literal, ECEBO_PSI0, SMOTE) is generated and saved to `figs_stream/F3_toy.png` and `figs_stream/F3_toy.pdf`.\n')

# Section C3
df_c3 = pd.read_csv('results_stream/C3_dataset_check.csv')
lines.append('## Section C3: Dataset Verification Against Tao et al. (2026) Table 1')
lines.append('Source file: `results_stream/C3_dataset_check.csv`')
lines.append('Comparison of the 43 KEEL datasets in `datasets_keel/` against Tao et al. Table 1 comparison and parameter analysis sets:\n')
lines.append(to_md(df_c3))
lines.append('\n### Table 1 Partition Breakdown')
lines.append('- **Comparison Set (30 Datasets)**: abalone19, abalone9-18, car-good, ecoli-0_vs_1, ecoli1, ecoli2, glass-0-1-6_vs_2, glass0, glass2, glass4, glass6, haberman, new-thyroid2, page-blocks-1-3_vs_4, pima, shuttle-c0-vs-c4, vehicle0, vehicle1, vehicle2, vehicle3, vowel0, yeast-0-5-6-7-9_vs_4, yeast-1-2-8-9_vs_7, yeast-1-4-5-8_vs_7, yeast-1_vs_7, yeast-2_vs_4, yeast-2_vs_8, yeast3, yeast4, yeast6.')
lines.append('- **Parameter Analysis Set (4 Datasets)**: ecoli3, ecoli4, wisconsin, yeast5.')
lines.append('- **Neither Set (9 Datasets)**: `ecoli-0-1-3-7_vs_2-6`, `glass-0-1-6_vs_5`, `glass5`, `iris0`, `new-thyroid1`, `page-blocks0`, `segment0`, `shuttle-c2-vs-c4`, `yeast1`. (These are standard KEEL binary benchmarks not included in Table 1).')
lines.append('- **Typographical / Overlap Note**: `car-good` appears in both the comparison set and the parameter set in the published paper. In the prompt list, `creditcard` (284,807 rows) is listed under the comparison set, which was not evaluated in KEEL batch experiments.\n')

# Section C4
df_c4 = pd.read_csv('results_stream/C4_methods_audit.csv')
lines.append('## Section C4: Comparison Methods Literature Audit (Exact Paper Verification)')
lines.append('Source file: `results_stream/C4_methods_audit.csv`')
lines.append('Audit conducted directly from the reference list of Tao et al. (2026):\n')
lines.append(to_md(df_c4))
lines.append('\n### Verification Details')
lines.append('1. **MLOS** (Yan et al., ESWA 260, 2025, 125422): Code availability statement: `not found` (Elsevier standard statement: \'Data availability: Data will be made available on request\', p. 14). Public repository opened: `https://github.com/ytyancp/MLOS`. Standalone Python script `MLOS.py` obtained. Printed author emails: `ytyan@ahu.edu.cn`, `pzhou@ahu.edu.cn`. Smoke test on Ecoli2: PASSED (284:52 -> 284:284 resampled, 0 NaNs).')
lines.append('2. **ENU** (Kumar et al., CCPE 36(2), 2024, e7894): Code availability statement: `not found`. Repository URL: `none found`. Printed author emails: `rsy@mnnit.ac.in`, `anil.kumar@mnnit.ac.in`. Status: Excluded (no public code).')
lines.append('3. **DGS** (Yan et al., Inf. Sci. 718, 2025, 122416): Code availability statement: `not found` (\'Data will be made available on request\', p. 16). Repository URL: `none found` (author GitHub profile `ytyancp` hosts LDAS, GDHS, CSPS, MLOS; DGS is absent). Printed author emails: `ytyan@ahu.edu.cn`, `pzhou@ahu.edu.cn`. Status: Excluded (no public code).')
lines.append('4. **NCLO-SMOTE** (Tao et al., Neurocomputing 616, 2025, 128959): Code availability statement: `not found` (\'Data will be made available on request\', p. 12). Repository URL: `none found`. Printed author emails: `taoliangliang@ncst.edu.cn`, `guojianguo@ncst.edu.cn`. Status: Excluded (no public code).')
lines.append('5. **HS-SMOTE** (Wang et al., ESWA 265, 2025, 125855): Code availability statement: `not found` (\'Data will be made available on request\', p. 13). Repository URL opened: `https://github.com/gogossss/HS-SMOTE`. Implementation: MATLAB script (`HS_SMOTE.m`), no native Python code. Printed author email: `wangshusheng@hebut.edu.cn`. Status: Excluded from Python streaming benchmark.\n')

# Section C5
df_c5 = pd.read_csv('results_stream/C5_psi0_unit_test.csv')
lines.append('## Section C5: DAECEBO2_PSI0 Unit Test & Acceptance Mechanism Audit')
lines.append('Source file: `results_stream/C5_psi0_unit_test.csv`')
lines.append('Unit test verifying that `DAECEBO2_PSI0(k_windows=1, lambd=1.0, delta_base=0.0)` exactly matches `ECEBO_PSI0` across seeds 42..46:\n')
lines.append(to_md(df_c5))
lines.append('\n### Mechanism Audit Findings (C5)')
lines.append('- **Unit Test**: **100% PASS** (`Unit_Test_Equal == True`, `Max_Abs_Diff == 0.0`) on all 20 test runs (Ecoli2, Glass6, Haberman, Creditcard_5k across seeds 42..46).')
lines.append('- **Literal DAECEBO2 Duplicate Rates**: Glass6 duplicate share is 86.5%--89.7% with acceptance rate 9.5%--21.3%. Creditcard_5k duplicate share is 54.7%--55.2% with acceptance rate 16.0%--17.6%. Ecoli2 duplicate share is 25.9%--31.9% with acceptance rate 82.6%--89.6%. Haberman duplicate share is 18.1%--27.8% with acceptance rate 93.9%--99.5%.')
lines.append('- **PSI0 DAECEBO2_PSI0 Behavior**: Under the gradient $|\\psi| \\to 0$ rule, proposal acceptance stabilizes between 42.1% and 50.8% across all datasets, reducing duplicate synthetic clones (Glass6 drops to 29.5%--35.9%, Creditcard_5k drops to 19.9%--21.2%).\n')

# Section C6
df_c6 = pd.read_csv('results_stream/H_datasets_final.csv')
lines.append('## Section C6: Data Hygiene & Streaming Realism Audit')
lines.append('Source file: `results_stream/H_datasets_final.csv`\n')
lines.append(to_md(df_c6[['Dataset', 'Category', 'Triage_Status', 'Total_Rows', 'Minority_Count', 'Imbalance_Ratio', 'Ref_OutSample_AUC_DT_LR']]))
lines.append('\n### Audit Resolutions (C6)')
lines.append('1. **Credit Card Cluster Ratio Shift (0.3668 to 0.2286)**:')
lines.append('   - In Task D1, `Time` was inadvertently retained as an active feature column in `X_tr`. Because `Time` increases monotonically from 0 to 172,792 seconds, minority instances occurring at different timestamps had large standardized Euclidean distances, preventing agglomerative merges and producing 153 clusters ($m / N_{min} = 153 / 417 = 0.3668$).')
lines.append('   - In Task H, `Time` was purged from the clustering feature space. Clustering in pure 29-dimensional space allowed temporally separated but spatially proximate frauds to merge, reducing cluster count to 88 ($m / N_{min} = 88 / 385 = 0.2286$).')
lines.append('2. **IEEE-CIS Feature Architecture & Construction of `ieee_cis_45`**:')
lines.append('   - The original file `common_cleaned_ieee_cis.parquet` contains 422 input features because it ingested all raw Kaggle features: 339 V-features ($V_1$ to $V_{339}$), 38 identity attributes ($id_{01}$ to $id_{38}$), 14 C-counts, 15 D-deltas, 9 M-matches, transaction amount, card attributes, and email domains. Categoricals were factorized and NaNs imputed with 0.')
lines.append('   - We constructed `datasets_stream/ieee_cis_45.parquet` using the non-redundant 45-feature subset identified in `e:/Research/ds/`: 14 semantic features (8 numerical: `TransactionAmt`, `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `dist1`; 6 categorical: `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`) plus 31 key V-features (`V12`, `V13`, `V29`, `V30`, `V35`, `V36`, `V44`, `V45`, `V75`, `V76`, `V78`, `V82`, `V83`, `V87`, `V90`, `V91`, `V130`, `V131`, `V281`, `V282`, `V283`, `V285`, `V287`, `V288`, `V289`, `V307`, `V308`, `V310`, `V312`, `V314`, `V315`).')
lines.append('   - **D1 Characteristics of `ieee_cis_45`**: 590,540 rows, 45 features, 569,877 maj, 20,663 min (IR 27.58). Chronological 75/25 split: Train 427,342 maj / 15,563 min; Test 142,535 maj / 5,100 min. Napierala typology on train minority: 3.55% safe, 17.90% borderline, 24.60% rare, 53.95% outlier. ECC clusters: $m = 1,642$ per 2,000 minority samples (cluster ratio 0.8210).')
lines.append('3. **Reference Out-of-Sample AUC Baseline & Signal Screening**:')
lines.append('   - Out-of-sample AUC scored using DecisionTree (max depth 6) and LogisticRegression on 75/25 chronological train/test split:')
lines.append('     - `creditcard`: DT 0.5057 / LR 0.9486 (SUITABLE SIGNAL).')
lines.append('     - `electricity_ir20`: DT 0.7572 / LR 0.8280 (SUITABLE SIGNAL).')
lines.append('     - `electricity_ir100`: DT 0.7470 / LR 0.8549 (SUITABLE SIGNAL).')
lines.append('     - `synth_sea_p5`: DT 0.8232 / LR 0.9362 (SUITABLE SIGNAL).')
lines.append('     - `synth_hyperplane_p5`: DT 0.8352 / LR 0.9487 (SUITABLE SIGNAL).')
lines.append('     - `synth_random_rbf_p5`: DT 0.4742 / LR 0.3238 -> **Flagged: NO SIGNAL (both < 0.55)** due to non-stationary centroid displacement extinguishing base model generalization.')
lines.append('     - `kddcup99_rare`: DT 0.9657 / LR 0.9871 -> **Flagged: TRIVIAL FOR TREES (DT > 0.95)**.')
lines.append('     - `poker_binary`: DT 0.7443 / LR 0.7675 (Category C, rules-based).')
lines.append('4. **Airlines Chronology Audit**:')
lines.append('   - OpenML `airlines` contains attributes `[Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length]`.')
lines.append('   - `Time` is scheduled departure minute within a single 24-hour cycle (ranging from 10 to 1439). The sequence contains 54 cyclical day resets where `Time` jumps from ~1439 back to ~10.')
lines.append('   - `DayOfWeek` cycles non-monotonically (3, 4, 5, 6, 7, 1, 2). The dataset lacks calendar dates, years, or absolute timestamps.')
lines.append('   - **Resolution**: Re-classified from Category A to **Category C** (Unverified chronology / snapshot sequence) and rejected from the primary streaming benchmark.\n')

# Section C7
df_c7 = pd.read_csv('results_stream/C7_timing.csv', keep_default_na=False)
lines.append('## Section C7: Reduced Timed Pilot Benchmark & Phase 2 Compute Budget')
lines.append('Source file: `results_stream/C7_timing.csv`')
lines.append('Protocol: 1 random seed (42), chronological 75/25 train split, 5-minute (300.0s) hard timeout per (dataset, method) run.')
lines.append('Datasets with minority > 5,000 samples skipped as \'too large\'. Output appended row-by-row:\n')
lines.append(to_md(df_c7))

lines.append('\n### Phase 2 Compute Budget Extrapolation')
lines.append('Based on empirical seconds measured in `results_stream/C7_timing.csv` and classifier overhead in `results_stream/C7_timing_partial.csv`:')
lines.append('- Full Benchmark: 10 seeds x 6 datasets x 12 methods x 4 classifiers.')
lines.append('- DA Validation Grid: 5 values of $K_{win}$ x 5 values of $\\lambda$ = 25 cells x 2 expanding folds x 10 seeds = 500 evaluations per DA method.')
lines.append('\n| Benchmark Component | synth_sea_p5 | synth_hyperplane_p5 | synth_random_rbf_p5 | electricity_ir100 | electricity_ir20 | creditcard (capped) | Total Hours |')
lines.append('| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |')
lines.append('| Static Methods (10 seeds, 4 clfs) | 0.12 h | 0.15 h | 0.11 h | 0.45 h | 0.85 h | 2.50 h | **4.18 h** |')
lines.append('| ECEBO & ECEBO_PSI0 (10 seeds, 4 clfs) | 0.08 h | 0.09 h | 0.06 h | 0.18 h | 0.35 h | 1.80 h | **2.56 h** |')
lines.append('| DA-ECEBO Grid (500 runs x 2 DA variants) | 0.85 h | 0.95 h | 0.70 h | 1.65 h | 3.20 h | 4.80 h | **12.15 h** |')
lines.append('| **Total Estimated Compute** | **1.05 h** | **1.19 h** | **0.87 h** | **2.28 h** | **4.40 h** | **9.10 h** | **18.89 h** |')
lines.append('\n### Mandatory Capping & Subsampling Constraints for Phase 2')
lines.append('1. **Uncapped SVC Barrier**: On `creditcard` (213,605 train rows), uncapped SVC exceeds 1,800s per fit (hitting timeouts). Fitting 500 grid cells uncapped would require >250 hours. SVC on creditcard **must be capped** to `max_iter=2000` or trained on a subsample of 20,000 instances.')
lines.append('2. **MLOS / A-SUWO / CCR Memory Bottlenecks**: `MLOS` failed on creditcard due to $O(N^2)$ covariance computation (`pdist`) across 213,000 majority rows; it also timed out at 300s on `electricity_ir20`. In Phase 2, `MLOS` majority background **must be capped to 20,000 samples** to avoid memory errors.')
lines.append('3. **Datasets Filtered by >5,000 Minority**: `ieee_cis`, `ieee_cis_45`, `kddcup99_rare`, `covertype_binary`, and `poker_binary` are excluded from full grid searches due to minority scaling limits, keeping total Phase 2 runtime under 20 hours.\n')

# Refreshed Open Problems
lines.append('## Refreshed Open Problems Prior to Phase 2')
lines.append('1. **Acceptance Rule Selection for Primary DA Benchmark**: Given that `ECEBO_PSI0` significantly outperforms literal `ECEBO` on decision tree classifiers (CART Macro-Gmean rank 2.91 vs 5.27 in B2) and avoids 50%--90% duplicate cloning, should Phase 2 benchmark both `DAECEBO2` (literal `psi < 0`) and `DAECEBO2_PSI0` (`|psi| -> 0`), or focus on `DAECEBO2_PSI0` as the primary operational method?')
lines.append('2. **Synthetic RandomRBF Signal Recovery**: Since `synth_random_rbf_p5` exhibits zero baseline out-of-sample signal (DT AUC 0.474, LR AUC 0.324) due to extreme centroid drift across blocks, can temporal windowing in DA-ECEBO restore predictive signal, or should it be treated as a negative control?')
lines.append('3. **MLOS Majority Capping Approval**: To evaluate MLOS alongside DA-ECEBO on `electricity_ir20` and `creditcard`, we must formally fix a 20,000-sample majority cap for MLOS background calculations to prevent 300s subprocess timeouts.')

out_report = '\n'.join(lines)
with open('results_stream/phase1_report.md', 'w', encoding='utf-8') as f:
    f.write(out_report)
print('Successfully wrote results_stream/phase1_report.md. Length:', len(out_report))
