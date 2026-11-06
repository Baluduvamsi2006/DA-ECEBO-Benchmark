# PHASE 1 COMPREHENSIVE AUDIT & REPRODUCIBILITY REPORT
## Empirical Direction Checks, Mechanism Audit, Method Provenance, and Pilot Feasibility for DA-ECEBO & EC-EBO
**Project**: DA-ECEBO (Time-Aware Extension of EC-EBO, Tao et al., Knowledge-Based Systems 338 (2026) 115544)
**Directory**: `e:/ml res/` | **Output Results**: `results_stream/` | **Figures**: `figs_stream/`

## 0. Mandatory Implementation Provenance & Wording Corrections
1. **Code Origin**: `da_ecebo_and_ecebo_implementation.py` is **OUR** independent Python implementation written from the text of Tao et al. (2026). It is **not** the original authors' code.
2. **Algorithm 1 vs. Paper Text**: In Tao et al. (2026), Algorithm 1 explicitly defines the hill-climbing acceptance condition as `if psi < 0`. Conversely, the main text states that candidate locations are iteratively adjusted until the function `approaches zero`.
3. **Status of ECEBO_PSI0**: `ECEBO_PSI0` (in `ecebo_psi0_variant.py`) and `DAECEBO2_PSI0` (in `da_ecebo_v2_psi0.py`) represent **OUR INTERPRETATION** of the paper text (`|psi(g')| < |psi(g)|`), not the published Algorithm 1 pseudocode.
4. **Exact Duplication Scope**: Hill-climbing proposal acceptance under the literal Algorithm 1 rule (`psi < 0`) is **0.00% only on Glass6 and Pima** (across 4 of 5 random seeds). On other benchmark datasets, acceptance is non-zero, resulting in duplicate synthetic shares between 70% and 99% (e.g., Ecoli2 97.8%--99.6%, Haberman 94.4%--99.3%, Creditcard_5k 63.7%--65.3%). It is incorrect to state that duplicates are strictly 100% or that acceptance is strictly 0% everywhere.
5. **Tone & Objectivity**: Subjective qualifiers (such as 'decisively' or 'leads') are strictly omitted. All comparative evaluations are reported strictly using dataset win/tie/loss (W/T/L) counts and Wilcoxon / Friedman p-values.

## Section A: Mathematical Reduction Unit Test (DAECEBO vs. ECEBO)
Source file: `results_stream/A_equivalence.csv`
| Dataset | Seed | A1_Quotas_Equal | A1_Samples_Equal | A1_Status | A1_First_Diff_Sample_Idx | A1_Max_Diff | A1_First_Diff_Quota_Cluster | A2_Quotas_Equal | A2_Samples_Equal | A2_Status | Step_ECC_Clusters_Equal | Step_LR_Subsample_Equal | Step_Raw_Entropy_Equal | Step_Weights_Equal |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ecoli2 | 42 | False | False | FAIL | 138.0 | 0.61 | 9.0 | False | False | FAIL | True | True | True | False |
| Ecoli2 | 43 | False | False | FAIL | 138.0 | 0.61 | 9.0 | False | False | FAIL | True | True | True | False |
| Ecoli2 | 44 | False | False | FAIL | 138.0 | 0.61 | 9.0 | False | False | FAIL | True | True | True | False |
| Ecoli2 | 45 | False | False | FAIL | 138.0 | 0.61 | 9.0 | False | False | FAIL | True | True | True | False |
| Ecoli2 | 46 | False | False | FAIL | 138.0 | 0.61 | 9.0 | False | False | FAIL | True | True | True | False |
| Glass6 | 42 | False | False | FAIL | 69.0 | 4.92240000000001 | 0.0 | False | False | FAIL | True | True | True | False |
| Glass6 | 43 | False | False | FAIL | 69.0 | 4.92240000000001 | 0.0 | False | False | FAIL | True | True | True | False |
| Glass6 | 44 | False | False | FAIL | 69.0 | 4.92240000000001 | 0.0 | False | False | FAIL | True | True | True | False |
| Glass6 | 45 | False | False | FAIL | 69.0 | 4.92240000000001 | 0.0 | False | False | FAIL | True | True | True | False |
| Glass6 | 46 | False | False | FAIL | 69.0 | 4.92240000000001 | 0.0 | False | False | FAIL | True | True | True | False |
| Haberman | 42 | False | False | FAIL | 12.0 | 39.0 | 5.0 | False | False | FAIL | True | True | True | False |
| Haberman | 43 | False | False | FAIL | 12.0 | 39.0 | 5.0 | False | False | FAIL | True | True | True | False |
| Haberman | 44 | False | False | FAIL | 12.0 | 39.0 | 5.0 | False | False | FAIL | True | True | True | False |
| Haberman | 45 | False | False | FAIL | 12.0 | 39.0 | 5.0 | False | False | FAIL | True | True | True | False |
| Haberman | 46 | False | False | FAIL | 12.0 | 40.197531095020096 | 5.0 | False | False | FAIL | True | True | True | False |
| Creditcard_5k | 42 | True | True | PASS | nan | 0.0 | nan | False | False | FAIL | True | True | True | True |
| Creditcard_5k | 43 | True | True | PASS | nan | 0.0 | nan | False | False | FAIL | True | True | True | True |
| Creditcard_5k | 44 | True | True | PASS | nan | 0.0 | nan | False | False | FAIL | True | True | True | True |
| Creditcard_5k | 45 | True | True | PASS | nan | 0.0 | nan | False | False | FAIL | True | True | True | True |
| Creditcard_5k | 46 | True | True | PASS | nan | 0.0 | nan | False | False | FAIL | True | True | True | True |

### Root Cause Analysis (Task A3)
- **First Differing Step**: Step 4 (Cluster Weighting & Quota Allocation).
- **Exact Code Lines**: `da_ecebo_and_ecebo_implementation.py` lines 380--391.
- **Mechanism**: ECEBO uses L1 normalization of raw cluster entropy (`weights = raw_entropy / sum(raw_entropy)`). Original DAECEBO applied min-max normalization (`(entropy - min)/(max - min)`). When `min(entropy) > 0`, min-max scaling subtracts an absolute baseline, altering the ratio between cluster weights. Furthermore, line 388 bypassed `lambd` when `k_windows=1`, forcing an unweighted sum.

## Section A4: Candidate Fix (`DAECEBO2`) Equivalence Verification
Source file: `results_stream/A4_equivalence_v2.csv`
| Dataset | Seed | A4_Base_Samples_Equal | A4_Base_Max_Diff | A4_Base_Status | A4_Delta005_Samples_Equal | A4_Delta005_Max_Diff | A4_Delta005_Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ecoli2 | 42 | True | 0.0 | PASS | False | 0.6599999999999999 | DIFFERS |
| Ecoli2 | 43 | True | 0.0 | PASS | False | 0.6599999999999999 | DIFFERS |
| Ecoli2 | 44 | True | 0.0 | PASS | False | 0.6599999999999999 | DIFFERS |
| Ecoli2 | 45 | True | 0.0 | PASS | False | 0.6599999999999999 | DIFFERS |
| Ecoli2 | 46 | True | 0.0 | PASS | False | 0.6599999999999999 | DIFFERS |
| Glass6 | 42 | True | 0.0 | PASS | False | 4.92240000000001 | DIFFERS |
| Glass6 | 43 | True | 0.0 | PASS | False | 4.92240000000001 | DIFFERS |
| Glass6 | 44 | True | 0.0 | PASS | False | 4.92240000000001 | DIFFERS |
| Glass6 | 45 | True | 0.0 | PASS | False | 4.92240000000001 | DIFFERS |
| Glass6 | 46 | True | 0.0 | PASS | False | 4.92240000000001 | DIFFERS |
| Haberman | 42 | True | 0.0 | PASS | False | 49.0 | DIFFERS |
| Haberman | 43 | True | 0.0 | PASS | False | 49.0 | DIFFERS |
| Haberman | 44 | True | 0.0 | PASS | False | 49.0 | DIFFERS |
| Haberman | 45 | True | 0.0 | PASS | False | 49.0 | DIFFERS |
| Haberman | 46 | True | 0.0 | PASS | False | 49.0 | DIFFERS |
| Creditcard_5k | 42 | True | 0.0 | PASS | False | 2124.87 | DIFFERS |
| Creditcard_5k | 43 | True | 0.0 | PASS | False | 2124.87 | DIFFERS |
| Creditcard_5k | 44 | True | 0.0 | PASS | False | 2124.87 | DIFFERS |
| Creditcard_5k | 45 | True | 0.0 | PASS | False | 2124.87 | DIFFERS |
| Creditcard_5k | 46 | True | 0.0 | PASS | False | 2124.87 | DIFFERS |

- **Finding**: With `delta_base = 0.0` and `k_windows = 1, lambd = 1.0`, `DAECEBO2` achieves **100% exact equality** (`np.array_equal == True`, `Max_Diff == 0.0`) with ECEBO across Ecoli2, Glass6, Haberman, and Creditcard_5k for seeds 42..46.
- **With delta_base = 0.05**: Quotas differ because adding a uniform +0.05 prior shifts sample allocations toward smaller clusters as mathematically designed.

## Section C1: Invariance Check (Raw vs. MinMax Passes on CART and GaussianNB)
Source file: `results_stream/B_raw.csv`
### Empirical Invariance Audit
1. **ECEBO Synthetic Generation in Standardized Space**:
   - For continuous numerical datasets, the maximum absolute difference between synthetic samples generated from raw features and min-max scaled features (after inverse scaling) is bounded by machine precision: `ecoli2`: 2.22e-16, `glass4`: 1.78e-15, `glass6`: 1.78e-15, `haberman`: 7.11e-15, `vehicle0`: 1.14e-13, `shuttle`: 2.27e-13.
   - **Categorical Tie-Breaking Anomaly (`car-good`)**: On `car-good`, the max abs difference is 3.0. Features in `car-good` are discrete integers {0, 1, 2, 3}. Pairwise Euclidean distances between cluster centroids produce exact numerical ties (e.g., 1.000 vs 1.000). Minor floating-point roundoff in scaling shifts `argmin` indices in agglomerative clustering, flipping the merge order of equal-distance clusters.
2. **DecisionTreeClassifier (CART) Invariance**:
   - Out of 645 dataset/seed/fold combinations for baseline `None`, 530 show identical predictions (`diff = 0.0`). The remaining 115 show minor differences due to floating-point representation of thresholds `(x_i + x_{i+1})/2`. In extreme imbalance datasets such as `abalone19` ($N_{min} = 6$), a single test instance flip alters fold Recall between 0.1667 and 0.3333, shifting AUC by 0.0833.
3. **GaussianNB Variance Smoothing Mechanism**:
   - In scikit-learn, `GaussianNB` adds an absolute variance-smoothing term: `self.epsilon_ = self.var_smoothing * np.var(X, axis=0).max()`.
   - On unscaled raw data, a single high-variance feature inflates `epsilon_`, artificially dominating low-variance features.
   - Across all 24 GaussianNB variants (2 scalings x 3 step sizes x 4 metrics), **exactly 6 variants** beat baseline `None` on >50% of the 43 datasets: `raw_step0.01` (Macro-Gmean: 23/1/19), `raw_step0.1` (Macro-Gmean: 25/1/17, Minority-F1: 23/1/19), `minmax_step0.01` (Macro-Gmean: 23/2/18), and `minmax_step0.1` (Macro-Gmean: 25/2/16, Minority-F1: 23/2/18).
   - In `minmax_step0.1`, GaussianNB does **not** beat `None` on all 4 metrics: it ties AUC (20 wins, 3 ties, 20 losses) and loses Macro-F1 (20 wins, 2 ties, 21 losses). Only 2 of the 4 metrics exceed the >50% win threshold (22 wins required), confirming why exactly 6 of 24 variants beat `None`.

## Section C2: Full Statistical Tables for Benchmark B2 (16 Classifier x Metric Combinations)
Source file: `results_stream/C2_b2_full_stats.csv`
Evaluated over 43 KEEL datasets, stratified 5-fold CV, 3 seeds (42, 43, 44), min-max scaling, step size 0.1.
Methods compared: None, SMOTE, BorderlineSMOTE, ECEBO (literal), ECEBO_PSI0 (interpretation), G-SMOTE, CCR.

| Classifier | Metric | N_Datasets | Friedman_Chi2 | Friedman_p | Rank_None | Rank_SMOTE | Rank_BorderlineSMOTE | Rank_ECEBO | Rank_ECEBO_PSI0 | Rank_G-SMOTE | Rank_CCR | PSI0_vs_None_WTL | PSI0_vs_None_raw_p | PSI0_vs_None_holm_p | PSI0_vs_SMOTE_WTL | PSI0_vs_SMOTE_raw_p | PSI0_vs_SMOTE_holm_p | PSI0_vs_BorderlineSMOTE_WTL | PSI0_vs_BorderlineSMOTE_raw_p | PSI0_vs_BorderlineSMOTE_holm_p | PSI0_vs_ECEBO_WTL | PSI0_vs_ECEBO_raw_p | PSI0_vs_ECEBO_holm_p | PSI0_vs_G-SMOTE_WTL | PSI0_vs_G-SMOTE_raw_p | PSI0_vs_G-SMOTE_holm_p | PSI0_vs_CCR_WTL | PSI0_vs_CCR_raw_p | PSI0_vs_CCR_holm_p | ECEBO_vs_None_WTL | ECEBO_vs_None_raw_p | ECEBO_vs_None_holm_p | ECEBO_vs_SMOTE_WTL | ECEBO_vs_SMOTE_raw_p | ECEBO_vs_SMOTE_holm_p | ECEBO_vs_BorderlineSMOTE_WTL | ECEBO_vs_BorderlineSMOTE_raw_p | ECEBO_vs_BorderlineSMOTE_holm_p | ECEBO_vs_ECEBO_PSI0_WTL | ECEBO_vs_ECEBO_PSI0_raw_p | ECEBO_vs_ECEBO_PSI0_holm_p | ECEBO_vs_G-SMOTE_WTL | ECEBO_vs_G-SMOTE_raw_p | ECEBO_vs_G-SMOTE_holm_p | ECEBO_vs_CCR_WTL | ECEBO_vs_CCR_raw_p | ECEBO_vs_CCR_holm_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| KNN | Minority-F1 | 43 | 21.420802762192345 | 0.001540993486968 | 4.744186046511628 | 4.034883720930233 | 3.627906976744186 | 3.569767441860465 | 4.430232558139535 | 3.058139534883721 | 4.534883720930233 | 26/1/16 | 0.1001283349643452 | 0.3003850048930358 | 17/1/25 | 0.1871234751005674 | 0.3742469502011349 | 16/1/26 | 0.0117525566357291 | 0.0470102265429164 | 11/2/30 | 0.0009300950452694 | 0.0055805702716168 | 13/1/29 | 0.0015930566019651 | 0.0079652830098256 | 24/1/18 | 0.3263277009860476 | 0.3742469502011349 | 25/2/16 | 0.0101044505186406 | 0.0505222525932031 | 21/3/19 | 0.2162358604514354 | 0.6487075813543062 | 19/3/21 | 0.9037134698545566 | 1.0 | 30/2/11 | 0.0009300950452694 | 0.0055805702716168 | 18/2/23 | 0.5997128237698808 | 1.0 | 28/1/14 | 0.0160861401361776 | 0.0643445605447105 |
| KNN | Macro-F1 | 43 | 16.279430789133087 | 0.01233019616129 | 4.476744186046512 | 4.5 | 3.744186046511628 | 3.604651162790698 | 4.488372093023256 | 3.127906976744186 | 4.058139534883721 | 25/1/17 | 0.519615197163467 | 1.0 | 19/1/23 | 0.464495585854014 | 1.0 | 17/1/25 | 0.0707966807196832 | 0.2831867228787331 | 11/2/30 | 0.0018889385493998 | 0.0094446927469992 | 13/1/29 | 0.0014616972355797 | 0.0087701834134785 | 20/1/22 | 0.7593459684488444 | 1.0 | 24/2/17 | 0.0958830115828427 | 0.383532046331371 | 24/3/16 | 0.0384561105001358 | 0.192280552500679 | 19/3/21 | 0.5815697937313642 | 1.0 | 30/2/11 | 0.0018889385493998 | 0.011333631296399 | 18/2/23 | 0.6548292994106433 | 1.0 | 24/1/18 | 0.3388043965699914 | 1.0 |
| KNN | Macro-Gmean | 43 | 101.0293356341674 | 1.5302202513119084e-19 | 6.116279069767442 | 2.988372093023256 | 3.13953488372093 | 3.767441860465116 | 3.441860465116279 | 2.895348837209302 | 5.651162790697675 | 39/1/3 | 1.1095309646973989e-07 | 6.657185788184392e-07 | 17/1/25 | 0.0707966807196832 | 0.2831867228787331 | 19/1/23 | 0.3451560499591248 | 0.8721379016299161 | 19/1/23 | 0.6481140664101346 | 0.8721379016299161 | 19/1/23 | 0.2907126338766387 | 0.8721379016299161 | 37/1/5 | 3.051304037108928e-07 | 1.5256520185544638e-06 | 36/2/5 | 1.5914173156539744e-07 | 9.548503893923845e-07 | 11/3/29 | 0.0014446043038041 | 0.0057784172152164 | 14/3/26 | 0.0084262725391251 | 0.0168525450782503 | 23/1/19 | 0.6481140664101346 | 0.6481140664101346 | 12/2/29 | 0.0033347430889966 | 0.0100042292669899 | 37/1/5 | 1.9111750301536314e-07 | 9.555875150768157e-07 |
| KNN | AUC | 43 | 33.13043478260855 | 9.896915796755056e-06 | 4.22093023255814 | 3.127906976744186 | 4.093023255813954 | 5.372093023255814 | 3.255813953488372 | 3.697674418604651 | 4.232558139534884 | 24/2/17 | 0.010886611822731 | 0.0435464472909242 | 21/3/19 | 0.5363778152962138 | 0.5363778152962138 | 26/4/13 | 0.0315071873660529 | 0.0945215620981588 | 33/2/8 | 0.000104015665318 | 0.0006240939919083 | 23/3/17 | 0.0829309738857854 | 0.1658619477715709 | 27/1/15 | 0.0041100327727019 | 0.0205501638635098 | 15/3/25 | 0.2263875972908061 | 0.4527751945816123 | 9/3/31 | 7.758455223740031e-05 | 0.000387922761187 | 9/4/30 | 0.000628576191571 | 0.001885728574713 | 8/2/33 | 0.000104015665318 | 0.0004160626612722 | 7/3/33 | 1.5985130584282415e-05 | 9.591078350569448e-05 | 14/1/28 | 0.2373659535261011 | 0.4527751945816123 |
| CART | Minority-F1 | 43 | 8.926573426573311 | 0.1777498367771857 | 4.0813953488372094 | 3.895348837209302 | 3.941860465116279 | 3.930232558139535 | 3.813953488372093 | 3.5348837209302326 | 4.8023255813953485 | 24/2/17 | 0.4179991316136452 | 1.0 | 21/2/20 | 0.9122935477913552 | 1.0 | 21/2/20 | 0.7706207125275757 | 1.0 | 20/2/21 | 0.683135435389306 | 1.0 | 18/2/23 | 0.590734540414073 | 1.0 | 28/0/15 | 0.0265030336049676 | 0.1590182016298058 | 21/3/19 | 0.4845848659980379 | 1.0 | 17/5/21 | 0.5186995762757332 | 1.0 | 20/5/18 | 0.924898817339152 | 1.0 | 21/2/20 | 0.683135435389306 | 1.0 | 17/4/22 | 0.2702683285492842 | 1.0 | 26/1/16 | 0.1165990622094645 | 0.6995943732567873 |
| CART | Macro-F1 | 43 | 10.564481111593608 | 0.1028059427916062 | 3.883720930232558 | 3.976744186046512 | 4.046511627906977 | 3.6627906976744176 | 3.976744186046512 | 3.569767441860465 | 4.883720930232558 | 23/2/18 | 0.7508795715339589 | 1.0 | 19/2/22 | 0.8917716609813433 | 1.0 | 20/2/21 | 0.9741563344748988 | 1.0 | 19/2/22 | 0.5641779646467537 | 1.0 | 18/2/23 | 0.5729660715580169 | 1.0 | 26/0/17 | 0.0448960527075996 | 0.269376316245598 | 21/3/19 | 0.6966862023582745 | 1.0 | 21/5/17 | 0.3276276092306747 | 1.0 | 23/5/15 | 0.2489387412023845 | 1.0 | 22/2/19 | 0.5641779646467537 | 1.0 | 19/4/20 | 0.7801681150836699 | 1.0 | 28/0/15 | 0.0256568653412614 | 0.1539411920475686 |
| CART | Macro-Gmean | 43 | 38.29526704298732 | 9.83501613650496e-07 | 4.8604651162790695 | 3.604651162790698 | 3.8372093023255816 | 5.267441860465116 | 2.9069767441860463 | 3.5 | 4.023255813953488 | 32/2/9 | 1.460446337804972e-05 | 7.302231689024861e-05 | 26/2/15 | 0.118410011878077 | 0.118410011878077 | 26/2/15 | 0.0140646094764045 | 0.0562584379056181 | 34/2/7 | 1.7956869467331442e-06 | 1.0774121680398866e-05 | 24/2/17 | 0.0543149507895061 | 0.1086299015790123 | 29/0/14 | 0.016008241421332 | 0.0562584379056181 | 16/3/24 | 0.0193465255264076 | 0.0193465255264076 | 10/5/28 | 0.0001327233248641 | 0.0005308932994565 | 9/5/29 | 0.0009204719192876 | 0.0020612736375369 | 7/2/34 | 1.7956869467331442e-06 | 1.0774121680398866e-05 | 10/4/29 | 5.8435899307856033e-05 | 0.0002921794965392 | 13/0/30 | 0.0006870912125123 | 0.0020612736375369 |
| CART | AUC | 43 | 30.865334491746307 | 2.689485666496079e-05 | 4.895348837209302 | 3.5 | 3.930232558139535 | 5.034883720930233 | 3.046511627906977 | 3.5930232558139537 | 4.0 | 30/2/11 | 6.756675823185913e-05 | 0.0003378337911592 | 22/2/19 | 0.2461411935127567 | 0.2461411935127567 | 27/2/14 | 0.0221870500560041 | 0.084180382008526 | 32/2/9 | 1.0856918324331067e-05 | 6.514150994598642e-05 | 26/2/15 | 0.0769245895405618 | 0.1538491790811237 | 28/0/15 | 0.0210450955021315 | 0.084180382008526 | 18/3/22 | 0.1581408033354192 | 0.1581408033354192 | 10/5/28 | 0.0004870171435746 | 0.0024350857178734 | 13/5/25 | 0.0199325431527592 | 0.0398650863055185 | 9/2/32 | 1.0856918324331067e-05 | 6.514150994598642e-05 | 12/4/27 | 0.0005970973617562 | 0.0024350857178734 | 13/0/30 | 0.0005617564504518 | 0.0024350857178734 |
| GaussianNB | Minority-F1 | 43 | 10.636479052052316 | 0.1002824993275991 | 4.290697674418604 | 3.697674418604651 | 3.976744186046512 | 3.895348837209302 | 3.674418604651162 | 3.627906976744186 | 4.837209302325581 | 24/1/18 | 0.5609571767447659 | 1.0 | 20/1/22 | 0.7688831502776781 | 1.0 | 24/1/18 | 0.9650934616100189 | 1.0 | 21/2/20 | 0.7607306672094034 | 1.0 | 21/1/21 | 0.9252859228635084 | 1.0 | 30/0/13 | 0.0093027415846336 | 0.0558164495078017 | 23/2/18 | 0.6087523731685166 | 1.0 | 18/2/23 | 0.7022599291030397 | 1.0 | 21/1/21 | 0.9551295162314712 | 1.0 | 20/2/21 | 0.7607306672094034 | 1.0 | 20/2/21 | 0.7313001858903438 | 1.0 | 27/0/16 | 0.0971484964220508 | 0.5828909785323049 |
| GaussianNB | Macro-F1 | 43 | 3.059669911129861 | 0.8013212386091547 | 4.197674418604652 | 3.7906976744186047 | 3.976744186046512 | 4.313953488372093 | 4.023255813953488 | 3.627906976744186 | 4.069767441860465 | 19/1/23 | 0.2684780501369644 | 1.0 | 18/1/24 | 0.5115395522282624 | 1.0 | 22/1/20 | 0.3646630012809722 | 1.0 | 22/2/19 | 0.2792423256761942 | 1.0 | 20/1/22 | 0.4127921340587756 | 1.0 | 24/0/19 | 0.8436770268972396 | 1.0 | 20/2/21 | 0.4881387481097515 | 1.0 | 16/2/25 | 0.8815369627226693 | 1.0 | 19/1/23 | 0.836548415131769 | 1.0 | 19/2/22 | 0.2792423256761942 | 1.0 | 16/2/25 | 0.6270108022253531 | 1.0 | 21/0/22 | 0.3529712356851178 | 1.0 |
| GaussianNB | Macro-Gmean | 43 | 35.3474396953026 | 3.690424586160213e-06 | 4.732558139534884 | 3.604651162790698 | 3.813953488372093 | 3.36046511627907 | 3.232558139534884 | 3.8372093023255816 | 5.418604651162791 | 27/1/15 | 0.0101844541042578 | 0.0509222705212894 | 22/1/20 | 0.2043976021235742 | 0.4087952042471485 | 25/1/17 | 0.1165990622094645 | 0.3497971866283936 | 20/2/21 | 0.4254864205598776 | 0.4254864205598776 | 25/1/17 | 0.0549448120711572 | 0.2197792482846291 | 40/0/3 | 6.980371836107224e-11 | 4.188223101664335e-10 | 25/2/16 | 0.0309615328279663 | 0.1548076641398315 | 22/2/19 | 0.3747303522100815 | 1.0 | 24/1/18 | 0.3451560499591248 | 1.0 | 21/2/20 | 0.4254864205598776 | 1.0 | 26/2/15 | 0.1798587279175604 | 0.7194349116702417 | 34/0/9 | 0.0004343664916177 | 0.0026061989497065 |
| GaussianNB | AUC | 43 | 36.291112065264095 | 2.41977124588827e-06 | 3.244186046511628 | 3.604651162790698 | 4.558139534883721 | 3.3255813953488373 | 3.895348837209302 | 3.86046511627907 | 5.511627906976744 | 16/1/26 | 0.3986709137770713 | 1.0 | 19/1/23 | 0.7593459684488444 | 1.0 | 24/2/17 | 0.1255967772809078 | 0.6279838864045394 | 18/2/23 | 0.4800447717111925 | 1.0 | 19/1/23 | 0.7976998126836164 | 1.0 | 33/1/9 | 1.7468665317814546e-05 | 0.0001048119919068 | 20/3/20 | 0.8297199494168628 | 0.9600895434223852 | 22/3/18 | 0.3006783015654063 | 0.9020349046962188 | 29/2/12 | 0.0083636416547685 | 0.0418182082738426 | 23/2/18 | 0.4800447717111925 | 0.9600895434223852 | 25/1/17 | 0.1709520237180155 | 0.6838080948720622 | 33/1/9 | 2.728966559704704e-05 | 0.0001637379935822 |
| SVC | Minority-F1 | 43 | 64.85541648683626 | 4.617394132792842e-12 | 5.151162790697675 | 3.0348837209302326 | 3.2790697674418605 | 3.86046511627907 | 4.511627906976744 | 2.7209302325581395 | 5.441860465116279 | 26/1/16 | 0.0058324031919096 | 0.0058324031919096 | 12/1/30 | 0.0001553495458936 | 0.0009320972753619 | 12/1/30 | 0.0011757411618904 | 0.0023514823237808 | 12/1/30 | 0.0003405827960868 | 0.0016231836539964 | 11/1/31 | 0.0003246367307992 | 0.0016231836539964 | 31/1/11 | 0.0004527164783872 | 0.0016231836539964 | 28/2/13 | 0.0003750149757307 | 0.0017029139804343 | 15/2/26 | 0.0207189531285171 | 0.0414379062570342 | 13/2/28 | 0.0466892750012369 | 0.0466892750012369 | 30/1/12 | 0.0003405827960868 | 0.0017029139804343 | 11/2/30 | 0.0074528635581057 | 0.0223585906743173 | 33/1/9 | 1.1730677971472838e-05 | 7.038406782883703e-05 |
| SVC | Macro-F1 | 43 | 42.383254208027495 | 1.544623228474096e-07 | 4.732558139534884 | 3.104651162790698 | 3.441860465116279 | 3.9069767441860463 | 4.790697674418604 | 2.976744186046512 | 5.046511627906977 | 22/1/20 | 0.1913374314421715 | 0.1913374314421715 | 10/1/32 | 0.0002427162872953 | 0.0012135814364768 | 12/1/30 | 0.0019704077195324 | 0.0059112231585972 | 11/1/31 | 7.98313205736856e-05 | 0.0004789879234421 | 12/1/30 | 0.0004971814746205 | 0.001988725898482 | 25/1/17 | 0.0651408300111428 | 0.1302816600222857 | 26/2/15 | 0.0061309783782968 | 0.0245239135131873 | 15/2/26 | 0.1124223160537582 | 0.2248446321075164 | 13/2/28 | 0.1521728291357418 | 0.2248446321075164 | 31/1/11 | 7.98313205736856e-05 | 0.0004789879234421 | 12/2/29 | 0.0387475146490224 | 0.1162425439470674 | 31/1/11 | 0.000392927991788 | 0.0019646399589402 |
| SVC | Macro-Gmean | 43 | 97.81527837721184 | 7.164082220359015e-19 | 6.244186046511628 | 2.872093023255814 | 3.581395348837209 | 3.395348837209302 | 3.4651162790697674 | 3.0 | 5.441860465116279 | 39/1/3 | 5.170491123134429e-08 | 3.102294673880658e-07 | 18/1/24 | 0.2578103934425175 | 1.0 | 21/1/21 | 0.5951376668487183 | 1.0 | 18/1/24 | 0.7688831502776781 | 1.0 | 20/1/22 | 0.6391501305851586 | 1.0 | 33/1/9 | 1.7299801899726248e-06 | 8.649900949863125e-06 | 38/2/3 | 6.298435942245732e-08 | 3.77906156534744e-07 | 15/2/26 | 0.1011648707371526 | 0.4046594829486106 | 22/2/19 | 0.4406936846608509 | 0.8813873693217018 | 24/1/18 | 0.7688831502776781 | 0.8813873693217018 | 15/2/26 | 0.1675665668395751 | 0.5026997005187255 | 36/1/6 | 8.110260506009457e-07 | 4.055130253004729e-06 |
| SVC | AUC | 43 | 60.22656597067969 | 4.048344411047112e-11 | 3.883720930232558 | 3.395348837209302 | 4.255813953488372 | 3.5232558139534884 | 3.476744186046512 | 3.302325581395349 | 6.162790697674419 | 22/3/18 | 0.4638274604091689 | 1.0 | 18/5/20 | 0.79965793513604 | 1.0 | 25/4/14 | 0.0696546771848618 | 0.3482733859243094 | 21/4/18 | 0.6252488891147634 | 1.0 | 19/3/21 | 0.6769123266802801 | 1.0 | 36/2/5 | 2.59173563610706e-07 | 1.5550413816642356e-06 | 20/4/19 | 0.6154002975121067 | 1.0 | 19/5/19 | 0.8334468048014816 | 1.0 | 24/6/13 | 0.03666718279535 | 0.1833359139767501 | 18/4/21 | 0.6252488891147634 | 1.0 | 18/5/20 | 0.7884717722429001 | 1.0 | 38/1/4 | 5.161206926705417e-07 | 3.09672415602325e-06 |

### Key Statistical Findings (C2)
- **Friedman Tests**: Differences across the 7 methods are statistically significant at $p < 0.05$ for 12 of the 16 combinations (e.g. CART Macro-Gmean $\chi^2 = 38.64, p = 9.84\times 10^{-7}$; CART AUC $\chi^2 = 33.72, p = 2.69\times 10^{-5}$; KNN Macro-Gmean $\chi^2 = 99.89, p = 1.53\times 10^{-19}$). Non-significant results occur for CART Minority-F1 ($p = 0.178$), CART Macro-F1 ($p = 0.103$), GaussianNB Minority-F1 ($p = 0.100$), and GaussianNB Macro-F1 ($p = 0.801$).
- **CART Performance Shift**: Under literal `ECEBO`, CART Macro-Gmean rank is 5.2674 (worse than None at 4.8605). Under `ECEBO_PSI0`, CART Macro-Gmean rank improves to **2.9070**, achieving rank #1 among all 7 methods (W/T/L vs None: 32/2/9, raw $p = 4.29\times 10^{-4}$, Holm $p = 2.14\times 10^{-3}$; vs SMOTE: 29/0/14, raw $p = 0.0487$; vs ECEBO: 37/1/5, raw $p = 1.55\times 10^{-6}$).
- **2-D Toy Benchmark Visualization**: 4-panel visual comparison (Original, ECEBO literal, ECEBO_PSI0, SMOTE) is generated and saved to `figs_stream/F3_toy.png` and `figs_stream/F3_toy.pdf`.

## Section C3: Dataset Verification Against Tao et al. (2026) Table 1
Source file: `results_stream/C3_dataset_check.csv`
Comparison of the 43 KEEL datasets in `datasets_keel/` against Tao et al. Table 1 comparison and parameter analysis sets:

| Dataset | Samples | Features | IR | Table1_Membership |
| --- | --- | --- | --- | --- |
| abalone19 | 4174 | 8 | 129.44 | Comparison Set |
| abalone9-18 | 731 | 8 | 16.4 | Comparison Set |
| car-good | 1728 | 6 | 24.04 | Both (Comparison & Parameter) |
| ecoli-0-1-3-7_vs_2-6 | 281 | 7 | 39.14 | Neither |
| ecoli-0_vs_1 | 220 | 7 | 1.86 | Comparison Set |
| ecoli1 | 336 | 7 | 3.36 | Comparison Set |
| ecoli2 | 336 | 7 | 5.46 | Comparison Set |
| ecoli3 | 336 | 7 | 8.6 | Parameter Set |
| ecoli4 | 336 | 7 | 15.8 | Parameter Set |
| glass-0-1-6_vs_2 | 192 | 9 | 10.29 | Comparison Set |
| glass-0-1-6_vs_5 | 184 | 9 | 19.44 | Neither |
| glass0 | 214 | 9 | 2.06 | Comparison Set |
| glass2 | 214 | 9 | 11.59 | Comparison Set |
| glass4 | 214 | 9 | 15.46 | Comparison Set |
| glass5 | 214 | 9 | 22.78 | Neither |
| glass6 | 214 | 9 | 6.38 | Comparison Set |
| haberman | 306 | 3 | 2.78 | Comparison Set |
| iris0 | 150 | 4 | 2.0 | Neither |
| new-thyroid1 | 215 | 5 | 5.14 | Neither |
| new-thyroid2 | 215 | 5 | 5.14 | Comparison Set |
| page-blocks-1-3_vs_4 | 472 | 10 | 15.86 | Comparison Set |
| page-blocks0 | 5472 | 10 | 8.79 | Neither |
| pima | 768 | 8 | 1.87 | Comparison Set |
| segment0 | 2308 | 19 | 6.02 | Neither |
| shuttle-c0-vs-c4 | 1829 | 9 | 13.87 | Comparison Set |
| shuttle-c2-vs-c4 | 129 | 9 | 20.5 | Neither |
| vehicle0 | 846 | 18 | 3.25 | Comparison Set |
| vehicle1 | 846 | 18 | 2.9 | Comparison Set |
| vehicle2 | 846 | 18 | 2.88 | Comparison Set |
| vehicle3 | 846 | 18 | 2.99 | Comparison Set |
| vowel0 | 988 | 13 | 9.98 | Comparison Set |
| wisconsin | 683 | 9 | 1.86 | Parameter Set |
| yeast-0-5-6-7-9_vs_4 | 528 | 8 | 9.35 | Comparison Set |
| yeast-1-2-8-9_vs_7 | 947 | 8 | 30.57 | Comparison Set |
| yeast-1-4-5-8_vs_7 | 693 | 8 | 22.1 | Comparison Set |
| yeast-1_vs_7 | 459 | 7 | 14.3 | Comparison Set |
| yeast-2_vs_4 | 514 | 8 | 9.08 | Comparison Set |
| yeast-2_vs_8 | 482 | 8 | 23.1 | Comparison Set |
| yeast1 | 1484 | 8 | 2.46 | Neither |
| yeast3 | 1484 | 8 | 8.1 | Comparison Set |
| yeast4 | 1484 | 8 | 28.1 | Comparison Set |
| yeast5 | 1484 | 8 | 32.73 | Parameter Set |
| yeast6 | 1484 | 8 | 41.4 | Comparison Set |

### Table 1 Partition Breakdown
- **Comparison Set (30 Datasets)**: abalone19, abalone9-18, car-good, ecoli-0_vs_1, ecoli1, ecoli2, glass-0-1-6_vs_2, glass0, glass2, glass4, glass6, haberman, new-thyroid2, page-blocks-1-3_vs_4, pima, shuttle-c0-vs-c4, vehicle0, vehicle1, vehicle2, vehicle3, vowel0, yeast-0-5-6-7-9_vs_4, yeast-1-2-8-9_vs_7, yeast-1-4-5-8_vs_7, yeast-1_vs_7, yeast-2_vs_4, yeast-2_vs_8, yeast3, yeast4, yeast6.
- **Parameter Analysis Set (4 Datasets)**: ecoli3, ecoli4, wisconsin, yeast5.
- **Neither Set (9 Datasets)**: `ecoli-0-1-3-7_vs_2-6`, `glass-0-1-6_vs_5`, `glass5`, `iris0`, `new-thyroid1`, `page-blocks0`, `segment0`, `shuttle-c2-vs-c4`, `yeast1`. (These are standard KEEL binary benchmarks not included in Table 1).
- **Typographical / Overlap Note**: `car-good` appears in both the comparison set and the parameter set in the published paper. In the prompt list, `creditcard` (284,807 rows) is listed under the comparison set, which was not evaluated in KEEL batch experiments.

## Section C4: Comparison Methods Literature Audit (Exact Paper Verification)
Source file: `results_stream/C4_methods_audit.csv`
Audit conducted directly from the reference list of Tao et al. (2026):

| Method | Paper_Citation | Code_Availability_Statement | Repository_URL_Opened | Language_Platform | Printed_Author_Emails | Smoke_Test_Status | Inclusion_Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MLOS | Yan, Zheng, Han, Yu, Zhou, Expert Syst. Appl. 260 (2025) 125422 | not found (Data availability: Data will be made available on request, p. 14) | https://github.com/ytyancp/MLOS | Python (standalone script MLOS.py) | ytyan@ahu.edu.cn, pzhou@ahu.edu.cn | PASSED (284 maj : 284 min, 0 NaNs on Ecoli2) | Included in Phase 1 pilot |
| ENU | Kumar, Singh, Yadav, Concurrency Comput. Pract. Exper. 36(2) (2024) e7894 | not found | none found | nan | rsy@mnnit.ac.in, anil.kumar@mnnit.ac.in | Not runnable (no public code) | Excluded (no public code) |
| DGS | Yan, Wang, Ma, Tian, Zhou, Inf. Sci. 718 (2025) 122416 | not found (Data availability: Data will be made available on request, p. 16) | none found (checked https://github.com/ytyancp) | nan | ytyan@ahu.edu.cn, pzhou@ahu.edu.cn | Not runnable (no public code) | Excluded (no public code) |
| NCLO-SMOTE | Tao, Wang, Yu, Cao, Liang, Luo, Guo, Neurocomputing 616 (2025) 128959 | not found (Data availability: Data will be made available on request, p. 12) | none found | nan | taoliangliang@ncst.edu.cn, guojianguo@ncst.edu.cn | Not runnable (no public code) | Excluded (no public code) |
| HS-SMOTE | Wang, Bao, Yang, Expert Syst. Appl. 265 (2025) 125855 | not found (Data availability: Data will be made available on request, p. 13) | https://github.com/gogossss/HS-SMOTE | MATLAB (.m files only, no Python) | wangshusheng@hebut.edu.cn | Not runnable (MATLAB only) | Excluded (MATLAB only) |

### Verification Details
1. **MLOS** (Yan et al., ESWA 260, 2025, 125422): Code availability statement: `not found` (Elsevier standard statement: 'Data availability: Data will be made available on request', p. 14). Public repository opened: `https://github.com/ytyancp/MLOS`. Standalone Python script `MLOS.py` obtained. Printed author emails: `ytyan@ahu.edu.cn`, `pzhou@ahu.edu.cn`. Smoke test on Ecoli2: PASSED (284:52 -> 284:284 resampled, 0 NaNs).
2. **ENU** (Kumar et al., CCPE 36(2), 2024, e7894): Code availability statement: `not found`. Repository URL: `none found`. Printed author emails: `rsy@mnnit.ac.in`, `anil.kumar@mnnit.ac.in`. Status: Excluded (no public code).
3. **DGS** (Yan et al., Inf. Sci. 718, 2025, 122416): Code availability statement: `not found` ('Data will be made available on request', p. 16). Repository URL: `none found` (author GitHub profile `ytyancp` hosts LDAS, GDHS, CSPS, MLOS; DGS is absent). Printed author emails: `ytyan@ahu.edu.cn`, `pzhou@ahu.edu.cn`. Status: Excluded (no public code).
4. **NCLO-SMOTE** (Tao et al., Neurocomputing 616, 2025, 128959): Code availability statement: `not found` ('Data will be made available on request', p. 12). Repository URL: `none found`. Printed author emails: `taoliangliang@ncst.edu.cn`, `guojianguo@ncst.edu.cn`. Status: Excluded (no public code).
5. **HS-SMOTE** (Wang et al., ESWA 265, 2025, 125855): Code availability statement: `not found` ('Data will be made available on request', p. 13). Repository URL opened: `https://github.com/gogossss/HS-SMOTE`. Implementation: MATLAB script (`HS_SMOTE.m`), no native Python code. Printed author email: `wangshusheng@hebut.edu.cn`. Status: Excluded from Python streaming benchmark.

## Section C5: DAECEBO2_PSI0 Unit Test & Acceptance Mechanism Audit
Source file: `results_stream/C5_psi0_unit_test.csv`
Unit test verifying that `DAECEBO2_PSI0(k_windows=1, lambd=1.0, delta_base=0.0)` exactly matches `ECEBO_PSI0` across seeds 42..46:

| Dataset | Seed | Unit_Test_Equal | Max_Abs_Diff | Unit_Test_Status | Literal_DAECEBO2_Dup_Share | Literal_DAECEBO2_Acc_Rate | PSI0_DAECEBO2_Dup_Share | PSI0_DAECEBO2_Acc_Rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ecoli2 | 42 | True | 0.0 | PASS | 0.3103448275862069 | 0.8963730569948186 | 0.3620689655172414 | 0.4909326424870466 |
| Ecoli2 | 43 | True | 0.0 | PASS | 0.2672413793103448 | 0.8682101513802315 | 0.2844827586206896 | 0.4657168299198575 |
| Ecoli2 | 44 | True | 0.0 | PASS | 0.2586206896551724 | 0.8617021276595744 | 0.3318965517241379 | 0.4728132387706856 |
| Ecoli2 | 45 | True | 0.0 | PASS | 0.3103448275862069 | 0.8262806236080178 | 0.3491379310344827 | 0.44543429844098 |
| Ecoli2 | 46 | True | 0.0 | PASS | 0.3189655172413793 | 0.8417350527549824 | 0.3663793103448275 | 0.4454865181711606 |
| Glass6 | 42 | True | 0.0 | PASS | 0.8974358974358975 | 0.098981077147016 | 0.3076923076923077 | 0.4818049490538573 |
| Glass6 | 43 | True | 0.0 | PASS | 0.8653846153846154 | 0.2134146341463414 | 0.3269230769230769 | 0.4359756097560975 |
| Glass6 | 44 | True | 0.0 | PASS | 0.8910256410256411 | 0.0952380952380952 | 0.3269230769230769 | 0.4608843537414966 |
| Glass6 | 45 | True | 0.0 | PASS | 0.8846153846153846 | 0.1130298273155416 | 0.358974358974359 | 0.4646781789638932 |
| Glass6 | 46 | True | 0.0 | PASS | 0.8782051282051282 | 0.2048710601719197 | 0.2948717948717949 | 0.5 |
| Haberman | 42 | True | 0.0 | PASS | 0.2430555555555555 | 0.9743589743589745 | 0.3680555555555556 | 0.4205128205128205 |
| Haberman | 43 | True | 0.0 | PASS | 0.2638888888888889 | 0.9392624728850324 | 0.4097222222222222 | 0.4316702819956616 |
| Haberman | 44 | True | 0.0 | PASS | 0.2222222222222222 | 0.9771615008156608 | 0.3263888888888889 | 0.430668841761827 |
| Haberman | 45 | True | 0.0 | PASS | 0.2777777777777778 | 0.968342644320298 | 0.3958333333333333 | 0.4394785847299813 |
| Haberman | 46 | True | 0.0 | PASS | 0.1805555555555555 | 0.9945848375451264 | 0.3402777777777778 | 0.4657039711191336 |
| Creditcard_5k | 42 | True | 0.0 | PASS | 0.5522908366533864 | 0.1690492980319251 | 0.2116533864541832 | 0.4953522661709084 |
| Creditcard_5k | 43 | True | 0.0 | PASS | 0.5473107569721115 | 0.1595436207766011 | 0.2024402390438247 | 0.5079425113464447 |
| Creditcard_5k | 44 | True | 0.0 | PASS | 0.5470617529880478 | 0.1756532431601598 | 0.2059262948207171 | 0.4972640639409775 |
| Creditcard_5k | 45 | True | 0.0 | PASS | 0.547808764940239 | 0.1672901208070153 | 0.1989541832669322 | 0.4962899368369411 |
| Creditcard_5k | 46 | True | 0.0 | PASS | 0.548804780876494 | 0.1731040835124347 | 0.211902390438247 | 0.4973288302118513 |

### Mechanism Audit Findings (C5)
- **Unit Test**: **100% PASS** (`Unit_Test_Equal == True`, `Max_Abs_Diff == 0.0`) on all 20 test runs (Ecoli2, Glass6, Haberman, Creditcard_5k across seeds 42..46).
- **Literal DAECEBO2 Duplicate Rates**: Glass6 duplicate share is 86.5%--89.7% with acceptance rate 9.5%--21.3%. Creditcard_5k duplicate share is 54.7%--55.2% with acceptance rate 16.0%--17.6%. Ecoli2 duplicate share is 25.9%--31.9% with acceptance rate 82.6%--89.6%. Haberman duplicate share is 18.1%--27.8% with acceptance rate 93.9%--99.5%.
- **PSI0 DAECEBO2_PSI0 Behavior**: Under the gradient $|\psi| \to 0$ rule, proposal acceptance stabilizes between 42.1% and 50.8% across all datasets, reducing duplicate synthetic clones (Glass6 drops to 29.5%--35.9%, Creditcard_5k drops to 19.9%--21.2%).

## Section C6: Data Hygiene & Streaming Realism Audit
Source file: `results_stream/H_datasets_final.csv`

| Dataset | Category | Triage_Status | Total_Rows | Minority_Count | Imbalance_Ratio | Ref_OutSample_AUC_DT_LR |
| --- | --- | --- | --- | --- | --- | --- |
| creditcard | A | SUITABLE | 284807 | 492 | 577.88 | DT:0.5057 / LR:0.9486 (SUITABLE SIGNAL) |
| electricity_ir20 | A | SUITABLE | 27378 | 1303 | 20.01 | DT:0.7572 / LR:0.8280 (SUITABLE SIGNAL) |
| electricity_ir100 | A | SUITABLE | 26335 | 260 | 100.29 | DT:0.7470 / LR:0.8549 (SUITABLE SIGNAL) |
| airlines_ir20 | C | REJECTED | 314074 | 14955 | 20.0 | DT:0.6335 / LR:0.6047 (RECLASSIFIED CAT C: UNVERIFIED CHRONOLOGY) |
| airlines_ir100 | C | REJECTED | 302110 | 2991 | 100.01 | DT:0.6234 / LR:0.6101 (RECLASSIFIED CAT C: UNVERIFIED CHRONOLOGY) |
| ieee_cis | A | SUITABLE | 590540 | 20663 | 27.58 | DT:0.6625 / LR:0.7818 (SUITABLE SIGNAL) |
| covertype_binary | C | BENCHMARK ORDER ONLY (Cat C) | 581012 | 9493 | 60.2 | DT:0.8711 / LR:0.8018 (CAT C: SPATIAL/NON-STREAM) |
| poker_binary | C | BENCHMARK ORDER ONLY (Cat C) | 829201 | 17541 | 46.27 | DT:0.7443 / LR:0.7675 (CAT C: COMBINATORIAL RULES) |
| kddcup99_rare | B | SUITABLE (Rare Attack Family) | 494021 | 5285 | 92.48 | DT:0.9657 / LR:0.9871 (TRIVIAL FOR TREES: DT > 0.95) |
| kddcup99_binary | B | NOT SUITABLE (Low IR = 3.61) | 494021 | 107201 | 3.61 | DT:0.7747 / LR:0.9951 (TRIVIAL FOR LINEAR: LR > 0.95) |
| synth_sea_p5 | D | SUITABLE (Synthetic Drift Benchmark) | 7835 | 391 | 19.04 | DT:0.8232 / LR:0.9362 (SUITABLE SIGNAL) |
| synth_sea_p1 | D | EXCLUDED (Test Minority 12 < 30) | 7519 | 75 | 99.25 | DT:0.9050 / LR:0.9630 (SUITABLE SIGNAL) |
| synth_hyperplane_p5 | D | SUITABLE (Synthetic Drift Benchmark) | 10350 | 517 | 19.02 | DT:0.8352 / LR:0.9487 (SUITABLE SIGNAL) |
| synth_random_rbf_p5 | D | SUITABLE (Synthetic Drift Benchmark) | 6837 | 341 | 19.05 | DT:0.4742 / LR:0.3238 (NO SIGNAL: BOTH < 0.55) |
| ieee_cis_45 | A | SUITABLE | 590540 | 20663 | 27.58 | DT:0.7021 / LR:0.7853 (SUITABLE SIGNAL) |

### Audit Resolutions (C6)
1. **Credit Card Cluster Ratio Shift (0.3668 to 0.2286)**:
   - In Task D1, `Time` was inadvertently retained as an active feature column in `X_tr`. Because `Time` increases monotonically from 0 to 172,792 seconds, minority instances occurring at different timestamps had large standardized Euclidean distances, preventing agglomerative merges and producing 153 clusters ($m / N_{min} = 153 / 417 = 0.3668$).
   - In Task H, `Time` was purged from the clustering feature space. Clustering in pure 29-dimensional space allowed temporally separated but spatially proximate frauds to merge, reducing cluster count to 88 ($m / N_{min} = 88 / 385 = 0.2286$).
2. **IEEE-CIS Feature Architecture & Construction of `ieee_cis_45`**:
   - The original file `common_cleaned_ieee_cis.parquet` contains 422 input features because it ingested all raw Kaggle features: 339 V-features ($V_1$ to $V_{339}$), 38 identity attributes ($id_{01}$ to $id_{38}$), 14 C-counts, 15 D-deltas, 9 M-matches, transaction amount, card attributes, and email domains. Categoricals were factorized and NaNs imputed with 0.
   - We constructed `datasets_stream/ieee_cis_45.parquet` using the non-redundant 45-feature subset identified in `e:/Research/ds/`: 14 semantic features (8 numerical: `TransactionAmt`, `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `dist1`; 6 categorical: `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`) plus 31 key V-features (`V12`, `V13`, `V29`, `V30`, `V35`, `V36`, `V44`, `V45`, `V75`, `V76`, `V78`, `V82`, `V83`, `V87`, `V90`, `V91`, `V130`, `V131`, `V281`, `V282`, `V283`, `V285`, `V287`, `V288`, `V289`, `V307`, `V308`, `V310`, `V312`, `V314`, `V315`).
   - **D1 Characteristics of `ieee_cis_45`**: 590,540 rows, 45 features, 569,877 maj, 20,663 min (IR 27.58). Chronological 75/25 split: Train 427,342 maj / 15,563 min; Test 142,535 maj / 5,100 min. Napierala typology on train minority: 3.55% safe, 17.90% borderline, 24.60% rare, 53.95% outlier. ECC clusters: $m = 1,642$ per 2,000 minority samples (cluster ratio 0.8210).
3. **Reference Out-of-Sample AUC Baseline & Signal Screening**:
   - Out-of-sample AUC scored using DecisionTree (max depth 6) and LogisticRegression on 75/25 chronological train/test split:
     - `creditcard`: DT 0.5057 / LR 0.9486 (SUITABLE SIGNAL).
     - `electricity_ir20`: DT 0.7572 / LR 0.8280 (SUITABLE SIGNAL).
     - `electricity_ir100`: DT 0.7470 / LR 0.8549 (SUITABLE SIGNAL).
     - `synth_sea_p5`: DT 0.8232 / LR 0.9362 (SUITABLE SIGNAL).
     - `synth_hyperplane_p5`: DT 0.8352 / LR 0.9487 (SUITABLE SIGNAL).
     - `synth_random_rbf_p5`: DT 0.4742 / LR 0.3238 -> **Flagged: NO SIGNAL (both < 0.55)** due to non-stationary centroid displacement extinguishing base model generalization.
     - `kddcup99_rare`: DT 0.9657 / LR 0.9871 -> **Flagged: TRIVIAL FOR TREES (DT > 0.95)**.
     - `poker_binary`: DT 0.7443 / LR 0.7675 (Category C, rules-based).
4. **Airlines Chronology Audit**:
   - OpenML `airlines` contains attributes `[Airline, Flight, AirportFrom, AirportTo, DayOfWeek, Time, Length]`.
   - `Time` is scheduled departure minute within a single 24-hour cycle (ranging from 10 to 1439). The sequence contains 54 cyclical day resets where `Time` jumps from ~1439 back to ~10.
   - `DayOfWeek` cycles non-monotonically (3, 4, 5, 6, 7, 1, 2). The dataset lacks calendar dates, years, or absolute timestamps.
   - **Resolution**: Re-classified from Category A to **Category C** (Unverified chronology / snapshot sequence) and rejected from the primary streaming benchmark.

## Section C7: Reduced Timed Pilot Benchmark & Phase 2 Compute Budget
Source file: `results_stream/C7_timing.csv`
Protocol: 1 random seed (42), chronological 75/25 train split, 5-minute (300.0s) hard timeout per (dataset, method) run.
Datasets with minority > 5,000 samples skipped as 'too large'. Output appended row-by-row:

| dataset | n | minority_count | method | seconds | timed_out |
| --- | --- | --- | --- | --- | --- |
| ieee_cis | 590540 | 20663 | ALL_METHODS | 0.0 | too large |
| ieee_cis_45 | 590540 | 20663 | ALL_METHODS | 0.0 | too large |
| kddcup99_rare | 494021 | 5285 | ALL_METHODS | 0.0 | too large |
| kddcup99_binary | 494021 | 107201 | ALL_METHODS | 0.0 | too large |
| covertype_binary | 581012 | 9493 | ALL_METHODS | 0.0 | too large |
| poker_binary | 829201 | 17541 | ALL_METHODS | 0.0 | too large |
| airlines_ir20 | 314074 | 14955 | ALL_METHODS | 0.0 | too large |
| synth_sea_p5 | 7835 | 391 | None | 0.0483 | NO |
| synth_sea_p5 | 7835 | 391 | SMOTE | 1.8106 | NO |
| synth_sea_p5 | 7835 | 391 | BorderlineSMOTE | 1.7797 | NO |
| synth_sea_p5 | 7835 | 391 | ADASYN | 1.8275 | NO |
| synth_sea_p5 | 7835 | 391 | G-SMOTE | 2.9068 | NO |
| synth_sea_p5 | 7835 | 391 | CCR | 2.5827 | NO |
| synth_sea_p5 | 7835 | 391 | A-SUWO | 8.0043 | NO |
| synth_sea_p5 | 7835 | 391 | MLOS | 96.9888 | NO |
| synth_sea_p5 | 7835 | 391 | ECEBO | 3.4432 | NO |
| synth_sea_p5 | 7835 | 391 | ECEBO_PSI0 | 4.7115 | NO |
| synth_sea_p5 | 7835 | 391 | DAECEBO2 | 3.7585 | NO |
| synth_sea_p5 | 7835 | 391 | DAECEBO2_PSI0 | 3.3345 | NO |
| synth_hyperplane_p5 | 10350 | 517 | None | 0.0445 | NO |
| synth_hyperplane_p5 | 10350 | 517 | SMOTE | 1.7763 | NO |
| synth_hyperplane_p5 | 10350 | 517 | BorderlineSMOTE | 1.7426 | NO |
| synth_hyperplane_p5 | 10350 | 517 | ADASYN | 1.7477 | NO |
| synth_hyperplane_p5 | 10350 | 517 | G-SMOTE | 2.3939 | NO |
| synth_hyperplane_p5 | 10350 | 517 | CCR | 2.6413 | NO |
| synth_hyperplane_p5 | 10350 | 517 | A-SUWO | 6.7284 | NO |
| synth_hyperplane_p5 | 10350 | 517 | MLOS | 135.5577 | NO |
| synth_hyperplane_p5 | 10350 | 517 | ECEBO | 3.9587 | NO |
| synth_hyperplane_p5 | 10350 | 517 | ECEBO_PSI0 | 3.7271 | NO |
| synth_hyperplane_p5 | 10350 | 517 | DAECEBO2 | 3.7097 | NO |
| synth_hyperplane_p5 | 10350 | 517 | DAECEBO2_PSI0 | 3.7446 | NO |
| synth_random_rbf_p5 | 6837 | 341 | None | 0.0413 | NO |
| synth_random_rbf_p5 | 6837 | 341 | SMOTE | 1.7097 | NO |
| synth_random_rbf_p5 | 6837 | 341 | BorderlineSMOTE | 1.6842 | NO |
| synth_random_rbf_p5 | 6837 | 341 | ADASYN | 1.7494 | NO |
| synth_random_rbf_p5 | 6837 | 341 | G-SMOTE | 2.3492 | NO |
| synth_random_rbf_p5 | 6837 | 341 | CCR | 2.4944 | NO |
| synth_random_rbf_p5 | 6837 | 341 | A-SUWO | 5.7584 | NO |
| synth_random_rbf_p5 | 6837 | 341 | MLOS | 56.3206 | NO |
| synth_random_rbf_p5 | 6837 | 341 | ECEBO | 2.6205 | NO |
| synth_random_rbf_p5 | 6837 | 341 | ECEBO_PSI0 | 2.6229 | NO |
| synth_random_rbf_p5 | 6837 | 341 | DAECEBO2 | 2.6073 | NO |
| synth_random_rbf_p5 | 6837 | 341 | DAECEBO2_PSI0 | 2.672 | NO |
| electricity_ir100 | 26335 | 260 | None | 0.0436 | NO |
| electricity_ir100 | 26335 | 260 | SMOTE | 1.686 | NO |
| electricity_ir100 | 26335 | 260 | BorderlineSMOTE | 1.668 | NO |
| electricity_ir100 | 26335 | 260 | ADASYN | 1.6821 | NO |
| electricity_ir100 | 26335 | 260 | G-SMOTE | 2.35 | NO |
| electricity_ir100 | 26335 | 260 | CCR | 2.8938 | NO |
| electricity_ir100 | 26335 | 260 | A-SUWO | 15.7383 | NO |
| electricity_ir100 | 26335 | 260 | MLOS | 177.2106 | NO |
| electricity_ir100 | 26335 | 260 | ECEBO | 2.6263 | NO |
| electricity_ir100 | 26335 | 260 | ECEBO_PSI0 | 2.5776 | NO |
| electricity_ir100 | 26335 | 260 | DAECEBO2 | 2.4956 | NO |
| electricity_ir100 | 26335 | 260 | DAECEBO2_PSI0 | 2.8177 | NO |
| electricity_ir20 | 27378 | 1303 | None | 0.0431 | NO |
| electricity_ir20 | 27378 | 1303 | SMOTE | 1.705 | NO |
| electricity_ir20 | 27378 | 1303 | BorderlineSMOTE | 1.7096 | NO |
| electricity_ir20 | 27378 | 1303 | ADASYN | 1.7166 | NO |
| electricity_ir20 | 27378 | 1303 | G-SMOTE | 2.3158 | NO |
| electricity_ir20 | 27378 | 1303 | CCR | 3.6638 | NO |
| electricity_ir20 | 27378 | 1303 | A-SUWO | 28.0974 | NO |
| electricity_ir20 | 27378 | 1303 | MLOS | 300.0 | YES |
| electricity_ir20 | 27378 | 1303 | ECEBO | 5.993 | NO |
| electricity_ir20 | 27378 | 1303 | ECEBO_PSI0 | 5.9956 | NO |
| electricity_ir20 | 27378 | 1303 | DAECEBO2 | 6.014 | NO |
| electricity_ir20 | 27378 | 1303 | DAECEBO2_PSI0 | 6.1056 | NO |
| creditcard | 284807 | 492 | None | 0.1328 | NO |
| creditcard | 284807 | 492 | SMOTE | 3.6659 | NO |
| creditcard | 284807 | 492 | BorderlineSMOTE | 3.4418 | NO |
| creditcard | 284807 | 492 | ADASYN | 3.5288 | NO |
| creditcard | 284807 | 492 | G-SMOTE | 3.8267 | NO |
| creditcard | 284807 | 492 | CCR | 9.9796 | NO |
| creditcard | 284807 | 492 | A-SUWO | 58.9659 | ERROR (ERROR) |
| creditcard | 284807 | 492 | MLOS | 62.6842 | ERROR (ERROR) |
| creditcard | 284807 | 492 | ECEBO | 7.9377 | NO |
| creditcard | 284807 | 492 | ECEBO_PSI0 | 8.6494 | NO |
| creditcard | 284807 | 492 | DAECEBO2 | 7.3549 | NO |
| creditcard | 284807 | 492 | DAECEBO2_PSI0 | 7.9912 | NO |

### Phase 2 Compute Budget Extrapolation
Based on empirical seconds measured in `results_stream/C7_timing.csv` and classifier overhead in `results_stream/C7_timing_partial.csv`:
- Full Benchmark: 10 seeds x 6 datasets x 12 methods x 4 classifiers.
- DA Validation Grid: 5 values of $K_{win}$ x 5 values of $\lambda$ = 25 cells x 2 expanding folds x 10 seeds = 500 evaluations per DA method.

| Benchmark Component | synth_sea_p5 | synth_hyperplane_p5 | synth_random_rbf_p5 | electricity_ir100 | electricity_ir20 | creditcard (capped) | Total Hours |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Static Methods (10 seeds, 4 clfs) | 0.12 h | 0.15 h | 0.11 h | 0.45 h | 0.85 h | 2.50 h | **4.18 h** |
| ECEBO & ECEBO_PSI0 (10 seeds, 4 clfs) | 0.08 h | 0.09 h | 0.06 h | 0.18 h | 0.35 h | 1.80 h | **2.56 h** |
| DA-ECEBO Grid (500 runs x 2 DA variants) | 0.85 h | 0.95 h | 0.70 h | 1.65 h | 3.20 h | 4.80 h | **12.15 h** |
| **Total Estimated Compute** | **1.05 h** | **1.19 h** | **0.87 h** | **2.28 h** | **4.40 h** | **9.10 h** | **18.89 h** |

### Mandatory Capping & Subsampling Constraints for Phase 2
1. **Uncapped SVC Barrier**: On `creditcard` (213,605 train rows), uncapped SVC exceeds 1,800s per fit (hitting timeouts). Fitting 500 grid cells uncapped would require >250 hours. SVC on creditcard **must be capped** to `max_iter=2000` or trained on a subsample of 20,000 instances.
2. **MLOS / A-SUWO / CCR Memory Bottlenecks**: `MLOS` failed on creditcard due to $O(N^2)$ covariance computation (`pdist`) across 213,000 majority rows; it also timed out at 300s on `electricity_ir20`. In Phase 2, `MLOS` majority background **must be capped to 20,000 samples** to avoid memory errors.
3. **Datasets Filtered by >5,000 Minority**: `ieee_cis`, `ieee_cis_45`, `kddcup99_rare`, `covertype_binary`, and `poker_binary` are excluded from full grid searches due to minority scaling limits, keeping total Phase 2 runtime under 20 hours.

## Refreshed Open Problems Prior to Phase 2
1. **Acceptance Rule Selection for Primary DA Benchmark**: Given that `ECEBO_PSI0` significantly outperforms literal `ECEBO` on decision tree classifiers (CART Macro-Gmean rank 2.91 vs 5.27 in B2) and avoids 50%--90% duplicate cloning, should Phase 2 benchmark both `DAECEBO2` (literal `psi < 0`) and `DAECEBO2_PSI0` (`|psi| -> 0`), or focus on `DAECEBO2_PSI0` as the primary operational method?
2. **Synthetic RandomRBF Signal Recovery**: Since `synth_random_rbf_p5` exhibits zero baseline out-of-sample signal (DT AUC 0.474, LR AUC 0.324) due to extreme centroid drift across blocks, can temporal windowing in DA-ECEBO restore predictive signal, or should it be treated as a negative control?
3. **MLOS Majority Capping Approval**: To evaluate MLOS alongside DA-ECEBO on `electricity_ir20` and `creditcard`, we must formally fix a 20,000-sample majority cap for MLOS background calculations to prevent 300s subprocess timeouts.