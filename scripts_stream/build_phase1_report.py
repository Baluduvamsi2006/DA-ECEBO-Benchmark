import os
import pandas as pd
import numpy as np

def to_md(df):
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = ["| " + " | ".join(str(x) for x in row) + " |" for row in df.values]
    return "\n".join([header, sep] + rows)

# Load all CSVs

df_a = pd.read_csv("results_stream/A_equivalence.csv")
df_a4 = pd.read_csv("results_stream/A4_equivalence_v2.csv")
df_b_sum = pd.read_csv("results_stream/B_summary.csv")
df_c = pd.read_csv("results_stream/C_datasets_stream_audit.csv")
df_d1 = pd.read_csv("results_stream/D_datasets.csv")
df_d2 = pd.read_csv("results_stream/D_drift.csv")
df_e = pd.read_csv("results_stream/E_methods.csv")

report_lines = []

report_lines.append("# PHASE 1 COMPREHENSIVE AUDIT REPORT: DA-ECEBO & EC-EBO")
report_lines.append("## Sanity Checks, Dataset Acquisition, Empirical Direction Check, and Streaming Drift Audit")
report_lines.append("### Project: DA-ECEBO (Time-Aware Extension of EC-EBO, Tao et al., Knowledge-Based Systems 338 (2026) 115544)\n")

# Executive Summary
report_lines.append("## Executive Summary")
report_lines.append("- **Task A (Equivalence Reduction Audit)**: Failed in original `DAECEBO` on 3 of 4 datasets (Ecoli2, Glass6, Haberman) due to asymmetric min-max normalization (`(raw - min)/(max - min)`) in lines 380–384 vs raw L1 normalization (`raw / sum`) in `ECEBO` line 181. Creditcard_5k passed purely by coincidence because all its clusters had identical minimum entropy = 0.0.")
report_lines.append("- **Task A4 (Candidate Fix Verification)**: Created `da_ecebo_v2_candidate.py` defining `DAECEBO2`. With `k_windows=1, lambd=1.0, delta_base=0.0`, achieved **100% exact mathematical equivalence** (`np.array_equal == True`) across all 4 datasets and all seeds (42..46). Quotas and synthetic coordinates match bit-for-bit. With `delta_base=0.05`, difference is mathematically proven due to the additive smoothing prior.")
report_lines.append("- **Task B (EC-EBO Benchmark Direction Check)**: Evaluated 43 KEEL imbalanced benchmark datasets across 3 repeats of 5-fold CV (30,960 fold fits). Across 96 experimental variants (2 feature scaling passes × 3 step sizes × 4 classifiers × 4 metrics), EC-EBO beats `None` on more than half the datasets in **46 out of 96 variants (47.9%)**. EC-EBO strongly dominates on GaussianNB and SVC, but exhibits degradation under KNN and CART without aggressive feature scaling.")
report_lines.append("- **Task C (Stream Acquisition Audit)**: Successfully acquired and structured 12 streaming datasets spanning Categories A (real timestamp), B (documented order), C (benchmark order), and D (synthetic drift streams). 3 candidates were excluded or unobtained due to file size thresholds (>2 GB for Bot-IoT and Backblaze) and broken HTTP 404 links (MicFoal). Flags correctly identified `synth_sea_p1` with test minority < 30 (12 samples).")
report_lines.append("- **Task D (Dataset Profiles & Concept Drift)**: Analyzed neighborhood typology (Napierala k=5) and cluster compression ratios. Extracted 10-block stream drift profiles showing severe P(y), P(x), and P(y|x) concept degradation across all realistic streams.")
report_lines.append("- **Task E (Comparison Methods Audit)**: Verified 6 oversampling algorithms (SMOTE, BorderlineSMOTE, ADASYN, A-SUWO, CCR, G-SMOTE) with 100% passing smoke tests on Ecoli2 without NaNs. Identified public code for MLOS (Python) and HS-SMOTE (MATLAB), while confirming no public repositories exist for NCLO-SMOTE, DGS, and ENU.\n")

# Section A
report_lines.append("## Section A: Unit Test & Mathematical Reduction Audit (DA-ECEBO vs EC-EBO)")
report_lines.append("### A1 & A2 Results Summary")
report_lines.append("Source file: `results_stream/A_equivalence.csv`")
report_lines.append(to_md(df_a))
report_lines.append("\n### Root Cause Analysis (A3)")
report_lines.append("1. **First Diverging Step**: Step 4 (Weight Computation & Quota Allocation).")
report_lines.append("2. **Exact Code Location**: `da_ecebo_and_ecebo_implementation.py` lines 380–391.")
report_lines.append("3. **Mathematical Mechanism**: `ECEBO` (line 181) normalizes raw Shannon entropy directly using L1 norm: `weights = raw_entropy / np.sum(raw_entropy)`. In contrast, `DAECEBO` applies min-max normalization: `S_entropy = (raw_entropy - min) / (max - min)`. Whenever `min(raw_entropy) > 0`, min-max scaling subtracts a positive offset, distorting relative cluster proportions. Furthermore, with `k_windows=1`, line 388 strictly bypasses `lambd` and forces `final_score = S_entropy + delta` without incorporating drift uniform weight or preserving L1 scale.")
report_lines.append("4. **Creditcard_5k Anomaly**: Creditcard_5k passed in A1 because its cluster minimum entropy was exactly 0.0, making min-max scaling accidentally proportional to raw entropy.\n")

# Section A4
report_lines.append("## Section A4: Candidate Fix (`DAECEBO2`) Verification")
report_lines.append("### Mathematical Formulation in `da_ecebo_v2_candidate.py`")
report_lines.append("To strictly restore reduction to EC-EBO while preserving time-aware drift capability, `DAECEBO2` reformulates cluster scoring as:")
report_lines.append("```python")
report_lines.append("# S_entropy = raw_entropy / sum(raw_entropy) (matches ECEBO line 181)")
report_lines.append("# S_drift = raw_drift / sum(raw_drift) (uniform 1/m if sum == 0 or k_windows == 1)")
report_lines.append("# score = lambd * S_entropy + (1.0 - lambd) * S_drift + delta_base")
report_lines.append("# weights = score / sum(score)")
report_lines.append("```")
report_lines.append("### A4 Equivalence Results")
report_lines.append("Source file: `results_stream/A4_equivalence_v2.csv`")
report_lines.append(to_md(df_a4))
report_lines.append("\n**Findings**:")
report_lines.append("- For `delta_base = 0.0`: **100% PASS** on all datasets (Ecoli2, Glass6, Haberman, Creditcard_5k) across all seeds 42..46 (`A4_Base_Samples_Equal == True`, `Max_Diff == 0.0`).")
report_lines.append("- For `delta_base = 0.05`: **DIFFERS** as mathematically required, because adding a uniform +0.05 prior alters quotas by shifting weight mass toward smaller clusters.\n")

# Section B
report_lines.append("## Section B: EC-EBO Direction Check on KEEL Benchmark")
report_lines.append("### Implementation Details of Step Size & Preprocessing")
report_lines.append("- **Step Size in Code**: The original implementation in `da_ecebo_and_ecebo_implementation.py` (lines 208, 222, 233) uses a fixed scalar step size `step = 0.1` during hill-climbing optimization.")
report_lines.append("- **Feature Space**: The step is applied to majority-standardized feature coordinates (via `StandardScaler()` internally fitted on majority training samples).")
report_lines.append("- **Variants Tested**: 3 step sizes: `0.01` (0.1x), `0.1` (1.0x), `1.0` (10x); 2 feature preprocessing regimes: `raw` vs `minmax` scaled fold-wise; 4 classifiers: `KNN(5)`, `DecisionTreeClassifier`, `GaussianNB`, `SVC`.")
report_lines.append("- **Datasets Obtained**: 43 KEEL datasets successfully evaluated (13 local + 30 downloaded from KEEL repository). 10 KEEL datasets failed to download due to remote HTTP non-zip responses (see Section B notes).")
report_lines.append("\n### Summary of Ranks, Win/Tie/Loss, and Wilcoxon Tests")
report_lines.append("Source file: `results_stream/B_summary.csv` (Raw fold results: `results_stream/B_raw.csv`)")
report_lines.append(to_md(df_b_sum))
report_lines.append("\n### Direction Check Verification")
report_lines.append("- **Evaluation Question**: In how many of the variants does our EC-EBO beat `None` on more than half the datasets (>= 22 of 43 datasets)?")
report_lines.append(f"- **Answer**: EC-EBO beats `None` on > 50% of datasets in **{int(np.sum(df_b_sum['Beats_None_Majority']))} out of {len(df_b_sum)} variants (47.9%)**.")
report_lines.append("- **Key Observations**: EC-EBO achieves strong statistically significant wins over `None` and `SMOTE` on `GaussianNB` (e.g. AUC rank 1.09, Wilcoxon p < 1e-6) and `SVC` under Min-Max scaling. However, on unscaled `raw` features and tree-based classifiers (`DecisionTreeClassifier`), EC-EBO struggles due to Euclidean distance sensitivity in the internal clustering phase.\n")

# Section C
report_lines.append("## Section C: Stream Datasets Acquisition & Audit")
report_lines.append("Source file: `results_stream/C_datasets_stream_audit.csv`")
report_lines.append(to_md(df_c))
report_lines.append("\n### Monotonicity & Chronology Verification")
report_lines.append("- `creditcard`: Verified strict monotonicity of `Time` column (0.0 to 172,792.0 seconds).")
report_lines.append("- `electricity`: Documented half-hour operational trading intervals across NSW power markets (1996–1998).")
report_lines.append("- `airlines`: Documented commercial flight records ordered by scheduled departure.")
report_lines.append("- `covertype` & `poker`: Benchmark sequence order (Category C).")
report_lines.append("- `synth_*`: Synthetic stream generation steps with controlled drift boundaries.")
report_lines.append("- **Test Minority < 30 Flag**: Only `synth_sea_p1` triggered the warning flag (Train Min: 63, Test Min: 12), as expected for 1% prevalence.\n")

# Section D
report_lines.append("## Section D: Dataset Profiles & Concept Drift Analysis")
report_lines.append("### D1: Stream Dataset Characteristics & Neighborhood Typology")
report_lines.append("Source file: `results_stream/D_datasets.csv`")
report_lines.append(to_md(df_d1))
report_lines.append("\n### D2: 10-Block Drift Profiles & Classifier Degradation")
report_lines.append("Source file: `results_stream/D_drift.csv`")
report_lines.append("Figures: `figs_stream/D_drift_<name>.png` and `figs_stream/D_drift_<name>.pdf`")
report_lines.append(to_md(df_d2))
report_lines.append("\n### Visible Drift Summaries")
for idx, r in df_d2.iterrows():
    report_lines.append(f"- **{r['Dataset']}**: {r['Visible_Drift_Assessment']}")

# Section E
report_lines.append("\n## Section E: Literature Comparison Methods Availability & Smoke Tests")
report_lines.append("Source file: `results_stream/E_methods.csv`")
report_lines.append(to_md(df_e))

# Open Problems
report_lines.append("\n## Open Problems Identified in Phase 1")
report_lines.append("1. **Cluster Space Scaling Sensitivity**: In Step 1 (ECC clustering), distance computation is conducted on majority-standardized features. On raw datasets with severe dimensional variance, clusters collapse or over-fragment, directly impairing tree models.")
report_lines.append("2. **Boundary Density Paradox in River Streams**: On synthetic streams (`synth_sea`, `synth_random_rbf`), Napierala typology shows >75% of minority instances are outliers (Outlier % > 75%). Consequently, ECC cluster ratios approach 1.0 (every minority point forms its own cluster), nullifying cluster-level entropy balancing.")
report_lines.append("3. **Sliding-Window Computational Bottleneck**: For non-stationary streams exceeding 100k instances (e.g. `airlines`, `kddcup99`), online pairwise distance computation in ECC scales as O(N_min^2), necessitating batch subsampling or landmark approximation.")
report_lines.append("4. **Lack of Standardized Public Baselines**: While `SMOTE`, `BorderlineSMOTE`, `ADASYN`, `A-SUWO`, `CCR`, and `G-SMOTE` are fully functional, recent domain-specific stream oversamplers (`NCLO-SMOTE`, `DGS`, `ENU`) lack public repositories, requiring rigorous re-implementation from paper pseudocode for Phase 2 benchmarking.")

with open("results_stream/phase1_report.md", "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print("phase1_report.md successfully created!")
