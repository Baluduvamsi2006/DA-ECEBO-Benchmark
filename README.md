# DA-ECEBO Empirical Verification Benchmark

Empirical verification and benchmark suite for **DA-ECEBO** (Drift-Aware Externally Constrained Ensemble Boundary Oversampling), extending Tao et al., *Knowledge-Based Systems* 338 (2026) 115544 ("EC-EBO").

## Environment & Requirements

Dependencies are pinned in `requirements.txt`:
```bash
pip install -r requirements.txt
```
Pinned versions:
- `numpy==2.3.2`
- `scipy==1.17.0`
- `scikit-learn==1.8.0`
- `pandas==2.3.2`
- `imbalanced-learn==0.14.2`
- `pyarrow==25.0.1`

Python version: `3.13` (Windows x64).

---

## Benchmark Scripts & Exact Execution Commands

All scripts use seeds `42..51` and the Largest Remainder integer allocation rule.

| Step | Script | Description | Measured Runtime | Output Files |
| :--- | :--- | :--- | :--- | :--- |
| **S1** | `python run_s1_cv_grid_10seeds.py` | 25-cell expanding-window CV grid ($K \in \{1..5\} \times \lambda \in \{0, 0.25, 0.5, 0.75, 1.0\}$) across 10 seeds (250 model tasks) | ~32 min (10 parallel workers) | `results/cv_grid_v3.csv`, `results/cv_grid_raw_10seeds.csv` |
| **S2 & S3** | `python run_s2_s3_experiments.py` | Permuted-timestamp control (4 variants x 10 seeds) + ECC cluster analysis + 5 non-overlapping rolling-origin blocks (4 methods x 10 seeds) | ~21 min (10 parallel workers) | `results/s2_permuted_control_summary.csv`, `results/s2_wilcoxon_pvalues.csv`, `results/s2_cluster_stats.csv`, `results/s3_rolling_blocks_summary.csv`, `results/s3_pooled_ci.csv` |
| **S4(c), S4(d), S5** | `python run_s4_s5_experiments.py` | Runtime benchmarks (5 runs, median + IQR), sklearn RBF SVC on stratified subsample (10 seeds), Haberman test bootstrap CI | ~5 min (10 parallel workers) | `results/runtime_benchmark_5runs.csv`, `results/rbf_svm_benchmark_summary.csv`, `results/haberman_bootstrap_ci.csv` |
| **All Tables** | `python make_all_tables.py` | Programmatic regenerator for every table directly from CSV data (zero hand-copying) | ~1.5 sec | Formatted tables to stdout |

---

## Reproducing All Tables Programmatically

To display every research table without manual copying:
```bash
python make_all_tables.py
```

---

## Core Findings Summary

1. **Locked Headline Configuration (S1)**:
   - Selected by CV argmax and 1-standard-error rule: **$K=2, \lambda=0.0$** (CV Composite Score: $0.5462 \pm 0.0262$).
   - Anomaly resolved: At $\lambda=1.0$, quota allocation zeroes drift, and spatial drift vector split `mid = K // 2` places both $K=2$ and $K=4$ at the exact 50% temporal midpoint, yielding bit-for-bit identical resampled datasets and identical validation scores ($0.5107 \pm 0.0312$).

2. **Permuted-Timestamp Control (S2)**:
   - Real timestamps: Minority-F1 = $0.6831 \pm 0.0162$.
   - Permuted timestamps: Minority-F1 = $0.6885 \pm 0.0239$ ($p = 0.6250$, Wilcoxon stat = 22.0, not significant).
   - Plain-language verdict: Real chronological timestamp information does not improve classification accuracy over randomly shuffled timestamps. The algorithm benefits from partitioning into multiple subsets ($K > 1$) with min-max scoring and baseline allocation, not temporal drift trajectory.

3. **Rolling-Origin Evaluation (S3)**:
   - 5 non-overlapping 10% test blocks covering 50% to 100% of the stream (22 to 91 test frauds per block).
   - DA-ECEBO beats EC-EBO in **4 / 5 blocks**.
   - DA-ECEBO beats Baseline in **3 / 5 blocks**.
   - Pooled mean difference (DA minus EC-EBO): **+0.0580** (95% Bootstrap CI: **[0.0348, 0.0826]**).

4. **Consistency & Model Scope (S4)**:
   - In Random Forest, SMOTE ($0.7557$) beats DA-ECEBO ($0.7359$).
   - Across 6 classifiers, DA-ECEBO achieves 4 wins vs EC-EBO, 2 wins vs SMOTE, and 1 win vs Baseline.
   - On sklearn RBF SVC ($N_{maj}=3,980$, 10:1 ratio), Baseline dominates ($F1 = 0.7353$), while all oversampling methods degrade precision.

5. **Haberman Evaluation (S5)**:
   - Chronological test split contains exactly **17 minority samples**.
   - Measured AUC is below 0.5 for all methods except EC-EBO (0.5270), with 95% bootstrap CIs spanning across 0.5.
