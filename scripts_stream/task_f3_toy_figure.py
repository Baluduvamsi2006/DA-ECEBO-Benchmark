import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.abspath('.'))
from da_ecebo_and_ecebo_implementation import ECEBO
from ecebo_psi0_variant import ECEBO_PSI0

# 1. Synthesize 2-D dataset mimicking Fig. 2 of Tao et al.
# Overlapping majority class + 3 minority sub-clusters of different sizes
rng_seed = 42
np.random.seed(rng_seed)

# Majority: 500 points in 2 overlapping blobs near center
X_maj1, _ = make_blobs(n_samples=300, n_features=2, centers=[[2.0, 2.5]], cluster_std=1.2, random_state=42)
X_maj2, _ = make_blobs(n_samples=200, n_features=2, centers=[[4.0, 3.5]], cluster_std=1.0, random_state=43)
X_maj = np.vstack([X_maj1, X_maj2])
y_maj = np.zeros(len(X_maj), dtype=int)

# Minority: 3 sub-clusters of different sizes:
# Cluster 1: 30 points (dense, slightly overlapping boundary)
X_min1, _ = make_blobs(n_samples=30, n_features=2, centers=[[1.5, 4.5]], cluster_std=0.5, random_state=44)
# Cluster 2: 15 points (sparse, near center)
X_min2, _ = make_blobs(n_samples=15, n_features=2, centers=[[4.5, 1.5]], cluster_std=0.4, random_state=45)
# Cluster 3: 10 points (small isolated group)
X_min3, _ = make_blobs(n_samples=10, n_features=2, centers=[[6.0, 4.0]], cluster_std=0.3, random_state=46)
X_min = np.vstack([X_min1, X_min2, X_min3])
y_min = np.ones(len(X_min), dtype=int)

X = np.vstack([X_maj, X_min])
y = np.hstack([y_maj, y_min])

print(f"2-D Toy Dataset created: {len(X)} points ({len(X_maj)} maj, {len(X_min)} min). IR = {len(X_maj)/len(X_min):.1f}")

# 2. Resample using ECEBO, ECEBO_PSI0, SMOTE
ecebo = ECEBO(iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=42)
X_ecebo, y_ecebo = ecebo.fit_resample(X, y)
X_syn_ecebo = X_ecebo[len(X):]

ecebo_psi0 = ECEBO_PSI0(iters=60, prob=0.2, alpha=1.0, k=7, step=0.1, random_state=42)
X_psi0, y_psi0 = ecebo_psi0.fit_resample(X, y)
X_syn_psi0 = X_psi0[len(X):]

smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)
X_syn_smote = X_smote[len(X):]

print(f"Generated synthetic points: ECEBO={len(X_syn_ecebo)}, ECEBO_PSI0={len(X_syn_psi0)}, SMOTE={len(X_syn_smote)}")

# 3. Create 4-panel figure like Fig. 2 of the paper
fig, axes = plt.subplots(1, 4, figsize=(20, 4.8), dpi=300)

panels = [
    ("Original Distribution", None, "No Oversampling"),
    ("EC-EBO (Author Implementation)", X_syn_ecebo, "psi < 0 rule"),
    ("EC-EBO_PSI0 (Interpretation)", X_syn_psi0, "|psi| -> 0 rule"),
    ("SMOTE (Baseline)", X_syn_smote, "Linear interpolation")
]

for ax, (title, X_syn_pts, subtitle) in zip(axes, panels):
    # Plot Majority
    ax.scatter(X_maj[:, 0], X_maj[:, 1], c='#a0aec0', label='Majority (Real)', alpha=0.55, edgecolors='none', s=25)
    # Plot Real Minority
    ax.scatter(X_min[:, 0], X_min[:, 1], c='#2b6cb0', label='Minority (Real)', alpha=0.9, edgecolors='black', linewidth=0.5, s=40)
    
    # Plot Synthetic Minority if present
    if X_syn_pts is not None:
        ax.scatter(X_syn_pts[:, 0], X_syn_pts[:, 1], c='#e53e3e', label='Minority (Synthetic)', alpha=0.75, marker='^', edgecolors='black', linewidth=0.4, s=35)
        
    ax.set_title(f"{title}\n({subtitle})", fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel("Feature 1", fontsize=10)
    ax.set_ylabel("Feature 2", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_xlim(-1.5, 7.5)
    ax.set_ylim(-0.5, 6.5)
    ax.legend(loc='lower left', fontsize=8, framealpha=0.9)

plt.tight_layout()
os.makedirs("figs_stream", exist_ok=True)
png_path = "figs_stream/F3_toy.png"
pdf_path = "figs_stream/F3_toy.pdf"
plt.savefig(png_path)
plt.savefig(pdf_path)
plt.close()
print(f"Fig F3 saved successfully: {png_path} and {pdf_path}")

