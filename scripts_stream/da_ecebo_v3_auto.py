"""
DAECEBO2_AUTO Implementation for Phase 3 Part 3
===============================================
Unified class supporting fair hyperparameter search:
  - k_windows in {1, 2, 4}
  - lambd in {0.0, 0.25, 0.5, 1.0}
  - rule in {'literal', 'psi0'}
  - rho in {0.1, 0.25, 0.5, 1.0} (share of full balancing quota Delta N)
  - step_scale in {0.5, 1.0, 2.0} (step = 0.1 * step_scale)

Guarantees:
  - Exact reduction to ECEBO (literal) when K=1, lambd=1.0, rule='literal', rho=1.0, step_scale=1.0 (diff = 0.0).
  - Exact reduction to ECEBO_PSI0 when K=1, lambd=1.0, rule='psi0', rho=1.0, step_scale=1.0 (diff = 0.0).
"""
import numpy as np
from da_ecebo_v2_candidate import DAECEBO2
from da_ecebo_v2_psi0 import DAECEBO2_PSI0

class DAECEBO2_AUTO:
    def __init__(self, k_windows=4, lambd=0.25, rule='psi0', rho=1.0, step_scale=1.0,
                 delta_base=0.0, random_state=42):
        self.k_windows = max(1, int(k_windows))
        self.lambd = float(lambd)
        self.rule = str(rule).lower()
        self.rho = float(rho)
        self.step_scale = float(step_scale)
        self.step = 0.1 * self.step_scale
        self.delta_base = delta_base
        self.random_state = random_state

    def fit_resample(self, X, y, timestamps=None):
        classes, counts = np.unique(y, return_counts=True)
        maj_c, min_c = (classes[0], classes[1]) if counts[0] > counts[1] else (classes[1], classes[0])
        n_maj, n_min = np.sum(y == maj_c), np.sum(y == min_c)
        
        target_min = n_min + self.rho * (n_maj - n_min)
        samp_strat = target_min / n_maj
        
        if self.rule == 'psi0':
            model = DAECEBO2_PSI0(k_windows=self.k_windows, lambd=self.lambd, step=self.step,
                                  delta_base=self.delta_base, random_state=self.random_state)
            return model.fit_resample(X, y, sampling_strategy=samp_strat)
        else:
            model = DAECEBO2(k_windows=self.k_windows, lambd=self.lambd, step=self.step,
                             delta_base=self.delta_base, random_state=self.random_state)
            return model.fit_resample(X, y, timestamps=timestamps, sampling_strategy=samp_strat)
