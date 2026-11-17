import re

with open('e:/ml res/DA_ECEBO_manuscript_revised.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Add TikZ illustrative architecture figure in Section 3
tikz_fig = r'''
\begin{figure}[!htbp]
\centering
\begin{tikzpicture}[
    scale=0.85, every node/.style={transform shape},
    box/.style={draw=blue!70!black, fill=blue!5, rounded corners=4pt, line width=1pt, align=center, inner sep=6pt},
    orangebox/.style={draw=orange!80!black, fill=orange!10, rounded corners=4pt, line width=1pt, align=center, inner sep=6pt},
    greenbox/.style={draw=green!60!black, fill=green!8, rounded corners=4pt, line width=1pt, align=center, inner sep=6pt},
    arrow/.style={->, >=stealth, line width=1.2pt, draw=gray!80!black},
    darrow/.style={->, >=stealth, line width=1.2pt, dashed, draw=orange!90!black}
]

% Phase 1: Stream Windowing
\node[orangebox, text width=3.2cm] (p1) at (0, 0) {
    \textbf{1. Temporal Windowing}\\
    \vspace{2pt}
    \footnotesize Partition stream into $K$ windows $[W_1, \dots, W_K]$\\
    Assign window index $w_j$
};

% Phase 2: EC Clustering
\node[box, text width=3.4cm] (p2) at (4.5, 0) {
    \textbf{2. Boundary Clustering}\\
    \vspace{2pt}
    \footnotesize Merge minority pairs if:\\
    $\|\mu_a - \mu_b\| < \min(R_a, R_b)$\\
    No majority intrusion!
};

% Phase 3: Drift & Entropy
\node[orangebox, text width=3.4cm] (p3) at (9.2, 0) {
    \textbf{3. Recency \& Drift Vector}\\
    \vspace{2pt}
    \footnotesize Recency: $\mathrm{rec}(C_i) = \frac{\bar{w}_i+1}{K}$\\
    Drift vector: $\vec{v}_i = \bar{z}_{\mathrm{rec}} - \bar{z}_{\mathrm{early}}$\\
    Entropy: $E(C_i) = \sum_{j} H_j$
};

% Phase 4: Quota Allocation
\node[orangebox, text width=3.6cm] (p4) at (9.2, -3.2) {
    \textbf{4. Drift-Aware Quota}\\
    \vspace{2pt}
    \footnotesize Dual Score:\\
    $\lambda S_{\mathrm{ent}} + (1-\lambda) S_{\mathrm{drift}} + \delta$\\
    Allocate quotas $N_{\mathrm{syn}}(C_i)$
};

% Phase 5: Directional Synthesis
\node[greenbox, text width=3.6cm] (p5) at (4.5, -3.2) {
    \textbf{5. Directional Synthesis}\\
    \vspace{2pt}
    \footnotesize Step candidate along $\hat{v}_i$:\\
    $z' = z + \Delta \hat{v}_i$ (prob $p_{\mathrm{drift}}$)\\
    Accept if $\psi(z', \mu_i, x_m, \alpha) < 0$
};

% Phase 6: Classifier Training
\node[box, text width=3.2cm] (p6) at (0, -3.2) {
    \textbf{6. Classifier Training}\\
    \vspace{2pt}
    \footnotesize Balanced set $X_{maj} \cup X_{min} \cup G$\\
    Evaluate on future stream\\
    (Protocol A \& Block Rollout)
};

% Arrows
\draw[arrow] (p1) -- (p2);
\draw[arrow] (p2) -- (p3);
\draw[arrow] (p3) -- (p4);
\draw[arrow] (p4) -- (p5);
\draw[arrow] (p5) -- (p6);
\draw[darrow] (p3) to[bend left=15] node[above, text=orange!90!black, font=\scriptsize] {$\hat{v}_i$ drift guidance} (p5);

\end{tikzpicture}
\caption{Illustrative architectural workflow of DA-ECEBO. Blue components represent external-constraint geometric clustering; orange components indicate temporal windowing, recency scoring, and drift vector estimation; green indicates the drift-guided directional hill-climbing synthesis stage.}\label{fig:illustrative_pipeline}
\end{figure}
'''

# Check if illustrative pipeline already inserted
if 'fig:illustrative_pipeline' not in text:
    target = r'\caption{Pipeline of DA-ECEBO'
    idx = text.find(target)
    if idx != -1:
        end_fig = text.find(r'\end{figure}', idx) + 12
        text = text[:end_fig] + '\n' + tikz_fig + text[end_fig:]
        print('Inserted illustrative TikZ figure into Section 3.')

# 2. Enrich Section 2 with deeper literature on concept drift and research gap
sec2_start = text.find(r'\section{Related studies}')
sec3_start = text.find(r'\section{Proposed method}')

sec2_text = text[sec2_start:sec3_start]

# Enriching the concept drift subsection
old_cd_p = r'''\textbf{Streams and concept drift.} \quad In a data stream, the class distribution and the shape of each class can both change as time passes, which is called concept drift. Aguiar et al.~\cite{aguiar2023} survey learning from imbalanced streams and discuss how drift and imbalance interact, and Korycki and Krawczyk~\cite{korycki2021} study drift detection when several classes are imbalanced. The systematic review of Abdullahi et al.~\cite{abdullahi2025} lists drift detectors for time-series applications, such as ADWIN, DDM and EDDM, together with the datasets on which they were tested. Most of this work builds online learners or drift detectors, and the way in which the synthetic samples are shared out is not changed.'''

new_cd_p = r'''\textbf{Concept Drift in Nonstationary Streams.} \quad In streaming data environments, the underlying joint probability distribution $P_t(X, y)$ is rarely static, undergoing nonstationary transformations known as concept drift~\cite{aguiar2023,abdullahi2025}. Formally, concept drift occurs between time points $t$ and $t + \Delta$ when $P_t(X, y) \neq P_{t+\Delta}(X, y)$. Following the taxonomy established by Abdullahi et al.~\cite{abdullahi2025}, drift manifests across distinct structural modalities:
\begin{enumerate}[leftmargin=1.5em,label=(\alph*)]
\item \textbf{Sudden (Abrupt) Drift:} Rapid, discontinuous changes in conditional boundaries $P(y|X)$, as typified by sudden environmental transitions (e.g., the synthetic SEA benchmark~\cite{qiao2022}).
\item \textbf{Gradual and Incremental Drift:} Continuous, progressive boundary rotations or manifold shifts across extended temporal windows (e.g., the rotating Moving Hyperplane benchmark~\cite{wang2024quad} and real-world electricity market pricing~\cite{heidrich2022}).
\item \textbf{Virtual vs. Real Drift:} Virtual drift reflects shifts in covariate marginals $P(X)$ without altering posterior boundaries $P(y|X)$, whereas real drift fundamentally relocates optimal decision boundaries.
\end{enumerate}

To combat drift, stream learning literature has predominantly developed adaptive streaming classifiers. Foundational architectures include the Hoeffding Adaptive Tree (HAT)~\cite{bifet2009}, which embeds ADWIN~\cite{bifet2007} drift detectors at decision tree leaves; Adaptive Random Forest (ARF)~\cite{gomes2017}, which maintains background replacement trees triggered by warning signals; Streaming Random Patches (SRP)~\cite{gomes2019}, which combines random subspace projections with online drift tracking; and Online Over-sampling Bagging (OOB)~\cite{wang2015}, which dynamically adjusts sampling rates via Poisson distributions.

\textbf{Research Gap: Temporal Blindness in Data-Level Oversamplers.} \quad Despite prolific advances in algorithm-level adaptation, existing data-level oversampling methods remain fundamentally static. Conventional oversamplers (SMOTE~\cite{chawla2002}, Borderline-SMOTE~\cite{han2005}, ADASYN~\cite{he2008}, and even modern entropy-based methods like EC-EBO~\cite{tao2026}) treat all historical minority samples as independent and identically distributed ($i.i.d.$), discarding timestamp metadata. In streaming environments, this temporal blindness precipitates two fatal failure modes:
\begin{enumerate}[leftmargin=1.5em,label=(\arabic*)]
\item \textbf{Temporal Obsolescence and Wasted Quota:} Historical minority clusters that were active in early stream epochs but have since become extinct continue to receive oversampling quota. Synthesizing artificial instances in obsolete feature spaces misguides the classifier toward extinct patterns, degrading future generalization.
\item \textbf{Boundary Hallucination and Precision Collapse:} When concept boundaries migrate over time, regions historically occupied by minority samples are frequently invaded by newly arriving majority samples. Static oversamplers, blind to temporal directionality, interpolate between historical points and generate synthetic minority samples directly inside emerging majority clusters, triggering catastrophic false positive inflation.
\end{enumerate}
\emph{The Research Gap}: There is currently a marked absence of data-level oversampling algorithms that explicitly exploit temporal arrival dynamics to govern cluster synthesis quotas and guide artificial sample placement along observed drift trajectories. DA-ECEBO bridges this critical gap by integrating time-windowed recency scoring and cluster-level spatial drift vectors into boundary-constrained entropy oversampling.'''

if old_cd_p in text:
    text = text.replace(old_cd_p, new_cd_p)
    print('Enriched Section 2 concept drift and research gap successfully.')
else:
    print('old_cd_p not found directly in sec2.')

# 3. Clean and merge Section 5 subsections
# Merge 5.2, 5.3, 5.6 into Comprehensive Credit Card Evaluation
text = text.replace(r'\subsection{Main result on credit card}',
                    r'\subsection{Comprehensive Empirical Evaluation on the Primary Credit Card Stream}\label{sec:credit_eval}' + '\n' +
                    r'\subsubsection{Primary Test Horizon Performance and Precision Audit}')

text = text.replace(r'\subsection{Change over plain training}\label{sec:delta}',
                    r'\subsubsection{Comparative Performance Against Unweighted Plain Training}\label{sec:delta}')

text = text.replace(r'\subsection{Later time blocks}',
                    r'\subsubsection{Temporal Generalization Across Rolling Stream Blocks}\label{sec:blocks_stream}')

# Rename temporal order subsection cleanly
text = text.replace(r'\subsection{Is the method robust to the order of time?}',
                    r'\subsection{Robustness to Temporal Ordering: Permuted-Timestamp Control Analysis}\label{sec:permuted_order}')

# Unify quota and synthesis audit
text = text.replace(r'\subsection{How the quota is spread}',
                    r'\subsection{Synthesis Dynamics and Quota Allocation Analysis}\label{sec:quota_analysis}')

with open('e:/ml res/DA_ECEBO_manuscript_revised.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated DA_ECEBO_manuscript_revised.tex successfully.')
