import re

with open('DA_ECEBO_manuscript_revised.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Expand Introduction (addressing Comment 1)
old_intro = r"""\section{Introduction}\label{sec:intro}
In many real problems one class is far smaller than the other. Credit card fraud, network intrusion and rare diseases are typical cases. A classifier trained on such data usually learns the large class well and misses the small one, and this is exactly the class that matters most to us~\cite{chen2024}. A popular remedy is oversampling, where new synthetic minority samples are generated and included in the training data prior to fitting the classifier.

Imbalance is not the only difficulty. Recent reviews show that overlap between the classes, and small separated groups of minority samples, make the problem much harder than the imbalance ratio alone suggests~\cite{santos2022,santos2023}. SMOTE, for example, draws a line between two minority neighbours and puts a new point on it. When the two neighbours sit in different groups, that point can land inside the majority class and cause false alarms. EC-EBO~\cite{tao2026} was proposed to deal with this: it builds minority clusters under an external constraint, so that two clusters are never merged across a majority sample, it gives clusters with higher entropy more synthetic samples, and it moves every new point step by step with a rule that keeps it away from the majority class. EC-EBO was tested on 60 benchmark datasets, all of them static, meaning that the order of the samples plays no role.

Many real data sets are streams, though. New transactions keep arriving, and what the minority class looks like can shift along the way~\cite{aguiar2023,korycki2021,abdullahi2025}. A fraud pattern that was common last week may be gone this week, and an oversampler that treats all minority samples as equal may spend its synthetic samples on old patterns.

The present work starts from a simple question: \emph{does the time at which a fraud occurred tell an oversampler where its synthetic samples should go?} To answer it, EC-EBO is extended to DA-ECEBO (drift-aware EC-EBO). The training time is cut into $K$ windows, every cluster is scored by how recent it is and by the direction in which it moved, and this score is combined with the entropy score of EC-EBO to set the quota of the cluster. The same direction may also be used as one choice for the step of the hill-climbing search. The question is asked on a stream with a real time column and is then tested with controls, so that the effect of time can be separated from the other changes made to the quota. The study asks whether DA-ECEBO improves Macro-F1 and Macro G-mean over EC-EBO and over the other oversamplers, whether the gain depends on the time information, where the synthetic samples go, and whether the results hold for later time blocks, other classifiers and static datasets.

The main contributions of this paper are summarised as follows:
\begin{itemize}
\item \textbf{A time-aware quota for EC-EBO.} Time windows, a recency score and a drift direction are added to the entropy-weighted quota and to the step of the hill-climbing search, and the relation of the result to EC-EBO is stated and tested (Section~\ref{sec:method}).
\item \textbf{A strict evaluation on a fraud stream.} Training uses the first 75\% of the stream and testing the future, with Macro-F1 and Macro G-mean as main metrics, ten seeds, five later blocks, seven classifiers, and 14 further datasets (Sections~\ref{sec:setup} and~\ref{sec:results}).
\item \textbf{Controls that separate time from spread.} Permuted timestamps, random cluster scores and restricted quotas show that the gain over EC-EBO comes from a more even spread of the quota over the clusters, so that the method is robust to the order of the timestamps (Section~\ref{sec:time}).
\item \textbf{An audit of the synthesis step.} The share of synthetic samples that are exact copies of real samples, and the share of accepted steps, are measured under the acceptance rule printed for EC-EBO and under a variant of it (Section~\ref{sec:synth}).
\end{itemize}"""

new_intro = r"""\section{Introduction}\label{sec:intro}
In real-world classification problems, severe class imbalance is ubiquitous, occurring in critical domains such as financial fraud detection, network intrusion monitoring, and medical diagnosis~\cite{chen2024}. When trained on datasets where the minority class constitutes a tiny fraction of instances, standard statistical classifiers inherently bias their decision boundaries toward the majority class, resulting in high nominal accuracy but catastrophic false negative rates on the rare instances that matter most. To mitigate this imbalance, data-level oversampling methods synthesize artificial minority instances to re-balance the training distribution prior to model fitting.

However, class imbalance is rarely an isolated challenge. Comprehensive surveys by Santos and co-authors~\cite{santos2022,santos2023} emphasize that class overlap, small disjuncts, and borderline noise exacerbate the problem far more severely than the nominal imbalance ratio alone. Conventional oversamplers frequently suffer from distinct architectural pitfalls:
\begin{enumerate}[leftmargin=1.5em,label=(\roman*)]
\item \textbf{Loss of Sample Diversity and Mode Collapse:} Basic random oversampling blindly replicates existing minority samples, inducing severe overfitting. While SMOTE~\cite{chawla2002} introduces linear interpolation between $k$-nearest neighbors, it generates samples along static line segments, failing to explore vacant high-density regions. Recent methods such as ASRDO~\cite{tao2024} and HS-SMOTE~\cite{wang2025} attempt to enhance diversity via regular hexagon scoring and safe regions, but remain constrained by static feature geometry.
\item \textbf{Borderline Overlap and Noise Amplification:} Borderline-SMOTE~\cite{han2005} and ADASYN~\cite{he2008} focus synthesis on difficult samples near the class boundary. However, in overlapping distributions, interpolating between borderline neighbors frequently places synthetic points directly inside majority manifolds, creating ambiguous regions that elevate false positive rates. Techniques like ENU~\cite{kumar2024}, NCLO-SMOTE~\cite{tao2025}, and MLOS~\cite{yan2025b} address this by employing local entropy cleaning or Mahalanobis distance metrics. Similarly, EC-EBO~\cite{tao2026} constructs minority clusters under an external constraint prohibiting clusters from merging across majority samples, and guides sample synthesis away from majority anchors via an entropy-balance function $\psi$.
\item \textbf{Temporal Obsolescence and Inability to Handle Concept Drift:} Crucially, almost all established oversamplers—including EC-EBO and its variants—are strictly static algorithms that treat all historical training observations as identically distributed and time-invariant. In streaming data streams, however, underlying data-generating distributions evolve over time, a phenomenon known as concept drift~\cite{aguiar2023,korycki2021,abdullahi2025}. A fraud topology or network intrusion pattern active in earlier windows may become obsolete, while novel minority clusters emerge downstream. An oversampler that treats all minority instances equally allocates valuable synthesis budget to extinct patterns, actively degrading classifier performance on future stream observations.
\end{enumerate}

To resolve this critical limitation, this study introduces \textbf{DA-ECEBO} (Drift-Aware Entropy-Balanced Oversampling). The method integrates temporal dynamics directly into boundary-constrained oversampling. By partitioning the historical stream into $K$ equal temporal windows, DA-ECEBO estimates the recency and directional spatial displacement vector ($\mathbf{v}_i$) of each externally constrained minority cluster. These temporal indicators are synthesized with local cluster entropy to formulate a drift-aware quota allocation mechanism and guide hill-climbing synthesis along the genuine trajectory of concept evolution.

The core contributions of this work are directly aligned with the aforementioned gaps:
\begin{itemize}[leftmargin=1.5em]
\item \textbf{Temporal Adaptation in Data-Level Oversampling:} We introduce a temporal windowing and recency-weighted quota formulation that prevents synthesis budget from being wasted on obsolete minority modes, prioritizing active and newly emerging drift concepts (Section~\ref{sec:method}).
\item \textbf{Directionally Guided Synthesis:} We augment the hill-climbing search of EC-EBO by incorporating the unit drift vector $\hat{\mathbf{v}}_i$ of each cluster, directing synthetic instance generation along the empirical trajectory of concept shift rather than arbitrary isotropic exploration (Section~\ref{sec:method}).
\item \textbf{Rigorous Benchmark on Real and Synthetic Streams:} We benchmark DA-ECEBO across genuine streaming benchmarks from the systematic survey of Abdullahi et al.~\cite{abdullahi2025} (including Credit Card Fraud, Moving Hyperplane, SEA Concept, and Electricity Market) and 14 supplementary benchmark datasets, comparing against traditional oversamplers and four dedicated streaming concept drift learners (ARF, SRP, HAT, OOB) across ten seeds (Section~\ref{sec:results}).
\item \textbf{Experimental Controls and Mechanism Auditing:} Through permuted-timestamp experiments, cluster score randomization, and an audit of hill-climbing step acceptance rates, we isolate the exact mechanism behind the observed performance gains (Section~\ref{sec:time} and Section~\ref{sec:synth}).
\end{itemize}"""

assert old_intro in text, "old_intro not found"
text = text.replace(old_intro, new_intro)
print("Expanded Introduction with explicit gap alignment (Comment 1)")

# 2. Expand Section 4.3 with citations for all classifiers (Comment 7)
old_impl = r"""The main classifier is the unpruned decision tree (CART) of scikit-learn, and in addition random forest, 5-nearest neighbours, Gaussian naive Bayes, logistic regression, a linear SVM and an RBF SVM are used. DA-ECEBO is compared with no oversampling, SMOTE, Borderline-SMOTE and ADASYN (all from the imbalanced-learn library with default settings), and with EC-EBO. Since EC-EBO is our own implementation, written from the description in the paper~\cite{tao2026}, our numbers can differ from those of the authors' original code, and the acceptance rule is implemented as printed in Algorithm~1 of that paper."""

new_impl = r"""The primary classifier employed is the unpruned Classification and Regression Tree (CART)~\cite{breiman1984} implemented via scikit-learn. To evaluate generalizability across diverse inductive biases, six additional classifier architectures are evaluated: Random Forest (RF)~\cite{breiman2001}, $k$-Nearest Neighbors ($k$-NN, $k=5$)~\cite{cover1967}, Gaussian Naive Bayes (GNB)~\cite{duda1973}, Logistic Regression (LR)~\cite{hosmer2000}, Linear Support Vector Machines (Linear SVM)~\cite{cortes1995}, and Radial Basis Function Support Vector Machines (RBF SVM)~\cite{scholkopf1997}. 

DA-ECEBO is benchmarked against standard data-level oversampling baselines: baseline unresampled training (None), SMOTE~\cite{chawla2002}, Borderline-SMOTE~\cite{han2005}, and ADASYN~\cite{he2008} (utilizing the imbalanced-learn framework with default hyperparameters), and EC-EBO~\cite{tao2026}. Furthermore, to address concept drift mitigation directly, four dedicated streaming algorithms from the river package are benchmarked: Adaptive Random Forest (ARF)~\cite{gomes2017}, Streaming Random Patches (SRP)~\cite{gomes2019}, Hoeffding Adaptive Tree with ADWIN (HAT)~\cite{bifet2009}, and Online Over-sampling Bagging (OOB)~\cite{wang2015}."""

assert old_impl in text, "old_impl not found"
text = text.replace(old_impl, new_impl)
print("Updated Section 4.3 with full baseline citations (Comment 7)")

# 3. Expand Section 4.4 with formal metric definitions and citations (Comment 8)
old_metrics = r"""Accuracy is not useful when frauds are rare, and for this reason the main metrics are the \emph{macro-F1} (the mean of the F1 of the two classes) and the \emph{macro G-mean} ($\sqrt{\text{recall}_{min}\times\text{recall}_{maj}}$). In addition, the minority F1 (the harmonic mean of precision and recall on the fraud class), the precision (the share of predicted frauds that are real frauds), the recall and the ROC-AUC are reported, and precision matters in particular because false alarms are the main cost in fraud work. Two remarks help in reading the tables. First, the F1 of the majority class is almost exactly 1 in our data, which means that macro-F1 is about (minority F1 + 1)/2, so it ranks the methods like the minority F1, with differences half as large. Second, the recall of the majority class is above 0.999 for every method, and therefore the macro G-mean is driven by the minority recall. A method that finds more frauds at the price of many false alarms can win on macro G-mean and lose on macro-F1, which is why precision is shown next to the two main metrics. Table~\ref{tab:main} shows the main metrics, and Table~\ref{tab:allmetrics} in Section~\ref{sec:results} shows all of them."""

new_metrics = r"""In highly skewed domains such as fraud detection, raw classification accuracy is misleading due to the accuracy paradox. Consequently, performance is evaluated using metrics robust to class imbalance~\cite{he2009}:
\begin{itemize}[leftmargin=1.5em]
\item \textbf{Precision and Recall:} Letting $TP, FP, TN, FN$ denote true positives, false positives, true negatives, and false negatives for the minority class, precision and recall are defined as:
\begin{equation}
\mathrm{Precision} = \frac{TP}{TP + FP}, \qquad \mathrm{Recall} = \frac{TP}{TP + FN}.
\end{equation}
\item \textbf{Minority F1 and Macro-F1:} The minority F1 is the harmonic mean of precision and recall: $\mathrm{F1}_{min} = \frac{2 \cdot \mathrm{Precision} \cdot \mathrm{Recall}}{\mathrm{Precision} + \mathrm{Recall}}$. Macro-F1 averages the F1 scores across both classes to provide equal importance to both classes regardless of sample size~\cite{santos2022}:
\begin{equation}
\text{Macro-F1} = \frac{\mathrm{F1}_{min} + \mathrm{F1}_{maj}}{2}.
\end{equation}
\item \textbf{Macro G-Mean:} The geometric mean of recalls assesses balanced class accuracy~\cite{kubat1997}:
\begin{equation}
\text{Macro G-mean} = \sqrt{\mathrm{Recall}_{min} \times \mathrm{Recall}_{maj}}, \quad \text{where } \mathrm{Recall}_{maj} = \frac{TN}{TN + FP}.
\end{equation}
\item \textbf{Area Under the ROC Curve (ROC-AUC):} Quantifies ranking discrimination across all classification thresholds~\cite{fawcett2006}.
\end{itemize}"""

assert old_metrics in text, "old_metrics not found"
text = text.replace(old_metrics, new_metrics)
print("Updated Section 4.4 with formal metric definitions and citations (Comment 8)")

# 4. Add citations for cover, hosmer, scholkopf, he2009 to the bibliography
extra_bibs = r"""\bibitem{cover1967} Thomas M. Cover and Peter E. Hart. Nearest neighbor pattern classification. \emph{IEEE Transactions on Information Theory}, 13(1):21--27, 1967.
\bibitem{he2009} Haibo He and Edwardo A. Garcia. Learning from imbalanced data. \emph{IEEE Transactions on Knowledge and Data Engineering}, 21(9):1263--1284, 2009.
\bibitem{hosmer2000} David W. Hosmer Jr., Stanley Lemeshow, and Rodney X. Sturdivant. \emph{Applied Logistic Regression}. John Wiley \& Sons, 2000.
\bibitem{scholkopf1997} Bernhard Sch\"olkopf, Kyoung-In Sung, Christopher J. C. Burges, Federico Girosi, Partha Niyogi, Tomaso Poggio, and Vladimir Vapnik. Comparing support vector machines with Gaussian kernels to radial basis function classifiers. \emph{IEEE Transactions on Signal Processing}, 45(11):2758--2765, 1997.
"""

pos_bib_end = text.find(r"\end{thebibliography}")
text = text[:pos_bib_end] + extra_bibs + text[pos_bib_end:]

with open('DA_ECEBO_manuscript_revised.tex', 'w', encoding='utf-8') as f:
    f.write(text)

print("Text expansion script completed successfully!")
