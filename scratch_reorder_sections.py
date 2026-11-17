with open('e:/ml res/DA_ECEBO_manuscript_revised.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Let's find the temporal generalization block:
# It starts at \subsubsection{Temporal Generalization Across Rolling Stream Blocks}\label{sec:blocks_stream}
# and ends right before \subsection{Directional Synthesis Dynamics and Acceptance Audit}

block_start = text.find(r'\subsubsection{Temporal Generalization Across Rolling Stream Blocks}\label{sec:blocks_stream}')
block_end = text.find(r'\subsection{Directional Synthesis Dynamics and Acceptance Audit}')

temporal_block = text[block_start:block_end]

# Remove it from its current position
text_without_block = text[:block_start] + text[block_end:]

# Now find where to insert it: right after \subsubsection{Comparative Performance Against Unweighted Plain Training}\label{sec:delta}
# specifically right before \subsection{Robustness to Temporal Ordering: Permuted-Timestamp Control Analysis}
insert_pos = text_without_block.find(r'\subsection{Robustness to Temporal Ordering: Permuted-Timestamp Control Analysis}')

text_reordered = text_without_block[:insert_pos] + temporal_block + '\n' + text_without_block[insert_pos:]

# Rename \subsection{Other classifiers} to \subsection{Generalization Across Alternative Base Classifiers}
text_reordered = text_reordered.replace(r'\subsection{Other classifiers}', r'\subsection{Generalization Across Alternative Base Classifiers}\label{sec:classifiers_eval}')

# Rename \subsection{Parameters and run time} to \subsection{Computational Complexity, Parameter Scaling, and Runtime Analysis}
text_reordered = text_reordered.replace(r'\subsection{Parameters and run time}', r'\subsection{Computational Complexity, Parameter Scaling, and Runtime Analysis}\label{sec:runtime_eval}')

# Rename \subsection{Design choices} to \subsection{Ablation Study and Design Choice Validation}
text_reordered = text_reordered.replace(r'\subsection{Design choices}', r'\subsection{Ablation Study and Design Choice Validation}\label{sec:ablation_eval}')

# Rename \subsection{Choosing $K$ and $\lambda$} to \subsection{Hyperparameter Sensitivity: Choosing $K$ and $\lambda$}
text_reordered = text_reordered.replace(r'\subsection{Choosing $K$ and $\lambda$}', r'\subsection{Hyperparameter Sensitivity: Choosing $K$ and $\lambda$}\label{sec:param_choice}')

with open('e:/ml res/DA_ECEBO_manuscript_revised.tex', 'w', encoding='utf-8') as f:
    f.write(text_reordered)

print('Reordered and renamed subsections successfully!')
