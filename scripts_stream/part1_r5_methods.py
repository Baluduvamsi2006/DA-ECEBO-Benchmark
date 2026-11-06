"""
Part 1: R5. Comparison Methods Audit from EC-EBO Paper PDF
===========================================================
Source PDF: Tao et al., Knowledge-Based Systems 338 (2026) 115544
Exact quotes extracted from Page 7 (Section 4.1), Page 12 (Table 2), Page 13 (Data Availability), Page 1 (Affiliations & Emails).

Output: results_stream/R5_comparison_methods.csv
"""
import os
import pandas as pd

def main():
    records = [
        {
            'Method_Number': 1,
            'Method_Name': 'SMOTE',
            'Printed_Name_in_Paper': 'SMOTE [24]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'SMOTE [24], Borderline-SMOTE (Bord) [25], A-SUWO [37], G-SMOTE [27], CCR [30], NCLO-SMOTE [42], MLOS [16], DGS [39], HS-SMOTE(HS) [43], and ENU [32].',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Public Python package available and installed (imbalanced-learn: imblearn.over_sampling.SMOTE).'
        },
        {
            'Method_Number': 2,
            'Method_Name': 'Borderline-SMOTE',
            'Printed_Name_in_Paper': 'Borderline-SMOTE (Bord) [25]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'Borderline-SMOTE (Bord) [25]',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Public Python package available and installed (imbalanced-learn: imblearn.over_sampling.BorderlineSMOTE).'
        },
        {
            'Method_Number': 3,
            'Method_Name': 'A-SUWO',
            'Printed_Name_in_Paper': 'A-SUWO [37]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'A-SUWO [37]',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Public Python package available and installed (smote_variants: smote_variants.A_SUWO).'
        },
        {
            'Method_Number': 4,
            'Method_Name': 'G-SMOTE',
            'Printed_Name_in_Paper': 'G-SMOTE [27]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'G-SMOTE [27]',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Public Python package available and installed (smote_variants: smote_variants.G_SMOTE).'
        },
        {
            'Method_Number': 5,
            'Method_Name': 'CCR',
            'Printed_Name_in_Paper': 'CCR [30]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'CCR [30]',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Public Python package available and installed (smote_variants: smote_variants.CCR).'
        },
        {
            'Method_Number': 6,
            'Method_Name': 'MLOS',
            'Printed_Name_in_Paper': 'MLOS [16]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'MLOS applies Mahalanobis distance in combination with local density ... in complex, highly imbalanced datasets [16].',
            'Status': 'RUN',
            'Why_Run_or_Not_Run': 'Official author GitHub repository located (https://github.com/ytyancp/MLOS, MLOS.py); integrated with majority background capped to 20,000 on large streams.'
        },
        {
            'Method_Number': 7,
            'Method_Name': 'HS-SMOTE',
            'Printed_Name_in_Paper': 'HS-SMOTE(HS) [43]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'HS-SMOTE(HS) [43]',
            'Status': 'NOT RUN',
            'Why_Run_or_Not_Run': 'Repository (https://github.com/gogossss/HS-SMOTE) provides MATLAB .m source code only; no native Python implementation exists in PyPI or public repositories.'
        },
        {
            'Method_Number': 8,
            'Method_Name': 'NCLO-SMOTE',
            'Printed_Name_in_Paper': 'NCLO-SMOTE [42]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'NCLO-SMOTE [42] employs a semi-supervised clustering mech- ... [40], and ENU [32], attempt to alleviate overlap by discarding am-',
            'Status': 'NOT RUN',
            'Why_Run_or_Not_Run': 'No public code found on GitHub or PyPI. Paper statement: "Data availability: Data will be made available on request, p. 12" (Neurocomputing 616 (2025) 128959).'
        },
        {
            'Method_Number': 9,
            'Method_Name': 'DGS',
            'Printed_Name_in_Paper': 'DGS [39]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'DGS [39] combines ... DGS, ENU, and HS-SMOTE have a complexity of O(n_min * n_maj * d)',
            'Status': 'NOT RUN',
            'Why_Run_or_Not_Run': 'No public code found on GitHub (checked author repo https://github.com/ytyancp). Paper statement: "Data availability: Data will be made available on request, p. 16" (Inf. Sci. 718 (2025) 122416).'
        },
        {
            'Method_Number': 10,
            'Method_Name': 'ENU',
            'Printed_Name_in_Paper': 'ENU [32]',
            'Paper_Page_and_Table': 'Page 7 (Section 4.1), Page 12 (Table 2)',
            'Exact_PDF_Quote': 'gions by removing select majority class samples. For example, ENU [32] ... ENU under the CART classifier, where the p-values slightly exceed',
            'Status': 'NOT RUN',
            'Why_Run_or_Not_Run': 'No public code repository found on GitHub or PyPI (Kumar, Singh, Yadav, Concurrency Comput. Pract. Exper. 36(2) (2024) e7894).'
        }
    ]

    df = pd.DataFrame(records)
    os.makedirs('results_stream', exist_ok=True)
    out_csv = 'results_stream/R5_comparison_methods.csv'
    df.to_csv(out_csv, index=False)
    print(f"Saved R5 methods audit to {out_csv}")
    print(df[['Method_Number', 'Method_Name', 'Status', 'Why_Run_or_Not_Run']].to_string())

if __name__ == '__main__':
    main()

