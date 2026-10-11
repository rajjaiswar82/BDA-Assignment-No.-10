# Fintech Fraud Detection Using Big Data Analytics

**BDA Assignment No. 10 — BE Computer Engineering, Semester 7**  
**Authors:** Raj Jaiswar and Adrain

## Project overview
This educational project builds a reproducible machine-learning baseline for identifying potentially fraudulent credit-card transactions. It uses a public, anonymized dataset and a class-weighted Logistic Regression pipeline. The goal is to demonstrate data preparation, rare-event classification, and appropriate evaluation—not to claim production readiness.

## Dataset
Download the **Credit Card Fraud Detection** dataset from Kaggle:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

The dataset contains 284,807 transactions and 492 fraud cases (about 0.172%). Features `V1`–`V28` are anonymized PCA components; `Time` and `Amount` are also provided. `Class=1` means fraud and `Class=0` means legitimate. Dataset description and counts are from the source listing; confirm details against the downloaded file.

**Do not commit `creditcard.csv` to GitHub.** It is large; link to the original dataset page instead.

## Repository structure
```text
bda-fintech-fraud-detection/
├── fraud_detection.py
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
├── .gitignore
└── creditcard.csv        # download locally; excluded from Git
```

## Setup and run
1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download `creditcard.csv` from Kaggle and place it in the repository root.
5. Run:
   ```bash
   
   
   ```

The script removes exact duplicate rows, creates a stratified train/test split, scales numeric inputs within a scikit-learn pipeline, trains class-weighted Logistic Regression, and reports a confusion matrix, precision/recall/F1, average precision, and ROC-AUC.

## Method and limitations
- **Problem:** fraudulent transactions are rare, so accuracy alone can be misleading.
- **Preprocessing:** check labels, remove exact duplicates, split before fitting preprocessing to avoid test-data leakage.
- **Model:** Logistic Regression with `class_weight="balanced"` is a transparent baseline.
- **Metrics:** inspect fraud-class precision and recall, F1, average precision, ROC-AUC, and false-negative/false-positive counts.
- **Limitations:** this dataset represents two days in 2013; most variables are anonymized; a random split may not represent real-time deployment. Validate with chronological splits, threshold/cost analysis, drift monitoring, and human review before any operational use.

## Expected output
The script prints metrics calculated from the actual dataset at runtime. No performance numbers are claimed in advance. Save the terminal output and insert the verified values into the paper if your instructor requires empirical results.

## References
1. A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, “Calibrating Probability with Undersampling for Unbalanced Classification,” *2015 IEEE Symposium Series on Computational Intelligence*, 2015.
2. Machine Learning Group, Université Libre de Bruxelles, “Credit Card Fraud Detection,” Kaggle dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
3. IEEE, “Manuscript Templates for Conference Proceedings”: https://www.ieee.org/conferences/publishing/templates.html

## Academic integrity
This is a baseline implementation for learning. Run it yourself, verify the output, and disclose any changes. Do not report unrun experiments as measured results.
