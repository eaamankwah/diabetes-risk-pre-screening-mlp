# Purpose of this Folder

This folder contains the **complete, working solution** for the Diabetes Risk Prediction MLP project. It is used by reviewers (and can be used by students as a reference) to verify expected notebook behavior, outputs, and results.

Everything needed to run the project end-to-end is self-contained here — no external downloads are required.

## Contents

```
starter-kit/
├── diabetes_prediction_mlp.ipynb   - Completed project notebook (all TODOs implemented)
├── README.md                        - This file
├── images/                          - Exported result plots (for GitHub README / portfolio use)
│   ├── loss_curve.png
│   ├── optimizer_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── experiment_comparison.png
└── data/
    ├── diabetes_data.csv                                  - Balanced dataset (50,000 rows, 50/50 split)
    ├── diabetes_binary_health_indicators_BRFSS2015.csv    - Original imbalanced dataset (253,680 rows, ~13.9% prevalence)
    └── data_dictionary.md                                 - Feature descriptions and clinical context
```

## What's implemented

- All 28 required TODOs (EDA, preprocessing, model design, training loop, evaluation, and the three required improvement experiments: dropout, learning rate tuning, architecture tuning)
- All three student-choice / stand-out sections:
  1. Combined optimization techniques (dropout + weight decay + learning-rate scheduling)
  2. Class-imbalance handling (class weighting and SMOTE, evaluated on the real imbalanced dataset)
  3. Feature engineering (clinically-motivated interaction terms)
- A dedicated diagnostic section (4.5) investigating and addressing validation-loss stability (optimizer choice, batch size), plus an empirical dropout-rate sweep

## How to run

From this directory:

```sh
pip install -r ../requirements.txt
jupyter notebook diabetes_prediction_mlp.ipynb
```

Run all cells from top to bottom — later cells depend on variables and functions defined earlier. A full run (100 epochs across all baseline, tuning, and stand-out experiments) takes roughly 30-45 minutes on CPU; a GPU will speed this up substantially if available (the notebook auto-detects CUDA).

## Notes for reviewers

- The notebook is provided with all cell outputs cleared, so grading reflects a genuine top-to-bottom execution rather than stale cached results.
- Section 4.5 and the dropout-rate sweep in Section 6.1 are additions beyond the base rubric, included to empirically justify specific hyperparameter choices (batch size, optimizer, dropout rate) rather than asserting them without evidence.
- The class-imbalance stand-out section uses the real original CDC BRFSS2015 dataset (`diabetes_binary_health_indicators_BRFSS2015.csv`), not a simulated approximation — see the markdown in that section for the data source.
