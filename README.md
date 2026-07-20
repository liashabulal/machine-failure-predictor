# Machine Failure Predictor

ML system for predicting machine failures from sensor data, benchmarking six 
classification models and deployed as an interactive Streamlit app.

## Overview

Built on the AI4I 2020 Predictive Maintenance dataset (10,000 rows, 3.4% failure 
rate). Rather than optimizing for accuracy, which is misleading on a heavily 
imbalanced target, models are compared on recall/precision tradeoffs, reflecting 
the real business cost asymmetry: a missed failure is more costly than a false alarm.

## Approach

- Benchmarked six classifiers: Logistic Regression, Decision Tree, Random Forest, 
  SVM, a scikit-learn Neural Network, and a PyTorch MLP, all under class-weighted 
  training.
- Swept decision thresholds from 0.10–0.50 for each model, selecting thresholds 
  that reflect the recall/precision tradeoff appropriate for failure prediction.
- Random Forest was selected as the best-performing model, threshold 0.30, 
  recall 0.72 / F1 0.74.
- Used Decision Tree feature importance to identify torque and rotational speed 
  as top failure drivers; flagged multicollinearity between air and process 
  temperature via correlation analysis.

## Extended Analysis: PCA, Deep Learning Benchmark & SQLite Ingestion

**PCA on sensor features:** Standardized the five sensor features (air temperature, 
process temperature, rotational speed, torque, tool wear) and ran PCA to check for 
redundancy following the earlier observation of air/process temperature correlation. 
Three of five components capture 95% of variance, crossing the 90% threshold between 
PC2 and PC3. PC1 and PC2 are nearly tied (38.2% and 36.8%), suggesting the redundancy 
is spread across features rather than concentrated in a single pair. Used as a 
diagnostic step rather than a preprocessing/reduction step, since the original five 
features remain more interpretable for SHAP-based explanations than abstract components.

**Deep learning benchmark:** Added a PyTorch MLP (3 hidden layers, BCEWithLogitsLoss 
with pos_weight to handle class imbalance) as a sixth model, evaluated with the same 
threshold sweep and metrics as the existing five. Best F1 in-range was 0.50 
(threshold 0.50, recall 0.94, precision 0.34), below Random Forest's 0.74. The 
strong class-weight correction pushed recall up but at a real precision cost. 
Random Forest remains the best-performing model for this dataset, consistent with 
tree-based models generally outperforming deep learning on small tabular datasets.

**SQLite ingestion layer:** Migrated data loading from a flat CSV to a local SQLite 
database (`data/sensor_data.db`), simulating a lightweight production-style 
ingestion step.

## Setup

```bash
git clone https://github.com/liashabulal/machine-failure-predictor.git
cd machine-failure-predictor
uv venv
.venv\Scripts\activate   # or source .venv/bin/activate on Mac/Linux
uv pip install -r requirements.txt
python setup_db.py       # generates local SQLite database from source data
```

## App

Run the Streamlit app locally:

```bash
streamlit run app.py
```

## Tech Stack

Python · scikit-learn · PyTorch · SQLite · pandas · SHAP · Streamlit · matplotlib

