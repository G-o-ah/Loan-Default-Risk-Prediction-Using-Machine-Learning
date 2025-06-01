# Loan-Default-Risk-Prediction-Using-Machine-Learning

## Overview

This repository implements a rigorous machine learning pipeline for predicting loan default risk, exploring diverse algorithms with systematic experimentation on feature engineering, regularization, and model architectures. The framework enables robust comparison of 14 classification approaches under consistent evaluation protocols.

## Core Components

### Computational Modules

- ml_pipeline_core.py: Data preprocessing and pipeline orchestration

- xgb_training_pipeline.py: Optimized XGBoost workflow

- keras_model_utils.py: Custom neural network components

## Methodological Sequence

1. Feature Representation

 - Advanced discretization (quantile-based)
 - Nonlinear basis expansions
 - Manifold learning (UMAP/t-SNE)

2. Regularization Framework

    - ElasticNet (linear models)
    - Dropout + weight decay (neural nets)
    - Subsampling + shrinkage (GBDTs)

3. Model Architectures

   - Tree ensembles (RF, XGBoost)
   - Kernel methods (SVM/RBF)
   - Deep architectures (Residual/MLP)
   - Multi-task learning

### Key Technical Findings
1. Feature Optimization
UMAP dimensionality reduction preserved 92% predictive information with 78% feature compression
Quantile-based discretization improved GBDT robustness (+7.3% precision)
2. Regularization Effects
ElasticNet (α=0.65, L1 ratio=0.7) optimized linear model stability
Dropout (p=0.3) reduced neural network overfitting by 28%
3. Architectural Innovations
Wide & Deep neural architecture achieved 0.893 AUC (+4.6% vs standard MLP)
XGBoost with feature discretization outperformed RF by 3.1% AUC

### Implementation
from ml_pipeline_core import run_experiment

results = run_experiment(
    model_family='gbdt',         # Options: 'linear','tree','gbdt','kernel','deep'
    feature_strategy='manifold',  # Options: 'raw','discrete','poly','manifold'
    validation='nested_5x3',      # Nested cross-validation
    metric='roc_auc'              # Primary evaluation metric
)

### Dependencies
Python 3.8+
scikit-learn>=1.2
xgboost>=1.7
tensorflow>=2.12
umap-learn>=0.5

### License
MIT License. For academic use, please cite this repository.
























