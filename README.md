# Zindi ML Classification Challenge

A machine learning pipeline developed for a Zindi data science competition.

## Overview

Implements a full preprocessing and model comparison workflow for binary classification. The pipeline covers binary column encoding, three categorical encoding strategies (One-Hot, Frequency, Target), correlation-based feature filtering, PCA dimensionality reduction (10 components), and evaluation of 8 classifiers with GridSearchCV hyperparameter tuning.

## Tech Stack

- Python
- scikit-learn
- pandas, numpy
- Jupyter Notebook

## Classifiers Evaluated

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- SVM
- Naive Bayes
- Neural Network (MLPClassifier)

## Dataset

Train.csv (1.1 GB) and Test.csv (933 MB) are excluded from this repository due to size. These are binary classification datasets used in the Zindi competition. A sample dataset (`show_dataset.csv`) is included for reference.

## Setup

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```
