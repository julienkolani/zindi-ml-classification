# Zindi ML Classification

Pipeline de machine learning développé pour une compétition Zindi de classification binaire.

## Pipeline

1. **Prétraitement** : encodage binaire de 33 colonnes, 3 stratégies d'encodage catégoriel (One-Hot, Fréquence, Target encoding)
2. **Sélection de features** : filtrage par corrélation (seuil 0,2), réduction dimensionnelle PCA (10 composantes)
3. **Modèles évalués** : Régression Logistique, KNN, Arbre de Décision, Random Forest, Gradient Boosting, SVM, Naive Bayes, MLP
4. **Optimisation** : GridSearchCV sur Random Forest avec validation croisée (cv=5)
5. **Sauvegarde** : persistance du modèle avec joblib

## Stack technique

- Python, scikit-learn, XGBoost, Pandas, NumPy, joblib

## Données

Les fichiers Train.csv (1,1 Go) et Test.csv (933 Mo) ne sont pas inclus en raison de leur taille.

## Installation

```bash
pip install -r requirements.txt
jupyter notebook main.ipynb
```
