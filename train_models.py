import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib
import openml as oml
# Charger les données
fmnist = oml.datasets.get_dataset(40996)
X, y, _, _ = fmnist.get_data(target=fmnist.default_target_attribute)

# Preprocessing
X = X / 255.0  # Normalisation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=0.1)

# Définir les modèles
classifiers = [LogisticRegression(max_iter=1500), LinearSVC(max_iter=1100, dual=False,tol=1e-4), KNeighborsClassifier()]

# Entraîner les modèles
for clf in classifiers:
    clf.fit(X_train, y_train)

# Sauvegarder les modèles
joblib.dump(classifiers[0], 'models/logistic_regression.pkl')
joblib.dump(classifiers[1], 'models/linear_svc.pkl')
joblib.dump(classifiers[2], 'models/knn.pkl')

print("Les modèles ont été entraînés et sauvegardés.")