# -*- coding: utf-8 -*-
"""
Created on Wed May 15 15:34:50 2024

@author: Emmanuel
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  
    X_normalized = (X - mean) / std
    return X_normalized

def train_svm(X_train, y_train, C=1.0, max_iter=1000, learning_rate=0.01):
    n_samples, n_features = X_train.shape

    weights = np.zeros(n_features)
    bias = 0

   
    for _ in range(max_iter):
        for idx, x in enumerate(X_train):
            condition = y_train[idx] * (np.dot(x, weights) - bias) >= 1
            if condition:
                weights -= learning_rate * (2 * C * weights)
            else:
                weights -= learning_rate * (2 * C * weights - np.dot(x, y_train[idx]))
                bias -= learning_rate * y_train[idx]

    return weights, bias

def predict(X, weights, bias):
    return np.sign(np.dot(X, weights) - bias)

def evaluate_model(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0

    return accuracy, precision, sensitivity, specificity, f1_score


data = pd.read_csv("zoo.csv")


X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values 
X = normalize(X)


np.random.seed(42) 
indices = np.random.permutation(len(X))
X_train, X_test = X[indices[:-20]], X[indices[-20:]]
y_train, y_test = y[indices[:-20]], y[indices[-20:]]

weights, bias = train_svm(X_train, y_train)

y_pred = predict(X_test, weights, bias)

accuracy, precision, sensitivity, specificity, f1_score = evaluate_model(y_test, y_pred)

# Imprimir métricas
print("Métricas del modelo:")
print(f"Precisión del modelo: {accuracy:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"Sensibilidad: {sensitivity:.2f}")
print(f"Especificidad: {specificity:.2f}")
print(f"Puntaje F1: {f1_score:.2f}")
