import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  
    X_normalized = (X - mean) / std
    return X_normalized

def calculate_class_probabilities(X, y, target_class):
    total_samples = len(y)
    class_indices = np.where(y == target_class)[0]
    class_samples = X[class_indices]
    class_probability = len(class_samples) / total_samples
    return class_samples, class_probability

def calculate_mean_std(features):
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    stds[stds == 0] = 1e-6  
    return means, stds

def calculate_gaussian_probability(x, mean, std):
    exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def train_naive_bayes(X_train, y_train):
    unique_classes = np.unique(y_train)
    class_probabilities = {}
    class_statistics = {}
    for target_class in unique_classes:
        class_samples, class_probability = calculate_class_probabilities(X_train, y_train, target_class)
        class_means, class_stds = calculate_mean_std(class_samples)
        class_probabilities[target_class] = class_probability
        class_statistics[target_class] = {'means': class_means, 'stds': class_stds}
    return class_probabilities, class_statistics


def predict_naive_bayes(X_test, class_probabilities, class_statistics):
    predictions = []

    for sample in X_test:
        best_class = None
        best_prob = -1
        for target_class, class_probability in class_probabilities.items():
            class_means = class_statistics[target_class]['means']
            class_stds = class_statistics[target_class]['stds']
            class_likelihood = np.prod(calculate_gaussian_probability(sample, class_means, class_stds))
            posterior_probability = class_probability * class_likelihood

            if posterior_probability > best_prob:
                best_prob = posterior_probability
                best_class = target_class
        predictions.append(best_class)
    return predictions

def evaluate_model(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if (TP + FP + TN + FN) == 0:
        accuracy = 0
    else:
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
class_probabilities, class_statistics = train_naive_bayes(X_train, y_train)
predictions = predict_naive_bayes(X_test, class_probabilities, class_statistics)

accuracy, precision, sensitivity, specificity, f1_score = evaluate_model(y_test, predictions)

print("Métricas del modelo:")
print(f"Precisión del modelo: {accuracy:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"Sensibilidad: {sensitivity:.2f}")
print(f"Especificidad: {specificity:.2f}")
print(f"Puntaje F1: {f1_score:.2f}")
