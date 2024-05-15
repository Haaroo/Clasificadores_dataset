import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  
    X_normalized = (X - mean) / std
    return X_normalized


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5 
    cost = (1 / m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
    return cost


def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    costs = []
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (1 / m) * X.T.dot(h - y)
        theta -= alpha * gradient
        cost = compute_cost(X, y, theta)
        costs.append(cost)
    return theta, costs


def predict(X, theta):
    h = sigmoid(X.dot(theta))
    return (h >= 0.5).astype(int)


def calculate_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN


def evaluate_model(y_true, y_pred):
    TP, FP, TN, FN = calculate_confusion_matrix(y_true, y_pred)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    return accuracy, precision, sensitivity, specificity, f1_score


data = pd.read_csv("zoo.csv")


X = data.iloc[:, 1:-1].values 
y = data.iloc[:, -1].values 


X = np.hstack((np.ones((X.shape[0], 1)), X))
X = normalize(X)
theta = np.zeros(X.shape[1])

alpha = 0.01
iterations = 1000

theta_final, costs = gradient_descent(X, y, theta, alpha, iterations)

print("Parámetros finales:")
print(theta_final)

plt.plot(range(iterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost vs. Iterations')
plt.show()

predictions = predict(X, theta_final)
accuracy, precision, sensitivity, specificity, f1_score = evaluate_model(y, predictions)
print(f"Precisión del modelo: {accuracy:.2f}")
print(f"Precisión: {precision:.2f}")
print(f"Sensibilidad: {sensitivity:.2f}")
print(f"Especificidad: {specificity:.2f}")
print(f"Puntaje F1: {f1_score:.2f}")
