import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def predict_classification(X_train, y_train, x_test, k):
    distances = []
    for i in range(len(X_train)):
        distance = euclidean_distance(x_test, X_train[i])
        distances.append((distance, y_train[i]))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[1]
        class_votes[label] = class_votes.get(label, 0) + 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

def calculate_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return TP, FP, TN, FN

def evaluate_model(X_train, y_train, X_test, y_test, k):
    y_pred = []
    for i in range(len(X_test)):
        prediction = predict_classification(X_train, y_train, X_test[i], k)
        y_pred.append(prediction)
    y_pred = np.array(y_pred)
    
    TP, FP, TN, FN = calculate_confusion_matrix(y_test, y_pred)
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    sensitivity = TP / (TP + FN) if (TP + FN) != 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) != 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) != 0 else 0
    
    return accuracy, precision, sensitivity, specificity, f1_score

data = pd.read_csv("zoo.csv")

X = data.iloc[:, 1:-1].values  
y = data.iloc[:, -1].values 

np.random.seed(42)
indices = np.random.permutation(len(X))
X_train, X_test = X[indices[:-20]], X[indices[-20:]]
y_train, y_test = y[indices[:-20]], y[indices[-20:]]

k = 5
accuracy, precision, sensitivity, specificity, f1_score = evaluate_model(X_train, y_train, X_test, y_test, k)

print("Métricas del modelo:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Sensitivity: {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")
print(f"F1 Score: {f1_score:.2f}")
