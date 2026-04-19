import csv
import math

# -----------------------
# Load dataset
# -----------------------
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        return [row for row in reader]

# -----------------------
# ONE-HOT ENCODING (IMPORTANT UPGRADE)
# -----------------------
def encode(row):
    age = int(row[0])

    # One-hot encoding for Income
    income_low = 1 if row[1].strip().title() == 'Low' else 0
    income_med = 1 if row[1].strip().title() == 'Medium' else 0
    income_high = 1 if row[1].strip().title() == 'High' else 0

    # Binary encoding
    student = 1 if row[2].strip().title() == 'Yes' else 0
    credit = 1 if row[3].strip().title() == 'Excellent' else 0

    # Label
    label = 1 if row[4].strip().title() == 'Yes' else -1

    features = [
        age,
        income_low,
        income_med,
        income_high,
        student,
        credit
    ]

    return features, label

# -----------------------
# Decision stump
# -----------------------
def train_stump(X, y, weights):
    n_features = len(X[0])
    best_error = float('inf')
    best = None

    for feature in range(n_features):
        thresholds = set([x[feature] for x in X])

        for thresh in thresholds:
            for polarity in [1, -1]:
                error = 0

                for i in range(len(X)):
                    pred = polarity * (1 if X[i][feature] <= thresh else -1)
                    if pred != y[i]:
                        error += weights[i]

                if error < best_error:
                    best_error = error
                    best = (feature, thresh, polarity)

    return best[0], best[1], best[2], best_error

# -----------------------
# AdaBoost Training
# -----------------------
def adaboost(X, y, T=10):  # increased iterations = better accuracy
    n = len(X)
    weights = [1/n] * n
    models = []

    for _ in range(T):
        feat, thresh, polarity, error = train_stump(X, y, weights)

        eps = 1e-10
        alpha = 0.5 * math.log((1 - error + eps) / (error + eps))

        models.append((feat, thresh, polarity, alpha))

        for i in range(n):
            pred = polarity * (1 if X[i][feat] <= thresh else -1)
            weights[i] *= math.exp(-alpha * y[i] * pred)

        total = sum(weights)
        weights = [w / total for w in weights]

    return models

# -----------------------
# Prediction
# -----------------------
def predict(x, models):
    result = 0

    for feat, thresh, polarity, alpha in models:
        pred = polarity * (1 if x[feat] <= thresh else -1)
        result += alpha * pred

    return 1 if result > 0 else -1

# -----------------------
# MAIN
# -----------------------
data = load_csv("dataset.csv")

X, y = [], []
for row in data:
    features, label = encode(row)
    X.append(features)
    y.append(label)

models = adaboost(X, y, T=10)

# -----------------------
# USER INPUT
# -----------------------
print("\nEnter input values:")

age = int(input("Age: "))
income = input("Income (Low/Medium/High): ").strip().title()
student = input("Student (Yes/No): ").strip().title()
credit = input("Credit (Excellent/Fair): ").strip().title()

# manual encoding for input (same one-hot logic)
income_low = 1 if income == 'Low' else 0
income_med = 1 if income == 'Medium' else 0
income_high = 1 if income == 'High' else 0

student = 1 if student == 'Yes' else 0
credit = 1 if credit == 'Excellent' else 0

x_test = [
    age,
    income_low,
    income_med,
    income_high,
    student,
    credit
]

pred = predict(x_test, models)

print("\nPrediction:", "Yes" if pred == 1 else "No")