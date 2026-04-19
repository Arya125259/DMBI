import csv
import random
import math

# ================= LOAD DATA =================
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        return [row for row in reader]

# ================= ENCODE =================
def encode_input(age, income, student, credit):
    income_map = {'low': 0, 'medium': 1, 'high': 2}
    age_map = {'youth': 0, 'middle': 1, 'senior': 2}

    # 🔥 Normalize inputs
    age = age.strip().lower()
    income = income.strip().lower()
    student = student.strip().lower()
    credit = credit.strip().lower()

    # Handle numeric OR categorical age
    try:
        age_num = int(age)
        if age_num <= 30:
            age_val = 0
        elif age_num <= 50:
            age_val = 1
        else:
            age_val = 2
    except:
        if age in age_map:
            age_val = age_map[age]
        else:
            raise ValueError("Invalid Age! Use number or Youth/Middle/Senior")

    if income not in income_map:
        raise ValueError("Invalid Income! Use Low/Medium/High")

    income_val = income_map[income]
    student_val = 1 if student == 'yes' else 0
    credit_val = 1 if credit == 'excellent' else 0

    return [age_val, income_val, student_val, credit_val]

def encode_row(row):
    x = encode_input(row[1], row[2], row[3], row[4])
    y = 1 if row[5].strip().lower() == 'yes' else -1
    return x, y

# ================= GINI =================
def gini(y):
    if len(y) == 0:
        return 0
    p = sum(1 for i in y if i == 1) / len(y)
    return 1 - p*p - (1-p)*(1-p)

# ================= SPLIT =================
def split(X, y, feature, threshold):
    left_X, left_y, right_X, right_y = [], [], [], []
    for i in range(len(X)):
        if X[i][feature] <= threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else:
            right_X.append(X[i])
            right_y.append(y[i])
    return left_X, left_y, right_X, right_y

# ================= BUILD TREE =================
def build_tree(X, y, depth=0, max_depth=5, min_size=2):

    if len(X) == 0:
        return 0

    if len(set(y)) == 1:
        return y[0]

    if depth >= max_depth or len(X) <= min_size:
        return max(set(y), key=y.count)

    n_features = len(X[0])
    features = random.sample(range(n_features), max(1, int(math.sqrt(n_features))))

    best_gini = float('inf')
    best_split = None

    for f in features:
        values = set([x[f] for x in X])
        for t in values:
            left_X, left_y, right_X, right_y = split(X, y, f, t)

            g = (len(left_y)/len(y))*gini(left_y) + (len(right_y)/len(y))*gini(right_y)

            if g < best_gini:
                best_gini = g
                best_split = (f, t, left_X, left_y, right_X, right_y)

    if best_split is None:
        return max(set(y), key=y.count)

    f, t, left_X, left_y, right_X, right_y = best_split

    if len(left_X) == 0 or len(right_X) == 0:
        return max(set(y), key=y.count)

    left_tree = build_tree(left_X, left_y, depth+1, max_depth, min_size)
    right_tree = build_tree(right_X, right_y, depth+1, max_depth, min_size)

    return ((f, t), left_tree, right_tree)

# ================= PREDICT =================
def predict_tree(tree, x):
    if isinstance(tree, int):
        return tree
    f, t = tree[0]
    if x[f] <= t:
        return predict_tree(tree[1], x)
    else:
        return predict_tree(tree[2], x)

def predict_forest(forest, x):
    preds = [predict_tree(tree, x) for tree in forest]
    return max(set(preds), key=preds.count)

# ================= RANDOM FOREST =================
def random_forest(X, y, n_trees=5):
    forest = []
    for _ in range(n_trees):
        sample_X, sample_y = [], []
        for _ in range(len(X)):
            i = random.randint(0, len(X)-1)
            sample_X.append(X[i])
            sample_y.append(y[i])
        forest.append(build_tree(sample_X, sample_y))
    return forest

# ================= MAIN =================
data = load_csv("dataset.csv")

X, y = [], []
for row in data:
    fx, label = encode_row(row)
    X.append(fx)
    y.append(label)

# Train model
forest = random_forest(X, y, n_trees=5)

# ================= ACCURACY =================
correct = 0
for i in range(len(X)):
    pred = predict_forest(forest, X[i])
    if pred == y[i]:
        correct += 1

print("Accuracy:", round(correct / len(X), 2))

# ================= USER INPUT =================
print("\nEnter details to predict:")

try:
    age = input("Age (number or Youth/Middle/Senior): ")
    income = input("Income (Low/Medium/High): ")
    student = input("Student (Yes/No): ")
    credit = input("Credit (Fair/Excellent): ")

    x_test = encode_input(age, income, student, credit)

    result = predict_forest(forest, x_test)

    print("\nEncoded Input:", x_test)
    print("Prediction:", "Yes" if result == 1 else "No")

except ValueError as e:
    print("Error:", e)