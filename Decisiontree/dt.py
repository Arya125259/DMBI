import math
import csv

# Load dataset (convert to lowercase)
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [[value.strip().lower() for value in row] for row in reader]
    return headers, data


# Entropy
def entropy(rows):
    total = len(rows)
    counts = {}

    for row in rows:
        label = row[-1]
        counts[label] = counts.get(label, 0) + 1

    ent = 0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)

    return ent


# Split dataset
def split_data(data, index, value):
    return [row for row in data if row[index] == value]


# Unique values
def unique_values(data, index):
    return list(set(row[index] for row in data))


# Best feature (Information Gain)
def best_feature(data):
    base_entropy = entropy(data)
    best_gain = -1
    best_index = -1

    for i in range(len(data[0]) - 1):
        values = unique_values(data, i)
        new_entropy = 0

        for val in values:
            subset = split_data(data, i, val)
            new_entropy += (len(subset)/len(data)) * entropy(subset)

        gain = base_entropy - new_entropy

        if gain > best_gain:
            best_gain = gain
            best_index = i

    return best_index


# Build tree
def build_tree(data, headers):

    labels = [row[-1] for row in data]

    if labels.count(labels[0]) == len(labels):
        return labels[0]

    if len(headers) == 1:
        return max(set(labels), key=labels.count)

    best_idx = best_feature(data)
    best_attr = headers[best_idx]

    tree = {best_attr: {}}

    for val in unique_values(data, best_idx):
        subset = split_data(data, best_idx, val)

        new_headers = headers[:best_idx] + headers[best_idx+1:]
        new_subset = [
            row[:best_idx] + row[best_idx+1:]
            for row in subset
        ]

        tree[best_attr][val] = build_tree(new_subset, new_headers)

    return tree


# Classify
def classify(tree, headers, instance):

    if not isinstance(tree, dict):
        return tree

    attr = next(iter(tree))
    index = headers.index(attr)

    value = instance[index]

    if value in tree[attr]:
        subtree = tree[attr][value]

        new_headers = headers[:index] + headers[index+1:]
        new_instance = instance[:index] + instance[index+1:]

        return classify(subtree, new_headers, new_instance)

    return "Unknown"


# -------- MAIN --------
headers, data = load_csv("dataset.csv")

tree = build_tree(data, headers)

print("\nDecision Tree:")
print(tree)

# Prepare valid values for each feature
valid_values = [set(row[i] for row in data) for i in range(len(headers)-1)]

# Take user input with validation
print("\nEnter values:")

user_input = []
for i in range(len(headers) - 1):
    while True:
        val = input(f"{headers[i]} {valid_values[i]}: ").strip().lower()

        if val == "":
            print("⚠️ Value cannot be empty.")
        elif val not in valid_values[i]:
            print("⚠️ Invalid value. Choose from given options.")
        else:
            user_input.append(val)
            break

# Prediction
result = classify(tree, headers[:-1], user_input)

print("\nPrediction:", result)