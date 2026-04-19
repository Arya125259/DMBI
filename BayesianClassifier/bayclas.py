import csv
from collections import defaultdict

# ================= LOAD DATA =================
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        return [row for row in reader]

# ================= TRAIN =================
def train(data):
    total = len(data)
    class_counts = defaultdict(int)
    feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for row in data:
        label = row[-1].strip().capitalize()
        class_counts[label] += 1

        for i in range(1, len(row)-1):
            feature = row[i].strip().capitalize()
            feature_counts[i][feature][label] += 1

    return total, class_counts, feature_counts

# ================= PREDICT =================
def predict(total, class_counts, feature_counts, input_data):

    probs = {}

    print("\n--- Class-wise Probabilities ---")

    for cls in class_counts:
        prob = class_counts[cls] / total   # Prior

        for i in range(len(input_data)):
            value = input_data[i]

            if value in feature_counts[i+1]:
                count = feature_counts[i+1][value][cls]
            else:
                count = 0

            prob *= (count + 1) / (class_counts[cls] + len(feature_counts[i+1]))

        probs[cls] = prob

        # 🔥 Rounded print
        print(f"{cls}: {round(prob, 4)}")

    return max(probs, key=probs.get), probs

# ================= MAIN =================
data = load_csv("dataset.csv")

total, class_counts, feature_counts = train(data)

print("Class Distribution:", dict(class_counts))

# ================= USER INPUT =================
print("\nEnter details:")

outlook = input("Outlook (Sunny/Overcast/Rain): ").strip().capitalize()
temp = input("Temperature (Hot/Mild/Cool): ").strip().capitalize()
humidity = input("Humidity (High/Normal): ").strip().capitalize()
wind = input("Wind (Weak/Strong): ").strip().capitalize()

input_data = [outlook, temp, humidity, wind]

# ================= PREDICT =================
result, probabilities = predict(total, class_counts, feature_counts, input_data)

# Rounded final probabilities
rounded_probs = {k: round(v, 4) for k, v in probabilities.items()}

print("\nFinal Probabilities:")
for k, v in rounded_probs.items():
    print(f"{k}: {v}")

print("\nPrediction:", result)