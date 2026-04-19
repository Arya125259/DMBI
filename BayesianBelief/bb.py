import csv
from collections import defaultdict

# ================= LOAD DATA =================
def load_csv(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        return [row for row in reader]

# ================= TRAIN (BUILD CPT) =================
def train(data):
    total = len(data)

    rain_count = defaultdict(int)
    traffic_given_rain = defaultdict(lambda: defaultdict(int))
    late_given_traffic = defaultdict(lambda: defaultdict(int))

    for rain, traffic, late in data:
        rain = rain.strip().capitalize()
        traffic = traffic.strip().capitalize()
        late = late.strip().capitalize()

        rain_count[rain] += 1
        traffic_given_rain[rain][traffic] += 1
        late_given_traffic[traffic][late] += 1

    return total, rain_count, traffic_given_rain, late_given_traffic

# ================= JOINT PROBABILITY =================
def joint_prob(rain, traffic, late, total, rain_count, traffic_given_rain, late_given_traffic):

    # P(Rain)
    p_r = rain_count[rain] / total

    # P(Traffic | Rain)
    p_t = traffic_given_rain[rain][traffic] / rain_count[rain]

    # P(Late | Traffic)
    total_t = sum(late_given_traffic[traffic].values())
    p_l = late_given_traffic[traffic][late] / total_t

    return p_r * p_t * p_l

# ================= MAIN =================
data = load_csv("dataset.csv")

total, rain_count, traffic_given_rain, late_given_traffic = train(data)

print("Rain Counts:", dict(rain_count))

# ================= USER INPUT =================
print("\nEnter values:")

rain = input("Rain (Yes/No): ").strip().capitalize()
traffic = input("Traffic (Heavy/Light): ").strip().capitalize()
late = input("Late (Yes/No): ").strip().capitalize()

prob = joint_prob(rain, traffic, late, total,
                  rain_count, traffic_given_rain, late_given_traffic)

print("\nJoint Probability =", round(prob, 4))