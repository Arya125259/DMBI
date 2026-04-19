import csv
from itertools import combinations

# Load dataset
def load_csv(filename):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = [row for row in reader]
    return header, data

# Encode transactions
def encode(data):
    transactions = []
    for row in data:
        transaction = set()
        for i, val in enumerate(row[1:]):  # skip ID
            if int(val) == 1:
                transaction.add(i)
        transactions.append(transaction)
    return transactions

# Generate candidates
def generate_candidates(prev_frequent, k):
    candidates = set()
    prev_list = list(prev_frequent)

    for i in range(len(prev_list)):
        for j in range(i + 1, len(prev_list)):
            union = prev_list[i] | prev_list[j]
            if len(union) == k:
                candidates.add(union)

    return candidates

# Calculate support
def calculate_support(candidates, transactions):
    support_count = {}
    for candidate in candidates:
        count = sum(1 for t in transactions if candidate.issubset(t))
        support_count[candidate] = count
    return support_count

# Prune
def prune(support_count, min_support, total_transactions):
    return {
        itemset: count
        for itemset, count in support_count.items()
        if count / total_transactions >= min_support
    }

# Generate rules
def generate_rules(frequent_itemsets, min_confidence):
    rules = set()

    for itemset in frequent_itemsets:
        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = frozenset(antecedent)
                consequent = itemset - antecedent

                confidence = (
                    frequent_itemsets[itemset] /
                    frequent_itemsets[antecedent]
                )

                if confidence >= min_confidence:
                    rules.add((antecedent, consequent, round(confidence, 2)))

    return rules

# Apriori main
def apriori(data, min_support, min_confidence):
    transactions = encode(data)
    total_transactions = len(transactions)

    # L1
    items = set()
    for t in transactions:
        for item in t:
            items.add(frozenset([item]))

    support_1 = calculate_support(items, transactions)
    frequent = prune(support_1, min_support, total_transactions)

    all_frequent = dict(frequent)
    k = 2

    while True:
        candidates = generate_candidates(set(frequent.keys()), k)
        support_k = calculate_support(candidates, transactions)
        frequent_k = prune(support_k, min_support, total_transactions)

        if not frequent_k:
            break

        all_frequent.update(frequent_k)
        frequent = frequent_k
        k += 1

    rules = generate_rules(all_frequent, min_confidence)

    return all_frequent, rules, total_transactions

# ✅ Improved Display Function
def display(frequent_itemsets, rules, header, total_transactions):
    print("\n========== FREQUENT ITEMSETS ==========\n")

    # Group itemsets by size
    grouped = {}
    for itemset, count in frequent_itemsets.items():
        k = len(itemset)
        grouped.setdefault(k, []).append((itemset, count))

    # Print grouped itemsets
    for k in sorted(grouped.keys()):
        print(f"L{k} (Size {k}):")
        print("-" * 40)
        for itemset, count in grouped[k]:
            items = [header[i + 1] for i in itemset]
            support = count / total_transactions
            print(f"{', '.join(items)}  |  Support = {support:.2f}")
        print()

    print("\n========== ASSOCIATION RULES ==========\n")

    # Sort rules by confidence
    sorted_rules = sorted(rules, key=lambda x: x[2], reverse=True)

    # Show top 20 rules (to avoid clutter)
    for antecedent, consequent, confidence in sorted_rules[:20]:
        ant = [header[i + 1] for i in antecedent]
        con = [header[i + 1] for i in consequent]

        print(f"{', '.join(ant)}  →  {', '.join(con)}  |  Confidence = {confidence:.2f}")

# ================= RUN =================

filename = "new_dataset.csv"
min_support = 0.2
min_confidence = 0.5

header, data = load_csv(filename)

frequent_itemsets, rules, total = apriori(data, min_support, min_confidence)

display(frequent_itemsets, rules, header, total)