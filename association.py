#Association Rule

import pandas as pd

# Load the dataset (adjust file path as needed)
df = pd.read_csv('generatedFiles/matrix-coho.csv')

# Select columns from index 14 onwards (items/products)
items_df = df.iloc[:, 14:]

# Create a binary (one-hot encoded) matrix indicating the presence of each item in an order
one_hot_encoded_df = items_df.apply(lambda x: x == True).astype(bool)
from mlxtend.frequent_patterns import apriori, association_rules

# Apply apriori algorithm to find frequent itemsets with a minimum support threshold
frequent_itemsets = apriori(one_hot_encoded_df, min_support=0.001, use_colnames=True)

# Generate association rules with a minimum lift threshold
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Display the top association rules based on lift
rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
#print(rules.sort_values(by='lift', ascending=False).to_string())
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph of associations
'''G = nx.DiGraph()

# Add edges based on association rules
for _, row in rules.iterrows():
    G.add_edge(str(row['antecedents']), str(row['consequents']), weight=row['lift'])

# Plot the graph
plt.figure(figsize=(100, 100))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=300, node_color="lightblue", font_size=8)
edge_labels = {(u, v): f'{d["weight"]:.2f}' for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Association Rule Network")
plt.savefig('association.png')
'''

def recommend_items(user_purchases, rules, top_n=5):

    recommended_items = set()

    # Go through the user's past purchases
    for item in user_purchases:
        # Find the rules where the item appears in the antecedents
        relevant_rules = rules[rules['antecedents'].apply(lambda x: item in x)]

        # For each of these rules, add the consequents (items bought together) to the recommendations
        for _, rule in relevant_rules.iterrows():
            consequents = rule['consequents']
            recommended_items.update(consequents)

    # Convert to list and sort by confidence or lift, assuming we want the top_n recommendations
    recommended_items = list(recommended_items)

    # Assuming you'd like to sort by 'lift', you can filter and sort by that:
    sorted_recommendations = []
    for recommended_item in recommended_items:
        sorted_recommendations.append(
            (recommended_item, rules[rules['consequents'].apply(lambda x: recommended_item in x)]['lift'].max()))

    # Sort based on lift, descending
    sorted_recommendations = sorted(sorted_recommendations, key=lambda x: x[1], reverse=True)

    # Return top N recommendations (filtering out already purchased items)
    top_recommendations = [item for item, _ in sorted_recommendations if item not in user_purchases][:top_n]

    return top_recommendations


# Example usage
user_purchases = {'Pepperoni Pizza'}  # The user has bought Burger and Fries
recommended_items = recommend_items(user_purchases, rules, top_n=5)

print("Recommended items for the user:", recommended_items)