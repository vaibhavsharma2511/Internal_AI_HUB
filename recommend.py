import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from a CSV file
df = pd.read_csv('generatedFiles/matrix-coho.csv')

# Create the co-occurrence matrix
item_columns = df.columns[14:]  # Adjust the index based on your data
cooccurrence_matrix = pd.DataFrame(index=item_columns, columns=item_columns, data=0)

for index, row in df.iterrows():
    items = row[item_columns][row[item_columns] == True].index
    for item1 in items:
        for item2 in items:
            if item1 != item2:
                cooccurrence_matrix.at[item1, item2] += 1

# Calculate the cosine similarity matrix
cosine_sim_matrix = pd.DataFrame(
    cosine_similarity(cooccurrence_matrix),
    index=cooccurrence_matrix.index,
    columns=cooccurrence_matrix.columns
)

print(cosine_sim_matrix)
cosine_sim_matrix.to_csv('generatedFiles/similarity.csv', encoding='utf-8')
def recommend(item, cosine_sim_matrix, top_n=5):
    return cosine_sim_matrix[item].sort_values(ascending=False).head(top_n).index.tolist()

# Example recommendation
product='EQLOK2HOBVFBH3HD6IRQTJ3Q-Pepperoni Pizza'
recommendations = recommend(product, cosine_sim_matrix)
print("Recommendations for "+product+":", recommendations)