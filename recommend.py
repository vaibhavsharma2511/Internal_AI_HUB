# Item-Based Collaborative Filtering (Using Cosine Similarity)

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
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
cooccurrence_matrix.to_csv('generatedFiles/cooccurance.csv', encoding='utf-8')
# Calculate the cosine similarity matrix
cosine_sim_matrix = pd.DataFrame(
    cosine_similarity(cooccurrence_matrix),
    index=cooccurrence_matrix.index,
    columns=cooccurrence_matrix.columns
)

# print(cosine_sim_matrix)
import seaborn as sns
# Convert to a sparse matrix
sparse_matrix = csr_matrix(cooccurrence_matrix)

# Compute cosine similarity efficiently
item_similarity = cosine_similarity(sparse_matrix.T, dense_output=False)

# Convert to DataFrame
item_sim_df = pd.DataFrame(item_similarity.toarray(), index=cooccurrence_matrix.columns, columns=cooccurrence_matrix.columns)

# Plot heatmap of item similarities
plt.figure(figsize=(100,100))
sns.heatmap(item_sim_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Item Similarity Heatmap")
plt.savefig('heatmap.png')

cosine_sim_matrix.to_csv('generatedFiles/similarity.csv', encoding='utf-8')


def recommend(item, cosine_sim_matrix, top_n=10):
    return cosine_sim_matrix[item].sort_values(ascending=False).head(top_n).index.tolist()


# Example recommendation
product = 'Pepperoni Pizza'
recommendations = recommend(product, cosine_sim_matrix)
print("Recommendations for " + product + ":", recommendations)
