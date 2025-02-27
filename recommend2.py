import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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

# Load the product data
product_df = pd.read_csv('generatedFiles/products-coho.csv')

# Define the features to be used (product category, price, etc.)
features = ['category', 'price']

# Create a preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['price']),
        ('cat', OneHotEncoder(), ['category'])
    ]
)

# Apply the preprocessing pipeline to the product data
feature_matrix = preprocessor.fit_transform(product_df[features])

# print(feature_matrix)

from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, hstack

# Normalize the co-occurrence matrix
cooccurrence_matrix_normalized = cooccurrence_matrix.div(cooccurrence_matrix.sum(axis=1), axis=0).fillna(0)

# Combine the co-occurrence matrix with the feature matrix
combined_matrix = hstack([csr_matrix(cooccurrence_matrix_normalized.values), csr_matrix(feature_matrix)])

# Calculate the cosine similarity
cosine_sim_matrix = cosine_similarity(combined_matrix)

# Convert to DataFrame for easier handling
cosine_sim_df = pd.DataFrame(
    cosine_sim_matrix,
    index=product_df['productName'],
    columns=product_df['productName']
)


# print(cosine_sim_df)

def recommend(product, cosine_sim, top_n=5):
    # Get the sorted list of recommended items
    recommendations = cosine_sim[product].sort_values(ascending=False).head(top_n + 1).index.tolist()

    # Remove the product itself from the recommendations
    if product in recommendations:
        recommendations.remove(product)

    # Return the top N recommendations excluding the product itself
    return recommendations[:top_n]


# Example recommendation
product='Pepperoni Pizza'
recommendations = recommend(product, cosine_sim_df)
print("Recommendations for "+product+":", recommendations)
