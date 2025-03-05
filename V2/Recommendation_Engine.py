import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load user-item matrix
df_final = pd.read_csv("V2/df_final.csv")

# Calculate co-occurrence matrix
def compute_co_occurrence_matrix(df):
    logging.info("Computing co-occurrence matrix...")
    x = df.to_numpy()
    y = x.T
    co_matrix = np.dot(y, x)
    np.fill_diagonal(co_matrix, 0)  # Remove self-co-occurrence
    
    return pd.DataFrame(co_matrix, columns=df.columns, index=df.columns)

df_co = compute_co_occurrence_matrix(df_final)
df_co.to_csv("V2/df_co.csv", index=False)
logging.info("Co-occurrence matrix saved successfully.")

# Item recommendation function
def recommend_items(order, top_n=10):
    recommended_items = {}

    for item in order:
        if item in df_co.index:
            similar_items = df_co.loc[item]
            top_recommendations = similar_items.sort_values(ascending=False).head(top_n)

            for rec_item, score in top_recommendations.items():
                if rec_item not in order:  # Avoid recommending already ordered items
                    recommended_items[rec_item] = recommended_items.get(rec_item, 0) + score

    if not recommended_items:
        logging.warning("No recommendations found. Check if input items exist in the dataset.")

    # Sort and return top recommendations
    return sorted(recommended_items, key=recommended_items.get, reverse=True)[:top_n]

# Example user order
user_order = ["13. Thai Noodle Soup", "Mango Pudding with Small Chewy Ball"]
recommendations = recommend_items(user_order, top_n=10)

logging.info(f"Recommended items: {recommendations}")