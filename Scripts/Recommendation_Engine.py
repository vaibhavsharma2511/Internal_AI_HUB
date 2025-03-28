import pandas as pd
import numpy as np
import logging

# Set up logging configuration to log to both console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # To log to console
        logging.FileHandler('logs_and_matrices/recommendation_process.log', mode='w')  # To log to file and overwrite the file each time
    ]
)

logging.info("Loading the User-Menu Matrix CSV file")
# Read the final user-item matrix from CSV
df_customer_item_matrix = pd.read_csv('logs_and_matrices/df_customer_item_matrix.csv')

# Extract the list of user phone numbers
user_list = df_customer_item_matrix['customer.phoneNumber'].to_list()
logging.info(f"Loaded {len(user_list)} users from df_customer_item_matrix")

# Drop the 'customer.phoneNumber' column for the matrix calculation
df_customer_item_matrix = df_customer_item_matrix.drop(columns=['customer.phoneNumber'])

# Step 1: Calculate co-occurrence matrix
logging.info("Calculating co-occurrence matrix")
x = np.array(df_customer_item_matrix)
y = np.array(df_customer_item_matrix.T)
co_matrix = np.dot(y, x)
np.fill_diagonal(co_matrix, 0)  # Fill diagonal with 0 to avoid self-co-occurrence

# Create a DataFrame for the co-occurrence matrix
columns_list = df_customer_item_matrix.columns
df_item_cooccurrence = pd.DataFrame(co_matrix, columns=columns_list, index=columns_list)

# Save the co-occurrence matrix to CSV
df_item_cooccurrence.to_csv('logs_and_matrices/df_item_cooccurrence.csv', index=False)
logging.info("Co-occurrence matrix saved as 'logs_and_matrices/df_item_cooccurrence.csv'")

# Step 2: Define function to recommend items based on co-occurrence
def recommend_items(order, top_n):
    # logging.info(f"Recommending items for the order: {order}")
    recommended_items = {}
    
    for item in order:
        if item in df_item_cooccurrence.index:
            # Get co-occurrence values for the ordered item
            similar_items = df_item_cooccurrence.loc[item]
            # Sort by highest co-occurrence and get top recommendations
            top_recommendations = similar_items.sort_values(ascending=False).head(top_n)

            # Add recommendations to the dictionary
            for rec_item, score in top_recommendations.items():
                if rec_item not in order:  # Avoid recommending items already ordered
                    recommended_items[rec_item] = recommended_items.get(rec_item, 0) + score

    # Sort final recommendations by score
    final_recommendations = sorted(recommended_items, key=recommended_items.get, reverse=True)
    
    # logging.info(f"Top {top_n} recommended items: {final_recommendations[:top_n]}")
    return final_recommendations[:top_n]

# Step 3: Create the user-item matrix using the co-occurrence matrix
logging.info("Creating user-item recommendation matrix")
user_matrix = np.dot(x, co_matrix)
# idx = np.nonzero(x)
# user_matrix[idx] = 0  # Remove already purchased items from recommendations
df_user_recommendations = pd.DataFrame(user_matrix, columns=columns_list, index=user_list)

# Save the user-item recommendation matrix
df_user_recommendations.to_csv('logs_and_matrices/df_user_recommendations.csv', index=True)
logging.info("User-item recommendation matrix saved as 'logs_and_matrices/df_user_recommendations.csv'")

# Step 4: Define function to recommend items for a specific user
def recommend_items_for_user(phone_number, df_user_recommendations, top_n=5):
    # logging.info(f"Recommending items for user: {phone_number}")
    if phone_number not in df_user_recommendations.index:
        logging.error(f"User {phone_number} not found in dataset.")
        return []

    # Get the recommendation scores for the user
    user_recommendations = df_user_recommendations.loc[phone_number]

    # Sort items by highest recommendation score and get top-N
    top_recommendations = user_recommendations.sort_values(ascending=False).head(top_n)
    
    # logging.info(f"Top {top_n} recommendations for user {phone_number}: {top_recommendations.index.tolist()}")
    return top_recommendations.index.tolist()

# Step 5: Define function to recommend popular items
def recommend_popular_items(df_customer_item_matrix, top_n=10):
    logging.info(f"Recommending top {top_n} popular items")
    item_popularity = df_customer_item_matrix.sum().sort_values(ascending=False)
    # Get the top N popular items
    return item_popularity.head(top_n).index.tolist()

# Display popular items recommendation
logging.info("----------------POPULAR ITEMS RECOMMENDATION-----------------------------")
popular_items = recommend_popular_items(df_customer_item_matrix, top_n=10)
logging.info(f"Popular items: {popular_items}")

# Display co-occurrence based item recommendation
logging.info("----------------------------ITEM RECOMMENDATION-----------------------------")
user_order = ['Grilled Chicken Bento (feeds 1-2)']
recommendations = recommend_items(user_order, top_n=10)
logging.info(f"Recommended items for order {user_order}: {recommendations}")

# Display user-specific personalized recommendation
logging.info("----------------USER PERSONALIZED RECOMMENDATION-----------------------------")
phone_number = '00975814695fafb3a7cb32b2dd307ff6a6e690a62aaa7eee0a814b64ae056426'
recommendations = recommend_items_for_user(phone_number, df_user_recommendations, top_n=10)
logging.info(f"Recommended items for user {phone_number}: {recommendations}")