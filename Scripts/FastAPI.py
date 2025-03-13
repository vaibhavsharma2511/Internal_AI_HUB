from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import uvicorn
from functools import lru_cache
import time
import threading

# Initialize FastAPI app
app = FastAPI(
    title="NextGen Restaurant Recommendation Engine",
    description="Provides personalized menu item recommendations based on user and cart data.",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()],
)

# Load precomputed matrices
try:
    logging.info("Loading user-item matrix...")
    df_customer_item_matrix = pd.read_csv("logs_and_matrices/df_customer_item_matrix.csv", index_col="customer.phoneNumber")

    logging.info("Loading item co-occurrence matrix...")
    df_item_cooccurrence = pd.read_csv("logs_and_matrices/df_item_cooccurrence.csv", index_col=0)

    logging.info("All data loaded successfully!")
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise RuntimeError("Failed to load recommendation system data")

# Define request model
class RecommendationRequest(BaseModel):
    user_id: str = None  # Optional for new users
    order: list = []  # Optional for empty cart users
    top_n: int = 5


@lru_cache(maxsize=1)
def get_popular_items_cached(top_n: int = 5):
    """Cache the result of recommend_popular_items for 1 hour."""
    return recommend_popular_items(top_n)

# Clear cache every hour
def clear_cache():
    """Clear the cache every hour."""
    while True:
        time.sleep(3600)  # 1 hour
        get_popular_items_cached.cache_clear()
        logging.info("Popular items cache cleared.")

# Start cache-clearing thread
threading.Thread(target=clear_cache, daemon=True).start()

# Recommendation endpoint
@app.post("/recommend", summary="Get recommendations", description="Provide recommendations based on user and cart data.")
async def recommend(data: RecommendationRequest):
    """Provide recommendations based on different use cases."""
    try:
        if not data.user_id and not data.order:
            # Use Case 1: New user with an empty cart (Recommend popular items)
            recommendations = recommend_popular_items(data.top_n)
            return {"use_case": "New User, Empty Cart", "recommended_items": recommendations}
        
        elif data.user_id and not data.order:
            # Use Case 2: Existing user with an empty cart (Recommend based on past orders)
            if data.user_id in df_customer_item_matrix.index:
                recommendations = recommend_items_for_user(data.user_id, data.top_n)
                return {"use_case": "Existing User, Empty Cart", "recommended_items": recommendations}
            else:
                raise HTTPException(status_code=404, detail="User not found")
        
        elif not data.user_id and data.order:
            # Use Case 3: New user adds an item (Recommend frequently bought together items)
            recommendations = recommend_items(data.order, data.top_n)
            return {"use_case": "New User, Added Item", "recommended_items": recommendations}
        
        elif data.user_id and data.order:
            # Use Case 4: Existing user adds an item (Hybrid approach)
            recommendations_user = recommend_items_for_user(data.user_id, data.top_n // 2)
            recommendations_order = recommend_items(data.order, data.top_n // 2)
            recommendations = list(set(recommendations_user + recommendations_order))[:data.top_n]
            return {"use_case": "Existing User, Added Item", "recommended_items": recommendations}
        
    except Exception as e:
        logging.error(f"Error in recommendation: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Recommendation functions
def recommend_items(order, top_n=5):
    recommended_items = {}
    for item in order:
        if item in df_item_cooccurrence.index:
            similar_items = df_item_cooccurrence.loc[item].sort_values(ascending=False).head(top_n)
            for rec_item, score in similar_items.items():
                recommended_items[rec_item] = recommended_items.get(rec_item, 0) + score
    return sorted(recommended_items, key=recommended_items.get, reverse=True)[:top_n]

def recommend_items_for_user(user_id, top_n=5):
    if user_id not in df_customer_item_matrix.index:
        logging.error(f"User {user_id} not found in dataset.")
        return []
    user_recommendations = df_customer_item_matrix.loc[user_id].sort_values(ascending=False).head(top_n)
    return user_recommendations.index.tolist()

def recommend_popular_items(top_n=10):
    item_popularity = df_customer_item_matrix.drop(columns=["customer.phoneNumber"], errors='ignore').sum().sort_values(ascending=False)
    return item_popularity.head(top_n).index.tolist()

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

