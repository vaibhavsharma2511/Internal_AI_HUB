from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import uvicorn
from functools import lru_cache
import time
import threading
from typing import Optional, Dict, List
import sys
from pathlib import Path

# Now we can import from recommendation_engine
from rev_recommendation_engine import (
    initialize_engine,
    recommend_items,
    recommend_items_for_user,
    recommend_popular_items,
    hybrid_recommendation_cooccurence,
    recommend_similar_items,
    user_based_recommendations,
    hybrid_recommendation,
    get_item_vector
)

# Initialize FastAPI app
app = FastAPI(
    title="NextGen Restaurant Recommendation Engine",
    description="Provides personalized menu item recommendations based on user and cart data.",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Initialize recommendation engine
RECOMMENDATION_MODE = 'embeddings'  # or 'embeddings'
engine_data = initialize_engine(cooccurrence_mode=(RECOMMENDATION_MODE == 'cooccurrence'))

# Extract data from engine
df_customer_item_matrix = engine_data['df_customer_item_matrix']
df_restaurant_menu = engine_data['df_restaurant_menu']
df_item_cooccurrence = engine_data['df_item_cooccurrence']
df_user_recommendations = engine_data['df_user_recommendations']
df_menu_vectors = engine_data.get('df_menu_vectors')
customer_item_matrix = engine_data.get('customer_item_matrix')

# Define request/response models
class RecommendationRequest(BaseModel):
    user_id: Optional[str] = None
    order: List[str] = []
    top_n: int = 10
    restaurant_id: Optional[str] = None
    recommendation_type: str = "cooccurrence"  # or "embeddings"

class RecommendationResponse(BaseModel):
    use_case: str
    recommended_items: Dict[str, Dict[str, float]]  # {restaurant_id: {item_name: score}}
    recommendation_type: str
    warning: Optional[str] = None

@lru_cache(maxsize=1)
def get_cached_popular_items(top_n: int = 5) -> Dict[str, float]:
    """Cache popular items with restaurant info"""
    items = recommend_popular_items(df_customer_item_matrix, top_n)
    return {item: float(score) for item, score in items.items()}

def clear_cache():
    """Clear cache periodically"""
    while True:
        time.sleep(3600)
        get_cached_popular_items.cache_clear()
        logger.info("Cleared recommendations cache")

threading.Thread(target=clear_cache, daemon=True).start()

def get_restaurant_id(item: str) -> Optional[str]:
    """Get restaurant ID for an item with error handling"""
    try:
        return df_restaurant_menu.loc[item, "restaurant_id"]
    except KeyError:
        logger.warning(f"Restaurant ID not found for item: {item}")
        return None

def organize_by_restaurant(item_scores: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Organize recommendations by restaurant"""
    result = {}
    for item, score in item_scores.items():
        if (restaurant_id := get_restaurant_id(item)) is not None:
            if restaurant_id not in result:
                result[restaurant_id] = {}
            result[restaurant_id][item] = score
    return result

def filter_by_restaurant(recommendations: Dict[str, Dict[str, float]], 
                        restaurant_id: str) -> Dict[str, Dict[str, float]]:
    """Filter recommendations to specific restaurant"""
    if restaurant_id not in df_restaurant_menu["restaurant_id"].values:
        raise HTTPException(status_code=404, detail="Restaurant not found")
    
    # Only return items from this restaurant
    filtered = {}
    for rest_id, items in recommendations.items():
        if rest_id == restaurant_id:
            filtered[rest_id] = items
    return filtered

def supplement_recommendations(recommendations: Dict[str, Dict[str, float]], 
                             top_n: int) -> Dict[str, Dict[str, float]]:
    """Ensure we have enough recommendations"""
    total = sum(len(items) for items in recommendations.values())
    if total >= top_n:
        return recommendations
    
    needed = top_n - total
    popular_items = get_cached_popular_items(needed)
    
    for item, score in popular_items.items():
        if (restaurant_id := get_restaurant_id(item)) is not None:
            if restaurant_id not in recommendations:
                recommendations[restaurant_id] = {}
            if item not in recommendations[restaurant_id]:
                recommendations[restaurant_id][item] = score
                needed -= 1
                if needed <= 0:
                    break
                    
    return recommendations

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(data: RecommendationRequest):
    """Main recommendation endpoint with robust restaurant filtering"""
    try:
        recommendations = {}
        use_case = ""
        warning = None
        restaurant_items = None
        
        # Validate restaurant exists and get its menu items
        if data.restaurant_id:
            if data.restaurant_id not in df_restaurant_menu["restaurant_id"].values:
                raise HTTPException(status_code=404, detail="Restaurant not found")
            
            restaurant_items = df_restaurant_menu[
                df_restaurant_menu["restaurant_id"] == data.restaurant_id
            ].index.tolist()
            
            # Only keep items that exist in the customer-item matrix
            restaurant_items = [item for item in restaurant_items 
                              if item in df_customer_item_matrix.columns]
            
            if not restaurant_items:
                return {
                    "use_case": "New User, Empty Cart",
                    "recommended_items": {},
                    "recommendation_type": data.recommendation_type,
                    "warning": f"No valid menu items found for restaurant {data.restaurant_id}"
                }

        # Determine recommendation strategy
        if not data.user_id and not data.order:
            use_case = "New User, Empty Cart"
            items_to_use = restaurant_items if restaurant_items else None
            recommendations = recommend_popular_items(
                df_customer_item_matrix[items_to_use] if items_to_use else df_customer_item_matrix,
                data.top_n
            )
            
        elif data.user_id and not data.order:
            use_case = "Existing User, Empty Cart"
            if data.user_id not in df_user_recommendations.index:
                raise HTTPException(status_code=404, detail="User not found")
            
            if data.recommendation_type == "cooccurrence":
                recommendations = recommend_items_for_user(
                    data.user_id, df_user_recommendations, data.top_n)
                if restaurant_items:
                    recommendations = {k: v for k, v in recommendations.items() 
                                     if k in restaurant_items}
            else:
                recommendations = user_based_recommendations(
                    data.user_id, customer_item_matrix, data.top_n)
                
        elif not data.user_id and data.order:
            use_case = "New User, Added Item"
            if data.recommendation_type == "cooccurrence":
                recommendations = recommend_items(
                    data.order, df_item_cooccurrence, data.top_n)
                if restaurant_items:
                    recommendations = {k: v for k, v in recommendations.items() 
                                     if k in restaurant_items}
            else:
                recommendations = recommend_similar_items(
                    data.order, df_menu_vectors, data.top_n)
                
        elif data.user_id and data.order:
            use_case = "Existing User, Added Item"
            if data.recommendation_type == "cooccurrence":
                recommendations = hybrid_recommendation_cooccurence(
                    data.user_id, data.order, 
                    df_user_recommendations, df_item_cooccurrence, data.top_n)
                if restaurant_items:
                    recommendations = {k: v for k, v in recommendations.items() 
                                     if k in restaurant_items}
            else:
                recommendations = hybrid_recommendation(
                    data.user_id, data.order, 
                    customer_item_matrix, df_menu_vectors, data.top_n)
        
        # Organize by restaurant and validate items exist
        organized_recs = {}
        missing_items = []
        for item, score in recommendations.items():
            if (restaurant_id := get_restaurant_id(item)) is not None:
                if restaurant_id not in organized_recs:
                    organized_recs[restaurant_id] = {}
                print (organized_recs)
                organized_recs[restaurant_id][item] = float(score)
            else:
                missing_items.append(item)
        
        # Apply strict restaurant filter if specified
        if data.restaurant_id:
            organized_recs = {data.restaurant_id: organized_recs.get(data.restaurant_id, {})}
            
            # Supplement with more items if needed
            if len(organized_recs[data.restaurant_id]) < data.top_n and restaurant_items:
                needed = data.top_n - len(organized_recs[data.restaurant_id])
                additional_items = recommend_popular_items(
                    df_customer_item_matrix[restaurant_items],
                    needed
                )
                for item, score in additional_items.items():
                    if item not in organized_recs[data.restaurant_id]:
                        organized_recs[data.restaurant_id][item] = float(score)
        
        # Generate warning if items were missing
        if missing_items:
            warning = f"{len(missing_items)} recommended items not found in menu data"
        
        return {
            "use_case": use_case,
            "recommended_items": organized_recs,
            "recommendation_type": data.recommendation_type,
            "warning": warning
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/reload-data")
async def reload_data():
    """Endpoint to reload all data"""
    try:
        global engine_data, df_customer_item_matrix, df_restaurant_menu
        global df_item_cooccurrence, df_user_recommendations, df_menu_vectors, customer_item_matrix
        
        engine_data = initialize_engine(cooccurrence_mode=(RECOMMENDATION_MODE == 'cooccurrence'))
        df_customer_item_matrix = engine_data['df_customer_item_matrix']
        df_restaurant_menu = engine_data['df_restaurant_menu']
        df_item_cooccurrence = engine_data['df_item_cooccurrence']
        df_user_recommendations = engine_data['df_user_recommendations']
        df_menu_vectors = engine_data.get('df_menu_vectors')
        customer_item_matrix = engine_data.get('customer_item_matrix')
        
        get_cached_popular_items.cache_clear()
        return {"status": "success", "message": "Data reloaded"}
    except Exception as e:
        logger.error(f"Data reload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Data reload failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)