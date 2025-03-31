# recommendation_engine.py
import pandas as pd
import numpy as np
import logging
from gensim.utils import simple_preprocess
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os

def initialize_engine(cooccurrence_mode=True):
    """Initialize and load all required data"""
    logging.info("Initializing recommendation engine...")
    
    # Load common data
    df_customer_item_matrix = pd.read_csv('..//logs_and_matrices//df_customer_item_matrix.csv')
    user_list = df_customer_item_matrix['customer.phoneNumber'].to_list()
    df_customer_item_matrix = df_customer_item_matrix.drop(columns=['customer.phoneNumber'])
    
    df_restaurant_menu = pd.read_csv('..//logs_and_matrices//df_restaurant_menu_with_restaurent_id.csv')
    df_restaurant_menu['index_name'] = df_restaurant_menu['name']
    df_restaurant_menu.set_index('index_name', inplace=True)
    
    # Initialize variables for both approaches
    df_item_cooccurrence = None
    df_user_recommendations = None
    df_menu_vectors = None
    customer_item_matrix = None
    word2vec_model = None
    
    if cooccurrence_mode:
        # Co-occurrence approach data
        x = np.array(df_customer_item_matrix)
        y = np.array(df_customer_item_matrix.T)
        co_matrix = np.dot(y, x)
        np.fill_diagonal(co_matrix, 0)
        
        columns_list = df_customer_item_matrix.columns
        df_item_cooccurrence = pd.DataFrame(co_matrix, columns=columns_list, index=columns_list)
        
        user_matrix = np.dot(x, co_matrix)
        df_user_recommendations = pd.DataFrame(user_matrix, columns=columns_list, index=user_list)
    else:
        # Embeddings approach data
        df = pd.read_csv('..//logs_and_matrices//df_item_cooccurrence.csv')
        menu_items = list(df_restaurant_menu.index.values)
        df_restaurant_menu_temp = pd.DataFrame(menu_items)
        df_restaurant_menu_temp['tokenized_name'] = df_restaurant_menu_temp[0].apply(lambda x: simple_preprocess(str(x)))
        sentences = df_restaurant_menu_temp['tokenized_name'].tolist()
        
        if os.path.exists("..//logs_and_matrices//word2vec_menu.model"):
            word2vec_model = Word2Vec.load("..//logs_and_matrices//word2vec_menu.model")
        else:
            word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
            word2vec_model.save("..//logs_and_matrices//word2vec_menu.model")
        
        if os.path.exists('..//logs_and_matrices//menu_item_vectors.csv'):
            df_menu_vectors = pd.read_csv('..//logs_and_matrices//menu_item_vectors.csv', index_col=0)
        else:
            menu_item_vectors = {item: get_item_vector(item, word2vec_model) for item in df_restaurant_menu.index}
            df_menu_vectors = pd.DataFrame.from_dict(menu_item_vectors, orient='index')
            df_menu_vectors.to_csv('..//logs_and_matrices//menu_item_vectors.csv')
        
        customer_item_matrix = pd.read_csv('..//logs_and_matrices//df_customer_item_matrix.csv')
        customer_item_matrix = customer_item_matrix.set_index('customer.phoneNumber')
    
    return {
        'df_customer_item_matrix': df_customer_item_matrix,
        'df_restaurant_menu': df_restaurant_menu,
        'df_item_cooccurrence': df_item_cooccurrence,
        'df_user_recommendations': df_user_recommendations,
        'df_menu_vectors': df_menu_vectors,
        'customer_item_matrix': customer_item_matrix,
        'word2vec_model': word2vec_model
    }

def get_item_vector(item_name, model):
    tokens = simple_preprocess(item_name)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Co-occurrence based functions
def recommend_items(order, df_item_cooccurrence, top_n=5):
    recommended_items = {}
    
    for item in order:
        if item in df_item_cooccurrence.index:
            similar_items = df_item_cooccurrence.loc[item]
            top_recommendations = similar_items.sort_values(ascending=False).head(top_n)

            for rec_item, score in top_recommendations.items():
                if rec_item not in order:
                    recommended_items[rec_item] = recommended_items.get(rec_item, 0) + score

    filtered_recommendations = {k: v for k, v in recommended_items.items() if v > 0}
    return dict(sorted(filtered_recommendations.items(), key=lambda item: item[1], reverse=True))

def recommend_items_for_user(phone_number, df_user_recommendations, top_n=10):
    recommended_item_for_user = {}
    if phone_number not in df_user_recommendations.index:
        logging.error(f"User {phone_number} not found in dataset.")
        return {}

    user_recommendations = df_user_recommendations.loc[phone_number]
    top_recommendations = user_recommendations.sort_values(ascending=False).head(top_n)
    
    for rec_item, score in top_recommendations.items(): 
        recommended_item_for_user[rec_item] = score
    
    return {k: v for k, v in recommended_item_for_user.items() if v > 0}

def hybrid_recommendation_cooccurence(user_id, cart_items, df_user_recommendations, df_item_cooccurrence, top_n=5):
    user_recs = recommend_items_for_user(user_id, df_user_recommendations, top_n)
    item_recs = recommend_items(cart_items, df_item_cooccurrence, top_n)

    combined = {}
    for item, score in user_recs.items():
        combined[item] = combined.get(item, 0) + score
    for item, score in item_recs.items():
        combined[item] = combined.get(item, 0) + score

    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n])

# Embeddings based functions
def recommend_similar_items(input_items, df_menu_vectors, top_n=5):
    if isinstance(input_items, str):
        input_items = [input_items]
    valid_items = [item for item in input_items if item in df_menu_vectors.index]
    if not valid_items:
        return {}
    input_vectors = np.array([df_menu_vectors.loc[item].values for item in valid_items])
    input_vector = np.mean(input_vectors, axis=0).reshape(1, -1)
    similarities = cosine_similarity(input_vector, df_menu_vectors.values)[0]
    similar_items = sorted(zip(df_menu_vectors.index, similarities), key=lambda x: x[1], reverse=True)
    return {item: score for item, score in similar_items if item not in valid_items[:top_n]}

def user_based_recommendations(user_id, customer_item_matrix, top_n=5):
    if user_id not in customer_item_matrix.index:
        return {}
    user_similarities = cosine_similarity(customer_item_matrix)
    similarity_df = pd.DataFrame(user_similarities, index=customer_item_matrix.index, columns=customer_item_matrix.index)
    similar_users = similarity_df[user_id].drop(user_id).sort_values(ascending=False).index[:2]
    similar_users_items = customer_item_matrix.loc[similar_users].sum(axis=0)
    user_purchased_items = customer_item_matrix.loc[user_id]
    recommended_items = similar_users_items[user_purchased_items == 0].sort_values(ascending=False).index[:top_n]
    return {item: 1.0 for item in recommended_items}

def hybrid_recommendation(user_id, cart_items, customer_item_matrix, df_menu_vectors, top_n=5):
    user_recs = user_based_recommendations(user_id, customer_item_matrix, top_n)
    item_recs = recommend_similar_items(cart_items, df_menu_vectors, top_n)
    
    combined = {}
    for item in list(user_recs.keys()) + list(item_recs.keys()):
        combined[item] = combined.get(item, 0) + 1
        
    return dict(sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_n])

# Common function
def recommend_popular_items(df_customer_item_matrix, top_n=10):
    item_popularity = df_customer_item_matrix.sum().sort_values(ascending=False).head(top_n)
    return {item: score for item, score in item_popularity.items()}