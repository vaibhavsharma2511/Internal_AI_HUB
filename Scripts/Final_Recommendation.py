from gensim.utils import simple_preprocess
import pandas as pd
import gensim
from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # To log to console
        logging.FileHandler('logs_and_matrices/Final_Recommendation.log', mode='w')  # To log to file and overwrite the file each time
    ]
)
# Load menu data
df = pd.read_csv('logs_and_matrices/df_item_cooccurrence.csv')
menu_items = df.columns

df_restaurant_menu = pd.DataFrame(menu_items)
df_restaurant_menu.rename(columns={0: "name"}, inplace=True)

# Tokenize using gensim's simple_preprocess
df_restaurant_menu['tokenized_name'] = df_restaurant_menu['name'].apply(lambda x: simple_preprocess(str(x)))

# Convert tokenized names into list of sentences
sentences = df_restaurant_menu['tokenized_name'].tolist()

# Train Word2Vec model
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("logs_and_matrices/word2vec_menu.model")

# Function to get average vector for a menu item
def get_item_vector(item_name, model):
    tokens = simple_preprocess(item_name)
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Compute vectors for all menu items
menu_item_vectors = {item: get_item_vector(item, word2vec_model) for item in df_restaurant_menu['name']}

# Convert to DataFrame
df_menu_vectors = pd.DataFrame.from_dict(menu_item_vectors, orient='index')
df_menu_vectors.to_csv('logs_and_matrices/menu_item_vectors.csv')

df_menu_vectors = pd.read_csv('logs_and_matrices/menu_item_vectors.csv', index_col=0)

# Function 1: Recommend based on Word2Vec embeddings
def recommend_similar_items(input_items, top_n=5):
    if isinstance(input_items, str):
        input_items = [input_items]
    valid_items = [item for item in input_items if item in df_menu_vectors.index]
    if not valid_items:
        return []
    input_vectors = np.array([df_menu_vectors.loc[item].values for item in valid_items])
    input_vector = np.mean(input_vectors, axis=0).reshape(1, -1)
    similarities = cosine_similarity(input_vector, df_menu_vectors.values)[0]
    similar_items = sorted(zip(df_menu_vectors.index, similarities), key=lambda x: x[1], reverse=True)
    recommendations = [item for item, score in similar_items if item not in valid_items][:top_n]
    return recommendations

# Function 2: Recommend based on user similarity
customer_item_matrix = pd.read_csv('logs_and_matrices/df_customer_item_matrix.csv')
customer_item_matrix = customer_item_matrix.set_index('customer.phoneNumber')

def user_based_recommendations(user_id, top_n=5):
    if user_id not in customer_item_matrix.index:
        return []
    user_similarities = cosine_similarity(customer_item_matrix)
    similarity_df = pd.DataFrame(user_similarities, index=customer_item_matrix.index, columns=customer_item_matrix.index)
    similar_users = similarity_df[user_id].drop(user_id).sort_values(ascending=False).index[:2]
    similar_users_items = customer_item_matrix.loc[similar_users].sum(axis=0)
    user_purchased_items = customer_item_matrix.loc[user_id]
    recommended_items = similar_users_items[user_purchased_items == 0].sort_values(ascending=False).index[:top_n]
    return list(recommended_items)

# Function 3: Hybrid recommendation (user similarity + item embeddings)
def hybrid_recommendation(user_id, cart_items, top_n=5):
    user_recommendations = user_based_recommendations(user_id, top_n=top_n)
    item_recommendations = recommend_similar_items(cart_items, top_n=top_n)
    hybrid_recommendations = list(set(user_recommendations + item_recommendations))[:top_n]
    return hybrid_recommendations

# Calling each function separately
logging.info("----------------------------ITEM RECOMMENDATION-----------------------------")
new_user_cart_item = "Grilled Chicken Bento (feeds 1-2)"
# print(f"Item-based recommendations: {recommend_similar_items(new_user_cart_item, top_n=5)}")
logging.info(f"Recommended items for order {new_user_cart_item}: {recommend_similar_items(new_user_cart_item, top_n=5)}")

logging.info("----------------USER PERSONALIZED RECOMMENDATION-----------------------------")
existing_user_id = "00975814695fafb3a7cb32b2dd307ff6a6e690a62aaa7eee0a814b64ae056426"
# print(f"User-based recommendations: {user_based_recommendations(existing_user_id, top_n=5)}")
logging.info(f"Recommended items for user {existing_user_id}: {user_based_recommendations(existing_user_id, top_n=5)}")

logging.info("----------------HYBRID RECOMMENDATION-----------------------------")
cart_items = ["Grilled Chicken Bento (feeds 1-2)", "Callister Cream Soda"]
# print(f"Hybrid recommendations: {hybrid_recommendation(existing_user_id, cart_items, top_n=5)}")
logging.info(f"Recommended items for user {existing_user_id} and with cart items {cart_items}: {hybrid_recommendation(existing_user_id, cart_items, top_n=10)}")