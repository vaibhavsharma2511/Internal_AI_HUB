import logging
import pandas as pd

# Set up logging configuration to log to both console and file
logging.basicConfig(
    filename='../logs/recommendation.log',  # Log file path
    level=logging.INFO,  # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(message)s',  # Log format (timestamp + message)
    filemode='a'  # Append the log file each time the program runs
)
def recommend(order, food_court, top_n=10):
    try:
        df=pd.read_csv('../generatedFiles/cooccurance-'+food_court+'.csv',index_col=0)
        candidates={}

        for item in order:
            if item not in df.index:
                logging.error(f"Item '{item}' not found in the {food_court} dataset.")

            # Extract the row corresponding to the item
            item_row = df.loc[item]

            # Sort the values in descending order and get the top n items
            top_items = item_row.sort_values(ascending=False).head(top_n)

            for candidate_item,occurance in top_items.items():
                if candidate_item not in order:
                    candidates[candidate_item] = candidates.get(candidate_item, 0) + occurance

        recommendations=sorted(candidates, key=candidates.get, reverse=True)
        return recommendations[:top_n]

    except Exception as e:
        logging.error(f"Error recommending product for food court {food_court}: {e}")


def recommend_most_popular(food_court, top_n=10):
    try:
        df = pd.read_csv('../generatedFiles/cooccurance-' + food_court + '.csv', index_col=0)
        popular=df.sum().sort_values(ascending=False)
        return popular.head(top_n).index.tolist()
    except Exception as e:
        logging.error(f"Error recommending most popular product for food court {food_court}: {e}")

# Example recommendation
product = ['Make it sundae']
food_court='coho'


products = recommend(product, food_court)
print("Recommendations for " + str(product) + ":", products)

print(recommend_most_popular('coho',5))


