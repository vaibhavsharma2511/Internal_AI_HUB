import pandas as pd


def recommend(item, food_court, top_n=10):

    df=pd.read_csv('generatedFiles/cooccurance-'+food_court+'.csv',index_col=0)
    if item not in df.index:
        return f"Item '{item}' not found in the dataset."

        # Extract the row corresponding to the item
    item_row = df.loc[item]

    # Sort the values in descending order and get the top n items
    top_items = item_row.sort_values(ascending=False).head(top_n)

    # Return the top n items
    return top_items.index.tolist()

# Example recommendation
product = 'Make it sundae'
food_court='coho'
recommendations = recommend(product, food_court)
print("Recommendations for " + product + ":", recommendations)

