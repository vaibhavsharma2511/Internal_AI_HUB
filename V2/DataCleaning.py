import pandas as pd
import json
import numpy as np
import logging

# Set up logging configuration to log to both console and file
logging.basicConfig(
    filename='V2/recommendation_logs.log',  # Log file path
    level=logging.INFO,  # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(message)s',  # Log format (timestamp + message)
    filemode='w'  # Overwrite the log file each time the program runs
)

# Define the path to the restaurant data
path = '/Users/vaibhavsharma/Documents/AI_HUB_Research_Assistant/NextGen_Kitchens/data-cleaned/v2/clubkitchen/'

# List of restaurant IDs to process
restaurants = ['FxPd8j5iWTbtvgmah8Z6','k4elsyVbBmSu24TpNT8u','kp87Hw65qxgTbabj66Q8','mMxlfT2CB6nlWSbwtLbl',
               'sO7hcYzRudGq9id9wOvs','ZIFwnuY716EUk2PG55IU','ZoVPhvcboPFTsoSpdLNH']

logging.info("Importing the MENU ITEM JSON Files of Different Restaurants")

# Initialize list to store menu items data
rows = []

# Iterate through each restaurant ID and load the corresponding MenuItems.json file
for restaurant_id in restaurants:
    try:
        logging.info(f"Processing menu items for restaurant: {restaurant_id}")
        with open(path+restaurant_id+'/MenuItems.json', "r") as file:
            data = json.load(file)
        
        # Extract items under the 'data' key
        items = data["data"]
        
        # Flatten the data into a list of dictionaries (one row for each menu item)
        for item_id, item in items.items():
            rows.append({
                "id": item.get("id", ""),
                "name": item.get("name", "")
            })
        logging.info(f"Successfully processed {len(items)} items for restaurant: {restaurant_id}")
    
    except Exception as e:
        logging.error(f"Error processing menu items for restaurant {restaurant_id}: {e}")

# Convert list of dictionaries into a DataFrame
df_menu_items = pd.DataFrame(rows)
logging.info(f"Menu items data loaded into DataFrame with {len(df_menu_items)} records")

# Save the menu items DataFrame to a CSV file
df_menu_items.to_csv('V2/df_menu_items.csv', index=False)
logging.info("Menu items CSV saved as 'V2/df_menu_items.csv'")

# Import Orders JSON file
logging.info("Importing Orders Files and making a matrix")
try:
    with open(path+'Orders-cleaned.json', "r") as file:
        data = json.load(file)

    # Extract order data from the JSON file
    orders = data.get("data", {})
    
    # Initialize list to store order-related data
    rows_orders = []
    
    # Flatten order data into a list of dictionaries
    for order_id, order in orders.items():
        order_items = order.get("orderItems", [])
        
        for item in order_items:
            rows_orders.append({
                "customer.phoneNumber": order.get("customer", {}).get("phoneNumber", ""),
                "orderItems.menuItemId": item.get("menuItemId", "")
            })
    logging.info(f"Processed {len(rows_orders)} order items from Orders file")

except Exception as e:
    logging.error(f"Error importing or processing Orders file: {e}")

# Convert the order data into a DataFrame
df_orders = pd.DataFrame(rows_orders)
logging.info(f"Orders data loaded into DataFrame with {len(df_orders)} records")

# Check for duplicate menuItem IDs in the menu items DataFrame
id_counts = df_menu_items['id'].value_counts()
multiple_id = id_counts[id_counts > 1]
if not multiple_id.empty:
    logging.warning(f"Multiple instances of the following menu item IDs found: {multiple_id}")
else:
    logging.info("No duplicate menu item IDs found")

# Merging orders and menu items based on menu item ID
df_merged = df_orders.merge(df_menu_items, left_on='orderItems.menuItemId', right_on='id', how='left')
logging.info("Merged orders data with menu items data")

# Step 2: Create pivot table using one-hot encoding for menu item names
df_pivot = pd.get_dummies(df_merged['name']).astype(int)
logging.info(f"Pivot table created with {df_pivot.shape[1]} one-hot encoded columns")

# Step 3: Combine the pivot table with the original order data, dropping the redundant menu item ID column
df_final = pd.concat([df_orders, df_pivot], axis=1).drop(columns=['orderItems.menuItemId'])
logging.info("Pivot table concatenated with orders data, redundant column removed")

# Group by customer phone number and sum up the item quantities
df_final = df_final.groupby('customer.phoneNumber', as_index=False).sum()
logging.info(f"Data grouped by customer phone number. Final dataset contains {df_final.shape[0]} unique customers")

# Save the final DataFrame to a CSV file
df_final.to_csv('V2/df_final.csv', index=False)
logging.info("User-to-menu item matrix CSV saved as 'V2/df_final.csv'")

# Completion message
logging.info("User to menu items matrix creation process completed")