import pandas as pd
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Define the data path
DATA_PATH = "/Users/david/OneDrive/Documents/Recommender_System/Internal_AI_HUB/V2/"
RESTAURANTS = [
    "FxPd8j5iWTbtvgmah8Z6", "k4elsyVbBmSu24TpNT8u", "kp87Hw65qxgTbabj66Q8",
    "mMxlfT2CB6nlWSbwtLbl", "sO7hcYzRudGq9id9wOvs", "ZIFwnuY716EUk2PG55IU", "ZoVPhvcboPFTsoSpdLNH"
]

# Function to load menu items from multiple restaurants
def load_menu_items():
    logging.info("Importing MENU ITEM JSON Files of Different Restaurants")
    menu_items = []

    for restaurant_id in RESTAURANTS:
        menu_path = os.path.join(DATA_PATH, restaurant_id, "MenuItems.json")
        
        try:
            with open(menu_path, "r", encoding="utf-8") as file:
                data = json.load(file)
                for item in data.get("data", {}).values():
                    menu_items.append({
                        "id": item.get("id", ""),
                        "name": item.get("name", "")
                    })
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.warning(f"Error loading {menu_path}: {e}")

    return pd.DataFrame(menu_items)

# Function to load orders data
def load_orders():
    logging.info("Importing Orders Files and making a matrix")
    orders_path = os.path.join(DATA_PATH, "Orders-cleaned.json")

    try:
        with open(orders_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            orders = []

            for order in data.get("data", {}).values():
                # Skip orders with null or empty phone numbers
                phone_number = order.get("customer", {}).get("phoneNumber", "")
                if not phone_number:  # Skip if phone number is null or empty
                    continue

                for item in order.get("orderItems", []):
                    orders.append({
                        "customer.phoneNumber": phone_number,
                        "orderItems.menuItemId": item.get("menuItemId", "")
                    })

            return pd.DataFrame(orders)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading {orders_path}: {e}")
        return pd.DataFrame(columns=["customer.phoneNumber", "orderItems.menuItemId"])

# Function to process and generate the user-to-menu items matrix
def generate_user_menu_matrix(df_orders, df_menu_items):
    logging.info("Merging order data with menu items")

    # Merge orders with menu items
    df_merged = df_orders.merge(df_menu_items, left_on="orderItems.menuItemId", right_on="id", how="left")

    # One-hot encode menu item names
    df_pivot = pd.get_dummies(df_merged["name"]).astype(int)

    # Aggregate purchases by customer
    df_final = df_orders[["customer.phoneNumber"]].join(df_pivot).groupby("customer.phoneNumber").sum()

    return df_final.reset_index(drop=True)

# Main execution
if __name__ == "__main__":
    df_menu_items = load_menu_items()
    df_orders = load_orders()

    if df_menu_items.empty or df_orders.empty:
        logging.error("Menu items or orders data is empty. Exiting...")
    else:
        # Ensure no null values in customer.phoneNumber
        if df_orders["customer.phoneNumber"].isnull().any():
            logging.warning("Null values found in customer.phoneNumber. Removing them...")
            df_orders = df_orders.dropna(subset=["customer.phoneNumber"])

        # df_menu_items.to_csv("V2/df_menu_items.csv", index=False)
        df_final = generate_user_menu_matrix(df_orders, df_menu_items)
        df_final.to_csv("V2/df_final.csv", index=False)

        logging.info("User-To-MenuItems Matrix Created Successfully")