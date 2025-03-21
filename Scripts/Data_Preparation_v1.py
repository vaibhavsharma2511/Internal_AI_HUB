import json
import logging
import sys

import pandas as pd

# New food courts will be added in the list
# Data for the new foodcourt will be added to Data/<foodcourt_name> folder

FOODCOURTS=["coho","foodrepublic","goodbowl","memphis"]
ROOT_DIR=sys.path[1]

# Lists to store file paths for input and output files.
inputMenuFiles = []    # Paths to menu files for each food court.
menuItemFiles = []     # Paths to menu item files for each food court.
ordersFiles = []       # Paths to order files for each food court.
productFiles = []      # Paths to output product CSV files for each food court.
cooccuranceFiles = []  # Paths to output co-occurrence matrix files for recommendations.

# Dynamically generate the paths for all input and output files based on the food court names.for foodcourt in FOODCOURTS:
for foodcourt in FOODCOURTS:
    inputMenuFiles.append(ROOT_DIR+'/Data/' + foodcourt + '/Menus.json')
    menuItemFiles.append(ROOT_DIR+'/Data/'+foodcourt+'/MenuItems.json')
    productFiles.append(ROOT_DIR+'/generatedFiles/products-' + foodcourt + '.csv')
    ordersFiles.append(ROOT_DIR+'/Data/'+foodcourt+'/Orders-cleaned.json')
    cooccuranceFiles.append(ROOT_DIR+'/generatedFiles/cooccurance-'+foodcourt+'.csv')

# Utility function to clean improperly formatted JSON strings.
# This replaces single quotes with double quotes and fixes any nested JSON structure inconsistencies.
def clean_json_string(json_str):
    cleaned_str = json_str.replace("'", '')  # Replace single quotes with double quotes
    cleaned_str = cleaned_str.replace('"{', '{').replace('}"', '}')  # Handle nested JSON
    return cleaned_str


# Parses the 'orderItems' JSON string from the order data.
# Handles potential errors in decoding JSON and logs problematic data.
def parse_order_items(order_items):
    try:
        cleaned_order_items = clean_json_string(order_items)
        parsed_items = json.loads(cleaned_order_items)
        logging.debug(f"Successfully parsed order items: {parsed_items}")
        return parsed_items
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        logging.error(f"Problematic data: {order_items}")
        return []  # Return an empty list if there's an error in parsing

# Processes menu and menu item files for each food court and generates product CSV files.
def createProductsCsv():
    for i, f in enumerate(inputMenuFiles):
        logging.info(f"Processing menu items from file: {f}")
        # Read and process the menu JSON file for each food court.
        try:
            with open(f, encoding='utf-8') as file:
                data = json.load(file)['data']
                rows = []
                # Extract product IDs and their categories from the menu content.
                for k,v in data.items():
                    for value in v.get('menuContent', []):
                        for item in value.get('itemIds', []):
                            rows.append({'productId': item, 'category': value.get('categoryName')})
                logging.debug(f"Menu rows extracted: {rows}")
        except Exception as e:
            logging.error(f"Error processing files for food court {FOODCOURTS[i]}: {e}")

        # Read and process the menu items JSON file for additional details about products.
        with open(menuItemFiles[i], encoding='utf-8') as productfile:
            logging.info(f"Processing foods from file: {menuItemFiles[i]}")
            products = json.load(productfile)['data']

            productRows = []
            # Extract product details such as name, ID, and price.
            for key, value in products.items():
                row = {
                    'productName': value.get('name'),
                    'productId': value.get('id'),
                    'price': value.get('retailPriceMoney', {}).get('amount'),
                }
                productRows.append(row)
            logging.debug(f"Product rows extracted: {productRows}")
            # Convert product details into a Pandas DataFrame and write to CSV.
            df_products = pd.DataFrame(productRows, columns=['productName','productId', 'price'])
            df_products.to_csv(productFiles[i], encoding='utf-8')
            logging.info(f"Products CSV written: {productFiles[i]}")

# Set up logging configuration to log to both console and file
logging.basicConfig(
    filename='../logs/data_preparation.log',  # Log file path
    level=logging.INFO,  # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(message)s',  # Log format (timestamp + message)
    filemode='w'  # Overwrite the log file each time the program runs
)

# Generate product CSV files for all food courts.
createProductsCsv()

# Process orders for each food court and create the co-occurrence file for recommendation
# Iterate through all foodcourts
for i, f in enumerate(ordersFiles):
    logging.info(f"Processing orders file: {f}")
    # Read and process the cleaned orders JSON file.
    try:
        with open(f, encoding='utf-8') as file:
            data = json.load(file)['data']   # Load the 'data' section of the JSON file.

            rows = []
            # Iterate through each order and extract relevant details.
            for key, value in data.items():
                row = {
                    'userId': value.get('userId'),
                    'appliedDiscount': value.get('appliedDiscount'),
                    'orderType': value.get('channelData', {}).get('orderType'),
                    'orderDate': value.get('orderDate', {}).get('__time__'),
                    'prepTime': value.get('prepTime'),
                    'discount': value.get('priceData', {}).get('discount'),
                    'organizationFee': value.get('priceData', {}).get('organizationFee'),
                    'organizationStripeFee': value.get('priceData', {}).get('organizationStripeFee'),
                    'stripeFee': value.get('priceData', {}).get('stripeFee'),
                    'subTotal': value.get('priceData', {}).get('subTotal'),
                    'taxes': value.get('priceData', {}).get('taxes'),
                    'total': value.get('priceData', {}).get('total'),
                    'status': value.get('status'),
                    'orderItems': json.dumps(value.get('orderItems', []))
                }
                rows.append(row)
            logging.debug(f"Order rows extracted: {rows}")

            # Convert the extracted order details into a Pandas DataFrame.
            df = pd.DataFrame(rows, columns=['userId', 'appliedDiscount', 'orderType', 'orderDate', 'prepTime', 'discount',
                                             'organizationFee', 'organizationStripeFee', 'stripeFee', 'subTotal', 'taxes',
                                             'total', 'status', 'orderItems'])
            df = df.dropna(subset=['orderItems'])  # Remove rows where 'orderItems' is missing.
            df['orderItemsParsed'] = df['orderItems'].apply(parse_order_items)  # Parse order items into structured data.
            logging.debug(f"DataFrame created: {df.head()}")
            # Extract unique menu item names to create column headers
            unique_menu_items = set()

            # Extract unique menu item names from the product file for column headers.
            products = pd.read_csv(productFiles[i])
            for index, row in products.iterrows():
                unique_menu_items.add(row['productName'])

            columns = ['userID', 'applied discount', 'order type', 'order date', 'prep time', 'discount',
                       'organization fee', 'organizationStripeFee', 'stripeFee', 'subtotal', 'taxes', 'total',
                       'status'] + list(unique_menu_items)
            matrix = pd.DataFrame(columns=columns)
            rows = []

            # Populate the matrix with user data and item interactions.
            for index, row in df.iterrows():
                user_id = row['userId']
                appliedDiscount = row['appliedDiscount']
                orderType = row['orderType']
                orderDate = row['orderDate']
                prepTime = row['prepTime']
                discount = row['discount']
                organizationFee = row['organizationFee']
                organizationStripeFee = row['organizationStripeFee']
                stripeFee = row['stripeFee']
                subtotal = row['subTotal']
                taxes = row['taxes']
                total = row['total']
                status = row['status']
                order_items = row['orderItemsParsed']
                menu_items_ordered = [item['name'] for item in order_items]

                matrix_row = {'userID': user_id, "applied discount": appliedDiscount, 'order type': orderType,
                              'order date': orderDate, 'prep time': prepTime, 'discount': discount,
                              'organization fee': organizationFee, 'organizationStripeFee': organizationStripeFee,
                              'stripeFee': stripeFee, 'subtotal': subtotal, 'taxes': taxes, 'total': total,
                              'status': status}

                for menu_item in unique_menu_items:
                    matrix_row[menu_item] = int(menu_item in menu_items_ordered)

                rows.append(matrix_row)

            matrix = pd.DataFrame(rows)

            # Create and save the co-occurrence matrix.
            item_columns = matrix.columns[14:]
            cooccurrence_matrix = pd.DataFrame(index=item_columns, columns=item_columns, data=0)

            # Increment co-occurrence counts for items ordered together.
            for index, row in matrix.iterrows():
                items = row[item_columns][row[item_columns] == 1].index
                for item1 in items:
                    for item2 in items:
                        if item1 != item2:
                            cooccurrence_matrix.at[item1, item2] += 1
            cooccurrence_matrix.to_csv(cooccuranceFiles[i], encoding='utf-8')
            logging.info(f"Co-occurrence CSV written: {cooccuranceFiles[i]}")
    except Exception as e:
        logging.error(f"Error processing orders for food court {FOODCOURTS[i]}: {e}")

