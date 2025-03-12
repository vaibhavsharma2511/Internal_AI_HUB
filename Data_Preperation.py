import json
import logging

import pandas as pd

# New food courts will be added in the list
# Data for the new foodcourt will be added to data/<foodcourt_name> folder

FOODCOURTS=["coho","foodrepublic","goodbowl","memphis"]

# create menu file names list
inputMenuFiles=[]
menuItemFiles=[]
ordersFiles=[]
productFiles = []
cooccuranceFiles=[]

# Create the required input and output file name lists.
for foodcourt in FOODCOURTS:
    inputMenuFiles.append('data/' + foodcourt + '/Menus.json')
    menuItemFiles.append('data/'+foodcourt+'/MenuItems.json')
    productFiles.append('generatedFiles/products-' + foodcourt + '.csv')
    ordersFiles.append('data/'+foodcourt+'/Orders-cleaned.json')
    cooccuranceFiles.append('generatedFiles/cooccurance-'+foodcourt+'.csv')

def clean_json_string(json_str):
    cleaned_str = json_str.replace("'", '')  # Replace single quotes with double quotes
    cleaned_str = cleaned_str.replace('"{', '{').replace('}"', '}')  # Handle nested JSON
    return cleaned_str



def parse_order_items(order_items):
    try:
        cleaned_order_items = clean_json_string(order_items)
        return json.loads(cleaned_order_items)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        logging.error(f"Problematic data: {order_items}")
        return []  # Return an empty list if there's an error in parsing

# generates product list files for each food court
def createProductsCsv():
    for i, f in enumerate(inputMenuFiles):
        with open(f, encoding='utf-8') as file:
            logging.info(f"Processing menu items from file: {f}")
            data = json.load(file)['data']
            rows = []
            for k,v in data.items():
                menuContent=v.get('menuContent')
                for value in menuContent:
                    itemIds=value.get('itemIds')
                    for item in itemIds:
                        row={'productId':item,'category':value.get('categoryName')}
                        rows.append(row)

        with open(menuItemFiles[i], encoding='utf-8') as productfile:
            logging.info(f"Processing foods from file: {menuItemFiles[i]}")
            products = json.load(productfile)['data']

            productRows = []
            for key, value in products.items():
                row = {
                    'productName': value.get('name'),
                    'productId': value.get('id'),
                    'price': value.get('retailPriceMoney', {}).get('amount'),
                }
                productRows.append(row)

            df_products = pd.DataFrame(productRows, columns=['productName','productId', 'price'])
            logging.info(f"Writing  file: {productFiles[i]}")
            df_products.to_csv(productFiles[i], encoding='utf-8')

# Set up logging configuration to log to both console and file
logging.basicConfig(
    filename='logs/data_preparation.log',  # Log file path
    level=logging.INFO,  # Log level (INFO, DEBUG, etc.)
    format='%(asctime)s - %(message)s',  # Log format (timestamp + message)
    filemode='w'  # Overwrite the log file each time the program runs
)

createProductsCsv()

# Process orders for each food court and create the co-occurrence file for recommendation

for i, f in enumerate(ordersFiles):
    with open(f, encoding='utf-8') as file:
        logging.info(f"Processing  file: {f}")
        data = json.load(file)['data']

        rows = []
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

        # Create the DataFrame
        df = pd.DataFrame(rows, columns=['userId', 'appliedDiscount', 'orderType', 'orderDate', 'prepTime', 'discount',
                                         'organizationFee', 'organizationStripeFee', 'stripeFee', 'subTotal', 'taxes',
                                         'total', 'status', 'orderItems'])
        df = df.dropna(subset=['orderItems'])
        df['orderItemsParsed'] = df['orderItems'].apply(parse_order_items)

        # Extract unique menu item names to create column headers
        unique_menu_items = set()

        products = pd.read_csv(productFiles[i])
        for index, row in products.iterrows():
            unique_menu_items.add(row['productName'])

        columns = ['userID', 'applied discount', 'order type', 'order date', 'prep time', 'discount',
                   'organization fee', 'organizationStripeFee', 'stripeFee', 'subtotal', 'taxes', 'total',
                   'status'] + list(unique_menu_items)
        matrix = pd.DataFrame(columns=columns)
        rows = []
        # Populate the matrix with user ID and True/False values
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

        # Create the co-occurrence matrix
        item_columns = matrix.columns[14:]  # Adjust the index based on your data
        cooccurrence_matrix = pd.DataFrame(index=item_columns, columns=item_columns, data=0)

        for index, row in matrix.iterrows():
            items = row[item_columns][row[item_columns] == 1].index
            for item1 in items:
                for item2 in items:
                    if item1 != item2:
                        cooccurrence_matrix.at[item1, item2] += 1
        cooccurrence_matrix.to_csv(cooccuranceFiles[i], encoding='utf-8')

