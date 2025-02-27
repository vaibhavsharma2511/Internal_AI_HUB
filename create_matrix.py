import pandas as pd
import json

df = pd.read_csv('../v1-coho/Orders-1738276168.csv')

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
        print(f"Problematic data: {order_items}")
        return []  # Return an empty list if there's an error in parsing


df_cleaned=df.copy()
df_cleaned = df_cleaned.dropna(subset=['orderItems'])
df_cleaned['orderItemsParsed'] = df_cleaned['orderItems'].apply(parse_order_items)
# Extract unique menu item names to create column headers
unique_menu_items = set()
for order_items in df_cleaned['orderItemsParsed']:
    for item in order_items:
        unique_menu_items.add(item['name'])

columns = ['userID','applied discount','order type','order date','prep time','discount','organization fee','organizationStripeFee','stripeFee','subtotal','taxes','total','status'] + list(unique_menu_items)
matrix = pd.DataFrame(columns=columns)
rows = []
# Populate the matrix with user ID and True/False values
for index, row in df_cleaned.iterrows():
    user_id = row['userId']
    appliedDiscount = row['appliedDiscount']
    orderType = row['channelData.orderType']
    orderDate = row['orderDate']
    prepTime = row['prepTime']
    discount = row['priceData.discount']
    organizationFee = row['priceData.organizationFee']
    organizationStripeFee = row['priceData.organizationStripeFee']
    stripeFee = row['priceData.stripeFee']
    subtotal = row['priceData.subTotal']
    taxes = row['priceData.taxes']
    total = row['priceData.total']
    status = row['status']
    order_items = row['orderItemsParsed']
    menu_items_ordered = [item['name'] for item in order_items]

    matrix_row = {'userID': user_id,"applied discount":appliedDiscount,'order type':orderType,'order date':orderDate,'prep time':prepTime,'discount':discount,'organization fee':organizationFee,'organizationStripeFee':organizationStripeFee,'stripeFee':stripeFee,'subtotal':subtotal,'taxes':taxes,'total':total,'status':status}
    for menu_item in unique_menu_items:
        matrix_row[menu_item] = menu_item in menu_items_ordered
    rows.append(matrix_row)
matrix = pd.concat([matrix, pd.DataFrame(rows)], ignore_index=True)

# Display the matrix
print(matrix)
matrix.to_csv('matrix-2.csv', encoding='utf-8')