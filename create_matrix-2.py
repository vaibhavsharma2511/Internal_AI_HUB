import pandas as pd
import json
from createProductsCsv import CreateProducts

inputFiles = ['data/coho/Orders-cleaned.json',
                  'data/foodrepublic/Orders-cleaned.json',
                  'data/goodbowl/Orders-cleaned.json',
                  'data/memphis/Orders-cleaned.json']
productFiles=['generatedFiles/products-coho.csv','generatedFiles/products-foodrepublic.csv','generatedFiles/products-goodbowl.csv','generatedFiles/products-memphis.csv']
outputFiles = ['generatedFiles/matrix-coho.csv', 'generatedFiles/matrix-foodrepublic.csv', 'generatedFiles/matrix-goodbowl.csv', 'generatedFiles/matrix-memphis.csv']


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


cv=CreateProducts()
cv.createProductsCsv()

for i, f in enumerate(inputFiles):
    with open(f, encoding='utf-8') as file:
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
        #for order_items in df['orderItemsParsed']:
        #    for item in order_items:
        #        unique_menu_items.add(item['name'])
        products=pd.read_csv(productFiles[i])
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
                matrix_row[menu_item] = menu_item in menu_items_ordered
            rows.append(matrix_row)
        matrix = pd.concat([matrix, pd.DataFrame(rows)], ignore_index=True)

        # Display the matrix
        print(matrix)
        matrix.to_csv(outputFiles[i], encoding='utf-8')
