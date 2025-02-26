import pandas as pd
import json
import numpy as np
path = '/Users/vaibhavsharma/Documents/AI HUB Research Assistant/NextGen Kitchens/data-cleaned/v2/clubkitchen/'

restaurants = ['FxPd8j5iWTbtvgmah8Z6','k4elsyVbBmSu24TpNT8u','kp87Hw65qxgTbabj66Q8','mMxlfT2CB6nlWSbwtLbl','sO7hcYzRudGq9id9wOvs','ZIFwnuY716EUk2PG55IU','ZoVPhvcboPFTsoSpdLNH']
rows = []
for restaurant_id in restaurants:
    with open(path+restaurant_id+'/MenuItems.json', "r") as file:
        data = json.load(file)
    
    # Extract items under "data"
    items = data["data"]
    
    # Flatten data into a list of dictionaries
    
    for item_id, item in items.items():
        rows.append({
            "id": item.get("id", ""),
            "name": item.get("name", ""),
            "description": item.get("description", ""),
            "posCategory": item.get("posCategory", ""),
            "retailPriceMoney": item.get("retailPriceMoney", {}),
            "channelData": item.get("channelData", {}),
            "itemImage": item.get("itemImage", {}),
            "restaurantId": item.get("restaurantId", ""),
            "isDeleted": item.get("isDeleted", False),
            "isSnoozed": item.get("isSnoozed", False),
            "modifierLists": item.get("modifierLists", []),
            "snoozeUntil": item.get("snoozeUntil", ""),
            "taxIds": item.get("taxIds", []),
            "__collections__": item.get("__collections__", {})
        })

# Convert list to DataFrame
df_menu_items = pd.DataFrame(rows)


print(df_menu_items.shape)
df_menu_items.to_csv('df_menu_items.csv', index=False)


# Read the orders JSON file
with open(path+'Orders-cleaned.json', "r") as file:
    data = json.load(file)

# Extract orders data
orders = data.get("data", {})

# Flatten data into a list of dictionaries
rows_orders = []
for order_id, order in orders.items():
    order_items = order.get("orderItems", [])
    
    for item in order_items:
        rows_orders.append({
            # "menuItemId": order.get("id", ""),
            # "customer.email": order.get("customer", {}).get("email", ""),
            # "customer.numberOfPrevOrders": order.get("customer", {}).get("numberOfPrevOrders", 0),
            "customer.phoneNumber": order.get("customer", {}).get("phoneNumber", ""),
            # "customer.registered": order.get("customer", {}).get("registered", False),
            # "customer.uid": order.get("customer", {}).get("uid", ""),
            # "date.__time__": order.get("date", {}).get("__time__", ""),
            # "isTest": order.get("isTest", False),
            # "loyaltyData": order.get("loyaltyData", {}),
            # "note": order.get("note", ""),
            # "notifications.push.send": order.get("notifications", {}).get("push", {}).get("send", False),
            # "notifications.sms.send": order.get("notifications", {}).get("sms", {}).get("send", False),
            # "number": order.get("number", ""),
            # "orderItems.id": item.get("id", ""),
            # "orderItems.menuId": item.get("menuId", ""),
            # "orderItems.menuItemCategory": item.get("menuItemCategory", ""),
            "orderItems.menuItemId": item.get("menuItemId", ""),
            # "orderItems.name": item.get("name", ""),
            # "orderItems.note": item.get("note", ""),
            # "orderItems.priceMoney.amount": item.get("priceMoney", {}).get("amount", 0),
            # "orderItems.priceMoney.currency": item.get("priceMoney", {}).get("currency", ""),
            # "orderItems.quantity": item.get("quantity", 0),
            # "orderItems.restaurantId": item.get("restaurantId", ""),
            # "orderItems.taxIds": item.get("taxIds", []),
            # "payment.amount": order.get("payment", {}).get("amount", 0),
            # "payment.currency": order.get("payment", {}).get("currency", ""),
            # "pickupDate.__time__": order.get("pickupDate", {}).get("__time__", ""),
            # "prepTime": order.get("prepTime", ""),
            # "promos": order.get("promos", []),
            # "receiptOptions": order.get("receiptOptions", {}),
            # "restaurantOrders": order.get("restaurantOrders", {}),
            # "scheduled": order.get("scheduled", False),
            # "source.channel": order.get("source", {}).get("channel", ""),
            # "source.type": order.get("source", {}).get("type", ""),
            # "status": order.get("status", ""),
            # "tableNumber": order.get("tableNumber", ""),
            # "timestamps.OrderConfirmed": order.get("timestamps", {}).get("OrderConfirmed", ""),
            # "timestamps.PlacingOrder": order.get("timestamps", {}).get("PlacingOrder", ""),
            # "timestamps.ReviewOrder": order.get("timestamps", {}).get("ReviewOrder", ""),
            # "timestamps.SelectReceipt": order.get("timestamps", {}).get("SelectReceipt", ""),
            # "transaction.GST": order.get("transaction", {}).get("GST", 0),
            # "transaction.PST": order.get("transaction", {}).get("PST", 0),
            # "transaction.bagFee": order.get("transaction", {}).get("bagFee", 0),
            # "transaction.bagFeeTax": order.get("transaction", {}).get("bagFeeTax", 0),
            # "transaction.subtotal": order.get("transaction", {}).get("subtotal", 0),
            # "transaction.total": order.get("transaction", {}).get("total", 0),
            # "transaction.totalTaxes": order.get("transaction", {}).get("totalTaxes", 0),
            # "type": order.get("type", "")
        })

# Convert list to DataFrame
df_orders = pd.DataFrame(rows_orders)
# print(df_orders.shape)
# df_orders.to_csv('df_orders.csv', index=False)


# To check duplicate menuItems

id_counts = df_menu_items['id'].value_counts()

multiple_id = id_counts[id_counts > 1]
print("Multiple ID's are = ",multiple_id)

# Merging tables
df_merged = df_orders.merge(df_menu_items, left_on='orderItems.menuItemId', right_on='id', how='left')

# Step 2: Create pivot table (one-hot encoding for names)
df_pivot = pd.get_dummies(df_merged['name'])
# print(df_merged['name'])
# Step 3: Combine with original df_orders (excluding duplicate columns)
df_final = pd.concat([df_orders, df_pivot], axis=1).drop(columns=['orderItems.menuItemId'])  
df_final.to_csv('df_final.csv', index=False)

# calculate co-occurrence matrix
x = np.array(df_final)
y = np.array(df_final.T)
co_matrix = np.dot(y,x)
np.fill_diagonal(co_matrix, 0)
columns_list = df_final.columns[1:]

df_co = pd.DataFrame(co_matrix, columns = columns_list, index = columns_list)
print(df_co)

# print(y)