import pandas as pd
import json

class CreateProducts:
    inputFiles = ['data/coho/MenuItems.json',
                  'data/foodrepublic/MenuItems.json',
                  'data/goodbowl/MenuItems.json',
                  'data/memphis/MenuItems.json']

    inputMenuFiles = ['data/coho/Menus.json',
                  'data/foodrepublic/Menus.json',
                  'data/goodbowl/Menus.json',
                  'data/memphis/Menus.json']

    outputFiles = ['generatedFiles/products-coho.csv', 'generatedFiles/products-foodrepublic.csv', 'generatedFiles/products-goodbowl.csv', 'generatedFiles/products-memphis.csv']

    def createProductsCsv(self):
        for i, f in enumerate(self.inputMenuFiles):
            with open(f, encoding='utf-8') as file:
                data = json.load(file)['data']

                rows = []
                for k,v in data.items():
                    menuContent=v.get('menuContent')
                    for value in menuContent:
                        itemIds=value.get('itemIds')
                        for item in itemIds:
                            row={'productId':item,'category':value.get('categoryName')}
                            rows.append(row)

            with open(self.inputFiles[i], encoding='utf-8') as productfile:
                products = json.load(productfile)['data']

                productRows = []
                for key, value in products.items():
                    row = {
                        'productName': value.get('name'),
                        'productId': value.get('id'),
                        'price': value.get('retailPriceMoney', {}).get('amount'),
                    }
                    productRows.append(row)

                # Create the DataFrame
               # df = pd.DataFrame(rows, columns=['productId', 'category'])
                df_products = pd.DataFrame(productRows, columns=['productName','productId', 'price'])
               # df_final=pd.merge(df,df_products,on='productId')
                df_products.to_csv(self.outputFiles[i], encoding='utf-8')