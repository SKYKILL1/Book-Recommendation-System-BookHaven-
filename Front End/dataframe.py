import pandas as pd
import json

dataframe = pd.read_csv('Books_merged.csv')
json.dump(dataframe,'data.json')
print(dataframe)