import pandas as pd
wine = pd.read_csv("wine.csv", sep=";")
# print(wine.head(5))
# print(wine.tail(5))
# print(wine.info())

# data cleaning
print(wine.isnull().sum())
print(wine.dtypes)
print(wine.duplicated().sum())
wine = wine.drop_duplicates().reset_index(drop=True)
# print(wine.duplicated().sum())
print("Dataset shape:", wine.shape)
print(wine.columns.tolist())

wine.to_csv("cleaned wine data.csv", index=False)
