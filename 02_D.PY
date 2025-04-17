import pandas as pd

# Reading the Titanic dataset
df = pd.read_csv('C:/Users/Student/Desktop/RaJ/DS/Practical 2/titanic.csv')
print("Original dataset:")
print(df.head(10))

# Dropping rows with missing values (NA)
print("Dataset after dropping NA values:")
df.dropna(inplace=True)
print(df)
