import pandas as pd

# Reading the Titanic dataset
df = pd.read_csv('C:/Users/Student/Desktop/RaJ/DS/Practical 2/titanic.csv')

# Display the first 10 rows
print("Original dataset:")
print(df.head(10))

# Filling missing values (NA) with 0
df2 = df.fillna(value=0)

# Display the dataset after filling NA values with 0
print("Dataset after filling NA values with 0:")
print(df2)
