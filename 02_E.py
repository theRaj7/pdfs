import pandas as pd

# Load Iris dataset
iris = pd.read_csv('C:/Users/Student/Desktop/RaJ/DS/Practical 2/Iris.csv')

# Filtering data based on a condition (setosa species)
setosa = iris[iris['Species'] == 'setosa']
print("Setosa samples:")
print(setosa.head())

# Sorting data by SepalLengthCm in descending order
sorted_iris = iris.sort_values(by='SepalLengthCm', ascending=False)
print("\nSorted iris dataset:")
print(sorted_iris.head())

# Grouping data by Species and calculating the mean of each group
grouped_species = iris.groupby('Species').mean()
print("\nMean measurements for each species:")
print(grouped_species)
