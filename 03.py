import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Reading the wine dataset
df = pd.read_csv(r'C:/Users/Student/Desktop/RaJ/DS/Practical 3/wine.csv', header=None, usecols=[0, 1, 2], skiprows=1)
df.columns = ['classlabel', 'Alcohol', 'Malic Acid']

# Displaying the original DataFrame
print("Original DataFrame:")
print(df)

# Min-Max Scaling
scaling = MinMaxScaler()
scaled_value = scaling.fit_transform(df[['Alcohol', 'Malic Acid']])
df[['Alcohol', 'Malic Acid']] = scaled_value

print("\nDataFrame after Min-Max Scaling:")
print(df)

# Standard Scaling
scaling = StandardScaler()
scaled_standard_value = scaling.fit_transform(df[['Alcohol', 'Malic Acid']])
df[['Alcohol', 'Malic Acid']] = scaled_standard_value

print("\nDataFrame after Standard Scaling:")
print(df)
