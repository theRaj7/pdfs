import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("C:/Users/Student/Desktop/RaJ/DS/Practical 8/wholesale.csv")

print(data.head())

# Define the categorical and continuous features
categorical_features = ['Channel', 'Region']
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']

# Get basic statistics of continuous features
print(data[continuous_features].describe())

# One-hot encode categorical features
for col in categorical_features:
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)

print(data.head())

# Check for NaN or infinite values in the data
if data.isnull().any().any():
    print("There are NaN values in the data.")
    print(data.isnull().sum())
else:
    print("No NaN values found.")

if (data == float('inf')).any().any():
    print("There are infinite values in the data.")
else:
    print("No infinite values found.")

# Scale the data using MinMaxScaler
scaler = MinMaxScaler()  # Make sure to avoid using 'fit' or 'transform' as variable names
scaler.fit(data)  # Apply fit on the entire dataset
data_transformed = scaler.transform(data)  # Use transform to scale the data

# Use the elbow method to find the optimal number of clusters (k)
sum_of_squared_distances = []
K = range(1, min(len(data), 15))  # Limit k to be less than or equal to the number of samples

for k in K:
    km = KMeans(n_clusters=k)
    km.fit(data_transformed)  # Fit the KMeans model
    sum_of_squared_distances.append(km.inertia_)  # Access inertia_ directly (no parentheses)

# Plot the elbow graph
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method for Optimal k')
plt.show()
    