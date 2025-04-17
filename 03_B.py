import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Reading the Iris dataset
iris = pd.read_csv("C:/Users/Student/Desktop/RaJ/DS/Practical 2/Iris.csv")
print("Original Iris dataset:")
print(iris)

# Label Encoding the 'Species' column
le = LabelEncoder()
iris['code'] = le.fit_transform(iris['Species'])

print("\nIris dataset with Label Encoded 'Species':")
print(iris)
