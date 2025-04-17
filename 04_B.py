import pandas as pd
import numpy as np
import seaborn as sb
import warnings
from scipy import stats
import matplotlib.pyplot as plt

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset 'mpg' from seaborn
df = sb.load_dataset('mpg')

# Display the first few rows of the dataframe
print(df)

# Show descriptive statistics of the 'horsepower' and 'model_year' columns
print(df['horsepower'].describe())
print(df['model_year'].describe())

# Creating bins for horsepower and categorize the values into new bins
bins = [0, 75, 150, 240]
df['horsepower_new'] = pd.cut(df['horsepower'], bins=bins, labels=['l', 'm', 'h'])

# Display the newly created 'horsepower_new' column
c = df['horsepower_new']
print(c)

# Creating bins for model year and categorize the values into new bins
ybins = [69, 72, 74, 84]
labels = ['t1', 't2', 't3']
df['modelyear_new'] = pd.cut(df['model_year'], bins=ybins, labels=labels)

# Display the newly created 'modelyear_new' column
newyear = df['modelyear_new']
print(newyear)

# Create a contingency table (cross-tabulation) between 'horsepower_new' and 'modelyear_new'
df_chi = pd.crosstab(df['horsepower_new'], df['modelyear_new'])

# Display the contingency table
print(df_chi)

# Perform the Chi-squared test of independence
chi2_stat, p_val, dof, expected = stats.chi2_contingency(df_chi)

# Print the results of the Chi-squared test
print(f"Chi2 Statistic: {chi2_stat}")
print(f"P-value: {p_val}")
print(f"Degrees of Freedom: {dof}")
print(f"Expected Frequencies Table:\n{expected}")
