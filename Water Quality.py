#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


gw = pd.read_csv('gw.csv')

gw.info()

gw.head()

# Remove the first row if it contains NaN values
gw = gw.dropna(how='all').reset_index(drop=True)

# Display the first few rows again to ensure the data is cleaned up
gw.head()

# Check for missing values across all columns
gw.isnull().sum()

gw.head()

gw.describe()

print(gw.columns)


gw['Fluoride (mg/L)'].unique()


gw.info()


import pandas as pd

# Assuming you have already loaded the data again as 'df', for example:
df = pd.read_csv("gw.csv")

# Converting all the columns except 'State Name' to float
columns_to_convert = df.columns[df.columns != 'State Name']

# Apply conversion
df[columns_to_convert] = df[columns_to_convert].apply(pd.to_numeric, errors='coerce')

# Verify the new data types
print(df.dtypes)

# Group by 'State Name' and get descriptive statistics for all water quality parameters
state_stats = df.groupby('State Name').describe()

# Display the state-wise descriptive statistics
print(state_stats)


# Getting the mean value of each water quality parameter by state
state_means = df.groupby('State Name').mean()

# Display the mean values
print(state_means)

import matplotlib.pyplot as plt
import seaborn as sns

# Assuming your data is stored in a pandas DataFrame named 'df'
parameters = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
              'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
              'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']

# Set the plot size
plt.figure(figsize=(12, 8))

# Create box plots for each parameter
for i, param in enumerate(parameters, 1):
    plt.subplot(3, 3, i)  # Create a grid of 3x3 for subplots
    sns.boxplot(x='State Name', y=param, data=df)
    plt.title(f'Box Plot of {param}')
    plt.xticks(rotation=90)
    plt.tight_layout()

# Show the plots
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

parameters = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
              'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
              'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']

# plot size for better readability
plt.figure(figsize=(18, 16))

for i, param in enumerate(parameters, 1):
    plt.subplot(3, 3, i)  # Create a grid of 3x3 for subplots
    sns.boxplot(x='State Name', y=param, data=df)
    
    plt.title(f'Box Plot of {param}', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.yticks(fontsize=12)
    
    # Add a label for the y-axis
    plt.ylabel(param, fontsize=12)

# Adjust the layout to avoid overlapping
plt.tight_layout()

# Show the plots
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Setting a large canvas for better readability
plt.figure(figsize=(20, 30))

# 1. Line plot for pH
plt.subplot(4, 2, 1)
sns.lineplot(x=df.index, y='pH', data=df, marker='o', label='pH')
plt.xticks(rotation=90, fontsize=10)
plt.title('pH Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('pH', fontsize=12)
plt.legend()

# 2. Bar plot for Conductivity
plt.subplot(4, 2, 2)
sns.barplot(x='State Name', y='Conductivity', data=df, palette='viridis')
plt.xticks(rotation=90, fontsize=10)
plt.title('Conductivity Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('Conductivity', fontsize=12)

# 3. Box plot for BOD (mg/L)
plt.subplot(4, 2, 3)
sns.boxplot(x='State Name', y='BOD (mg/L)', data=df, palette='coolwarm')
plt.xticks(rotation=90, fontsize=10)
plt.title('BOD (mg/L) Distribution Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('BOD (mg/L)', fontsize=12)



# 4. Violin plot for Faecal Coliform (MPN/100ml)
plt.subplot(4, 2, 5)
sns.violinplot(x='State Name', y='Faecal Coliform (MPN/100ml)', data=df, palette='rocket')
plt.xticks(rotation=90, fontsize=10)
plt.title('Faecal Coliform (MPN/100ml) Distribution Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('Faecal Coliform (MPN/100ml)', fontsize=12)

# 5. Scatter plot for Total Coliform (MPN/100ml)
plt.subplot(4, 2, 6)
sns.scatterplot(x=df.index, y='Total Coliform (MPN/100ml)', data=df, s=100, color='green', label='Total Coliform')
plt.xticks(rotation=90, fontsize=10)
plt.title('Total Coliform (MPN/100ml) Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('Total Coliform (MPN/100ml)', fontsize=12)
plt.legend()

# 6. Histogram for Total Dissolved Solids (mg/L)
plt.subplot(4, 2, 7)
sns.histplot(df['Total Dissolved Solids (mg/L)'], bins=15, kde=True, color='purple')
plt.title('Total Dissolved Solids (mg/L) Distribution', fontsize=14)
plt.xlabel('Total Dissolved Solids (mg/L)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# 7. Line plot for Fluoride (mg/L)
plt.subplot(4, 2, 8)
sns.lineplot(x=df.index, y='Fluoride (mg/L)', data=df, marker='s', label='Fluoride')
plt.xticks(rotation=90, fontsize=10)
plt.title('Fluoride (mg/L) Across States', fontsize=14)
plt.xlabel('State Name', fontsize=12)
plt.ylabel('Fluoride (mg/L)', fontsize=12)
plt.legend()

# Adjust the layout to prevent overlap
plt.tight_layout()
plt.show()


# In[9]:


import pandas as pd

# Loading the dataset (replace 'file_path' with the actual path to your file)
data = pd.read_csv('gw.csv')

# Inspect first few rows of the dataset
print(data.head())

# Check for missing values and erroneous entries
print(data.isnull().sum())  # Number of missing values
print(data.describe())  # Summary statistics to spot anomalies

# Check for non-numeric values like '#DIV/0!' that need cleaning
print(data.isin(['#DIV/0!']).sum())  # Check for erroneous values

# Convert erroneous values to NaN
data.replace('#DIV/0!', pd.NA, inplace=True)

# Convert relevant columns to numeric types
cols_to_convert = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 'Faecal Coliform (MPN/100ml)', 
                   'Total Coliform (MPN/100ml)', 'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']
data[cols_to_convert] = data[cols_to_convert].apply(pd.to_numeric, errors='coerce')

# Check data after cleaning
print(data.head())
print(data.isnull().sum())  # Recheck for missing values



import pandas as pd

# Load the dataset (make sure you update the file path)
water_quality_data = pd.read_csv('gw.csv')

# Check the first few rows of the dataset
print(water_quality_data.head())


# Fill missing values using the median for each column
water_quality_data_filled = water_quality_data.copy()

# Fill missing values with the median
for column in water_quality_data_filled.columns:
    if water_quality_data_filled[column].dtype != 'object':  # Only apply to numerical columns
        median_value = water_quality_data_filled[column].median()
        water_quality_data_filled[column].fillna(median_value, inplace=True)

# Check if the missing values have been filled
print(water_quality_data_filled.isnull().sum())


# Convert the numerical columns to numeric, coercing errors to NaN
numerical_columns = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
                     'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
                     'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']

# Coerce non-numeric values to NaN
for column in numerical_columns:
    water_quality_data[column] = pd.to_numeric(water_quality_data[column], errors='coerce')

# Fill missing values for numerical columns with the median
for column in numerical_columns:
    water_quality_data[column].fillna(water_quality_data[column].median(), inplace=True)

# Check the data to ensure missing values are filled
print(water_quality_data.isnull().sum())


# Descriptive statistics for the numerical columns
descriptive_stats = water_quality_data.describe()
print(descriptive_stats)



# Descriptive statistics for the numerical columns
descriptive_stats = water_quality_data.describe()
print(descriptive_stats)


import matplotlib.pyplot as plt

# Plotting histograms for each numerical column
numerical_columns = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
                     'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
                     'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']

plt.figure(figsize=(14, 10))
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)
    water_quality_data[column].hist(bins=20, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


# Plot histograms for each numerical column
numerical_columns = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
                     'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
                     'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']

# Set up the plot grid
plt.figure(figsize=(14, 10))

# Loop through each numerical column and plot a histogram
for i, column in enumerate(numerical_columns, 1):
    plt.subplot(3, 3, i)  # 3x3 grid for the histograms
    plt.hist(water_quality_data[column], bins=20, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Compute the correlation matrix for numerical columns
correlation_matrix = water_quality_data[numerical_columns].corr()

# Display the correlation matrix
print(correlation_matrix)

import seaborn as sns

# Create a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Water Quality Parameters')
plt.show()


# Generate pairwise plot to explore relationships between variables
sns.pairplot(water_quality_data[numerical_columns])
plt.suptitle('Pairwise Plot of Water Quality Parameters', y=1.02)
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

feature_columns = ['pH', 'Conductivity', 'BOD (mg/L)', 'Nitrate N (mg/L)', 
                   'Faecal Coliform (MPN/100ml)', 'Total Coliform (MPN/100ml)', 
                   'Total Dissolved Solids (mg/L)', 'Fluoride (mg/L)']


water_quality_data['Water_Quality'] = water_quality_data.apply(
    lambda row: 'Safe' if (row['pH'] >= 6 and row['pH'] <= 8 and row['BOD (mg/L)'] < 5) else 'Unsafe', axis=1)

# Splitting data into features (X) and target (y)
X = water_quality_data[feature_columns]
y = water_quality_data['Water_Quality']

# Splitting data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred = knn.predict(X_test_scaled)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[30]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Let's predict pH as an example
X = water_quality_data[feature_columns]
y = water_quality_data['pH']  # Or choose any other continuous target

# Splitting data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizing the feature data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)
knn_regressor.fit(X_train_scaled, y_train)

# Making predictions on the test set
y_pred_reg = knn_regressor.predict(X_test_scaled)

# Evaluating the model
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred_reg))


# Accuracy:
# Overall Accuracy: 96.35%. This indicates that the model correctly predicted water quality for 96.35% of the test samples.
# Classification Report:
# Precision: Measures how many of the predicted "Safe" or "Unsafe" water quality labels were actually correct.
# 
# For "Safe" water: 96% precision — 96% of the times the model predicted "Safe," it was correct.
# For "Unsafe" water: 97% precision — 97% of the times the model predicted "Unsafe," it was correct.
# Recall: Measures how many of the actual "Safe" or "Unsafe" water quality labels were correctly identified by the model.
# 
# For "Safe" water: 99% recall — The model correctly identified 99% of all "Safe" water samples.
# For "Unsafe" water: 82% recall — The model correctly identified 82% of all "Unsafe" water samples.
# F1-Score: The harmonic mean of precision and recall, providing a balanced measure of the model's performance.
# 
# "Safe": 98% F1-score.
# "Unsafe": 89% F1-score.
# Support: The number of samples for each class in the test data.
# 
# "Safe": 179 samples.
# "Unsafe": 40 samples.
# Observations:
# The model has excellent performance with high precision, recall, and F1-scores, especially for the "Safe" water classification (which is likely the majority class).
# However, the recall for the "Unsafe" category is lower (82%), indicating that the model might be missing some of the "Unsafe" water quality cases. This can be improved by experimenting with different values of k, or by using techniques like class weighting or resampling to balance the class distribution.

# The Mean Squared Error (MSE) value of 0.0141 indicates the average squared difference between the predicted values and the actual values in the regression model.
# 
# Interpretation of MSE:
# The MSE is a common metric used to evaluate the performance of regression models. A lower MSE value suggests better predictive accuracy of the model.
# In this case, an MSE of 0.0141 indicates that, on average, the squared difference between the predicted water quality values (or any regression output in your case) and the actual values is 0.0141.
# Benefits of this result:
# The MSE is relatively low, suggesting that your KNN model (in regression mode) is performing well with minimal error.
# Since you're predicting continuous values (like water quality measurements), having a small MSE means that the model's predictions are very close to the actual values.



# Checking column names in the dataset
print(water_quality_data.columns)



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Dropping non-numerical columns and set the target variable (pH)
X = water_quality_data.drop(columns=['State Name', 'pH', 'Water_Quality'])  # Dropping non-numerical columns
y = water_quality_data['pH']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features (important for KNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Initializing the KNN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune n_neighbors

# Training the model
knn_regressor.fit(X_train, y_train)

# Initialize the KNN Regressor
knn_regressor = KNeighborsRegressor(n_neighbors=5)  # You can tune n_neighbors

# Training the model
knn_regressor.fit(X_train, y_train)


# Predicting on the test data
y_pred = knn_regressor.predict(X_test)

# Evaluating the performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared: {r2}")


# The results of your KNN Regressor model are:
# 
# Mean Absolute Error (MAE): 0.33
# R-squared (R²): 0.47
# Interpretation:
# MAE of 0.33: This indicates that, on average, the predicted pH values are off by 0.33 units from the actual values. This is relatively low, but it depends on the scale and range of pH values in your dataset.
# 
# R² of 0.47: This means the model explains 47% of the variance in the target variable (pH). This is a moderate performance, suggesting that there is some predictive power but there's room for improvement.



from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#  features (excluding the target column 'Water_Quality')
X = water_quality_data.drop(columns=['Water_Quality'])

# target column
y = water_quality_data['Water_Quality']

#  categorical columns (e.g., 'State Name') using One-Hot Encoding
categorical_columns = ['State Name']

# Creating a column transformer to apply OneHotEncoding to categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ], 
    remainder='passthrough'  # Keep the remaining columns as they are (e.g., numeric columns)
)

# Creating a pipeline with preprocessing followed by RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Splitting data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision for Safe: 1.00 (perfect precision for predicting safe water quality)
# Recall for Safe: 0.99 (99% of the actual safe samples were correctly identified)
# Precision for Unsafe: 0.95 (some false positives for unsafe water, but still good)
# Recall for Unsafe: 1.00 (all unsafe water samples were correctly identified)
# F1-Score: Overall, the F1-scores are strong for both classes, especially for safe water, indicating good balance between precision and recall.
# The model seems very effective at identifying both safe and unsafe water quality instances, especially with the high recall for the unsafe class. The slight dip in precision for the unsafe class is typical when dealing with an imbalanced dataset, but overall performance is excellent.



