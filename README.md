# ML_ASSIGNMENT_HARIOM
# Data Exploration, Splitting, and Linear Regression in Python

This repository contains three Python scripts that illustrate key concepts in data science and machine learning using the Iris dataset and a sample dataset for linear regression. The scripts cover dataset exploration, data splitting, and linear regression model training.

## Table of Contents
1. [Iris Dataset Exploration](#iris-dataset-exploration)
2. [Iris Dataset Splitting](#iris-dataset-splitting)
3. [Linear Regression on Salary Data](#linear-regression-on-salary-data)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [License](#license)

## Iris Dataset Exploration

### Overview

The first script, `iris_exploration.py`, is designed to load the Iris dataset, explore its structure, and display key statistics. The Iris dataset is a classic dataset in machine learning, containing 150 samples of iris flowers, with features such as sepal length, sepal width, petal length, and petal width.

### Code Explanation

- **Loading the Iris Dataset:**
  ```python
  from sklearn.datasets import load_iris
  import pandas as pd
  
  # Load the Iris dataset
  iris = load_iris()
# Create a DataFrame from the data
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("First five rows of the dataset:")
print(iris_df.head())
print("\nShape of the dataset:")
print(iris_df.shape)

print("\nSummary statistics for each feature:")
print(iris_df.describe())
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Number of samples in the training set: {X_train.shape[0]}")
print(f"Number of samples in the testing set: {X_test.shape[0]}")
Number of samples in the training set: 120
Number of samples in the testing set: 30
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset (YearsExperience vs. Salary)
data = {
    'YearsExperience': [1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7, 3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0, 6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5],
    'Salary': [39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189, 63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940, 91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872]
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)
# Define features (X) and target variable (y)
X = df[['YearsExperience']]
y = df['Salary']

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict salaries on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error (RMSE)

# Print the Mean Squared Error and Root Mean Squared Error
print(f"Mean Squared Error on the test set: {mse}")
print(f"Root Mean Squared Error on the test set: {rmse}")
Mean Squared Error on the test set: 49830096.85590839
Root Mean Squared Error on the test set: 7059.76736897757
git clone https://github.com/yourusername/machine-learning-examples.git
cd machine-learning-examples
pip install pandas scikit-learn numpy
python iris_exploration.py
python iris_data_split.py
python linear_regression_salary.py

### Instructions:

1. Replace `yourusername` with your GitHub username in the clone command.
2. Save the content above in a `README.md` file in your project directory.
3. Customize any sections as needed for your specific project.

This `README.md` provides detailed explanations of each script and how to use them, offering clarity to anyone who might use or review your repository.
