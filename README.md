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
