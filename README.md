# House-Price-Prediction

## Overview
This project aims to predict the median value of owner-occupied homes in the Boston Standard Metropolitan Statistical Area (SMSA) based on various socio-economic and environmental factors using different machine learning models. The dataset used is drawn from the 1970 Boston Housing dataset. The goal is to build and compare several machine learning models to identify the best performing model for predicting house prices.

## Dataset
The dataset contains 14 features, with the MEDV variable as the target (the median value of owner-occupied homes in 1000's dollars [k$]). Below is the list of input features:

CRIM: Per capita crime rate by town.
ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
INDUS: Proportion of non-retail business acres per town.
CHAS: Charles River dummy variable (1 if the tract bounds the river; 0 otherwise).
NOX: Nitric oxides concentration (parts per 10 million).
RM: Average number of rooms per dwelling.
AGE: Proportion of owner-occupied units built prior to 1940.
DIS: Weighted distances to five Boston employment centers.
RAD: Index of accessibility to radial highways.
TAX: Full-value property tax rate per 10,000 dollars.
PTRATIO: Pupil-teacher ratio by town.
B: Calculated as 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town.
LSTAT: Percentage of lower status of the population.
MEDV: Median value of owner-occupied homes in 1000's [Target variable].

## Machine Learning Models
The following models were used to predict the target variable (MEDV):

* Linear Regression: A basic regression model that fits a linear relationship between the features and the target.
* Random Forest: An ensemble method using multiple decision trees to improve the model's performance and accuracy.
* Ridge Regression: A linear regression model with L2 regularization to prevent overfitting by penalizing large coefficients.
* XGBoost: An optimized gradient boosting algorithm that often outperforms other algorithms in regression tasks.
* Random Forest with Recursive Feature Elimination (RFE): Feature selection technique that recursively removes less important features to improve the model's performance.

## Results
After evaluating all the models, XGBoost achieved the highest accuracy for predicting the median housing prices in the dataset. 

## Installation & Requirements
To run this project, you will need the following packages:

* Python 3.x
* scikit-learn
* xgboost
* numpy
* pandas
* matplotlib (optional, for data visualization)
* seaborn (optional, for data visualization)


Clone this repository:

Copy code
git clone https://github.com/your_username/boston-housing-prediction.git
cd boston-housing-prediction
Run the Jupyter notebook or Python script to train and evaluate models:

bash
Copy code
jupyter notebook

You can also view the detailed analysis, training process, and evaluation metrics within the notebook.

## Conclusion
The project highlights the importance of selecting the right machine learning model for a regression task. XGBoost outperformed other models in terms of prediction accuracy, making it the best model for this dataset.
