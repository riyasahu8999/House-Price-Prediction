# House Price Prediction

## QSkill Python Development Internship – Task 2

### Project Overview

This project predicts house prices using Machine Learning and Linear Regression.

The model uses house features such as area, number of bedrooms, bathrooms, age, and location to predict the price of a house.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset Features

The dataset contains:

- Area
- Bedrooms
- Bathrooms
- Age
- Location
- Price

## Project Steps

1. Load the house price dataset using Pandas.
2. Check dataset information and missing values.
3. Select input features and target variable.
4. Encode categorical location data using One-Hot Encoding.
5. Split the dataset into training and testing data.
6. Train a Linear Regression model.
7. Predict house prices.
8. Evaluate the model using MAE, RMSE, and R² Score.
9. Visualize Actual vs Predicted house prices.
10. Predict the price of a new house.

## Model Evaluation

The model is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Visualization

The project generates an Actual vs Predicted House Prices scatter plot.

## How to Run

Install required libraries:

```bash
pip install pandas numpy matplotlib scikit-learn
