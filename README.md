# Food-Demand

## Introduction
Genpact, a food delivery company, is facing financial losses due to understocking key food items, causing customers to leave when products like chicken are unavailable. To address this issue, they have prioritized demand forecasting to accurately predict customer needs and stock the right amount of food. They have hired you, a data scientist, `to analyze their data and provide recommendations on future food demand to ensure proper stocking and cost savings.`

## Aim of Project
The project aims to analyze Genpact's demand dataset and build a complex algorithm to predict future food demand. By the project's end, the algorithm will be deployed as an API using FastAPI, allowing it to be integrated into web or mobile applications for real-world use.

## Source of Project
This project was derived from a [research paper](https://doi.org/10.1109/ACCESS.2023.3266275).

## Dataset description
The ['Food Demand Forecasting' dataset](https://github.com/ikoghoemmanuell/Forecasting_of_Food_Demand_Supply_Chain/tree/main/assets/data) comprises 145 weeks’ worth of weekly orders for 50 distinct meals. The data is spread across 3 files, with about 450,000 entries and 15 features. 

`The task` is to predict demand for the next 10 weeks (Weeks 146–155) using:
- historical demand data (Weeks 1–145)
- meal features (category, sub-category, price, discount)
- fulfillment center details (area, city, etc.).

**1. weekly Demand data (train.csv)**: test.csv doesn't contain the target variable.

**2. fulfilment_center_info.csv**:

**3. meal_info.csv**:

## Content
**EDA:**
- Univariate Analysis
  - The distribution of features
  - Check outliers.
  - Which type of center received the most orders?
  - What are the top 5 centers?
  - ...
- Multivariate Analysis: Is there multicollinearity in our variables?
- Feature Engineering:
  - Create new features.
  - Handle missing values by interpolation.
  - Handling Skewness.
  - Check Normality of All Numerical Columns.
  - Features Scaling and Encoding.

**ML Models for Forecasting**
- Random Forest
- Gradient Boosting Machine (GBM)
- Light GBM
- XGBoost
- CatBoost
- LSTM
- Bi-LSTM

**API deployment**

If there are any errors or contributions, please [email](dangcaoho151202gmail.com) me.
