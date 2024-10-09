from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import uvicorn
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from category_encoders import BinaryEncoder

# Call the app
app = FastAPI(title="Product Demand Prediction API")

# Load the model
with open("./models/forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Define columns
categorical_cols = ['center_id', 'meal_id', 'emailer_for_promotion', 'homepage_featured', 'city_code', 'region_code', 'center_type', 'category', 'cuisine']
numeric_cols = ['week', 'base_price', 'discount', 'op_area']

# Fit transformers
encoder = BinaryEncoder(drop_invariant=False, return_df=True)

quantile_transformer = QuantileTransformer(output_distribution='normal')

scaler = StandardScaler()
scaler.set_output(transform="pandas")

# Define your predict function
def predict(df, endpoint="simple"):
    # Preprocess input data
    df_cat = encoder.fit_transform(df[categorical_cols])
    df_num_quantile = quantile_transformer.fit_transform(df[numeric_cols])
    df_num_quantile = pd.DataFrame(df_num_quantile, columns=numeric_cols)
    df_num_scaled = scaler.fit_transform(df_num_quantile)
    
    # Concatenate encoded categorical and scaled numerical data
    preprocessed_df = pd.concat([df_num_scaled, df_cat], axis=1)

    # Ensure the DataFrame has all the columns that the model was trained on
    model_columns = preprocessed_df.columns.tolist()
    preprocessed_df = preprocessed_df.reindex(columns=model_columns, fill_value=0)

    # Prediction
    prediction = model.predict(preprocessed_df)  # Make predictions using the pre-trained model

    response = []
    for num_orders in prediction:
        # Convert NumPy float to Python native float
        num_orders = int(num_orders)
        # Create a response for each prediction with the predicted number of orders
        output = {
            "predicted_num_orders": num_orders
        }
        response.append(output)  # Add the response to the list of responses

    return response  # Return the list of responses

class Demand(BaseModel):
    week: int
    center_id: str
    meal_id: str
    base_price: float
    emailer_for_promotion: int
    homepage_featured: int
    discount: float
    city_code: str
    region_code: str
    center_type: str
    op_area: float
    category: str
    cuisine: str

class Demands(BaseModel):
    all_demands: list[Demand]

    @classmethod
    def return_list_of_dict(cls, demands: "Demands"):
        demand_list = []
        for demand in demands.all_demands:  # for each item in all_demands
            demand_dict = demand.dict()  # convert to a dictionary
            demand_list.append(demand_dict)  # add it to the empty list called demand_list
        return demand_list

# Endpoints
# Root Endpoint
@app.get("/")
def root():
    return {"message": "Welcome to the Product Demand Prediction API! This API provides endpoints for predicting product demand based on input data."}

# Prediction endpoint
@app.post("/predict")
def predict_demand(demand: Demand):
    # Make prediction
    data = pd.DataFrame(demand.dict(), index=[0])
    predicted_demand = predict(df=data)
    return predicted_demand

# Multiple Prediction Endpoint
@app.post("/predict_multiple")
def predict_demand_for_multiple_demands(demands: Demands):
    """Make prediction with the passed data"""
    data = pd.DataFrame(Demands.return_list_of_dict(demands))
    predicted_demand = predict(df=data, endpoint="multi")
    return {"predicted_demand": predicted_demand}

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)