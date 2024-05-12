from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import xgboost as xgb
import pickle
import pandas as pd

# Load the XGBoost models
with open('prediction_model.pkl', 'rb') as file:
    loan_approval_model = pickle.load(file)

with open('approved.pkl', 'rb') as file:
    approved_percentage_model = pickle.load(file)

with open('interest.pkl', 'rb') as file:
    interest_rate_model = pickle.load(file)

# Load the test data
test_data = pd.read_csv('TEST_DATA.csv')

# Define the input data model for loan approval prediction
class LoanApprovalRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float

# Define the output data model for loan approval prediction
class LoanApprovalResponse(BaseModel):
    prediction: int

# Define the input data model for getting approved percentage and interest rate
class ApprovedPercentageAndInterestRateRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float

# Define the output data model for getting approved percentage and interest rate
class ApprovedPercentageAndInterestRateResponse(BaseModel):
    approved_percentage: float
    interest_rate: float

# Initialize FastAPI app
app = FastAPI()

# Route to predict loan approval
@app.post("/predict_loan_approval/")
async def predict_loan_approval(request: LoanApprovalRequest):

    # Extract TAN ID, tenure, and requested amount
    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row=tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested']=requested_amount
    tan_row['Loan_Tenure_Months']=tenure
    
    input_data = np.array(tan_row)
    

    # Make prediction

    prediction = loan_approval_model.predict([input_data])
    return LoanApprovalResponse(prediction=prediction)
    
    # Return prediction
   

# Route to get approved percentage and interest rate
@app.post("/get_approved_percentage_and_interest_rate/")
async def get_approved_percentage_and_interest_rate(request: ApprovedPercentageAndInterestRateRequest):

    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row=tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested']=requested_amount
    tan_row['Loan_Tenure_Months']=tenure
    input_data = np.array(tan_row)
    
    # Make predictions for approved percentage and interest rate
    approved_percentage = approved_percentage_model.predict([input_data])[0]
    interest_rate = interest_rate_model.predict([input_data])[0]
    
    # Return approved percentage and interest rate
    return ApprovedPercentageAndInterestRateResponse(approved_percentage=approved_percentage, interest_rate=interest_rate)
# Define the input data model for getting top 5 rules
class Top5RulesRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float

# Define the output data model for getting top 5 rules
class Top5RulesResponse(BaseModel):
    top_5_rules: list

# Route to get top 5 rules
@app.post("/get_top_5_rules/")
async def get_top_5_rules(request: Top5RulesRequest):
    # Assuming your XGBoost model is named 'model'

    # Define the test row (you can modify this based on your use case)
    
    # Extract TAN ID, tenure, and requested amount
    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row=tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested']=requested_amount
    tan_row['Loan_Tenure_Months']=tenure
    test_row=tan_row
    input_data = np.array(tan_row)
    
    model=loan_approval_model
    # Get the prediction probabilities from your model for the selected row
    prediction_probs = model.predict_proba([test_row])

    predicted_class = prediction_probs.argmax(axis=1)

    # Get the booster of the model
    booster = model.get_booster()

    # Get the indices of the top 5 most important features
    importance_scores = model.feature_importances_
    top5_indices =  importance_scores.argsort()[-5:]

    # Initialize a list to store the top 5 rules
    top5_rules = []

    # Get the tree associated with the predicted class
    tree = booster.get_dump()[0]

    # Parse the tree to extract the split conditions based on the input data
    for line in tree.split('\n'):
        for i in top5_indices:
            feature_name = test_data.columns[i]
            if feature_name in line:
                feature_value = test_row.iloc[0, i]
                split_info = line.strip()
                # Split the line by whitespace
                parts = split_info.split()
                for part in parts:
                    if '<=' in part :
                        operator='<='
                    elif '<' in part:
                        operator='<'
                    elif '>=' in part:
                        operator='>='
                    elif '>' in part:
                        operator='>'
                    
                    split_value = float(parts[0][parts[0].index(operator)+1:-1])

                    # Check if the split condition is satisfied
                    if (operator == "<=" and feature_value <= split_value) or \
                        (operator == "<" and feature_value < split_value) or \
                        (operator == ">=" and feature_value >= split_value) or \
                        (operator == ">" and feature_value > split_value):
                        top5_rules.append(split_info)
                        break  # No need to check further if condition is satisfied

    # Return top 5 rules
    return Top5RulesResponse(top_5_rules=top5_rules)