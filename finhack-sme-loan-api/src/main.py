from firebase_admin import initialize_app, credentials, storage
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, ORJSONResponse, Response
from fastapi import FastAPI
# from fastapi import FastAPI, Query, HTTPException, Depends, status, Path
import os
import numpy as np
# import xgboost as xgb
import pickle
import pandas as pd
from loggers import info
from datamodel import (UserData, LoanApplication, LoanApprovalResponse, LoanApprovalRequest,
                       ApprovedPercentageAndInterestRateRequest, ApprovedPercentageAndInterestRateResponse,
                       Top5RulesRequest, Top5RulesResponse)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(SCRIPT_DIR, "env", )
firebase_config = os.path.join(ENV_DIR, "angelhack-finhack-2024-firebase-adminsdk-gsrjl-841da46da0.json")

app = FastAPI(title="CRONUS - SME Loan Processing APIs", version="1.0.0")

# Initialize Firebase App
# Use a service account.
cred = credentials.Certificate(firebase_config)  # todo
firebase_app = initialize_app(cred, {
    'storageBucket': 'angelhack-finhack-2024.appspot.com'
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://example.com",  # Replace with your actual origin
        "http://localhost:8000",  # Localhost with port 8000
        "http://127.0.0.1:8000",  # Also include 127.0.0.1 if needed,
        "http://localhost:3000",  # Localhost with port 8000
        "http://127.0.0.1:3000",  # Also include 127.0.0.1 if needed,
        "https://go2.video"
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Or specify methods like ["GET", "POST", "PUT", "DELETE"]
    allow_headers=["*"],  # Or specify headers
)


def read_pickle_file_from_path(pickle_path: str):
    try:
        # Create a file-like object from the Firebase Storage path
        bucket = storage.bucket()
        blob = bucket.blob(pickle_path)
        file_obj = blob.open("rb")  # Open in binary read mode

        # Load the pickle directly from the file-like object
        loaded_data = pickle.load(file_obj)

        file_obj.close()  # Close the file object
        return {"message": "Pickle loaded successfully", "data": loaded_data}

    except FileNotFoundError:
        return {"message": "Pickle file not found", "status": 404}

    except pickle.UnpicklingError:
        return {"message": "Error unpickling data", "status": 500}


def read_csv_from_storage(csv_path: str):
    try:
        # Create a file-like object from the Firebase Storage path
        bucket = storage.bucket()
        blob = bucket.blob(csv_path)
        file_obj = blob.open("rb")

        # Read the CSV directly from the file-like object into a DataFrame
        df = pd.read_csv(file_obj)

        # # Convert DataFrame to a list of dictionaries for easy serialization
        # data = df.to_dict(orient="records")

        file_obj.close()
        return {"message": "CSV file read successfully", "data": df}

    except FileNotFoundError:
        return {"message": "CSV file not found", "status": 404}

    except pd.errors.EmptyDataError:
        return {"message": "CSV file is empty", "status": 400}

    except Exception as e:
        # Catch general exceptions for unexpected errors
        return {"message": f"Error reading CSV file: {e}", "status": 500}


loan_approval_model = read_pickle_file_from_path('prediction_model.pkl').get('data', None)
approved_percentage_model = read_pickle_file_from_path('approved.pkl').get('data', None)
interest_rate_model = read_pickle_file_from_path('interest.pkl').get('data', None)
test_data = read_csv_from_storage('TEST_DATA.csv').get('data', None)


@app.get("/", response_class=HTMLResponse)
async def welcome_loan_api_app():
    html_content = f"""
    <p>
      <strong>Hello: </strong>Welcome to FinHack - Loan Processing APIs  
    </p>

    """
    return HTMLResponse(html_content)


@app.post("/validate_tan_pan_din")
def validate_user(user_data: UserData):
    # Implement your validation logic here
    info(f"Received user data: {user_data}")
    if len(user_data.TAN) == 10 and len(user_data.PAN) == 10 and len(user_data.DIN) == 8:
        return format_response("Success", {"valid_user": True, "mobile_number": "123-456-7890"}, status_code=200)
    else:
        return format_response("Invalid TAN, PAN, or DIN provided", {}, status_code=400)


@app.post("/process_loan_application")
def process_loan(application: LoanApplication):
    # Implement your loan approval logic here
    if application.AnnualRevenue > 500000 and application.MissedPayments < 2:
        return format_response("Loan Application Processed",
                               {"Loan Approval Status": "Approved"}, status_code=200)
    else:
        return format_response("Loan Application Processed",
                               {"Loan Approval Status": "Rejected"}, status_code=200)


@app.post("/loan_application_approved_amount")
def loan_amount(application: LoanApplication):
    # Implement your loan amount calculation logic here
    if application.AnnualRevenue > 500000 and application.MissedPayments < 2:
        # Example: Loan amount could be 20% of annual revenue
        loan_amount_val = application.AnnualRevenue * 0.20
        return format_response("Loan Application Processed",
                               {"Loan Approved Amount": loan_amount_val}, status_code=200)
    else:
        return format_response("Loan Application Processed",
                               {"Loan Approved Amount": 0}, status_code=200)


# Route to predict loan approval
@app.post("/predict_loan_approval/")
async def predict_loan_approval(request: LoanApprovalRequest):
    # Extract TAN ID, tenure, and requested amount
    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    info(f"Received TAN ID: {TAN_ID}, Tenure: {tenure}, Requested Amount: {requested_amount}")
    # Check if TAN ID exists in the test data
    if TAN_ID not in test_data['TAN_ID'].values:
        return format_response("Invalid TAN Number Provided", {
            "prediction": "Rejected",
            "approved_percentage": 0,
            "interest_rate": 0
        }, status_code=400)

    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row = tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
                       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
                       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
                       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
                       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
                       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
                       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
                       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested'] = requested_amount
    tan_row['Loan_Tenure_Months'] = tenure

    input_data = np.array(tan_row)
    data = {}
    # Make prediction
    prediction = loan_approval_model.predict([input_data])
    if prediction and prediction[0] == 1:
        prediction = "Approved"
        temp = get_approved_percentage_and_interest_rate(TAN_ID=TAN_ID, tenure=tenure,
                                                  requested_amount=requested_amount)
        data.update(temp)
        data['prediction'] = prediction
    else:
        prediction = "Rejected"
        data.update({
            "prediction": prediction,
            "approved_percentage": 0,
            "interest_rate": 0
        })
    info(f"Loan Approval Prediction: {data}")
    # return LoanApprovalResponse(prediction=prediction)
    return format_response("Loan Application Processed", data, status_code=200)


def get_approved_percentage_and_interest_rate(
        TAN_ID: int = 0, tenure: int = 0, requested_amount: float = 0
):
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row = tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
                       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
                       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
                       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
                       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
                       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
                       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
                       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested'] = requested_amount
    tan_row['Loan_Tenure_Months'] = tenure
    input_data = np.array(tan_row)

    # Make predictions for approved percentage and interest rate
    approved_percentage = round(approved_percentage_model.predict([input_data])[0], 2)
    interest_rate = round(interest_rate_model.predict([input_data])[0], 2)
    approved_amount = int(approved_percentage * requested_amount)

    info(f"Approved Amount: {approved_amount}, Interest Rate: {interest_rate}")

    return {
        "approved_amount": approved_amount,
        "interest_rate": interest_rate
    }


# Route to get approved percentage and interest rate
# @app.post("/get_loan_amount_and_interest_rate/")
async def get_approved_percentage_and_interest_rate_route(request: ApprovedPercentageAndInterestRateRequest):
    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row = tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
                       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
                       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
                       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
                       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
                       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
                       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
                       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested'] = requested_amount
    tan_row['Loan_Tenure_Months'] = tenure
    input_data = np.array(tan_row)

    # Make predictions for approved percentage and interest rate
    approved_percentage = approved_percentage_model.predict([input_data])[0]
    interest_rate = interest_rate_model.predict([input_data])[0]

    # Return approved percentage and interest rate
    return ApprovedPercentageAndInterestRateResponse(approved_percentage=approved_percentage,
                                                     interest_rate=interest_rate)


# Route to get top 5 rules
# @app.post("/get_top_5_rules/")
async def get_top_5_rules(request: Top5RulesRequest):
    # Assuming your XGBoost model is named 'model'

    # Define the test row (you can modify this based on your use case)

    # Extract TAN ID, tenure, and requested amount
    TAN_ID = request.TAN_ID
    tenure = request.Tenure
    requested_amount = request.Requested_Amount
    # Fetch other features from test data based on TAN ID
    tan_row = test_data[test_data['TAN_ID'] == TAN_ID].iloc[0]
    tan_row = tan_row[['Industry_Type', 'Annual_Sales_(Revenue)', 'Net_Income', 'Total_Assets',
                       'Current_Ratio', 'Quick_Ratio', 'Debt_to_Equity_Ratio',
                       'Return_on_Assets_(ROA)', 'Return_on_Equity_(ROE)',
                       'Gross_Profit_Margin', 'Operating_Profit_Margin', 'Net_Profit_Margin',
                       'Total_Liabilities', 'Total_Shareholder_Equity', 'Number_of_Employees',
                       'Age_of_the_Company', 'UrbanORRural', 'NewExist', 'Credit_Score',
                       'Number_of_Existing_Loans', 'Percentage_of_On-time_Payments',
                       'Number_of_Missed_or_Late_Payments']]
    tan_row['Loan_Amount_Requested'] = requested_amount
    tan_row['Loan_Tenure_Months'] = tenure
    test_row = tan_row
    input_data = np.array(tan_row)
    model = loan_approval_model
    # Get the prediction probabilities from your model for the selected row
    prediction_probs = model.predict_proba([test_row])

    predicted_class = prediction_probs.argmax(axis=1)

    # Get the booster of the model
    booster = model.get_booster()

    # Get the indices of the top 5 most important features
    importance_scores = model.feature_importances_
    top5_indices = importance_scores.argsort()[-5:]

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
                operator = None
                for part in parts:
                    if '<=' in part:
                        operator = '<='
                    elif '<' in part:
                        operator = '<'
                    elif '>=' in part:
                        operator = '>='
                    elif '>' in part:
                        operator = '>'

                    split_value = float(parts[0][parts[0].index(operator) + 1:-1])

                    # Check if the split condition is satisfied
                    if (operator == "<=" and feature_value <= split_value) or \
                            (operator == "<" and feature_value < split_value) or \
                            (operator == ">=" and feature_value >= split_value) or \
                            (operator == ">" and feature_value > split_value):
                        top5_rules.append(split_info)
                        break  # No need to check further if condition is satisfied

    # Return top 5 rules
    return Top5RulesResponse(top_5_rules=top5_rules)


# FastAPI route to display content of a file
@app.get("/read-firebase-file/")
async def read_firebase_file():
    try:
        file_location = 'model.txt'
        file_content = read_file_from_storage(file_location)
        return Response(content=file_content, media_type="text/plain")

    except Exception as e:
        return format_response(str(e), {}, status_code=400)


# FastAPI route to display content of a file
@app.get("/read-firebase-file/")
async def read_firebase_file():
    try:
        file_location = 'model.txt'
        file_content = read_file_from_storage(file_location)
        return Response(content=file_content, media_type="text/plain")

    except Exception as e:
        return format_response(str(e), {}, status_code=400)


def format_response(message: str, data: dict, status_code: int):
    json_compatible_item_data = jsonable_encoder(data)
    info(f"JSON Response with message: {message} data: {data} status code: {status_code}")
    return ORJSONResponse(content={"message": message, "data": json_compatible_item_data}, status_code=status_code)


def read_file_from_storage(blob_name):
    # Create a bucket object for your storage bucket
    bucket = storage.bucket()

    # Create a blob object from the filepath
    blob = bucket.blob(blob_name)

    # Download the file as bytes
    file_bytes = blob.download_as_bytes()
    return file_bytes.decode('utf-8')


if __name__ == "__main__":
    import uvicorn

    print(loan_approval_model)
    print(approved_percentage_model)
    print(interest_rate_model)
    print(test_data.head())

    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
