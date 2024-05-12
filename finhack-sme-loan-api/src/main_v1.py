from firebase_admin import initialize_app, credentials, storage
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, ORJSONResponse, Response
from fastapi.security import HTTPBasicCredentials, HTTPBasic
from fastapi import FastAPI, Query, HTTPException, Depends, status, Path
import os
from loggers import info
from datamodel import UserData, LoanApplication

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_DIR = os.path.join(SCRIPT_DIR, "env", )
firebase_config = os.path.join(ENV_DIR, "angelhack-finhack-2024-firebase-adminsdk-gsrjl-841da46da0.json")

app = FastAPI(title="Loan Processing APIs", version="1.0.0")

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
        loan_amount = application.AnnualRevenue * 0.20
        return format_response("Loan Application Processed",
                               {"Loan Approved Amount": loan_amount}, status_code=200)
    else:
        return format_response("Loan Application Processed",
                               {"Loan Approved Amount": 0}, status_code=200)


# Function to read file from Firebase Storage


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

    uvicorn.run(app, port=int(os.environ.get("PORT", 8000)), host="0.0.0.0")
