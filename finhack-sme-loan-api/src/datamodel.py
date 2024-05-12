from pydantic import BaseModel


class UserData(BaseModel):
    TAN: str
    DIN: str
    PAN: str


class LoanApplication(BaseModel):
    AnnualRevenue: float
    NumberOfDirectors: int
    MissedPayments: int


# Define the input data model for getting approved percentage and interest rate
class ApprovedPercentageAndInterestRateRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float


# Define the output data model for getting approved percentage and interest rate
class ApprovedPercentageAndInterestRateResponse(BaseModel):
    approved_percentage: float
    interest_rate: float


# Define the output data model for loan approval prediction
class LoanApprovalResponse(BaseModel):
    prediction: int


# Define the input data model for loan approval prediction
class LoanApprovalRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float


# Define the input data model for getting top 5 rules
class Top5RulesRequest(BaseModel):
    TAN_ID: int
    Tenure: int
    Requested_Amount: float


# Define the output data model for getting top 5 rules
class Top5RulesResponse(BaseModel):
    top_5_rules: list
