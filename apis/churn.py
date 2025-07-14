import joblib
import numpy as np

def predictChurn(MonthlyCharges, TotalCharges, InternetService, tenure, Contract):
    # Dictionaries to map strings to numerical values
    internet_services = {"Fiber optic": 0, "DSL": 1, "No": 2}
    contracts = {"Month-to-month": 1, "One year": 12, "Two year": 24}
    
    # Convert string inputs to corresponding numerical values
    internet_service = internet_services[InternetService]
    contract = contracts[Contract]
    
    # Load the saved model
    model = joblib.load('models/churn.joblib')
    
    # Make predictions using the model
    prediction = model.predict([[MonthlyCharges, TotalCharges, internet_service, tenure, contract]])[0]
    
    # Convert numpy.int64 to Python int
    return int(prediction)
