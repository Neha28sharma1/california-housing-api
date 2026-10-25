from contextlib import asynccontextmanager
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from house_model import load_housing_model
from olama_service import explain_prediction
from pydantic import BaseModel, ConfigDict, Field

MODEL_NAME = "shalev396/california-housing"


# 2.2 Define the input schema for user input validation
class HousingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    MedInc: float = Field(gt=0, description="Median income in $10,000 units")
    HouseAge: float = Field(ge=0, description="Median housing age")
    AveRooms: float = Field(gt=0, description="Average rooms per household")
    AveBedrms: float = Field(ge=0, description="Average bedrooms per household")
    Population: float = Field(gt=0, description="District population")
    AveOccup: float = Field(gt=0, description="Average occupants per household")
    Latitude: float = Field(ge=32, le=42, description="California latitude")
    Longitude: float = Field(ge=-125, le=-114, description="California longitude")


# Define the response schema for the prediction endpoint
class PredictionResponse(BaseModel):
    predicted_price_usd: float
    model: str


# Load the Model Once When the App Starts and Store It in the App State
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading Hugging Face model...")
    # stores model state in the app's state object, which can be accessed when handling requests instead of loading the model for each request
    app.state.predictor = load_housing_model()
    print("Model loaded successfully!")

    yield

    app.state.predictor = None


# 2.1 Create FastAPI application
app = FastAPI(
    title="California Housing Price Prediction API",
    description="Predict median housing values using a pretrained Hugging Face XGBoost model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok"}


# def get_check():
#     assert status == "ok"


# 2.3 Implement POST /predict
# error handling to the predict endpoint to return a 500 error if the prediction fails
@app.post("/predict", response_model=PredictionResponse)
def predict_house_price(
    housing: HousingInput,
    request: Request,
):
    try:
        input_data = housing.model_dump()

        predictor = request.app.state.predictor
        result = predictor.predict(input_data)

        return PredictionResponse(
            predicted_price_usd=float(result["price_usd"]),
            model=result["model"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a housing price prediction.",
        ) from exc


# response model and endpoint for prediction explanation


class ExplanationResponse(BaseModel):
    predicted_price_usd: float
    model: str
    explanation: str


#####################3need to fix the error handling
@app.post(
    "/prediction_explanation",
    response_model=ExplanationResponse,
)
def prediction_explanation(
    housing: HousingInput,
    request: Request,
):
    try:
        # Get the housing inputs
        # Convert validated Pydantic input into a dictionary
        input_data = housing.model_dump()

        # Predict using the Hugging Face model

        predictor = request.app.state.predictor
        result = predictor.predict(input_data)

        predicted_price = float(result["price_usd"])

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate a housing price prediction.",
        ) from exc

    try:
        # Ask Ollama to explain the prediction
        explanation = explain_prediction(
            input_data,
            predicted_price,
        )
    # Handle Ollama connection errors
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=503,
            detail="The explanation service is unavailable. Please check that Ollama is running.",
        ) from exc
    # Return the final responses
    return ExplanationResponse(
        predicted_price_usd=predicted_price,
        model=result["model"],
        explanation=explanation,
    )
