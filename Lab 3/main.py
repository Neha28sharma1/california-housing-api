from contextlib import asynccontextmanager
from pathlib import Path
from house_model import load_housing_model
from fastapi import FastAPI, Request

from pydantic import BaseModel, ConfigDict, Field

MODEL_NAME = "shalev396/california-housing"


# 2.2 Define the input schema
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


# Define the response schema
class PredictionResponse(BaseModel):
    predicted_price_usd: float
    model: str


# 2.4 Load the model once when the application starts
load_housing_model()


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
@app.post("/predict", response_model=PredictionResponse)
def predict_house_price(
    housing: HousingInput,
    request: Request,
):
    # Convert validated Pydantic input into a dictionary
    input_data = housing.model_dump()

    # Use the pretrained model
    predictor = request.app.state.predictor
    result = predictor.predict(input_data)

    # 2.5 Return the predicted price as JSON
    return PredictionResponse(
        predicted_price_usd=float(result["price_usd"]),
        model=result["model"],
    )
