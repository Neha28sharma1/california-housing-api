from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from main import HousingInput

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"
app = FastAPI()


def explain_prediction(housing_data, predicted_price):
    prompt = f"""
    You are explaining a California housing price prediction
    to someone without a technical background.

    Housing features:
    {housing_data}

    Predicted median house value: ${predicted_price:,.2f}

    Explain the prediction in simple language.
    Discuss how the housing features might relate to
    the estimated value.

    Do not claim to know the exact reasons behind
    the machine learning model's prediction.
    Keep the explanation short and clear.
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    # change this to raise an exception if the request fails
    response.raise_for_status()
    # return the explanation text from the response , convert the response to json and return the "response" field
    return response.json()["response"]


# response model


class ExplanationResponse(BaseModel):
    predicted_price_usd: float
    model: str
    explanation: str


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
        input_data = housing.model_dump()

        # Predict using the Hugging Face model
        # Convert validated Pydantic input into a dictionary
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
