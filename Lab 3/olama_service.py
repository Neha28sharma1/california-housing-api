from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import requests
from main import HousingInput

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"


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
    # #  response is in json format , convert the response to dict and return the "response" field
    return response.json()["response"]
