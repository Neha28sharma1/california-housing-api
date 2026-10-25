### 1.1 Locate a pretrained model

For this project, I selected the pretrained **California Housing Price Predictor** model from Hugging Face.

- **Repository:** `shalev396/california-housing`
- **URL:** https://huggingface.co/shalev396/california-housing
- **Task:** Tabular Regression
- **Algorithm:** XGBoost
- **Dataset:** California Housing dataset based on the 1990 census

The model predicts the median house value of a California district using eight numerical housing features. I selected it because it is already trained, provides numerical predictions, and can be downloaded using the Hugging Face ecosystem.

The model is intended for historical district-level estimates rather than current individual property valuations.

## Installation and Setup
**Install dependencies:**
uv sync
**Running the Application**
- **Start the Ollama server if it is not already running:**
   ollama serve
- **Start the FastAPI backend:**
   uvicorn main:app --reload  
   The backend will normally be available at: API: http://127.0.0.1:8000
- **Start the Streamlit frontend:**
    streamlit run streamlit_main.py
    The frotend will normally be available at:http://127.0.0.1:8501
**The FastAPI backend and Ollama server must be running for the relevant frontend features to work.**

How our API works:

**Connecting Prediction and Explanation:** We use one FastAPI application in `main.py`, which contains two endpoints: `/predict` and `/prediction_explanation`. The `/prediction_explanation` endpoint calls the pretrained Hugging Face model to generate a housing price prediction. It then calls the `explain_prediction()` function from `ollama_service.py`, which communicates with Ollama's existing REST API using `requests.post()`. Finally, FastAPI returns the predicted house value and the generated explanation as JSON to the Streamlit client.

How the components communicate in our project:
### How the Components Communicate in Our Project

1. **Hugging Face Model:** First, we download a pretrained housing price prediction model from Hugging Face and load it into our Python application.

2. **FastAPI Backend:** We create a FastAPI server in `main.py` with a `/predict` endpoint to handle housing price predictions.

3. **Streamlit Client:** Streamlit acts as the frontend client. When the user enters housing information and clicks Predict, Streamlit sends an HTTP POST request to the FastAPI server. FastAPI then calls the loaded Hugging Face model using a normal Python function to generate the prediction.

4. **Ollama Integration:** Ollama has its own API for generating text. In `ollama_service.py`, we use `requests.post()` to send the housing features and predicted price to Ollama. The HTTP response is stored in the `response` variable, and we extract the generated explanation using `response.json()["response"]`.

5. **Connecting Prediction and Explanation:** In `main.py`, we add another FastAPI endpoint called `/prediction_explanation`. This endpoint calls the Hugging Face model to predict the price and then calls the function in `ollama_service.py` to generate an explanation.

6. **Returning Results:** Finally, FastAPI returns the predicted house value and Ollama-generated explanation as JSON. Streamlit receives this response and displays the results to the user.

## Which AI tools did you use?

I used ChatGPT as an AI programming assistant.I used it to explain unfamiliar Python concepts, help structure the FastAPI and Streamlit components, understand testing techniques, and investigate errors encountered during development like locating a pretrained Hugging Face model,how to write pytest tests for the model,explain monkeypatch.setattr()
AI-generated suggestions were not always correct on the first attempt and needed to be checked against the actual project. Issues involved like communication between the frontend and backend.