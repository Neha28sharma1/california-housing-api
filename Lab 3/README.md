### 1.1 Locate a pretrained model

For this project, I selected the pretrained **California Housing Price Predictor** model from Hugging Face.

- **Repository:** `shalev396/california-housing`
- **URL:** https://huggingface.co/shalev396/california-housing
- **Task:** Tabular Regression
- **Algorithm:** XGBoost
- **Dataset:** California Housing dataset based on the 1990 census

The model predicts the median house value of a California district using eight numerical housing features. I selected it because it is already trained, provides numerical predictions, and can be downloaded using the Hugging Face ecosystem.

The model is intended for historical district-level estimates rather than current individual property valuations.

uv sync
uvicorn main:app --reload  to run the server and test the endpoints.
http://127.0.0.1:8000/docs to access the Swagger UI for testing the API endpoints.
ollama pull llama3.2
