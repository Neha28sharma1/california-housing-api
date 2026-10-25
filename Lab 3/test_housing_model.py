import math
import pytest
from fastapi.testclient import TestClient
from house_model import load_housing_model

from main import app


# Load the pretrained model once for all tests
@pytest.fixture(scope="module")
def predictor():
    """
    Load the pretrained housing prediction model for testing.

    Returns:
        object: The loaded model with a predict() method.
    """
    return load_housing_model()


@pytest.fixture
def housing_data():
    """
    Provide sample housing features for model tests.

    Returns:
        dict: A dictionary containing the eight features
        required by the California housing model.
    """
    return {
        "MedInc": 4.445,
        "HouseAge": 52,
        "AveRooms": 5.5346,
        "AveBedrms": 1.1509,
        "Population": 742,
        "AveOccup": 2.3333,
        "Latitude": 37.77,
        "Longitude": -122.43,
    }


def test_model_loads(predictor):
    """
    Verify that the model loads and exposes a predict() method.
    """
    assert predictor is not None
    assert callable(predictor.predict)


def test_prediction_structure(predictor, housing_data):
    """
    Verify that predictions contain the expected dictionary fields.
    """
    # Generate a prediction using the sample housing features.
    result = predictor.predict(housing_data)

    assert isinstance(result, dict)
    assert "price_usd" in result
    assert "model" in result
    assert result["model"] == "xgboost"


def test_prediction_is_positive(predictor, housing_data):
    """ 
    Verify that the predicted price is finite and positive.
    """
    result = predictor.predict(housing_data)

    price = result["price_usd"]

    assert isinstance(price, (int, float))
    assert math.isfinite(price)
    assert price > 0


def test_known_prediction(predictor, housing_data):
    """
    Compare the prediction with the documented Hugging Face example.
    """
    result = predictor.predict(housing_data)

    assert result["price_usd"] == pytest.approx(368791.17, abs=100)


def test_same_input_same_prediction(predictor, housing_data):
    """
    Check prediction consistency.
    """
    result1 = predictor.predict(housing_data)
    result2 = predictor.predict(housing_data)

    assert result1["price_usd"] == pytest.approx(result2["price_usd"], abs=0.01)


# Checks three housing examples
@pytest.mark.parametrize(
    "income, age, rooms, latitude, longitude",
    [
        (2.5, 25, 4.5, 34.05, -118.24),
        (4.5, 35, 5.5, 37.77, -122.43),
        (7.5, 20, 6.5, 37.33, -121.89),
    ],
)
def test_different_housing_inputs(
    predictor,
    housing_data,
    income,
    age,
    rooms,
    latitude,
    longitude,
):
    """
    Check predictions for different housing districts.
    """
    data = housing_data.copy()

    data["MedInc"] = income
    data["HouseAge"] = age
    data["AveRooms"] = rooms
    data["Latitude"] = latitude
    data["Longitude"] = longitude

    # Generate a prediction for this housing district.
    result = predictor.predict(data)
    price = result["price_usd"]

    assert math.isfinite(price)
    assert 0 < price < 1_000_000


def test_income_changes_prediction(predictor, housing_data):
    """
    Check model sensitivity to median income.
    """
    # Create independent copies for the low- and high-income cases.
    low_income = housing_data.copy()
    high_income = housing_data.copy()

    low_income["MedInc"] = 2.0
    high_income["MedInc"] = 8.0

    low_price = predictor.predict(low_income)["price_usd"]
    high_price = predictor.predict(high_income)["price_usd"]

    assert math.isfinite(low_price)
    assert math.isfinite(high_price)
    assert low_price != pytest.approx(high_price, abs=0.01)


#  //////////////////////////////////////////////API test//////////////////////////////////////////////

# Create a TestClient to send HTTP requests directly to the API.
@pytest.fixture(scope="module")
def client():
    """
    Start the FastAPI application for the API tests.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_checkpost(client,housing_data):
    """
    Verify that POST /predict accepts valid housing data
    and returns the expected predicted price.

    Expected:
        HTTP 200 and a predicted price close to the
        documented reference value.
    """
    response = client.post("/predict", json=housing_data)
    result = response.json()

    # Verify that the API request succeeded.
    assert response.status_code == 200

    assert result["predicted_price_usd"] == pytest.approx(368791.17, abs=100)

    assert isinstance(result["predicted_price_usd"], float)


def test_prediction_explanation(client, housing_data, monkeypatch):
    """
    Verify that the explanation endpoint returns both the predicted
    price and the AI-generated explanation.
    """

    # Mock the Ollama explanation service so the test does not
    # require Ollama to be running.
    expected_explanation = (
        "The estimated value reflects the district's income, "
        "housing characteristics, and location."
    )

    def fake_explain_prediction(housing_data, predicted_price):
        """
        Return predictable explanation text for testing.
        """
        assert isinstance(housing_data, dict)
        assert isinstance(predicted_price, float)
        return expected_explanation

    # Replace the real explanation function used by main.py.
    monkeypatch.setattr(
        "main.explain_prediction",
        fake_explain_prediction,
    )

    # Send housing features to the prediction-and-explanation endpoint.
    response = client.post(
        "/prediction_explanation",
        json=housing_data,
    )

    # Check that the API request succeeded.
    assert response.status_code == 200

    result = response.json()

    # Verify that both required response fields are present.
    assert "predicted_price_usd" in result
    assert "explanation" in result

    # Verify that the predicted price is a valid, positive number.
    price = result["predicted_price_usd"]

    assert isinstance(price, (int, float))
    assert math.isfinite(price)
    assert price > 0

    # Verify that the explanation matches the mocked response.
    assert result["explanation"] == expected_explanation
