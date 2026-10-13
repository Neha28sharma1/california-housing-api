import importlib.util
import math
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from house_model import load_housing_model
from huggingface_hub import snapshot_download

from main import app


# Load the pretrained model once for all tests
@pytest.fixture(scope="module")
def predictor():
    return load_housing_model()


@pytest.fixture
def housing_data():
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
    """Check that the pretrained model loads."""
    assert predictor is not None
    assert callable(predictor.predict)


def test_prediction_structure(predictor, housing_data):
    """Check the prediction response format."""
    result = predictor.predict(housing_data)

    assert isinstance(result, dict)
    assert "price_usd" in result
    assert "model" in result
    assert result["model"] == "xgboost"


def test_prediction_is_positive(predictor, housing_data):
    """Check that the predicted price is valid."""
    result = predictor.predict(housing_data)

    price = result["price_usd"]

    assert isinstance(price, (int, float))
    assert math.isfinite(price)
    assert price > 0


def test_known_prediction(predictor, housing_data):
    """Check the known Hugging Face example."""
    result = predictor.predict(housing_data)

    assert result["price_usd"] == pytest.approx(368791.17, abs=100)


def test_same_input_same_prediction(predictor, housing_data):
    """Check prediction consistency."""
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
    """Check predictions for different housing districts."""
    data = housing_data.copy()

    data["MedInc"] = income
    data["HouseAge"] = age
    data["AveRooms"] = rooms
    data["Latitude"] = latitude
    data["Longitude"] = longitude

    result = predictor.predict(data)

    price = result["price_usd"]

    assert math.isfinite(price)
    assert 0 < price < 1_000_000


def test_income_changes_prediction(predictor, housing_data):
    """Check model sensitivity to median income."""
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


client = TestClient(app)


def test_get_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_checkpost():

    housingInput = {
        "MedInc": 4.445,
        "HouseAge": 52,
        "AveRooms": 5.5346,
        "AveBedrms": 1.1509,
        "Population": 742,
        "AveOccup": 2.3333,
        "Latitude": 37.77,
        "Longitude": -122.43,
    }
    with TestClient(app) as client:
        response = client.post("/predict", json=housingInput)

    result = response.json()

    assert response.status_code == 200

    assert result["predicted_price_usd"] == pytest.approx(368791.17, abs=100)

    assert isinstance(result["predicted_price_usd"], float)
