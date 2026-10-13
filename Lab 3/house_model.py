import importlib.util
from pathlib import Path

from importlib import import_module

MODEL_NAME = "shalev396/california-housing"


def load_housing_model():
    # Download model from Hugging Face
    snapshot_download = import_module("huggingface_hub").snapshot_download
    model_path = snapshot_download(repo_id=MODEL_NAME)

    # Locate the model.py file
    model_file = Path(model_path) / "model.py"

    # Import the downloaded model dynamically
    spec = importlib.util.spec_from_file_location("hf_housing_model", model_file)

    if spec is None or spec.loader is None:
        raise ImportError("Could not load Hugging Face model")

    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)

    # Load pretrained model
    predictor = model_module.load(model_path, device="cpu")

    return predictor
