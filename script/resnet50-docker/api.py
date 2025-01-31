import onnxruntime as ort
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import google.cloud.logging
import logging

# Initialize the FastAPI app
api = FastAPI()

# Configure Google Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()  # Routes Python logs to Cloud Logging

# Create a custom logger
logger = logging.getLogger("onnx-api-logger")
logger.setLevel(logging.INFO)

# Load the ONNX model
onnx_model_path = "/home/arsenalducvy/Desktop/resnet50_model.onnx"  # Path to your ONNX model
try:
    session = ort.InferenceSession(onnx_model_path)
    logger.info("Model loaded successfully: %s", onnx_model_path)
except Exception as e:
    logger.error("Failed to load model: %s", e)
    raise

# Define the request body structure
class InputData(BaseModel):
    input: list

# Route for health check
@api.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {"status": "ok"}

# Route for inference
@api.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input list to a NumPy array
        input_data = np.array(data.input, dtype=np.float32)

        # Run the ONNX model
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_data})

        logger.info("Prediction successful. Input: %s | Output: %s", data.input, result[0].tolist())
        # Return the prediction result as JSON
        return {"prediction": result[0].tolist()}

    except Exception as e:
        logger.error("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Model inference failed")

# CORS middleware configuration
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
