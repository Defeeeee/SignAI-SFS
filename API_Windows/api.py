from fastapi import Query
from fastapi.responses import JSONResponse
import dotenv
import os
import logging
import sys

# Add project root to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_core.app import create_app
from api_core.prediction import run_prediction_async
from api_core.gemini import process_video_translation

dotenv.load_dotenv(dotenv.find_dotenv())

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Only load the prediction router for Windows/GPU API
app = create_app(title="Sign Language Prediction API (Windows/GPU)", include_routers=["prediction"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API (Windows/GPU). Use /predict?video_url=<url> to get predictions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
