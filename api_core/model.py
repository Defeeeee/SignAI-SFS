import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from .config import get_model_args
import sys
import os

# Add project root to sys.path for module import if not already there
# This is needed because prediction.predict is in the root
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from prediction.predict import load_model

logger = logging.getLogger(__name__)

# Global variables for model
model = None
gloss_dict = None
device = None

def load_model_global():
    global model, gloss_dict, device
    if model is not None:
        return
        
    logger.info("Loading model...")
    try:
        args = get_model_args()
        model, gloss_dict, device = load_model(args)
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

def get_model_components():
    """Returns the global model components."""
    return model, gloss_dict, device

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    try:
        load_model_global()
    except Exception as e:
        logger.error(f"Startup model loading failed: {e}")
        # We don't raise here to allow the app to start, 
        # requests will try to load the model again and fail then if it persists.
        
    yield
    
    # Clean up resources
    logger.info("Shutting down...")
    global model, gloss_dict, device
    model = None
    gloss_dict = None
    device = None
