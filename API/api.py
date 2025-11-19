from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from video_predict import video_predict
import dotenv
import os
import aiohttp  # For making asynchronous HTTP requests
import asyncio
import platform
import argparse
from contextlib import asynccontextmanager
import torch
from gemini import glosses_to_text, custom_prompt

# Add project root to sys.path for module import
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prediction.predict import load_model

dotenv.load_dotenv(dotenv.find_dotenv())

from fastapi.middleware.cors import CORSMiddleware

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model
model = None
gloss_dict = None
device = None

def get_model_args():
    # Set paths based on system
    if platform.system() == 'Darwin':  # Mac
        base_dir = '/Users/defeee/Documents/GitHub/SignAI-SFS'
        config_path = os.path.join(base_dir, 'configs/phoenix2014-T.yaml')
        dict_path = os.path.join(base_dir, 'preprocess/phoenix2014-T/gloss_dict.npy')
        weights_path = os.path.join(base_dir, 'best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt')
    else:  # Linux or other
        config_path = './configs/phoenix2014-T.yaml'
        dict_path = './preprocess/phoenix2014-T/gloss_dict.npy'
        weights_path = 'best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
        
    # Determine device
    device_str = 'cpu'
    if torch.cuda.is_available():
        device_str = 'cuda:0'
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_str = 'mps'
        
    # Set number of threads for CPU inference
    if device_str == 'cpu':
        # On ARM/Mac, using too many threads can hurt performance due to overhead
        # 4 is usually a sweet spot for M1/M2/M3 chips
        torch.set_num_threads(4)
        logger.info("Set torch num threads to 4 for CPU optimization")

    args = argparse.Namespace(
        config=config_path,
        dict_path=dict_path,
        weights=weights_path,
        device=device_str
    )
    return args

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
    global model
    del model
    del gloss_dict
    del device
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API. Use /predict?video_url=<url> to get predictions."}

async def run_prediction_async(video_url: str):
    """Run prediction in a separate thread to avoid blocking the event loop."""
    # Ensure model is loaded (lazy loading fallback)
    if model is None:
        logger.info("Model not loaded, attempting lazy load...")
        load_model_global()
        
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        lambda: video_predict(
            video_url,
            model_instance=model,
            gloss_dict_instance=gloss_dict,
            device_instance=device
        )
    )

@app.get("/predict")
async def predict(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict_gemini")
async def predict_gemini(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        logger.info(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        # Prepare the prompt for Gemini API

        gemini_response = await glosses_to_text(prediction)

        logger.info(f"Done calling Gemini API")

        if gemini_response:
            translation = gemini_response
            gemini_summary = await custom_prompt(f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            if gemini_summary:
                summary = gemini_summary
            return JSONResponse(content={"translation": translation,
                                         "summary": summary if 'summary' in locals() else "No summary generated"
                                         })
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"}, status_code=500)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/v2/slowfast/predict_gemini')
async def predict_gemini_v2(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        prediction = await run_prediction_async(video_url)

        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        logger.info(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        # Prepare the prompt for Gemini API

        gemini_response = await glosses_to_text(prediction)

        logger.info(f"Done calling Gemini API")

        if gemini_response:
            translation = gemini_response
            gemini_summary = await custom_prompt(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            if gemini_summary:
                summary = gemini_summary
            return JSONResponse(content={"translation": translation,
                                         "summary": summary if 'summary' in locals() else "No summary generated"
                                         })
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"},
                                status_code=500)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/v2/slowfast/predict")
async def predict_v2(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# forward request to localhost 8000
@app.get("/v2/keypoints/predict")
def predict_v2_keypoints(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        import requests
        response = requests.get("http://localhost:8000/predict", params={"video_url": video_url})
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            return JSONResponse(content={"error": "Failed to get prediction from keypoints API"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Remote prediction endpoints
REMOTE_API_URL = "http://100.71.0.60:8082"

@app.get("/v2/GPU/predict")
async def remote_predict(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REMOTE_API_URL}/predict", params={"video_url": video_url}) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Remote API error: {response.status} - {error_text}")
                    return JSONResponse(content={"error": f"Remote API error: {response.status}"}, status_code=response.status)
    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/v2/GPU/predict_gemini")
async def remote_predict_gemini(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        # 1. Get prediction from remote API
        prediction_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REMOTE_API_URL}/predict", params={"video_url": video_url}) as response:
                if response.status == 200:
                    data = await response.json()
                    prediction_text = data.get("prediction", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Remote API error: {response.status} - {error_text}")
                    return JSONResponse(content={"error": f"Remote API error: {response.status}"}, status_code=response.status)
        
        if not prediction_text:
            return JSONResponse(content={"error": "No prediction received from remote API"}, status_code=400)

        logger.info(f"Remote model made prediction: {prediction_text}. This will be sent to local Gemini API for translation.")

        gemini_response = await glosses_to_text(prediction_text)

        if gemini_response:
            translation = gemini_response
            gemini_summary = await custom_prompt(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            if gemini_summary:
                summary = gemini_summary
            return JSONResponse(content={"translation": translation,
                                         "summary": summary if 'summary' in locals() else "No summary generated"
                                         })
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API"}, status_code=500)

    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
