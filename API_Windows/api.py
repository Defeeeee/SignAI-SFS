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

app = create_app(title="Sign Language Prediction API (Windows/GPU)")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API (Windows/GPU). Use /predict?video_url=<url> to get predictions."}

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
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        logger.info(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        result = await process_video_translation(prediction)
        
        if result:
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"}, status_code=500)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict_gemini_de")
async def predict_gemini_de(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        result = await process_video_translation(prediction, target_language="German")
        
        if result:
            # The original endpoint only returned translation, not summary
            return JSONResponse(content={"translation": result["translation"]})
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"}, status_code=500)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/v2/slowfast/predict_gemini')
async def predict_gemini_v2(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        logger.info(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        result = await process_video_translation(prediction)
        
        if result:
            return JSONResponse(content=result)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
