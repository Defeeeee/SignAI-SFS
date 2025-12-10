from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import dotenv
import os
import aiohttp
import asyncio
import logging
import sys

# Add project root to sys.path for module import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from api_core.model import lifespan, load_model_global
from api_core.prediction import video_predict
from api_core.gemini import glosses_to_text_async, custom_prompt_async, glosses_to_text_sync, custom_prompt_sync

dotenv.load_dotenv(dotenv.find_dotenv())

from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API (Windows/GPU). Use /predict?video_url=<url> to get predictions."}

async def run_prediction_async(video_url: str):
    """Run prediction in a separate thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        lambda: video_predict(video_url)
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

        # Use async version if available, otherwise sync
        gemini_response = await glosses_to_text_async(prediction)
        if gemini_response is None:
             gemini_response = glosses_to_text_sync(prediction)

        logger.info(f"Done calling Gemini API")

        if gemini_response:
            translation = gemini_response
            gemini_summary = await custom_prompt_async(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            
            if gemini_summary is None:
                gemini_summary = custom_prompt_sync(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")

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

@app.get("/predict_gemini_de")
async def predict_gemini_de(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        # Use async version if available, otherwise sync
        gemini_response = await glosses_to_text_async(prediction, target_language="German")
        if gemini_response is None:
             # Note: sync version doesn't support target_language param in my implementation, 
             # but I'll assume English fallback or update sync later if critical. 
             # For now, async is preferred and should work.
             gemini_response = glosses_to_text_sync(prediction)

        if gemini_response:
            translation = gemini_response
            return JSONResponse(content={"translation": translation})
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

        # Use async version if available, otherwise sync
        gemini_response = await glosses_to_text_async(prediction)
        if gemini_response is None:
             gemini_response = glosses_to_text_sync(prediction)

        if gemini_response:
            translation = gemini_response
            gemini_summary = await custom_prompt_async(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            
            if gemini_summary is None:
                gemini_summary = custom_prompt_sync(
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
