from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import aiohttp
import logging
import asyncio
from .prediction import video_predict
from .gemini import process_video_translation

logger = logging.getLogger(__name__)

router = APIRouter()

# Remote prediction endpoints
REMOTE_API_URL = "http://100.71.0.60:8082"
REMOTE_2_API_URL = "http://100.102.136.67:8082"

async def run_prediction_async(video_url: str):
    """Run prediction in a separate thread to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, 
        lambda: video_predict(video_url)
    )

@router.get("/v2/keypoints/predict")
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

@router.get("/v3/GPU/predict")
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

@router.get("/v3/GPU")
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

        result = await process_video_translation(prediction_text)
        
        if result:
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API"}, status_code=500)

    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/v3/NPU")
async def remote_predict_gemini2(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        # 1. Get prediction from remote API
        prediction_text = ""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REMOTE_2_API_URL}/predict", params={"video_url": video_url}) as response:
                if response.status == 200:
                    data = await response.json()
                    prediction_text = data.get("prediction", "")
                else:
                    error_text = await response.text()
                    logger.error(f"Remote API error: {response.status} - {error_text}")
                    return JSONResponse(content={"error": f"Remote API error: {response.status}"},
                                        status_code=response.status)

        if not prediction_text:
            return JSONResponse(content={"error": "No prediction received from remote API"}, status_code=400)

        logger.info(
            f"Remote model made prediction: {prediction_text}. This will be sent to local Gemini API for translation.")

        result = await process_video_translation(prediction_text)
        
        if result:
            return JSONResponse(content=result)
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API"}, status_code=500)

    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
