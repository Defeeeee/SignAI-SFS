from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import aiohttp
import logging
from api_core.gemini import process_video_translation
from api_core.schemas import PredictionResponse, TranslationResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Remote prediction endpoints
REMOTE_API_URL = "http://100.71.0.60:8082"
REMOTE_2_API_URL = "http://100.102.136.67:8082"

@router.get("/GPU/predict", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
async def remote_predict(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{REMOTE_API_URL}/predict", params={"video_url": video_url}) as response:
                if response.status == 200:
                    data = await response.json()
                    return PredictionResponse(**data)
                else:
                    error_text = await response.text()
                    logger.error(f"Remote API error: {response.status} - {error_text}")
                    raise HTTPException(status_code=response.status, detail=f"Remote API error: {response.status}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/GPU", response_model=TranslationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
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
                    raise HTTPException(status_code=response.status, detail=f"Remote API error: {response.status}")
        
        if not prediction_text:
            raise HTTPException(status_code=400, detail="No prediction received from remote API")

        logger.info(f"Remote model made prediction: {prediction_text}. This will be sent to local Gemini API for translation.")

        result = await process_video_translation(prediction_text)
        
        if result:
            return TranslationResponse(**result)
        else:
            raise HTTPException(status_code=500, detail="Failed to get translation from Gemini API")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/NPU", response_model=TranslationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
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
                    raise HTTPException(status_code=response.status, detail=f"Remote API error: {response.status}")

        if not prediction_text:
            raise HTTPException(status_code=400, detail="No prediction received from remote API")

        logger.info(
            f"Remote model made prediction: {prediction_text}. This will be sent to local Gemini API for translation.")

        result = await process_video_translation(prediction_text)
        
        if result:
            return TranslationResponse(**result)
        else:
            raise HTTPException(status_code=500, detail="Failed to get translation from Gemini API")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Remote prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
