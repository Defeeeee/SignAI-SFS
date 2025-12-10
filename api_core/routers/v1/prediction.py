from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import logging
from api_core.prediction import run_prediction_async
from api_core.gemini import process_video_translation
from api_core.schemas import PredictionResponse, TranslationResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/predict", response_model=PredictionResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        return PredictionResponse(prediction=prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict_gemini", response_model=TranslationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_gemini(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            raise HTTPException(status_code=400, detail="No prediction made")

        logger.info(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        result = await process_video_translation(prediction)
        
        if result:
            return TranslationResponse(**result)
        else:
            raise HTTPException(status_code=500, detail="Failed to get translation from Gemini API, try /predict")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/predict_gemini_de", response_model=TranslationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def predict_gemini_de(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            raise HTTPException(status_code=400, detail="No prediction made")

        result = await process_video_translation(prediction, target_language="German")
        
        if result:
            return TranslationResponse(**result)
        else:
            raise HTTPException(status_code=500, detail="Failed to get translation from Gemini API, try /predict")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

