from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
import logging
from api_core.schemas import ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/keypoints/predict", responses={500: {"model": ErrorResponse}})
def predict_v2_keypoints(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        import requests
        response = requests.get("http://localhost:8000/predict", params={"video_url": video_url})
        if response.status_code == 200:
            return JSONResponse(content=response.json())
        else:
            raise HTTPException(status_code=500, detail="Failed to get prediction from keypoints API")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
