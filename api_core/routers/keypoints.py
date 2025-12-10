from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

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
