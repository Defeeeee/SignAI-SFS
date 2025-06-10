from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from video_predict import video_predict

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API. Use /predict?video_url=<url> to get predictions."}

@app.get("/predict")
def predict(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        prediction = video_predict(
            video_url,
            weights='best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
        )
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)