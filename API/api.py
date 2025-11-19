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

# Add project root to sys.path for module import
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prediction.predict import load_model

dotenv.load_dotenv(dotenv.find_dotenv())

from fastapi.middleware.cors import CORSMiddleware

# Global variables for model
model = None
gloss_dict = None
device = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    global model, gloss_dict, device
    print("Loading model on startup...")
    
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
        print("Set torch num threads to 4 for CPU optimization")

    args = argparse.Namespace(
        config=config_path,
        dict_path=dict_path,
        weights=weights_path,
        device=device_str
    )
    
    try:
        model, gloss_dict, device = load_model(args)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        
    yield
    
    # Clean up resources
    print("Shutting down...")
    del model
    del gloss_dict
    del device

async def call_gemini_api(prompt: str):
    """Calls the Gemini API with the given prompt."""
    api_key = os.getenv("GEMINI_API_KEY")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    request_body = {
      "contents": [
        {
          "parts": [
            {
              "text": prompt  # Use the provided prompt
            }
          ]
        }
      ]
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, headers=headers, json=request_body) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                print(f"Error calling Gemini API: {response.status}")
                print(error_text)
                return None

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sign Language Prediction API. Use /predict?video_url=<url> to get predictions."}

async def run_prediction_async(video_url: str):
    """Run prediction in a separate thread to avoid blocking the event loop."""
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
        if model is None:
            return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
            
        prediction = await run_prediction_async(video_url)
        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict_gemini")
async def predict_gemini(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        if model is None:
            return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
            
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        print(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        # Prepare the prompt for Gemini API
        prompt = f"""You are a specialized translator for German Sign Language (DGS) glosses to English.

Task: Translate the following DGS glosses into fluent, natural English.

Context: DGS glosses are written representations of sign language where:
- Words appear in their base form
- Grammar markers are often omitted
- Word order follows DGS syntax, not English syntax
- Special notation may be used (e.g., POSS for possessive)

Instructions:
1. Translate the meaning, not word-for-word
2. Use proper English grammar and sentence structure
3. Maintain the original meaning and intent
4. Return ONLY the translated English text, nothing else
5. Do not include explanations, notes, or any text besides the translation
6. Use complete, grammatically correct English sentences

DGS Glosses to translate: {prediction}"""
        gemini_response = await call_gemini_api(prompt)

        print(f"Done calling Gemini API")

        if gemini_response and 'candidates' in gemini_response and gemini_response['candidates']:
            translation = gemini_response['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
            gemini_summary = await call_gemini_api(f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            if gemini_summary and 'candidates' in gemini_summary and gemini_summary['candidates']:
                summary = gemini_summary['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
            return JSONResponse(content={"translation": translation,
                                         "summary": summary if 'summary' in locals() else "No summary generated"
                                         })
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/predict_gemini_de")
async def predict_gemini_de(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        if model is None:
            return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
            
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        # Prepare the prompt for Gemini API
        prompt = f"""You are a specialized translator for German Sign Language (DGS) glosses to German.

Task: Translate the following DGS glosses into fluent, natural German.

Context: DGS glosses are written representations of sign language where:
- Words appear in their base form
- Grammar markers are often omitted
- Word order follows DGS syntax, not German syntax
- Special notation may be used (e.g., POSS for possessive)

Instructions:
1. Translate the meaning, not word-for-word
2. Use proper German grammar and sentence structure
3. Maintain the original meaning and intent
4. Return ONLY the translated German text, nothing else
5. Do not include explanations, notes, or any text besides the translation
6. Use complete, grammatically correct German sentences

DGS Glosses to translate: {prediction}"""
        gemini_response = await call_gemini_api(prompt)

        if gemini_response and 'candidates' in gemini_response and gemini_response['candidates']:
            translation = gemini_response['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
            return JSONResponse(content={"translation": translation})
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"}, status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get('/v2/slowfast/predict_gemini')
async def predict_gemini_v2(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        if model is None:
            return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
            
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        print(f"Model made prediction: {prediction}. This will be sent to Gemini API for translation.")

        # Prepare the prompt for Gemini API
        prompt = f"""You are a specialized translator for German Sign Language (DGS) glosses to English.

        Task: Translate the following DGS glosses into fluent, natural English.

        Context: DGS glosses are written representations of sign language where:
        - Words appear in their base form
        - Grammar markers are often omitted
        - Word order follows DGS syntax, not English syntax
        - Special notation may be used (e.g., POSS for possessive)

        Instructions:
        1. Translate the meaning, not word-for-word
        2. Use proper English grammar and sentence structure
        3. Maintain the original meaning and intent
        4. Return ONLY the translated English text, nothing else
        5. Do not include explanations, notes, or any text besides the translation
        6. Use complete, grammatically correct English sentences

        DGS Glosses to translate: {prediction}"""
        gemini_response = await call_gemini_api(prompt)

        if gemini_response and 'candidates' in gemini_response and gemini_response['candidates']:
            translation = gemini_response['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
            gemini_summary = await call_gemini_api(
                f"""Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {translation}""")
            if gemini_summary and 'candidates' in gemini_summary and gemini_summary['candidates']:
                summary = gemini_summary['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
            return JSONResponse(content={"translation": translation,
                                         "summary": summary if 'summary' in locals() else "No summary generated"
                                         })
        else:
            return JSONResponse(content={"error": "Failed to get translation from Gemini API, try /predict"},
                                status_code=500)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
@app.get("/v2/slowfast/predict")
async def predict_v2(video_url: str = Query(..., description="URL of the video to predict")):
    try:
        if model is None:
            return JSONResponse(content={"error": "Model not loaded"}, status_code=503)
            
        prediction = await run_prediction_async(video_url)
        
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

        return JSONResponse(content={"prediction": prediction})
    except Exception as e:
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
