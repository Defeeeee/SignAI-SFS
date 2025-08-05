from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from video_predict import video_predict
import dotenv
import os
import aiohttp  # For making asynchronous HTTP requests

dotenv.load_dotenv(dotenv.find_dotenv())

from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=['*'],
    allow_credentials=True, allow_methods=['*'], allow_headers=['*'])

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

@app.get("/predict_gemini")
async def predict_gemini(video_url: str = Query(..., description="URL of the video to predict")):
    # run model prediction and then call Gemini API with the glosses and tell it to generate a natural language translation
    # the result from gemini api will be returned and its expected to be a glosses --> text translation
    try:
        prediction = video_predict(
            video_url,
            weights='best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
        )
        if not prediction:
            return JSONResponse(content={"error": "No prediction made"}, status_code=400)

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
        prediction = video_predict(
            video_url,
            weights='best_checkpoints/phoenix2014-T_dev_17.66_test_18.71.pt'
        )
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8082)
