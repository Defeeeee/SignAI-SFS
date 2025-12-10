import os
import aiohttp
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Google GenAI client if available
try:
    from google import genai
    client = genai.Client()
except ImportError:
    logger.warning("google.genai module not found. Using REST API fallback.")
    client = None
except Exception as e:
    logger.warning(f"Failed to initialize Google GenAI client: {e}")
    client = None

async def call_gemini_api_rest(prompt: str):
    """Calls the Gemini API using REST (useful if the library is not available or for async)."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return None
        
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {
        "Content-Type": "application/json"
    }
    request_body = {
      "contents": [
        {
          "parts": [
            {
              "text": prompt
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
                logger.error(f"Error calling Gemini API: {response.status}")
                logger.error(error_text)
                return None

def glosses_to_text_sync(glosses):
    """Synchronous version using the library (from original gemini.py)."""
    if not client:
        return None
        
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

    DGS Glosses to translate: {glosses}"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error in glosses_to_text_sync: {e}")
        return None

def custom_prompt_sync(prompt):
    """Synchronous version using the library (from original gemini.py)."""
    if not client:
        return None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error in custom_prompt_sync: {e}")
        return None

async def glosses_to_text_async(glosses, target_language="English"):
    """Asynchronous version using REST API."""
    if target_language == "German":
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

DGS Glosses to translate: {glosses}"""
    else:
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

DGS Glosses to translate: {glosses}"""

    response = await call_gemini_api_rest(prompt)
    if response and 'candidates' in response and response['candidates']:
        return response['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
    return None

async def custom_prompt_async(prompt):
    """Asynchronous version using REST API."""
    response = await call_gemini_api_rest(prompt)
    if response and 'candidates' in response and response['candidates']:
        return response['candidates'][0]['content']['parts'][0]['text'].rstrip('\n')
    return None
