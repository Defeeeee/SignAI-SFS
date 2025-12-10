import os
import logging
import asyncio
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize Google GenAI client
try:
    from google import genai
    client = genai.Client()
except ImportError:
    logger.error("google.genai module not found.")
    client = None
except Exception as e:
    logger.error(f"Failed to initialize Google GenAI client: {e}")
    client = None

def glosses_to_text(glosses):
    if not client:
        logger.error("Google GenAI client not initialized.")
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
        logger.error(f"Error in glosses_to_text: {e}")
        return None

def custom_prompt(prompt):
    if not client:
        logger.error("Google GenAI client not initialized.")
        return None
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite", contents=prompt
        )
        return response.text
    except Exception as e:
        logger.error(f"Error in custom_prompt: {e}")
        return None

async def process_video_translation(prediction_text: str, target_language: str = "English"):
    """
    High-level function to process the translation flow using the synchronous Gemini functions
    wrapped in asyncio.to_thread for non-blocking execution.
    """
    if not prediction_text:
        return None

    # Run synchronous glosses_to_text in a separate thread
    gemini_response = await asyncio.to_thread(glosses_to_text, prediction_text)

    if not gemini_response:
        return None

    result = {"translation": gemini_response}

    # Generate summary (only for English, as per original logic and prompt limitations)
    if target_language == "English":
        summary_prompt = f"Make a really brief summary encapsling all the content of the following text in one sentence of between two and 4 words: {gemini_response}"
        gemini_summary = await asyncio.to_thread(custom_prompt, summary_prompt)
        
        if gemini_summary:
            result["summary"] = gemini_summary
        else:
            result["summary"] = "No summary generated"

    return result
