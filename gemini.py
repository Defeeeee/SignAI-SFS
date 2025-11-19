from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

# response = client.models.generate_content(
#     model="gemini-2.5-flash-lite", contents="Explain the concept of polymorphism in object-oriented programming."
# )
#
# print(response.text)

def glosses_to_text(glosses):
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

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )

    return response.text

def custom_prompt(prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite", contents=prompt
    )
    return response.text

if __name__ == "__main__":
    sample_glosses = "DIENSTAG BESONDERS REGION MEHR FREUNDLICH LANG ABER AUCH DABEI SCHAUER"
    translation = glosses_to_text(sample_glosses)
    print("Translated Text:", translation)

