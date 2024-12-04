import os
from openai import OpenAI
import json

# Initialize the OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def analyze_review_sentiment(review):
    """
    Analyze a single review for emotional sentiment using OpenAI API.
    """
    if not review.strip():
        return {}

    sentiment_prompt = (
        f"Analyze the following Amazon review and return the emotional sentiment analysis as a JSON object. "
        f"Include the fields 'Happy', 'Sadness', 'Anger', and 'Disgust'.\n\n"
        f"Review: \"{review}\"\n\n"
        "Format the response as:\n"
        "{\n"
        "    \"Sentiment Data\": {\n"
        "        \"Happy\": <score>,\n"
        "        \"Sadness\": <score>,\n"
        "        \"Anger\": <score>,\n"
        "        \"Disgust\": <score>\n"
        "    }\n"
        "}"
    )

    try:
        # Make the API call
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant for analyzing Amazon reviews."},
                {"role": "user", "content": sentiment_prompt}
            ],
            model="gpt-4o-mini",
        )

        # Extract and sanitize the content
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty response content.")

        # Ensure the content is valid JSON
        content = content.replace("```json", "").replace("```", "").strip()
        sentiment_data = json.loads(content).get("Sentiment Data", {})
        return sentiment_data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response content: {content}")
        return {}
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return {}

def extract_review_keywords(review):
    """
    Extract keywords from a single review using OpenAI API.
    """
    if not review.strip():
        return []

    keyword_prompt = (
        "Extract the most relevant keywords from the following review, focusing on terms that capture the product's features, benefits, or issues.\n\n"
        "Return the keywords as a JSON array in the following format:\n"
        "{\n"
        "    \"Keywords\": [\"<keyword1>\", \"<keyword2>\", \"<keyword3>\", ...]\n"
        "}\n\n"
        f"Review: \"{review}\"\n\n"
        "Ensure the keywords are specific, concise, and relevant to the review content."
    )

    try:
        # Make the API call
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant for analyzing Amazon reviews."},
                {"role": "user", "content": keyword_prompt}
            ],
            model="gpt-4o-mini",
        )

        # Extract and sanitize the content
        content = response.choices[0].message.content.strip()
        if not content:
            raise ValueError("Empty response content.")

        # Ensure the content is valid JSON
        content = content.replace("```json", "").replace("```", "").strip()
        keywords = json.loads(content).get("Keywords", [])
        return keywords
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response content: {content}")
        return []
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []