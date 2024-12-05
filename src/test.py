import json
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_sentiments(review_text):
    system_prompt = "You are a helpful assistant designed to analyze Amazon "+\
                    "reviews. Your task is to extract up to 10 concise keywords "+\
                    "and phrases that represent the product's features, benefits, "+\
                    "or issues from the provided text. Only include keywords directly "+\
                    "found in the text."
    
    schema = {
        "name": "get_keywords",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
            "problem_keywords": {
                "type": "array",
                "items": {
                "type": "string"
                },
                "description": "Keywords identifying specific issues and features mentioned in the review."
            }
            },
            "required": [
            "problem_keywords"
            ],
            "additionalProperties": False
        }
    }
    
    # OpenAI API Call
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": review_text},
            ],
            functions=[{"name": schema["name"], "parameters": schema["schema"]}],
            function_call={"name": schema["name"]},
            temperature=0.0,
            max_tokens=300,
        )
        # Extract the function_call arguments
        sentiment_scores = json.loads(response.choices[0].message.function_call.arguments)
        return sentiment_scores
    except Exception as e:
        print(f"Error: {e}")
        return None
    
# Test the function
if __name__ == "__main__":
    review_text = "I hate this chinese crap, rather buy anything else, save your dollars what the fuck what the hell"
    sentiment_scores = get_sentiments(review_text)
    print(sentiment_scores)