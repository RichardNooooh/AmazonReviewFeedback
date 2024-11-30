import re
import json
from collections import defaultdict

MAX_REVIEW_LENGTH = 512

def clean_text(text):
    """
    Cleans the text by replacing certain Unicode characters with ASCII equivalents
    and removing unsupported Unicode characters.
    """
    # Remove [[VIDEOID:...]] and other tags like that
    text = re.sub(r'\[\[.*?\]\]', '', text)
    text = re.sub(r'<br\s*/?>', r'\\n', text)

    # Replace specific Unicode characters with ASCII equivalents
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "..."  # Ellipsis
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Remove all remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]', '', text)

    # Truncate text if it exceeds the maximum length
    if len(text) > MAX_REVIEW_LENGTH:
        text = text[:MAX_REVIEW_LENGTH] + "..."

    return text

def clean_and_condense_reviews(input_file: str, output_file: str):
    """
    Cleans and groups reviews by ASIN and then by rating for better traceability.

    Args:
        input_file (str): Path to the input selected_reviews.json file.
        output_file (str): Path to save the cleaned and condensed JSON file.
    """
    # Load the selected reviews
    with open(input_file, "r") as f:
        reviews = json.load(f)

    # Group reviews by ASIN and then by rating
    grouped_reviews = defaultdict(lambda: defaultdict(list))
    for review in reviews:
        asin = review["asin"]
        rating = review["rating"]

        # Clean the title and text
        cleaned_title = clean_text(review["title"])
        cleaned_text = clean_text(review["text"])

        # Append cleaned review details to the appropriate group
        grouped_reviews[asin][rating].append({
            "title": cleaned_title,
            "text": cleaned_text
        })

    # Convert grouped reviews to a list for JSON serialization
    condensed_reviews = [
        {
            "asin": asin,
            "ratings": dict(ratings)  # Convert defaultdict to a regular dict
        }
        for asin, ratings in grouped_reviews.items()
    ]

    # Save the condensed reviews to a new JSON file
    with open(output_file, "w") as f:
        json.dump(condensed_reviews, f, indent=4)

    print(f"Cleaned and condensed reviews saved to {output_file}")


# Example usage
input_file = "../data/raw/selected_reviews.json"
output_file = "../data/raw/cleaned_reviews.json"
clean_and_condense_reviews(input_file, output_file)
