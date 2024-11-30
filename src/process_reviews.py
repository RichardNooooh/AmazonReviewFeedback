import re
import json
MAX_REVIEW_LENGTH = 512
def clean_text(text):
    """
    Cleans the text by replacing certain Unicode characters with ASCII equivalents
    and removing unsupported Unicode characters.
    """
    # remove [[VIDEOID:...]] and other tags like that.
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

    if len(text) > MAX_REVIEW_LENGTH:
        text = text[:MAX_REVIEW_LENGTH] + "..."

    return text

# Load the JSON file
input_path = '../data/raw/selected_reviews.json'
output_path = '../data/raw/cleaned_reviews.json'

with open(input_path, 'r') as file:
    reviews = json.load(file)

# Clean the reviews
for review in reviews:
    review['title'] = clean_text(review['title'])
    review['text'] = clean_text(review['text'])

# Save the cleaned reviews to a new file
with open(output_path, 'w') as file:
    json.dump(reviews, file, indent=4)