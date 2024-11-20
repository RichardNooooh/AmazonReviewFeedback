import json
import pandas as pd
import re
from tqdm import tqdm
from transformers import LlamaTokenizer
import os

AUTH_TOKEN = os.getenv("HF_TOKEN")
MAX_REVIEW_LENGTH = 1000

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


def process_json_to_string(data):
    output = f"Title: {data['title']}\n"
    output += f"Average Rating: {data['average_rating']} stars\n\n"
    
    for rating in range(1, 6):  # Loop through 1 to 5 star reviews
        reviews = data['reviews'].get(str(rating))
        if reviews is None:
            continue

        output += f"{rating} star reviews:\n"
        for review in reviews[:2]:
            output += f"- Review title: {clean_text(review['title'])}\n"
            output += f"  - Text: {clean_text(review['text'])}\n"
            output += f"  - Number of votes: {review['helpful_vote']}\n"
        output += "\n"
    
    return output.strip()


if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat", token=AUTH_TOKEN)

    file_folder = "./data/selected/"
    file_names = ["Amazon_Fashion", "Appliances", "Electronics", "Handmade_Products", "Health_and_Household"]  # Add your file names here
    processed_data = []

    max_length_review = -1
    max_review = None
    total_token_count, total_entry_count = 0, 0
    for file_name in file_names:
        print(f"Processing {file_name}")
        with open(f"{file_folder}{file_name}.json", 'r') as file:
            json_data = json.load(file)
            for asin, asin_data in tqdm(json_data.items()):
                processed_string = process_json_to_string(asin_data)

                tokens = tokenizer.tokenize(processed_string)
                num_tokens = len(tokens)
                
                if num_tokens > max_length_review:
                    max_length_review = num_tokens
                    max_review = processed_string
                total_token_count += num_tokens
                total_entry_count += 1

                processed_data.append({'ASIN': asin, 'Reviews': processed_string})

    print(f"Obtained dataset of {len(processed_data)} Amazon items")
    print(f"Longest entry is {max_length_review} tokens long:\n{max_review}")
    print(f"Average tokens: {total_token_count / total_entry_count}")
    
    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)
    print(f"Total processed items: {df.shape}")

    # Save the DataFrame as a TSV file
    output_file = './data/processed/amazon_reviews.tsv'
    df.to_csv(output_file, sep='\t', index=False)

    print(f"Data successfully processed and saved to {output_file}.")