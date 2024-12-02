import json
import pandas as pd
import re
from tqdm import tqdm
from transformers import AutoTokenizer
import tiktoken
import os

AUTH_TOKEN = os.getenv("HF_TOKEN")
MAX_REVIEW_LENGTH = 512


def clean_text(text):
    """
    Cleans the text by replacing certain Unicode characters with ASCII equivalents
    and removing unsupported Unicode characters.
    """
    # remove [[VIDEOID:...]] and other tags like that.
    text = re.sub(r"\[\[.*?\]\]", "", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\n", " ", text)

    # Replace specific Unicode characters with ASCII equivalents
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote
        "\u201c": '"',  # Left double quote
        "\u201d": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2026": "...",  # Ellipsis
    }
    for unicode_char, ascii_char in replacements.items():
        text = text.replace(unicode_char, ascii_char)

    # Remove all remaining non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]", "", text)

    if len(text) > MAX_REVIEW_LENGTH:
        text = text[:MAX_REVIEW_LENGTH] + "..."

    return text


def process_json_to_string(asin, asin_data):
    delimiter = " ||| "  # for processing during runtime

    review_data = asin_data["reviews"]
    try:
        reviews = {
            f"rating_{n}_review": 
                f"{clean_text(review_data[str(n)]['title'])}{delimiter}"
                + f"{clean_text(review_data[str(n)]['text'])}{delimiter}"
                + f"{review_data[str(n)]['helpful_vote']}"
            for n in range(1, 6)
        }
    except Exception as e:
        # ignoring all items that do not have at least 1 review in each rating 
        # (should already be handled in select_raw_data.py)
        print(f"An error occurred with ASIN {asin}: {e}. Skipping this item.")
        return None

    entry_dict = {
        "ASIN": asin,
        "item_title": clean_text(asin_data["title"]),
        "avg_rating": str(asin_data["average_rating"]),
        **reviews,
    }

    return entry_dict


if __name__ == "__main__":
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", token=AUTH_TOKEN
    )
    gpt_tokenizer = tiktoken.encoding_for_model("gpt-4o")  # should be same for mini
    max_gpt_tokens, max_qwen_tokens = -1, -1
    total_gpt_count, total_qwen_count = 0, 0
    total_entry_count = 0

    file_folder = "../data/selected"
    file_names = [
        "Amazon_Fashion",
        "Appliances",
        "Electronics",
        "Handmade_Products",
        "Health_and_Household",
    ]  # Add your file names here
    processed_data = []

    for file_name in file_names:
        print(f"Processing {file_name}")
        with open(f"{file_folder}/{file_name}.json", "r") as file:
            json_data = json.load(file)
            for item in tqdm(json_data.items()):
                processed_entry = process_json_to_string(*item)
                if processed_entry is None:
                    continue

                processed_data.append(processed_entry)

                # token count estimation for price
                all_text = "\n".join(processed_entry.values())
                qwen_tokens = qwen_tokenizer.tokenize(all_text)
                gpt_tokens = gpt_tokenizer.encode(all_text)

                max_qwen_tokens = max(max_qwen_tokens, len(qwen_tokens))
                max_gpt_tokens = max(max_gpt_tokens, len(gpt_tokens))
                total_gpt_count += len(gpt_tokens)
                total_qwen_count += len(qwen_tokens)
                total_entry_count += 1

    print(f"Obtained dataset of {len(processed_data)} Amazon items")
    print(f"Max GPT Tokens: {max_gpt_tokens}\nMax qwen tokens: {max_qwen_tokens}")
    print(f"Total GPT tokens: {total_gpt_count}\nTotal qwen tokens: {total_qwen_count}")
    print(f"Total number of entries: {total_entry_count}")

    # Create a DataFrame from the processed data
    df = pd.DataFrame(processed_data)
    print(f"Total processed items: {df.shape}")

    # Save the DataFrame as a TSV file
    output_file = "../data/processed/amazon_reviews.tsv"
    df.to_csv(output_file, sep="\t", index=False)

    print(f"Data successfully processed and saved to {output_file}.")
