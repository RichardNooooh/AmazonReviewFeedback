# Used to evaluate our model's outputs
from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json

def process_input_data(df: pd.DataFrame, processed_location: str) -> pd.DataFrame:
    if df.shape[1] != 8:
        print(f"Input dataframe does not have the correct number of columns (8). Received dataframe with {df.shape[1]} columns")
    
    new_df = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # we want gpt-4o-mini data from each review
        for n in range(1, 6):
            review_title, review_text, helpful_votes = row[f"rating_{n}_review"].split(" ||| ")
            new_df.append({
                "id": f"{row["ASIN"]}_{n}", # Key: {asin}_{rating}
                "user_input": f"{review_title}\n{review_text}"
            })
    
    new_df = pd.DataFrame(new_df)
    
    # makes batches ~the same size since some reviews in certain categories are very long
    new_df = new_df.sample(frac=1, random_state=42).reset_index(
        drop=True
    )

    new_df.to_csv(processed_location, sep="\t", index=False)

    return new_df

def process_output_data(output_data: dict):
    reviews_groupings = dict()
    for key, keywords_string in output_data.items():
        asin, rating = key.split("_")
        sentiments = json.loads(keywords_string)
        sentiments_list = [f"{sentiment}: {"{:.1f}".format(score)}" for sentiment, score in sentiments.items()]
        sentiments_str = ", ".join(sentiments_list)
        if asin in reviews_groupings:
            reviews_groupings[asin][rating] = sentiments_str
        else:
            reviews_groupings[asin] = {rating: sentiments_str}

    result = []
    for asin, review_list in reviews_groupings.items():
        rating_columns = dict()
        for n in range(1, 6):
            sentiments = review_list[str(n)]
            rating_columns.update({f"rating_{str(n)}_sentiments": sentiments})
            
        result.append({
                "ASIN": asin,
                **rating_columns
            })
    df = pd.DataFrame(result)
    df.to_csv("../data/processed/amazon_sentiments.tsv", sep="\t", index=False)



if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    file_handler = logging.FileHandler("../logs/batchrunner_sentiment.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(fmt)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    load_dotenv("./.env")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    system_prompt = "You are an emotionally intelligent agent designed to evaluate various sentiments of a given piece of text. Given the following review for an Amazon item, please score the text for each sentiment type between 0.0 and 1.0 (inclusive), where 0.0 is the complete absence of that sentiment and 1.0 is a strong indication of that sentiment."
    schema = {
        "name": "sentiment_scores",
        "description": "Sentiments of the given text scored between 0.0 to 1.0 (inclusive)",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "satisfaction": {
                    "type": "number",
                    "description": "Contentment or fulfillment with the product or service."
                },
                "gratitude": {
                    "type": "number",
                    "description": "Thankfulness or appreciation."
                },
                "trust": {
                    "type": "number",
                    "description": "Confidence in the product or brand."
                },
                "disappointment": {
                    "type": "number",
                    "description": "Unmet expectations or dissatisfaction."
                },
                "frustration": {
                    "type": "number",
                    "description": "Annoyance or irritation experienced."
                },
                "confusion": {
                    "type": "number",
                    "description": "Lack of clarity or understanding."
                },
                "fear": {
                    "type": "number",
                    "description": "Concern or worry about potential risks."
                },
                "regret": {
                    "type": "number",
                    "description": "Wishing one hadn't made the purchase or choice."
                },
                "indifference": {
                    "type": "number",
                    "description": "Lack of strong feelings or interest."
                },
                "informative": {
                    "type": "number",
                    "description": "Providing information about the product without strong feelings."
                }
            },
            "required": [
                "satisfaction",
                "gratitude",
                "trust",
                "disappointment",
                "frustration",
                "confusion",
                "fear",
                "regret",
                "indifference",
                "informative"
            ],
            "additionalProperties": False
        }
    }
    
    
    input_file = "../data/processed/amazon_reviews.tsv"
    processed_file_location = "../data/processed/formatted_reviews.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, json_schema=schema, input_file=input_file,
                               batch_input_folder="../data/batch_sentiment/batch_input/",
                               batch_output_folder="../data/batch_sentiment/batch_output/",
                               id_folder="../data/batch_sentiment/ids/")
    # runner.create_jsonl_batches(process_input_data, processed_file_location, batch_size=2500)
    # runner.upload_batch_files()
    # runner.submit_batch_jobs()
    # runner.check_status_and_download()
    # runner.delete_data_files()

    output_data = runner.get_data()
    process_output_data(output_data)
