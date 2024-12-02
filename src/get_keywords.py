# Used to evaluate our model's outputs
from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

# assumes the dataframe is in the format defined by `process_raw_data.py`
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


if __name__ == "__main__":
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    file_handler = logging.FileHandler("../logs/batchrunner.log", mode="w")
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
    
    input_file = "../data/processed/amazon_reviews.tsv"
    processed_file_location = "../data/processed/formatted_reviews.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, json_schema=schema, input_file=input_file, batch_input_folder="./test/")
    runner.create_jsonl_batches(process_input_data, processed_file_location, batch_size=2500)
    # runner.upload_batch_files()
    # runner.submit_batch_jobs()
    # runner.check_status_and_download()
    # runner.delete_data_files()
    # output_data = runner.get_data()


    # reviews_groupings = dict()
    # for key, keywords in output_data.items():
    #     asin, rating = key.split("_")
    #     if asin in reviews_groupings:
    #         reviews_groupings.append({rating: keywords})
    #     else:
    #         reviews_groupings[asin] = [keywords]

    # result = []
    # for asin, review_list in reviews_groupings.items():
    #     rating_columns = dict()
    #     for n in range(1, 6):
    #         keywords = review_list[str(n)]
    #         rating_columns.update({f"rating_{str(n)}_keywords": keywords})
            
    #     result.append({
    #             "ASIN": asin,
    #             **rating_columns
    #         })
    # df = pd.DataFrame(result)
    # df.to_csv("../data/processed/amazon_keywords.tsv", sep="\t")

