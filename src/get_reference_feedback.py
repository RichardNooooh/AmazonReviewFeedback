# Used to evaluate our model's outputs
from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# assumes the dataframe is in the format defined by `process_raw_data.py`
def process_input_data(df: pd.DataFrame, processed_location: str) -> pd.DataFrame:
    if df.shape[1] != 8:
        print(f"Input dataframe does not have the correct number of columns (8). Received dataframe with {df.shape[1]} columns")
    
    new_df = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        new_text = f"{row["item_title"]}\nAverage Rating: {row["avg_rating"]}\n\n"
        
        reviews = []
        for n in range(1, 6):
            # ignoring helpful_votes since that field seems to be bugged on CanopyAPI
            review_title, review_text, helpful_votes = row[f"rating_{n}_review"].split(" ||| ")
            review_formatted = f"{n}-star review:\n- Review Title: {review_title}" +\
                                f"\n- Review Text: {review_text}"
            reviews.append(review_formatted)

        new_text += "\n\n".join(reviews)

        new_df.append({
            "id": row["ASIN"],
            "user_input": new_text
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

    system_prompt = (
        "You are a helpful assistant for a business. "
        + "You are given a set of Amazon reviews for a given item, one for each rating out of 5, "
        + "and tasked with providing actionable feedback to help improve this item. "
        + "Please format your response into concise sentences, one for each actionable feedback. "
        + "Place each feedback on a bulletpoint."
    )
    input_file = "../data/processed/amazon_reviews.tsv"
    processed_file_location = "../data/processed/formatted_baseline.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, input_file=input_file)
    # runner.create_jsonl_batches(process_input_data, processed_file_location)
    # runner.upload_batch_files()
    # runner.submit_batch_jobs()
    # runner.check_status_and_download()
    # runner.delete_data_files()
    output_data = runner.get_data()

    df = pd.read_csv(processed_file_location, delimiter="\t", index_col=False)
    df['reference'] = df['id'].map(output_data)

    # for finetuning baseline data
    df.to_csv("../data/final/baseline.tsv", sep="\t", index=False)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('../data/final/baseline_train.tsv', index=False, sep="\t")
    test_df.to_csv('../data/final/baseline_test.tsv', index=False, sep="\t")

