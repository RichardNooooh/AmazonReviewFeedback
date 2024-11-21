# Used to evaluate our model's outputs
from batchapi_runner import OpenAIBatchRunner
import logging
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split


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
        + "You are given a set of Amazon reviews for a given item, grouped by their ratings out of 5, "
        + "and tasked with providing actionable feedback to help improve this item. "
        + "Please format your response into concise sentences, one for each actionable feedback. "
        + "Place each feedback on a bulletpoint."
    )
    input_file = "../data/processed/amazon_reviews.tsv"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, input_file)
    runner.create_jsonl_batches()
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.delete_data_files()
    output_data = runner.get_data()

    df = pd.read_csv(input_file, delimiter="\t")
    df['Reference'] = df['ASIN'].map(output_data)

    df.to_csv("../data/final/baseline.tsv", sep="\t")

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('../data/final/baseline_train.tsv', index=False, sep="\t")
    test_df.to_csv('../data/final/baseline_test.tsv', index=False, sep="\t")


