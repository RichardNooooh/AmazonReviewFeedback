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
        + "You are given a set of Amazon reviews for a given item, grouped by their ratings out of 5. "
        + "For each review, you must: "
        + "1) Provide a brief actionable feedback, and "
        + "2) Analyze and return the sentiment of the review as a JSON object with fields: 'Happy', 'Joy', 'Anger', etc. "
        + "3) Extract and return a list of the most relevant keywords from the review text that highlight the key points."
        + "Format the response as: "
        + "{  'Actionable Feedback': <feedback>, 'Sentiment Data': { 'Happy': <score>, 'Joy': <score>, 'Anger': <score>, ... }, 'Keywords': [<keyword1>, <keyword2>, ...]}"    )
    
    input_file = "../data/processed/amazon_reviews.tsv"
    
    batch_input_folder = "../data/batch_temp/batch_input"
    batch_output_folder = "../data/batch_temp/batch_output"
    id_folder = "../data/batch_temp/ids"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, input_file, batch_input_folder, batch_output_folder, id_folder)
    runner.create_jsonl_batches()
    runner.upload_batch_files()
    runner.submit_batch_jobs()
    runner.check_status_and_download()
    runner.delete_data_files()
    output_data = runner.get_data()

    # # Load the original review data
    # df = pd.read_csv(input_file, delimiter="\t")
    # # Map the output data (reference + sentiment) to the corresponding reviews
    # def append_sentiment_data(asin):
    #     data = output_data.get(asin, {})
    #     if not data:
    #         return None
    #     # Extract Actionable Feedback and Sentiment Data
    #     return {
    #         "Actionable Feedback": data.get("Actionable Feedback", "No feedback available"),
    #         "Sentiment Data": data.get("Sentiment Data", {})
    #     }

#     # df['Reference'] = df['ASIN'].map(append_sentiment_data)

#     # # Save the updated dataframe to the final baseline file
#     # df.to_csv("../data/final/baseline_with_sentiment.tsv", sep="\t", index=False)

#     # train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
#     # train_df.to_csv('../data/final/baseline_train.tsv', index=False, sep="\t")
#     # test_df.to_csv('../data/final/baseline_test.tsv', index=False, sep="\t")
