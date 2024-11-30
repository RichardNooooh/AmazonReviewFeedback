# Used to evaluate our model's outputs
from sentiment_batchapi_runner import OpenAIBatchRunner
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

    system_prompt = "You are a helpful assistant for a business. Your task is to analyze Amazon reviews and return the emotional sentiment analysis as a JSON object. Include the fields 'Happy', 'Sadness', 'Anger', and 'Disgust'."
    input_file = "../data/raw/cleaned_reviews.json"
    batch_input_folder = "batch_input/"
    batch_output_folder = "batch_output/"
    id_folder = "ids/"
    runner = OpenAIBatchRunner(OPENAI_API_KEY, system_prompt, input_file, batch_input_folder, batch_output_folder, id_folder)

    runner.create_jsonl_batches()
    # runner.upload_batch_files()
    # runner.submit_batch_jobs()
    # runner.check_status_and_download()
    # runner.delete_data_files()
    # output_data = runner.get_data()

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
