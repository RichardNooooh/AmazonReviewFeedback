from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    data_folder = "../data/processed"
    
    reviews_df = pd.read_csv(f"{data_folder}/amazon_reviews.tsv", delimiter="\t")
    keywords_df = pd.read_csv(f"{data_folder}/amazon_keywords.tsv", delimiter="\t")
    sentiments_df = pd.read_csv(f"{data_folder}/amazon_sentiments.tsv", delimiter="\t")
    references_df = pd.read_csv(f"{data_folder}/amazon_references.tsv", delimiter="\t")

    merged_df = reviews_df.merge(keywords_df, on="ASIN")\
                            .merge(sentiments_df, on="ASIN")\
                            .merge(references_df, on="ASIN")
    merged_df.to_csv(f"{data_folder}/amazon_complete.tsv", sep="\t", index=False)

    # Construct datasets
    # Similar to what we did in `get_reference_feedback.py`, we'll combine the text into a natural format
    combined_df = []
    for _, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
        item_text = f"{row["item_title"]}\n" +\
                   f"Average Rating: {row["avg_rating"]}\n\n"

        baseline_reviews, aug_kw_reviews, aug_sm_reviews, aug_full_reviews = [], [], [], []
        for n in range(1, 6):
            # ignoring helpful_votes since that field seems to be bugged on CanopyAPI
            # also, it seems that the reviews with the highest helpful votes are hidden
            # on Amazon.com, since they are usually older reviews that may not be relevant
            # for a given item listing that may have been updated. not sure though
            review_title, review_text, _ = row[f"rating_{n}_review"].split(" ||| ")

            # Standard review format
            base_text = f"{n}-star review:\n- Review Title: {review_title}\n" +\
                                f"- Review Text: {review_text}"
            
            # Augmented data format
            keyword_text   = f"- Keywords: {row[f"rating_{n}_keywords"]}"
            sentiment_text = f"- Sentiment scores: {row[f"rating_{n}_sentiments"]}"
            
            # review append
            baseline_reviews.append(base_text)
            aug_kw_reviews.append(f"{base_text}\n{keyword_text}")
            aug_sm_reviews.append(f"{base_text}\n{sentiment_text}")
            aug_full_reviews.append(f"{base_text}\n{keyword_text}\n{sentiment_text}")

        combined_df.append({
            "ASIN": row["ASIN"],
            "baseline": item_text + "\n\n".join(baseline_reviews),
            "augmented_keyword": item_text + "\n\n".join(aug_kw_reviews),
            "augmented_sentiment": item_text + "\n\n".join(aug_sm_reviews),
            "augmented_full": item_text + "\n\n".join(aug_full_reviews),
            "labels": row["feedback"]
        })

    combined_df = pd.DataFrame(combined_df)

    # Save train/test splitted data
    # This is done first so that all models are trained and tested on the same reviews
    output_folder = "../data/final"
    train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)    

    train_baseline_df = train_df[["ASIN", "baseline", "labels"]].rename(columns={'baseline': 'text'})
    train_augmented_kw_df = train_df[["ASIN", "augmented_keyword", "labels"]].rename(columns={'augmented_keyword': 'text'})
    train_augmented_sm_df = train_df[["ASIN", "augmented_sentiment", "labels"]].rename(columns={'augmented_sentiment': 'text'})
    train_augmented_full_df = train_df[["ASIN", "augmented_full", "labels"]].rename(columns={'augmented_full': 'text'})

    test_baseline_df = test_df[["ASIN", "baseline", "labels"]].rename(columns={'baseline': 'text'})
    test_augmented_kw_df = test_df[["ASIN", "augmented_keyword", "labels"]].rename(columns={'augmented_keyword': 'text'})
    test_augmented_sm_df = test_df[["ASIN", "augmented_sentiment", "labels"]].rename(columns={'augmented_sentiment': 'text'})
    test_augmented_full_df = test_df[["ASIN", "augmented_full", "labels"]].rename(columns={'augmented_full': 'text'})

    train_baseline_df.to_csv(f"{output_folder}/train/baseline.tsv", index=False, sep="\t")
    train_augmented_kw_df.to_csv(f"{output_folder}/train/augmented_keywords.tsv", index=False, sep="\t")
    train_augmented_sm_df.to_csv(f"{output_folder}/train/augmented_sentiments.tsv", index=False, sep="\t")
    train_augmented_full_df.to_csv(f"{output_folder}/train/augmented_full.tsv", index=False, sep="\t")

    test_baseline_df.to_csv(f"{output_folder}/test/baseline.tsv", index=False, sep="\t")
    test_augmented_kw_df.to_csv(f"{output_folder}/test/augmented_keywords.tsv", index=False, sep="\t")
    test_augmented_sm_df.to_csv(f"{output_folder}/test/augmented_sentiments.tsv", index=False, sep="\t")
    test_augmented_full_df.to_csv(f"{output_folder}/test/augmented_full.tsv", index=False, sep="\t")
