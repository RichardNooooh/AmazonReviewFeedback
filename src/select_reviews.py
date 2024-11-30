import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from datetime import datetime

def load_reviews(review_file_path, max_items=10_000_000):
    """
    Load individual reviews from a JSONL file and return them as a pandas DataFrame.
    """
    reviews = []
    count = 0
    with open(review_file_path, 'r') as file:
        for line in tqdm(file):
            if count >= max_items:
                break
            try:
                review = json.loads(line)
                asin = review.get('asin')
                rating = review.get('rating')
                title = review.get('title')
                text = review.get('text')
                helpful_vote = review.get('helpful_vote')
                verified = review.get('verified_purchase')
                timestamp = review.get('timestamp')

                if asin and rating and text:
                    reviews.append({
                        'asin': asin,
                        'rating': rating,
                        'title': title,
                        'text': text,
                        'helpful_vote': helpful_vote,
                        'verified_purchase': verified,
                        'timestamp': timestamp
                    })
            except Exception as e:
                print(f"Error in JSON decoding: {e}")
                continue
            count += 1
    return pd.DataFrame(reviews)

def fetch_reviews_for_sentiment_analysis(review_df, reviews_per_rating=2):
    """
    Fetch up to 2 reviews per star rating for each product and prepare for sentiment analysis.
    """
    product_reviews = defaultdict(lambda: defaultdict(list))
    date_string = "2019-01-01"
    threshold_timestamp = int(datetime.strptime(date_string, "%Y-%m-%d").timestamp() * 1000)

    # Group reviews by product and star rating
    for _, review in tqdm(review_df.iterrows(), total=len(review_df)):
        asin = review['asin']
        if review['timestamp'] > threshold_timestamp and review['verified_purchase']:
            rating = str(int(review['rating']))
            product_reviews[asin][rating].append({
                "title": review['title'],
                "text": review['text'],
                "helpful_vote": review.get('helpful_vote', 0)
            })

    # Fetch up to `reviews_per_rating` reviews for each star rating
    selected_reviews = []
    for asin, ratings_dict in product_reviews.items():
        for star_rating, reviews in ratings_dict.items():
            # Sort by helpful votes
            sorted_reviews = sorted(reviews, key=lambda x: x.get('helpful_vote', 0), reverse=True)
            # Fetch the top `reviews_per_rating` reviews
            selected_reviews.extend([
                {
                    "asin": asin,
                    "rating": star_rating,
                    "title": review["title"],
                    "text": review["text"]
                }
                for review in sorted_reviews[:reviews_per_rating]
            ])
    
    return selected_reviews

def main(review_file_path, output_json_path, reviews_per_rating=2, max_items=10_000_000):
    """
    Main function to process reviews, select top reviews, and save output for sentiment analysis.
    """
    print(f"Loading reviews from {review_file_path}...")
    review_df = load_reviews(review_file_path, max_items=max_items)
    print(f"Loaded {len(review_df)} reviews.")

    print("Selecting reviews for sentiment analysis...")
    selected_reviews = fetch_reviews_for_sentiment_analysis(review_df, reviews_per_rating=reviews_per_rating)

    print(f"Saving selected reviews to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(selected_reviews, f, indent=4)
    print(f"Selected reviews saved to {output_json_path}.")

# Example usage
if __name__ == "__main__":
    review_file = "../data/raw/test_review.jsonl"  # Replace with your file path
    output_json = "../data/raw/selected_reviews.json"
    main(review_file, output_json, reviews_per_rating=2, max_items=1000)