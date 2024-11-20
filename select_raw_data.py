import json
import pandas as pd
from collections import defaultdict
import gzip
from tqdm import tqdm
from datetime import datetime
import gc

def load_metadata(metadata_file_path, max_items=10_000_000):
    """
    Load metadata from a JSONL file and return it as a pandas DataFrame.
    """
    metadata = []
    count = 0
    with gzip.open(metadata_file_path, 'rb') as file:
        for line in tqdm(file):
            if count > max_items:
                break
            try:
                product = json.loads(line)
                asin = product.get('parent_asin')
                title = product.get('title')
                rating_number = product.get('rating_number')
                average_rating = product.get('average_rating')
                
                metadata.append({
                    'asin': asin,
                    'title': title,
                    'rating_number': rating_number,
                    'average_rating': average_rating
                })
            except Exception as e:
                print(f"Error in JSON decoding: {e}")
                continue
            count += 1
    return pd.DataFrame(metadata)

def load_reviews(review_file_path, max_items=10_000_000):
    """
    Load individual reviews from a JSONL file and return it as a pandas DataFrame.
    """
    reviews = []
    count = 0
    with gzip.open(review_file_path, 'rb') as file:
        for line in tqdm(file):
            if count > max_items:
                break
            try:
                review = json.loads(line)
                asin = review.get('parent_asin')
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
            count += 1
    return pd.DataFrame(reviews)

def get_top_reviewed_products(metadata_df, top_n=100, min_reviews=100):
    """
    Get the top N products with the highest number of reviews from metadata.
    """
    filtered_df = metadata_df[metadata_df["rating_number"] >= min_reviews]
    top_products_df = filtered_df.sort_values(by='rating_number', ascending=False).head(top_n)
    return top_products_df

def fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5):
    """
    Fetch reviews grouped by ASIN and rating, and return them in a nested JSON format.
    """
    product_reviews = defaultdict(lambda: defaultdict(list))
    date_string = "2019-01-01"
    threshold_timestamp = int(datetime.strptime(date_string, "%Y-%m-%d").timestamp() * 1000)

    # Group reviews by product and star rating for the top ASINs
    top_asins_set = set(top_asins['asin'])
    for _, review in tqdm(review_df.iterrows(), total=len(review_df)):
        asin = review['asin']
        if asin in top_asins_set and \
                review['timestamp'] > threshold_timestamp and \
                review['verified_purchase']:
            rating = str(int(review['rating']))
            product_reviews[asin][rating].append({
                "title": review['title'],
                "text": review['text'],
                "helpful_vote": review['helpful_vote']
            })

    # Fetch up to 5 reviews per rating for each product
    nested_reviews = {}
    for asin, ratings_dict in tqdm(product_reviews.items()):
        temp = dict()
        # nested_reviews[asin] = {}
        temp["title"] = top_asins.set_index('asin').loc[asin, 'title']
        temp["average_rating"] = top_asins.set_index('asin').loc[asin, 'average_rating']
        temp["reviews"] = dict()
        is_valid = True
        for star_rating, reviews in ratings_dict.items():
            if len(reviews) < reviews_per_rating:
                is_valid = False
                break
            # Sort by helpful votes
            sorted_reviews = sorted(reviews, key=lambda x: (x.get('helpful_vote')), reverse=True)
            # Fetch the top 'reviews_per_rating' reviews for each star rating
            temp["reviews"][star_rating] = sorted_reviews[:reviews_per_rating]
        
        if is_valid:
            nested_reviews[asin] = temp
    
    return nested_reviews

def main(metadata_file, review_file, output_name):
    # Load the datasets
    print(f"Loading metadata from {metadata_file}")
    metadata_df = load_metadata(metadata_file)
    print(f"Loaded {len(metadata_df)} metadata entries.")
    
    print(f"Loading reviews from {review_file}")
    review_df = load_reviews(review_file)
    print(f"Loaded {len(review_df)} reviews.")
    
    # Get the top 10000 products by review count
    print("Selecting top reviewed products...")
    top_asins = get_top_reviewed_products(metadata_df, top_n=10000, min_reviews=50)
    print(f"Selected {len(top_asins)} top products.")
    
    # Fetch reviews for these top products and convert to nested JSON
    print("Fetching reviews for top products...")
    nested_reviews = fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5)
    
    # Save the nested JSON to a file
    print(f"Writing JSON output to {output_name}")
    with open(output_name, 'w') as json_file:
        json.dump(nested_reviews, json_file, indent=4)

    del metadata_df
    del review_df
    del top_asins
    del nested_reviews
    gc.collect()

if __name__ == "__main__":
    data_folder = "./data/raw"
    categories = ["Amazon_Fashion", "Appliances", "Electronics", "Handmade_Products", "Health_and_Household"]
    for category in categories:
        metadata_file = f"{data_folder}/meta_{category}.jsonl.gz"
        review_file = f"{data_folder}/{category}.jsonl.gz"
        main(metadata_file, review_file, f"data/selected/{category}.json")
