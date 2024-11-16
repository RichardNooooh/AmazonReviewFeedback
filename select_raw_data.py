import json
import pandas as pd
from collections import defaultdict

def load_metadata(metadata_file_path):
    """
    Load metadata from a JSONL file and return it as a pandas DataFrame.
    """
    metadata = []
    with open(metadata_file_path, 'r') as file:
        for line in file:
            try:
                product = json.loads(line)
                asin = product.get('parent_asin', product.get('asin'))
                title = product.get('title')
                rating_number = product.get('rating_number', 0)
                average_rating = product.get('average_rating')
                
                metadata.append({
                    'asin': asin,
                    'title': title,
                    'rating_number': rating_number,
                    'average_rating': average_rating
                })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(metadata)

def load_reviews(review_file_path):
    """
    Load individual reviews from a JSONL file and return it as a pandas DataFrame.
    """
    reviews = []
    with open(review_file_path, 'r') as file:
        for line in file:
            try:
                review = json.loads(line)
                asin = review.get('asin')
                rating = review.get('rating')
                title = review.get('title')
                text = review.get('text')
                helpful_vote = review.get('helpful_vote', 0)
                verified = review.get('verified_purchase', False)

                if asin and rating and text:
                    reviews.append({
                        'asin': asin,
                        'rating': rating,
                        'title': title,
                        'text': text,
                        'helpful_vote': helpful_vote,
                        'verified_purchase': verified
                    })
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(reviews)

def get_top_reviewed_products(metadata_df, top_n=100):
    """
    Get the top N products with the highest number of reviews from metadata.
    """
    top_products_df = metadata_df.sort_values(by='rating_number', ascending=False).head(top_n)
    return top_products_df['asin'].tolist()

def fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5):
    """
    Fetch reviews grouped by ASIN and rating, and return them in a nested JSON format.
    """
    product_reviews = defaultdict(lambda: defaultdict(list))

    # Group reviews by product and star rating for the top ASINs
    for _, review in review_df.iterrows():
        asin = review['asin']
        if asin in top_asins:
            rating = str(int(review['rating']))
            product_reviews[asin][rating].append({
                "title": review['title'],
                "text": review['text']
            })

    # Fetch up to 5 reviews per rating for each product
    nested_reviews = {}
    for asin, ratings_dict in product_reviews.items():
        nested_reviews[asin] = {}
        for star_rating, reviews in ratings_dict.items():
            # Sort by helpful votes and verified status
            sorted_reviews = sorted(reviews, key=lambda x: (x.get('helpful_vote', 0), x.get('verified_purchase', False)), reverse=True)
            # Fetch the top 'reviews_per_rating' reviews for each star rating
            nested_reviews[asin][star_rating] = sorted_reviews[:reviews_per_rating]
    
    return nested_reviews

def main(metadata_file, review_file):
    # Load the datasets
    print("Loading metadata...")
    metadata_df = load_metadata(metadata_file)
    print(f"Loaded {len(metadata_df)} metadata entries.")
    
    print("Loading reviews...")
    review_df = load_reviews(review_file)
    print(f"Loaded {len(review_df)} reviews.")
    
    # Get the top 100 products by review count
    print("Selecting top reviewed products...")
    top_asins = get_top_reviewed_products(metadata_df, top_n=100)
    print(f"Selected {len(top_asins)} top products.")
    
    # Fetch reviews for these top products and convert to nested JSON
    print("Fetching reviews for top products...")
    nested_reviews = fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5)
    
    # Save the nested JSON to a file
    output_file = 'top_products_reviews.json'
    with open(output_file, 'w') as json_file:
        json.dump(nested_reviews, json_file, indent=4)
    print(f"Nested reviews saved to {output_file}.")

if __name__ == "__main__":
    # Replace these with the paths to your full datasets
    metadata_file = 'meta_CDs_and_Vinyl.jsonl'
    review_file = 'CDs_and_Vinyl.jsonl'
    main(metadata_file, review_file)
