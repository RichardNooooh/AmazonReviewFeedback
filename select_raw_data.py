import json
import pandas as pd
from collections import defaultdict

def load_metadata(metadata_file_path):
    """
    Load the metadata from a JSONL file and return it as a pandas DataFrame.
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
    Load the individual reviews from a JSONL file and return it as a pandas DataFrame.
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
    Get the top N products with the highest number of reviews from the metadata.
    """
    # Sort by rating_number in descending order and select the top N products
    top_products_df = metadata_df.sort_values(by='rating_number', ascending=False).head(top_n)
    return top_products_df['asin'].tolist()

def fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5):
    """
    Fetch a specified number of reviews per star rating for the top reviewed products.
    """
    product_reviews = defaultdict(lambda: defaultdict(list))

    # Group reviews by product and star rating for only the top ASINs
    for _, review in review_df.iterrows():
        asin = review['asin']
        if asin in top_asins:
            rating = int(review['rating'])
            product_reviews[asin][rating].append(review)

    # Filter and fetch reviews based on requirements
    filtered_reviews = []
    for asin in top_asins:
        ratings_dict = product_reviews.get(asin, {})
        for star_rating in range(5, 0, -1):
            reviews = ratings_dict.get(star_rating, [])
            
            # Sort reviews by helpful votes and verified status
            sorted_reviews = sorted(reviews, key=lambda x: (x['helpful_vote'], x['verified_purchase']), reverse=True)
            
            # Fetch the top 'reviews_per_rating' reviews for each star rating
            selected_reviews = sorted_reviews[:reviews_per_rating]
            for review in selected_reviews:
                filtered_reviews.append({
                    'asin': asin,
                    'rating': star_rating,
                    'title': review['title'],
                    'text': review['text'],
                    'helpful_vote': review['helpful_vote'],
                    'verified_purchase': review['verified_purchase']
                })

    return pd.DataFrame(filtered_reviews)

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
    
    # Fetch reviews for these top products
    print("Fetching reviews for top products...")
    filtered_reviews_df = fetch_reviews_for_top_products(review_df, top_asins, reviews_per_rating=5)
    
    # Save the filtered reviews to a CSV file
    output_file = 'top_products_reviews.csv'
    filtered_reviews_df.to_csv(output_file, index=False)
    print(f"Filtered reviews saved to {output_file}.")
    
    # Display the first few rows of the filtered reviews
    print(filtered_reviews_df.head(10))

if __name__ == "__main__":
    # Replace these with the paths to your full datasets
    metadata_file = 'meta_CDs_and_Vinyl.jsonl' # Can replace with other datasets
    review_file = 'CDs_and_Vinyl.jsonl' # Same for this
    main(metadata_file, review_file)
