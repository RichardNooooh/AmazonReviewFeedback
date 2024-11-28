from sklearn.feature_extraction.text import TfidfVectorizer

# Example data (actionable feedback)
reviews_data = {
    "ASIN": "B01195G5FS",
    "Reviews": [
        {
            "Star": 5,
            "Review title": "wife loves them",
            "Text": "Continue providing quality sound and design that users enjoy.",
            "Number of votes": 0,
        },
        {
            "Star": 1,
            "Review title": "Sound",
            "Text": "Consider improving sound quality, as this is a recurring issue among reviewers.",
            "Number of votes": 2,
        },
    ],
}

# Collect all actionable feedback texts
feedback_texts = [review["Text"] for review in reviews_data["Reviews"]]

# Extract keywords using TF-IDF
vectorizer = TfidfVectorizer(max_features=5, stop_words="english")  # Top 5 keywords
X = vectorizer.fit_transform(feedback_texts)
feature_names = vectorizer.get_feature_names_out()

# Add keywords to each review
for i, review in enumerate(reviews_data["Reviews"]):
    review["Keywords"] = feature_names[X[i].indices]

# Print updated reviews
for review in reviews_data["Reviews"]:
    print(f"Star: {review['Star']}")
    print(f"Review title: {review['Review title']}")
    print(f"Text: {review['Text']}")
    print(f"Keywords: {', '.join(review['Keywords'])}\n")
