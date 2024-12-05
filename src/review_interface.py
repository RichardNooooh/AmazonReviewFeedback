import gradio as gr
from openai_api_calls import analyze_review_sentiment, extract_review_keywords

import gradio as gr
from openai_api_calls import analyze_review_sentiment, extract_review_keywords

def process_feedback(title, review1title, review1, review2title, review2, review3title, review3, review4title, review4, review5title, review5):
    """
    Process feedback by analyzing reviews for sentiment and keywords.
    """
    combined_feedback = f"Item Title: {title}\n\n"
    reviews = [review1title, review1, review2title, review2, review3title, review3, review4title, review4, review5title, review5]

    for i, review in enumerate(reviews, start=1):
        if review.strip():
            # Initialize feedback placeholders
            sentiment = {}
            keywords = []

            # Attempt to get sentiment and keywords
            try:
                sentiment = analyze_review_sentiment(review)
                if not sentiment:
                    sentiment = {"error": "No sentiment data returned"}
            except Exception as e:
                sentiment = {"error": str(e)}

            try:
                keywords = extract_review_keywords(review)
                if not keywords:
                    keywords = ["No keywords returned"]
            except Exception as e:
                keywords = [f"Error: {str(e)}"]

            # Append feedback for the review
            combined_feedback += (
                f"Feedback for Review {i}:\n"
                f"Sentiment Analysis: {sentiment}\n"
                f"Keywords: {keywords}\n\n"
            )
        else:
            combined_feedback += f"Feedback for Review {i}: No review provided.\n\n"

    return combined_feedback


# Create the Gradio interface
interface = gr.Interface(
    fn=process_feedback,
    inputs=[
        gr.Textbox(label="Amazon Item Title", placeholder="Enter the Amazon item title here"),
        gr.Textbox(label="Average Review", placeholder="Enter the average star rating here", lines=1),
        gr.Textbox(label="Review Title 1", placeholder="Enter review 1 title here", lines=1),
        gr.Textbox(label="Review 1", placeholder="Enter review 1 text here", lines=3),
        gr.Textbox(label="Review Title 2", placeholder="Enter review 2 title here", lines=1),
        gr.Textbox(label="Review 2", placeholder="Enter review 2 text here", lines=3),
        gr.Textbox(label="Review Title 3", placeholder="Enter review 3 title here", lines=1),
        gr.Textbox(label="Review 3", placeholder="Enter review 3 text here", lines=3),
        gr.Textbox(label="Review Title 4", placeholder="Enter review 4 title here", lines=1),
        gr.Textbox(label="Review 4", placeholder="Enter review 4 text here", lines=3),
        gr.Textbox(label="Review Title 5", placeholder="Enter review 5 title here", lines=1),
        gr.Textbox(label="Review 5", placeholder="Enter review 5 text here", lines=3),
    ],
    outputs=gr.Textbox(label="Generated Feedback", lines=20),
    title="Amazon Review Feedback Generator",
    description="Input the Amazon item title and up to 5 reviews to generate actionable feedback with sentiment analysis and keywords.",
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
