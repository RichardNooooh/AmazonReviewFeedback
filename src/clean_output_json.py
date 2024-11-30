import json

class OutputCleaner:

    def clean_keyword(self, input_file: str, output_file: str):
        """
        Cleans up the output JSONL file by retaining only the ASIN, star rating, and extracted keywords.

        Args:
            input_file (str): Path to the input JSONL file with the original API responses.
            output_file (str): Path to save the cleaned JSON file.
        """
        cleaned_data = []

        # Open and read the input JSONL file line by line
        with open(input_file, "r") as infile:
            for line in infile:
                # Parse each line as JSON
                response = json.loads(line)

                # Extract ASIN and star rating from the custom_id
                custom_id = response["custom_id"]
                try:
                    asin = custom_id.split("_")[1]  # Extract ASIN
                    star_rating = custom_id.split("_")[4]
                except IndexError:
                    continue  # Skip if custom_id format is invalid

                # Extract keywords from the API response
                try:
                    content = response["response"]["body"]["choices"][0]["message"]["content"]
                    keywords = json.loads(content)["Keywords"]  # Parse the keywords JSON
                except (KeyError, json.JSONDecodeError):
                    keywords = []  # Default to an empty list if parsing fails

                # Append the cleaned data
                cleaned_data.append({
                    "ASIN": asin,
                    "Star Rating": star_rating,
                    "Keywords": keywords
                })

        # Save the cleaned data as a JSON file
        with open(output_file, "w") as outfile:
            json.dump(cleaned_data, outfile, indent=4)

        print(f"Cleaned output saved to {output_file}")


    def clean_emotion_output(self, input_file: str, output_file: str):
        """
        Cleans up the emotional sentiment output JSONL file by retaining only the ASIN, star rating, and sentiment scores.

        Args:
            input_file (str): Path to the input JSONL file with the original API responses.
            output_file (str): Path to save the cleaned JSON file.
        """
        cleaned_data = []

        # Open and read the input JSONL file line by line
        with open(input_file, "r") as infile:
            for line in infile:
                # Parse each line as JSON
                response = json.loads(line)

                # Extract ASIN and star rating from the custom_id
                custom_id = response["custom_id"]
                try:
                    asin = custom_id.split("_")[1]  # Extract ASIN
                    star_rating = custom_id.split("_")[4]  # Extract star rating
                except IndexError:
                    continue  # Skip if custom_id format is invalid

                # Extract emotional sentiment data from the API response
                try:
                    content = response["response"]["body"]["choices"][0]["message"]["content"]
                    sentiment_data = json.loads(content)["Sentiment Data"]  # Parse the sentiment data JSON
                except (KeyError, json.JSONDecodeError):
                    sentiment_data = {}  # Default to an empty dictionary if parsing fails

                # Append the cleaned data
                cleaned_data.append({
                    "ASIN": asin,
                    "Star Rating": star_rating,
                    "Sentiment Data": sentiment_data
                })

        # Save the cleaned data as a JSON file
        with open(output_file, "w") as outfile:
            json.dump(cleaned_data, outfile, indent=4)

        print(f"Cleaned emotional sentiment output saved to {output_file}")



cleaner = OutputCleaner()
cleaner.clean_keyword("test_keyword_output.jsonl", "cleaned_keyword_output.json")
cleaner.clean_emotion_output("test_sentiment_output.jsonl", "cleaned_sentiment_output.json")