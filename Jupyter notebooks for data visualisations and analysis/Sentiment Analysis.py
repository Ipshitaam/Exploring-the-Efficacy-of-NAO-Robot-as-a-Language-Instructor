import pandas as pd
import re
import torch 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def perform_sentiment_analysis(csv_file, text_column, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
  """
  Performs sentiment analysis on a CSV file with a column containing text.

  Args:
      csv_file (str): Path to the CSV file.
      text_column (str): Name of the column containing the text data.
      model_name (str, optional): Name of the Transformers model for sentiment analysis.
          Defaults to "distilbert-base-uncased-finetuned-sst-2-english".

  Returns:
      pandas.DataFrame: A new DataFrame with added sentiment columns.
  """
  csv_file = "C:\\Users\\SAMEER\\Participants for NAO.csv"

  # Read CSV data
  df = pd.read_csv(csv_file)

  # Load Transformers model and tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSequenceClassification.from_pretrained(model_name)

  # Preprocess text (optional, adjust based on your data)
  def preprocess_text(text):
    text = text.strip()  # Remove leading/trailing whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation (optional)
    return text

  df[text_column] = df['What did you like best about the NAO and how much did you enjoy the session?'].apply(preprocess_text)

  # Prepare input for Transformers model
  def prepare_inputs(text):
    encoded_text = tokenizer(text, return_tensors="pt")
    return encoded_text

  encoded_texts = df[text_column].apply(prepare_inputs)

  # Perform sentiment analysis with Transformers model
  with torch.no_grad():
    outputs = model(**encoded_texts)
    predictions = torch.argmax(outputs.logits, dim=-1)

  # Add sentiment labels based on model output
  df["sentiment_transformer"] = predictions.tolist()
  df["sentiment_label_transformer"] = df["sentiment_transformer"].apply(
      lambda x: "Positive" if x == 1 else "Negative" if x == 0 else "Neutral"
  )

  # Optionally, use NLTK VADER for additional insights
  vader = SentimentIntensityAnalyzer()
  df["sentiment_vader"] = df[text_column].apply(vader.polarity_scores)
  df[["vader_pos", "vader_neu", "vader_neg", "vader_compound"]] = df["sentiment_vader"].explode()

  return df

if __name__ == "__main__":
  csv_file = "your_data.csv"  # Replace with your CSV file path
  text_column = "text_column"  # Replace with the column name containing text

  df = perform_sentiment_analysis(csv_file, text_column)

  # Save results to a new CSV file
  df.to_csv("output.csv", index=False)

  print("Sentiment analysis results saved to output.csv")