
from transformers import pipeline
import pandas as pd

from review_summarization.dataset_loading import load_roto_dataset


# TODO: What "structure" do the tutors want? Separate functions? God-class?
# Instantiate the sentiment analysis pipeline
def load_sentiment_model():
    sent_pipe_default = pipeline(task='sentiment-analysis')


# Predict the sentiment of an input sentence
def predict_sentiment(sent_pipe, sentence):
    result = sent_pipe(sentence)


# distilbert-base-uncased-finetuned-sst-2-english classifies into "POSITIVE" or "NEGATIVE" + score how certain
# bert-base-multilingual-uncased-sentiment classifies into "1 star", "2 stars", ... "5 stars" + score how certain
AVAILABLE_MODELS = ['distilbert-base-uncased-finetuned-sst-2-english', 'bert-base-multilingual-uncased-sentiment']


def classify_sentiment(data: pd.DataFrame, classifier: str):
    # Cast column dtype and then convert to list returning a numpy array
    review_contents = data["review_content"].astype(str).values.tolist()

    print("Using pretrained sentiment analysis model: " + classifier)

    # Load the sentiment analysis model
    sent_pipe = pipeline(task='sentiment-analysis', model=classifier, tokenizer=classifier)

    # Predict the sentiments
    sentiment_default = sent_pipe(review_contents)

    # Extract two lists from the predictions containing the labels and the scores
    sentiment_labels = [item['label'] for item in sentiment_default]
    sentiment_scores = [item['score'] for item in sentiment_default]

    # Add the predictions to the dataframe
    data['sentiment_label'] = sentiment_labels
    data['sentiment_score'] = sentiment_scores

    # Print most extreme reviews
    print("Most positive review:", data.loc[data['sentiment_score'].idxmax()]['review_content'])
    print("Most negative review:", data.loc[data['sentiment_score'].idxmin()]['review_content'])
    print()
