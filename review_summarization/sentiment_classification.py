import torch
from torch import nn
from transformers import pipeline, DistilBertForSequenceClassification
import pandas as pd
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.pipelines.text_classification import TextClassificationPipeline, softmax

# distilbert-base-uncased-finetuned-sst-2-english classifies into "POSITIVE" or "NEGATIVE" + score how certain
# bert-base-multilingual-uncased-sentiment classifies into "1 star", "2 stars", ... "5 stars" + score how certain
AVAILABLE_MODELS = ['distilbert-base-uncased-finetuned-sst-2-english', 'nlptown/bert-base-multilingual-uncased-sentiment']

PRETRAINED_MODEL_PATH = './model/checkpoint-42000'


class SentimentClassifier():
    def __init__(self):
        pretrained_model = DistilBertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_PATH)
        self.pre_classifier = pretrained_model.pre_classifier
        self.classifier = pretrained_model.classifier
        self.id2label = pretrained_model.config.id2label

    def forward(self, x):
        x = self.pre_classifier(x)
        x = nn.ReLU()(x)
        logits = self.classifier(x).detach().numpy()
        scores = softmax(logits)
        dict_scores = [
            {"label": self.id2label[score.argmax().item()], "score": score.max().item()} for score in scores
        ]
        return dict_scores


def classify_sentiment(data: pd.DataFrame, classifier: str, common_embedding: bool = False):
    # Cast column dtype and then convert to list returning a numpy array
    review_contents = data["review_content"].astype(str).values.tolist()

    if common_embedding:
        print("Using pretrained embeddings with custom classification decoder")
        classifier = SentimentClassifier()
        sentiment_predictions = classifier.forward(torch.stack(data['embedding'].values.tolist()))
    else:
        print("Using pretrained sentiment analysis model: " + classifier)

        # Load the sentiment analysis model
        sent_pipe = pipeline(task='sentiment-analysis', model=classifier, tokenizer=classifier)

        # Predict the sentiments
        sentiment_predictions = sent_pipe(review_contents)

    # Extract two lists from the predictions containing the labels and the scores
    sentiment_labels = [item['label'] for item in sentiment_predictions]
    sentiment_scores = [item['score'] for item in sentiment_predictions]

    # Add the predictions to the dataframe
    data['sentiment_label'] = sentiment_labels
    data['sentiment_score'] = sentiment_scores

    # Print number of reviews per label
    print("Results of classification:")
    label_classes = data['sentiment_label'].unique()
    for label in label_classes:
        print(f"\t{label}: {len(data[data['sentiment_label'] == label])} reviews")

    if len(label_classes) == 2:
        print(f"Review score: {round(len(data[data['sentiment_label'] == 'POSITIVE']) / len(data) * 100)}% positive")

    # Print most extreme reviews
    print("Most positive review:", data.loc[data['sentiment_score'].idxmax()]['review_content'])
    print("Most negative review:", data.loc[data['sentiment_score'].idxmin()]['review_content'])
    print()
