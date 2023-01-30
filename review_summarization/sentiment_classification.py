import torch
from transformers import pipeline
import pandas as pd
from transformers.pipelines.text_classification import TextClassificationPipeline


# distilbert-base-uncased-finetuned-sst-2-english classifies into "POSITIVE" or "NEGATIVE" + score how certain
# bert-base-multilingual-uncased-sentiment classifies into "1 star", "2 stars", ... "5 stars" + score how certain
AVAILABLE_MODELS = ['distilbert-base-uncased-finetuned-sst-2-english', 'nlptown/bert-base-multilingual-uncased-sentiment']


class EmbeddingTextClassificationPipeline(TextClassificationPipeline):
    def postprocess(self, model_outputs, function_to_apply=None, top_k=1, _legacy=True):
        dict_scores = super().postprocess(model_outputs, function_to_apply, top_k, _legacy)
        last_transformer_hidden_state = model_outputs.hidden_states[-1]
        dict_scores['embedding'] = torch.squeeze(last_transformer_hidden_state[:, 0])  # class token pooling
        return dict_scores


def classify_sentiment(data: pd.DataFrame, classifier: str):
    # Cast column dtype and then convert to list returning a numpy array
    review_contents = data["review_content"].astype(str).values.tolist()

    print("Using pretrained sentiment analysis model: " + classifier)

    # Load the sentiment analysis model
    sent_pipe = pipeline(task='sentiment-analysis', model=classifier, tokenizer=classifier, model_kwargs={'output_hidden_states': True}, pipeline_class=EmbeddingTextClassificationPipeline)

    # Predict the sentiments
    sentiment_default = sent_pipe(review_contents)

    # Extract two lists from the predictions containing the labels and the scores
    sentiment_labels = [item['label'] for item in sentiment_default]
    sentiment_scores = [item['score'] for item in sentiment_default]
    embeddings = [item['embedding'] for item in sentiment_default]

    # Add the predictions to the dataframe
    data['sentiment_label'] = sentiment_labels
    data['sentiment_score'] = sentiment_scores
    data['embedding'] = embeddings

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
