import numpy as np
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertConfig, TrainingArguments, Trainer, DistilBertTokenizer, \
    DataCollator, DataCollatorWithPadding
from datasets import load_dataset

from review_summarization.dataset_loading import load_roto_dataset, load_score_dataset
from review_summarization.encoder import SentenceEncoder
from review_summarization.sentiment_classification import classify_sentiment, AVAILABLE_MODELS


def train_model():
    """Train the classification head for the sentiment analysis task"""
    config = DistilBertConfig.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    model = DistilBertForSequenceClassification(config)
    model.distilbert.from_pretrained('distilbert-base-uncased')
    for param in model.distilbert.parameters():
        param.requires_grad = False

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    dataset = load_dataset('sst2')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    tokenized_dataset = dataset.map(lambda d: tokenizer(d['sentence']), batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['validation'],
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(eval_results)


def evaluate_on_labelled_data(common_embedding=False):
    """This function evaluates a sentiment classifier on the aggregated data of Rotten Tomatoes"""
    data = load_roto_dataset()[:1000]
    score_dataset = load_score_dataset()
    concat_data = pd.concat([data, score_dataset], axis=1, join='inner')

    if common_embedding:
        encoder = SentenceEncoder()
        sentence_data = encoder.encode_sentences(data["review_content"].values.tolist())
        concat_data["embedding"] = list(sentence_data.detach())

    classify_sentiment(concat_data, AVAILABLE_MODELS[0], common_embedding=common_embedding)

    accuracy = np.sum(concat_data.sentiment_label == concat_data.normalised_pone) / len(concat_data)
    print('Accuracy:', accuracy)
    true_positives = np.sum((concat_data.sentiment_label == 'POSITIVE') & (concat_data.normalised_pone == 'POSITIVE'))
    precision = true_positives / np.sum(concat_data.sentiment_label == 'POSITIVE')
    recall = true_positives / np.sum(concat_data.normalised_pone == 'POSITIVE')
    print('Precision:', precision)
    print('Recall:', recall)


if __name__ == '__main__':
    train_model()
    #evaluate_on_labelled_data()
