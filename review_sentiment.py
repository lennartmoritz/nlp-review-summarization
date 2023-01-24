#!/usr/bin/env python3
# Authors: 
# Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler

from transformers import pipeline
import argparse
import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

#TODO: What "structure" do the tutors want? Separate functions? God-class?
# Instantiate the sentiment analysis pipeline
def load_sentiment_model():
  sent_pipe_default = pipeline(task = 'sentiment-analysis')

# Predict the sentiment of an input sentence
def predict_sentiment(sent_pipe, sentence):
  result = sent_pipe(sentence)
# => Better just inside a for-loop inside of main(?)

if __name__ == "__main__":
  # Read data
  data = pd.read_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data.csv", encoding='unicode_escape')

  # Number of words and distribution of review lenghts
  #print("Entire Data")
  #data['review_length'] = data['review_content'].str.count(' ') + 1
  #print(data['review_length'].value_counts())

  # Select roto (Rotten Tomatoes) reviews
  data = data[data['dataset_id'] == "roto"]
  
  # Number of words and distribution of review lenghts
  #print("Only roto (Rotten-Tomatoes) Data")
  #data['review_length'] = data['review_content'].str.count(' ') + 1
  #print(data['review_length'].value_counts())

  # Testing on small subset
  data = data[:1000]
  
  # Cast column dtype and then convert to list returning a numpy array
  review_contents = data["review_content"].astype(str).values.tolist()
  
  # TODO: Selection of models and data (maybe subsets of data as grouped by "dataset_id"- or based on word-count)
  # Accepting arguments to select sentiment analysis models
  input_parser = argparse.ArgumentParser()
  input_parser.add_argument('-s', '--sentiment', default='default', choices=['distilbert-base-uncased-finetuned-sst-2-english', 'bert-base-multilingual-uncased-sentiment'])
  # 'distilbert-base-uncased-finetuned-sst-2-english' is the 'default' model for sentiment analysis
  # Used like: python script.py -s modelName 

  args = input_parser.parse_args()
  
  ##############################################################################
  # Model A: Resulting in sentiment labels "POSITIVE" or "NEGATIVE" + score how certain
  # https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
  print('Using pretrained default sentiment')
  sent_pipe = pipeline(task = 'sentiment-analysis')
  
  # Names for columns in final dataframe
  current_label = "distilbertbaseuncasedsst2_label"
  current_score = "distilbertbaseuncasedsst2_score"
  
  # Predict the sentiments
  sentiment_default = sent_pipe(review_contents)

  # Extract two lists from the predictions containing the labels and the scores  
  sentiment_default_labels = [item['label'] for item in sentiment_default]
  sentiment_default_scores = [item['score'] for item in sentiment_default]

  # Add predicted labels and corresponding scored to the dataframe
  data['sentiment_default-label'] = pd.Series(sentiment_default_labels).values
  data['sentiment_default-score'] = pd.Series(sentiment_default_scores).values

  data.to_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data-sentiment-distilbertbaseuncasedsst2.csv")
  
  ##############################################################################
  # https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
  # Model B: Resulting in sentiment labels "1 star", "2 stars", ... "5 stars" + score how certain  
  print('Using pretrained bert-base-multilingual-uncased-sentiment')
  sent_pipe = pipeline(task = 'sentiment-analysis', model = 'nlptown/bert-base-multilingual-uncased-sentiment', tokenizer = 'nlptown/bert-base-multilingual-uncased-sentiment')
  
  # Names for columns in final dataframe
  current_label = "bertbasemultiuncased_label"
  current_score = "bertbasemultiuncased_score"
  
  # Predict the sentiments
  sentiment_default = sent_pipe(review_contents)

  # Extract two lists from the predictions containing the labels and the scores  
  sentiment_default_labels = [item['label'] for item in sentiment_default]
  sentiment_default_scores = [item['score'] for item in sentiment_default]

  # Add predicted labels and corresponding scored to the dataframe
  data['sentiment_default-label'] = pd.Series(sentiment_default_labels).values
  data['sentiment_default-score'] = pd.Series(sentiment_default_scores).values

  data.to_csv("/home/christianschuler/data/NLP-Project-Review-Summaries/data-sentiment-bertbasemultiuncased.csv")

