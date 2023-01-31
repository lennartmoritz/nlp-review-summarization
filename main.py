#!/usr/bin/env python3
# Authors: Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler

import argparse
import random

import pandas as pd

from review_summarization.dataset_loading import load_roto_dataset, print_dataset_information
from review_summarization.sentiment_classification import classify_sentiment, AVAILABLE_MODELS
from review_summarization.similarity_detection import BiEncoderSimilarity
from review_summarization.topic_modelling import model_topics


def aggregate_reviews(data: pd.DataFrame):
    print_dataset_information(data)

    classify_sentiment(data, args.sentiment)
    topic_data = model_topics(data, print_topics=False)

    similarity_detection = BiEncoderSimilarity()
    for topic in topic_data.topic.unique():
        if topic == -1:
            # skip list of unclassified reviews
            continue
        reviews_for_topic = topic_data[topic_data.topic == topic]
        similarity_detection.set_data(reviews_for_topic)
        avg_review = similarity_detection.get_average_review()
        print(f"\t{avg_review}")


if __name__ == '__main__':
    # TODO: Selection of data (maybe subsets of data as grouped by "dataset_id"- or based on word-count)
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--sentiment', default=AVAILABLE_MODELS[0], choices=AVAILABLE_MODELS)
    group = input_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-r', '--random', action='store_true', help='Aggregate review for a random movie')
    group.add_argument('-a', '--all', action='store_true', help='Aggregate review for all movies')
    group.add_argument('-s', '--search', help='Search for a specific movie')

    args = input_parser.parse_args()

    data = load_roto_dataset()

    # Filter dataset to only include the movies with enough reviews
    movie_title_counts = data.movie_title.value_counts()
    movie_titles = movie_title_counts[movie_title_counts > 50].index.values

    movies = []
    if args.random:
        movies = [random.choice(movie_titles)]
    elif args.all:
        movies = movie_titles
    elif args.search:
        if args.search in movie_titles:
            movies = args.search
        else:
            movies = list(filter(lambda title: title.startswith(args.search), movie_titles))
        if not movies:
            movies = list(filter(lambda title: args.search in title, movie_titles))

    for movie in movies:
        movie_reviews = data[data.movie_title == movie]
        aggregate_reviews(movie_reviews)
