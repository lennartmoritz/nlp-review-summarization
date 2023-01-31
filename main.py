#!/usr/bin/env python3
# Authors: Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler

import argparse
import random
import sys

import pandas as pd

from review_summarization.dataset_loading import load_roto_dataset, print_dataset_information
from review_summarization.encoder import SentenceEncoder
from review_summarization.sentiment_classification import classify_sentiment, AVAILABLE_MODELS
from review_summarization.similarity_detection import BiEncoderSimilarity
from review_summarization.topic_modelling import model_topics


def aggregate_reviews(data: pd.DataFrame, common_embedding=False):
    print_dataset_information(data)

    if common_embedding:
        # encode the reviews
        encoder = SentenceEncoder()
        sentence_data = encoder.encode_sentences(data["review_content"].values.tolist())
        data["embedding"] = list(sentence_data.detach())

    # classify the reviews by sentiment
    classify_sentiment(data, args.sentiment, common_embedding=common_embedding)

    model_topics(data, common_embedding=common_embedding, print_topics=False)

    similarity_detection = BiEncoderSimilarity(common_embedding=common_embedding)
    for topic in data.topic.unique():
        if topic == -1:
            # skip list of unclassified reviews
            continue
        reviews_for_topic = data[data.topic == topic]
        similarity_detection.set_data(reviews_for_topic)
        avg_review = similarity_detection.get_average_review()
        print(f"\t{avg_review}")


if __name__ == '__main__':
    # TODO: Selection of data (maybe subsets of data as grouped by "dataset_id"- or based on word-count)
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('--sentiment', default=AVAILABLE_MODELS[0], choices=AVAILABLE_MODELS)
    group = input_parser.add_mutually_exclusive_group()
    group.add_argument('-r', '--random', action='store_true', help='Aggregate review for a random movie', default=True)
    group.add_argument('-a', '--all', action='store_true', help='Aggregate review for all movies')
    group.add_argument('-s', '--search', help='Search for a specific movie')
    input_parser.add_argument('-c', '--common-embedding', action='store_true', help='Use a common encoder for all the tasks')

    args = input_parser.parse_args()

    data = load_roto_dataset()

    # Filter dataset to only include the movies with enough reviews
    movie_title_counts = data.movie_title.value_counts()
    movie_titles = movie_title_counts[movie_title_counts > 50].index.values

    movies = []
    if args.all:
        movies = movie_titles
    elif args.search:
        if args.search in movie_titles:
            movies = [args.search]
        else:
            movies = list(filter(lambda title: title.lower().startswith(args.search.lower()), movie_titles))
        if not movies:
            movies = list(filter(lambda title: args.search.lower() in title.lower(), movie_titles))
    elif args.random:
        movies = [random.choice(movie_titles)]
    else:
        sys.exit("No movie provided")

    for movie in movies:
        movie_reviews = data[data.movie_title == movie]
        aggregate_reviews(movie_reviews, common_embedding=args.common_embedding)
