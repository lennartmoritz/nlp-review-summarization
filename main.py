#!/usr/bin/env python3
# Authors: Carlotta Mahncke, Lennart Joshua Moritz, Timon Engelke and Christian Schuler

import argparse

from review_summarization.dataset_loading import load_roto_dataset, print_dataset_information
from review_summarization.sentiment_classification import classify_sentiment, AVAILABLE_MODELS
from review_summarization.similarity_detection import detect_similarity_bi_encoder
from review_summarization.topic_modelling import model_topics

if __name__ == '__main__':
    # TODO: Selection of data (maybe subsets of data as grouped by "dataset_id"- or based on word-count)
    input_parser = argparse.ArgumentParser()
    input_parser.add_argument('-s', '--sentiment', default=AVAILABLE_MODELS[0], choices=AVAILABLE_MODELS)

    args = input_parser.parse_args()

    data = load_roto_dataset()

    # Testing on small subset
    data = data[:100]

    print_dataset_information(data)

    classify_sentiment(data, args.sentiment)
    detect_similarity_bi_encoder(data)
    model_topics(data)
