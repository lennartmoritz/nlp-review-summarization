import os

import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data.csv')


def load_dataset():
    data = pd.read_csv(DATASET_PATH, encoding='unicode_escape')
    data.dropna(subset=['review_content'], inplace=True)
    return data


def load_roto_dataset():
    """
    The Rotten Tomatoes dataset consists of short reviews with and without scores.
    All data points have a movie_title and review_content.
    No data points are classified with a sentiment.
    """
    data = load_dataset()
    return data[data['dataset_id'] == "roto"]


def load_imdb_dataset():
    """
    The IMDB dataset consists of longer reviews without scores or movie titles.
    Some reviews are already classified with a negative sentiment.
    """
    data = load_dataset()
    return data[data['dataset_id'] == "imdb"]


def print_dataset_information(data):
    """
    Prints information about the dataset.
    """
    print(f"Dataset has {len(data)} rows.")
    print(f"Dataset has {len(data['movie_title'].unique())} unique movie titles.")
    print(f"Dataset has {len(data['review_content'].unique())} unique reviews.")

    data['review_length'] = data['review_content'].str.count(' ') + 1
    print(f"Average review length: {data['review_length'].mean()} words.")
    print()
