import os

import pandas as pd

DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data.csv')
NORMALISED_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data-sentiment-normalised.csv')


def load_dataset():
    data = pd.read_csv(DATASET_PATH, encoding='unicode_escape', dtype={
        "review_content": str, "review_sentiment": str, "dataset_id": str, "review_id": str, "review_score": str,
        "author_id": str, "movie_title": str
    })
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


def load_score_dataset():
    data = pd.read_csv(NORMALISED_DATASET_PATH)
    data = data.set_index('ID')
    return data


def print_dataset_information(data):
    """
    Prints information about the dataset.
    """
    if len(data['movie_title'].unique()) == 1:
        print(f" === {data.iloc[0]['movie_title']} ===")
    else:
        print(f"Dataset has {len(data['movie_title'].unique())} unique movie titles.")

    print(f"Aggregating {len(data)} reviews.")

    data['review_length'] = data['review_content'].str.count(' ') + 1
    print(f"Average review length: {data['review_length'].mean():.2f} words.")
    print()
