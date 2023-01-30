import numpy as np
import torch
from bertopic import BERTopic
import pandas as pd


def model_topics(data: pd.DataFrame, print_topics: bool = True):
    # convert pd.nan entries from float to string to work with model
    docs = list(data["review_content"].astype('str').values)
    embeddings = torch.stack(data["embedding"].values.tolist()).numpy()

    # Execute the topic modeling
    topic_model = BERTopic(language="english")
    topics, probs = topic_model.fit_transform(docs, embeddings)
    topics = np.array(topics)

    # Print topic results. The -1 topic refers to all outlier documents and is typically ignored
    topic_count = len(np.unique(topics[topics != -1]))

    print(f"The reviews were clustered into {topic_count} topics.")

    if print_topics:
        # Print some representative reviews for each topic
        for topic, representative_reviews in topic_model.get_representative_docs().items():
            print(f"Topic {topic}:")
            for review in representative_reviews:
                print(f"\t{review}")
            print()
        print()

    return pd.DataFrame({"review_content": docs, "topic": topics})
