from bertopic import BERTopic
import pandas as pd


def model_topics(data: pd.DataFrame, print_topics: bool = True):
    # convert pd.nan entries from float to string to work with model
    docs = list(data.loc[:, "review_content"].astype('str').values)

    # Execute the topic modeling
    topic_model = BERTopic(language="english")
    topics, probs = topic_model.fit_transform(docs)

    # Print topic results. The -1 topic refers to all outlier documents and is typically ignored
    topic_info = topic_model.get_topic_info()

    print(f"The reviews were clustered into {len(topic_info) - 1} topics.")

    if print_topics:
        # Print some representative reviews for each topic
        for topic, representative_reviews in topic_model.get_representative_docs().items():
            print(f"Topic {topic}:")
            for review in representative_reviews:
                print(f"\t{review}")
            print()
        print()

    return pd.DataFrame({"review_content": docs, "topic": topics})
