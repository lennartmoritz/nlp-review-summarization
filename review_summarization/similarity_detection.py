#!/usr/bin/env python3
from scipy.spatial import KDTree
from sentence_transformers import CrossEncoder, SentenceTransformer
import pandas as pd


def detect_similarity_crossencoder(data: pd.DataFrame):
    model = CrossEncoder('cross-encoder/stsb-roberta-base')

    max_similarity = (0, (0, 0))
    min_similarity = (1, (0, 0))
    for i, review1 in enumerate(data["review_content"]):
        for j, review2 in enumerate(data["review_content"]):
            if i > j:
                score = model.predict([review1, review2])
                if score > max_similarity[0]:
                    max_similarity = (score, (i, j))
                if score < min_similarity[0]:
                    min_similarity = (score, (i, j))
                print(f"Review {i} and {j} have a similarity score of {score}")

    print(f"Max similarity: {max_similarity}")
    print(f"Most similar reviews: {data.iloc[max_similarity[1][0]]['review_content']} and {data.iloc[max_similarity[1][1]]['review_content']}")


def detect_similarity_bi_encoder(data: pd.DataFrame):
    model = SentenceTransformer('stsb-roberta-large')

    # Encode all movie reviews
    embeddings = model.encode(data["review_content"].values.tolist(), convert_to_tensor=True)

    # A K-D tree can be used for efficient nearest neighbor queries
    # If we normalize the embeddings, k-d trees are equivalent to cosine similarity
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)

    # find two most similar reviews
    tree = KDTree(embeddings)
    dist, ind = tree.query(embeddings, k=2)

    # find most similar review to each review
    most_similar = []
    for i, review in enumerate(data["review_content"]):
        most_similar.append(data.iloc[ind[i][1]]["review_content"])
    data["most_similar_review"] = most_similar

    print("Some examples for similar reviews:")
    for i in range(5):
        print(f"Review {i}: {data.iloc[i]['review_content']}")
        print(f"Most similar review: {data.iloc[i]['most_similar_review']}")
        print()

    # find the closest review to average review
    avg_embedding = embeddings.mean(dim=0)
    _, closest_to_avg_idx = tree.query(avg_embedding, k=1)
    closest_to_avg = data.iloc[closest_to_avg_idx]["review_content"]
    print(f"Closest review to average review: {closest_to_avg}")
    print()
