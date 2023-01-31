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


class BiEncoderSimilarity:
    def __init__(self):
        self.model = SentenceTransformer('stsb-roberta-large')
        self.data = None
        self.embeddings = None
        self.tree = None

    def set_data(self, data: pd.DataFrame):
        self.data = data
        # Encode given movie reviews
        self.embeddings = self.model.encode(data["review_content"].values.tolist(), convert_to_tensor=True)
        self.embeddings = self.embeddings.cpu()

        # A K-D tree can be used for efficient nearest neighbor queries
        # If we normalize the embeddings, k-d trees are equivalent to cosine similarity
        self.embeddings = self.embeddings / self.embeddings.norm(dim=1, keepdim=True)

        self.tree = KDTree(self.embeddings)

    def get_average_review(self):
        """Find the closest review to the average review"""
        assert self.data is not None, "Call set_data() before getting reviews"

        # Calculate average embedding
        avg_embedding = self.embeddings.mean(dim=0)
        # Query tree
        _, closest_to_avg_idx = self.tree.query(avg_embedding, k=1)
        closest_to_avg = self.data.iloc[closest_to_avg_idx]["review_content"]
        return closest_to_avg

    def get_closest_review_to_each(self):
        """Find the most similar review to each of the reviews"""
        assert self.data is not None, "Call set_data() before getting reviews"

        # find two most similar reviews (first is always review itself)
        dist, ind = self.tree.query(self.embeddings, k=2)

        # find most similar review to each review
        most_similar = []
        for i, review in enumerate(self.data["review_content"]):
            most_similar.append(self.data.iloc[ind[i][1]]["review_content"])
        self.data["most_similar_review"] = most_similar

        print("Some examples for similar reviews:")
        for i in range(5):
            print(f"Review {i}: {self.data.iloc[i]['review_content']}")
            print(f"Most similar review: {self.data.iloc[i]['most_similar_review']}")
            print()
