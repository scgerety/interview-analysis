#!/usr/bin/env python

import glob
import json
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

model = SentenceTransformer('all-MiniLM-L6-v2')
output_file = os.path.abspath(sys.argv[1])
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "data")
helper_dir = os.path.join(this_dir, "Supporting-Docs")
if not os.path.exists(helper_dir):
    os.mkdir(helper_dir)


def main():
    files = find_docs()
    data = parse_docs()
    # score_matrix = find_distances(data)
    # plot_dendrogram(score_matrix)
    agglomerative(data)


def find_docs():
    return glob.glob(data/*)


def parse_docs():
    with open(input_file, "r") as doc:
        rows = [row.split(",") for row in doc]
        header = rows[1] # The data is from Qualtrics. When I don't design the surveys with
                         # meaningful short variables, row 1 is more meaningful than row 0.
        data = rows[3:]  # But row 2 is almost always useless for my purposes, so skip to row 3.

    comment = header.index("Please provide any additional comments you would like to add here.")
    session = header.index("session") - len(header)
    # Indexing session this way works because the comment column comes before the session column
    
    data = [{"id": idx, "session": row[session].replace(r'"', ''), "comment": ",".join(row[comment:session]).replace(r'"', '')} \
            for idx, row in enumerate(data) \
            if len(",".join(row[comment:session-1])) > 0] # Putting session as the limit on comment works FOR THIS DATASET
                                              # because session happens to come right after comment.
                                              # I'm rejoining comments together, because I've split them up by
                                              # commas: it was a csv.
    
    return data


def find_distances(data):
    score_matrix = dict()
    data_tfs = [{
        "id": response["id"],
        "tf": model.encode(response["comment"]),
        "session": response["session"]
        } for response in data]

    for response in data_tfs:
        score_matrix[response['id']] = [{
            "other_id": other["id"],
            "score": 1 - distance.cosine(response["tf"], other["tf"]),
            "session": other["session"]
            } for other in data_tfs if other["id"] != response["id"]
        ]

    return score_matrix


def plot_dendrogram(score_matrix):
    arr = []
    for idx, response in score_matrix.items():
        closest_pair_score, session, closest_pair_id = sorted([(
            score_info["score"],
            score_info["session"],
            score_info["other_id"]
            ) for score_info in response])[-1]
        arr.append({idx: {"other_id": closest_pair_id, "score": closest_pair_score, "session": session}})

    print(arr)
    

def agglomerative(data):
    """
    agglomerative is directly lifted from https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/clustering/agglomerative.py
    """
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1.5,
            )

    corpus = [row["comment"] for row in data if len(row["comment"]) > 0]
    corpus_embeddings = embedder.encode(corpus)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(corpus[sentence_id])

    for i, cluster in clustered_sentences.items():
        print("Cluster ", i + 1)
        print(cluster)
        print("")



if __name__ == "__main__":
    main()
