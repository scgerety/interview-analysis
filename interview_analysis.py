#!/usr/bin/env python

import glob
import json
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re
import sys
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

PUNCT = r'.!?'

model = SentenceTransformer('all-MiniLM-L6-v2')
output_file = os.path.abspath(sys.argv[1])
this_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(this_dir, "data")
helper_dir = os.path.join(this_dir, "Supporting-Docs")

if not os.path.exists(helper_dir):
    os.mkdir(helper_dir)


def main():
    files = find_docs()
    data = []
    for input_file in files:
        lines = parse_docs(input_file)
        data.append(lines)
    # score_matrix = find_distances(data)
    # plot_dendrogram(score_matrix)
    agglomerative(data)


def find_docs():
    return glob.glob(f"{data_dir}/*")


def parse_docs(input_file):
    with open(input_file, "r") as doc:
        lines = [line for line in doc if line[0] not in ["[", "1"]]

    lines = "".join(lines)
    lines = re.split(f"{PUNCT}", lines)

    return lines


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

    corpus_embeddings = embedder.encode(data)
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
