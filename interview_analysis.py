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

PUNCT = r'\n.!?'

model = SentenceTransformer('all-MiniLM-L6-v2')
output_file = os.path.abspath(sys.argv[1])
figname = f"{sys.argv[2]}"
title = f"{sys.argv[3]}"
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
        data.extend(lines)
    # score_matrix = find_distances(data)
    # plot_dendrogram(score_matrix)
    model = train_model(data)
    fig, ax = plt.subplots()
    sentence_matrix = plot_dendrogram(model, truncate_mode="level", p=5)
    plt.title(f"{title}")
    plt.xlabel("Index")
    plt.ylabel("Disance")
    plt.savefig(f"{os.path.join(helper_dir, figname)}")
    save_sentences(sentence_matrix)


def find_docs():
    return glob.glob(f"{data_dir}/*")


def parse_docs(input_file):
    with open(input_file, "r") as doc:
        lines = [line for line in doc if line[0] not in ["[", "1"]]

    lines = " ".join(lines)
    lines = re.split(f"{PUNCT}", lines)

    return lines
    

def train_model(data):
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=0,
            compute_distances=True,
            )

    corpus_embeddings = embedder.encode(np.array(data))
    
    clustering_model.fit(corpus_embeddings)

    return clustering_model


def plot_dendrogram(model, **kwargs):
    """
    plot_dendrogram is lifted directly from https://www.geeksforgeeks.org/machine-learning/hierarchical-clustering/
    """
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]).astype(float)
    dendrogram(linkage_matrix, **kwargs)
    sentence_matrix = model.labels_

    return sentence_matrix


def save_sentences(sentence_matrix):
    with open(output_file, "w") as out:
        sentences = [sentence for sentence in sentence_matrix]

        for line in sentences:
            out.write("Children, Distances, Counts\n")
            out.write(f"{line}\n")


if __name__ == "__main__":
    main()
