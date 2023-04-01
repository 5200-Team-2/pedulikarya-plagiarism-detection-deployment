from cosine_similarity import calc_cosine_similarity_from_embeddings
from pickle import load
import torch
from collections import OrderedDict
import numpy as np
from tqdm import tqdm

with open('../data/indosum/df.pkl', 'rb') as f:
    corpus = load(f)

with open('../data/indosum/corpus_embeddings.pkl', 'rb') as f:
    corpus_embeddings = load(f)

with open('../data/indosum/corpus_embeddings_cluster_labels2.pkl', 'rb') as f:
    corpus_embeddings_cluster_labels = load(f)

with open('../data/indosum/kmeans_model2.pkl', 'rb') as f:
    kmeans = load(f)

def get_cluster(sus_doc_embeddings):
    flattened = torch.flatten(sus_doc_embeddings)
    flattened = flattened.flatten().numpy()
    cluster_label = kmeans.predict(np.array([flattened]))[0]
    return cluster_label

def get_source_doc_indexes(sus_doc_cluster_label):
    indexes = []
    for i, label in enumerate(corpus_embeddings_cluster_labels):
        if label == sus_doc_cluster_label:
            indexes.append(i)
    return indexes

def get_similar_docs_by_cos_sims(sus_doc_embeddings, n=5):
    sus_doc_cluster_label = get_cluster(sus_doc_embeddings)
    same_cluster_indexes = get_source_doc_indexes(sus_doc_cluster_label)
    cosine_similarities = {}
    for index in same_cluster_indexes:
        e1 = corpus_embeddings[index].cpu()
        # e1 = corpus_embeddings[index]
        e2 = sus_doc_embeddings
        cosine_similarities[index] = calc_cosine_similarity_from_embeddings(e1, e2)
    
    sorted_ = OrderedDict(sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True))
    sliced_ = OrderedDict(list(sorted_.items())[:n])
    return sliced_

def get_similar_docs_by_cos_sims_without_clustering(sus_doc_embeddings, n=5):
    cosine_similarities = {}
    idx = 0
    for e in tqdm(corpus_embeddings):
        source_embed = e.cpu()
        cosine_similarities[idx] = calc_cosine_similarity_from_embeddings(source_embed, sus_doc_embeddings)
        idx += 1
    
    sorted_ = OrderedDict(sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True))
    sliced_ = OrderedDict(list(sorted_.items())[:n])
    return sliced_