from get_embeddings import document_mean_pooling_bert
from scipy.spatial.distance import cosine
import torch

def calc_cosine_similarity(text1, text2):
    embeddings1 = document_mean_pooling_bert(text1)
    embeddings2 = document_mean_pooling_bert(text2)

    embeddings1 = torch.flatten(embeddings1)
    embeddings2 = torch.flatten(embeddings2)
    
    similarity = 1 - cosine(u=embeddings1, v=embeddings2)
    return similarity

def calc_cosine_similarity_from_embeddings(embeddings1, embeddings2):
    embeddings1 = torch.flatten(embeddings1)
    embeddings2 = torch.flatten(embeddings2)
    
    similarity = 1 - cosine(u=embeddings1, v=embeddings2)
    return similarity
