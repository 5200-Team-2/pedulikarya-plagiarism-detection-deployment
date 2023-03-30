import torch
from transformers import BertTokenizer, BertModel
from get_embeddings import mean_pooling_bert
from cosine_similarity import calc_cosine_similarity_from_embeddings
from nltk.tokenize import sent_tokenize

# tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
# model = BertModel.from_pretrained('indobenchmark/indobert-base-p1', output_hidden_states=True)

def compare_sentences(src_sent, sus_sent):
    src_embedded = mean_pooling_bert(src_sent).cpu()
    sus_embedded = mean_pooling_bert(sus_sent).cpu()
    cos_sim = calc_cosine_similarity_from_embeddings(src_embedded, sus_embedded)
    return cos_sim

def compare_documents(src_doc, sus_doc):
    src_sents = src_doc
    sus_sents = sent_tokenize(sus_doc)
    score_pairs = []
    print(f'[INFO] Length of source: {len(src_sents)}')
    print(f'[INFO] Length of sus: {len(sus_sents)}')
    for i, src_sent in enumerate(src_sents):
        for j, sus_sent in enumerate(sus_sents):
            # print(F'[INFO] Checking current source sentence no {i} with sus sentence no {j}...')
            cos_sim = compare_sentences(src_sent, sus_sent)
            score_pairs.append((src_sent, sus_sent, cos_sim))
    return score_pairs

def get_plagiarised_pairs(src_doc, sus_doc, threshold=.9):
    plagiarised_pairs = []
    score_pairs = compare_documents(src_doc, sus_doc)
    for src_sent, sus_sent, cos_sim in score_pairs:
        if cos_sim >= threshold:
            # plagiarised_pairs.append((src_sent, sus_sent, cos_sim))
            plagiarised_pairs.append({'source_sentence': src_sent, 'input_sentence': sus_sent, 'similarity_score': cos_sim})
    return plagiarised_pairs



# text1 = 'Versi web Google Calendar memiliki sesuatu yang baru untuk ditawarkan. Penampilannya telah sepenuhnya diperbarui, mengadopsi gaya desain modern yang sebelumnya tersedia pada aplikasi Google Calendar di perangkat mobile.'
# text2 = 'Ada yang baru dari Google Calendar versi web. Tampilannya kini telah dirombak habis, mengadopsi gaya desain modern yang sebelumnya sudah tersedia pada aplikasi Google Calendar di perangkat mobile.'
# print(get_plagiarised_pairs(text1, text2, .85))
    
