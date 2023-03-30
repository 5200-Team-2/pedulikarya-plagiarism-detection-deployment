'''
TODO:
- Transform suspicious text to embeddings
- Similar Documents Retrieval: get top 5 similar documents
- Detailed Analysis: get sentences pairs that have high cosine similarity
'''
from get_embeddings import mean_pooling_bert
from similar_documents_retrieval import get_similar_docs_by_cos_sims
from detailed_analysis import get_plagiarised_pairs
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
from pickle import load, dump

with open('../data/indosum/df.pkl', 'rb') as f:
    corpus = load(f)

def transform_suspicious_text(sus_text):
    print('[INFO] Transforming suspicious text into BERT embeddings...')
    embeddings = mean_pooling_bert(text=sus_text)
    return embeddings

def get_n_similar_documents(sus_text, n=5):
    embeddings = transform_suspicious_text(sus_text=sus_text)
    print(f'[INFO] Getting {n} similar documents by cosine similarity...')
    similar_docs = get_similar_docs_by_cos_sims(sus_doc_embeddings=embeddings, n=n)
    return similar_docs

def detailed_plagiarism(sus_text, n=5, threshold=.9):
    sent_tok = sent_tokenize(sus_text)
    similar_sources = get_n_similar_documents(sus_text=sus_text, n=n)
    similar_source_docs = {}
    suspected_plagiarism_cases = {}
    print('[INFO] Getting source documents..')
    for src_idx in similar_sources:
        similar_source_docs[src_idx] = corpus['paragraphs_sent_tok'][src_idx]
        print(f'[INFO] Source document {src_idx} appended')

    print('[INFO] Performing detailed analysis...')
    list_of_source_doc_urls = []
    suspected_plagiarised_sentences = []
    sum_cos_sim = 0
    for doc_idx, doc in similar_source_docs.items():
        # print(doc)
        src_document_url = corpus['source_url'][src_idx]
        if src_document_url not in list_of_source_doc_urls:
            list_of_source_doc_urls.append(src_document_url)
            plagiarised_pairs = get_plagiarised_pairs(src_doc=doc, sus_doc=sus_text, threshold=threshold)
            for pair in plagiarised_pairs:
                input_sentence = pair['input_sentence']
                if input_sentence not in suspected_plagiarised_sentences:
                    suspected_plagiarised_sentences.append({input_sentence: pair['similarity_score']})
                    sum_cos_sim += pair['similarity_score']
                else:
                    idx_to_be_deleted = 0
                    for idx, item in enumerate(suspected_plagiarised_sentences):
                        if pair['similarity_score'] > item[input_sentence]:
                            idx_to_be_deleted = idx
                            del suspected_plagiarised_sentences[idx_to_be_deleted]
                            suspected_plagiarised_sentences.append({input_sentence: pair['similarity_score']})
                            sum_cos_sim += pair['similarity_score']

            suspected_plagiarism_cases[doc_idx] = {
                'source_doc_url': corpus['source_url'][src_idx],
                'plagiarised_pairs': plagiarised_pairs
            }
        print(f'[INFO] Source document {doc_idx} checked')

    return {'overall_plagiarism_score': sum_cos_sim/len(sent_tok), 'suspected_sentences': suspected_plagiarism_cases}


# test
# document = '''Versi web Google Calendar memiliki sesuatu yang baru untuk ditawarkan. Penampilannya telah sepenuhnya diperbarui, mengadopsi gaya desain modern yang sebelumnya tersedia pada aplikasi Google Calendar di perangkat mobile. Secara keseluruhan, versi Calendar baru terlihat lebih rapi dan lebih mudah untuk diarahkan. Layout responsif yang digunakan berarti penampilannya akan disesuaikan secara otomatis dengan ukuran layar dan jendela browser. Bukan hanya berubah secara visual, versi Calendar baru juga membawa beberapa fitur yang berguna. Salah satunya adalah kemampuan untuk memformat dan menambahkan tautan pada undangan, sehingga pengguna dapat membuat jadwal yang lebih detail dan memberikan akses kepada materi yang dibutuhkan oleh peserta lain sebelum rapat dimulai. Calendar sekarang juga membuat lebih mudah untuk melihat dan mengelola beberapa kalender dari akun yang berbeda dalam tampilan harian. Fitur ini tentunya akan sangat membantu pekerja yang harus menjadwalkan rapat untuk anggota tim mereka. Terakhir, saat membuat jadwal rapat baru, pengguna dapat menambahkan informasi yang lebih detail tentang ruang rapat, termasuk lokasi, kapasitas, peralatan audio dan video, dan informasi tentang aksesibilitas bagi peserta yang berkedudukan kursi roda. Bagi yang menggunakan Calendar untuk akun pribadi mereka, mereka bisa mengaktifkan tampilan baru dengan mengklik tombol "Use new Calendar" di ujung kanan atas. Administrator G Suite juga dapat mengaktifkan tampilan baru Calendar mulai hari ini. Sumber: Google. DailySocial.id adalah portal berita startup dan inovasi teknologi. Kamu bisa menjadi anggota komunitas startup dan inovasi DailySocial.id, mengunduh laporan riset dan statistik seputar teknologi secara gratis, dan mengikuti berita startup dan gadget terbaru di Indonesia.'''
# # sentences = sent_tokenize(document)
# # print(sentences)
# start = datetime.now()
# plagiarism_cases = detailed_plagiarism(sus_text=document, n=3, threshold=.9)
# print(plagiarism_cases)
# end = datetime.now()
# total_time = end-start
# print(f"total time: {total_time}")








