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

def transform_suspicious_text(sus_text):
    embeddings = mean_pooling_bert(text=sus_text)
    return embeddings

def get_n_similar_documents(sus_text, n=5):
    embeddings = transform_suspicious_text(sus_text=sus_text)
    similar_docs = get_similar_docs_by_cos_sims(sus_doc_embeddings=embeddings, n=n)
    return similar_docs

def detailed_plagiarism(sus_text, n=5, threshold=.9):
    similar_source_docs = get_n_similar_documents(sus_text=sus_text, n=n)
    suspected_plagiarism_cases = {}
    for doc_index, doc in enumerate(similar_source_docs):
        suspected_plagiarism_cases[doc_index] = get_plagiarised_pairs(src_doc=doc, sus_doc=sus_text, threshold=threshold)
    return suspected_plagiarism_cases


# test
document = '''Versi web Google Calendar memiliki sesuatu yang baru untuk ditawarkan. Penampilannya telah sepenuhnya diperbarui, mengadopsi gaya desain modern yang sebelumnya tersedia pada aplikasi Google Calendar di perangkat mobile. Secara keseluruhan, versi Calendar baru terlihat lebih rapi dan lebih mudah untuk diarahkan. Layout responsif yang digunakan berarti penampilannya akan disesuaikan secara otomatis dengan ukuran layar dan jendela browser. Bukan hanya berubah secara visual, versi Calendar baru juga membawa beberapa fitur yang berguna. Salah satunya adalah kemampuan untuk memformat dan menambahkan tautan pada undangan, sehingga pengguna dapat membuat jadwal yang lebih detail dan memberikan akses kepada materi yang dibutuhkan oleh peserta lain sebelum rapat dimulai. Calendar sekarang juga membuat lebih mudah untuk melihat dan mengelola beberapa kalender dari akun yang berbeda dalam tampilan harian. Fitur ini tentunya akan sangat membantu pekerja yang harus menjadwalkan rapat untuk anggota tim mereka. Terakhir, saat membuat jadwal rapat baru, pengguna dapat menambahkan informasi yang lebih detail tentang ruang rapat, termasuk lokasi, kapasitas, peralatan audio dan video, dan informasi tentang aksesibilitas bagi peserta yang berkedudukan kursi roda. Bagi yang menggunakan Calendar untuk akun pribadi mereka, mereka bisa mengaktifkan tampilan baru dengan mengklik tombol "Use new Calendar" di ujung kanan atas. Administrator G Suite juga dapat mengaktifkan tampilan baru Calendar mulai hari ini. Sumber: Google. DailySocial.id adalah portal berita startup dan inovasi teknologi. Kamu bisa menjadi anggota komunitas startup dan inovasi DailySocial.id, mengunduh laporan riset dan statistik seputar teknologi secara gratis, dan mengikuti berita startup dan gadget terbaru di Indonesia.'''
start = datetime.now()
plagiarism_cases = detailed_plagiarism(sus_text=document, n=1, threshold=.9)
print(plagiarism_cases)
end = datetime.now()
total_time = end-start
print(f"total time: {total_time}")








