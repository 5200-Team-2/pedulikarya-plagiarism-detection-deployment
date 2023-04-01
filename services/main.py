'''
TODO:
- Transform suspicious text to embeddings
- Similar Documents Retrieval: get top 5 similar documents
- Detailed Analysis: get sentences pairs that have high cosine similarity
'''
from get_embeddings import mean_pooling_bert
from similar_documents_retrieval import get_similar_docs_by_cos_sims, get_similar_docs_by_cos_sims_without_clustering
from detailed_analysis import get_plagiarised_pairs
from datetime import datetime, timedelta
from nltk.tokenize import sent_tokenize
from pickle import load, dump

with open('../data/indosum/df.pkl', 'rb') as f:
    corpus = load(f)

def transform_suspicious_text(sus_text):
    print('[INFO] Transforming suspicious text into IndoBERT embeddings...')
    embeddings = mean_pooling_bert(text=sus_text)
    return embeddings

def get_n_similar_documents(sus_text, n=5, clustering=True):
    embeddings = transform_suspicious_text(sus_text=sus_text)
    print(f'[INFO] Getting {n} similar documents by cosine similarity...')
    if clustering == True:
        similar_docs = get_similar_docs_by_cos_sims(sus_doc_embeddings=embeddings, n=n)
    else:
        similar_docs = get_similar_docs_by_cos_sims_without_clustering(sus_doc_embeddings=embeddings, n=n)

    return similar_docs

def detailed_plagiarism(sus_text, n=5, threshold=.9, clustering=True):
    sent_tok = sent_tokenize(sus_text)
    similar_sources = get_n_similar_documents(sus_text=sus_text, n=n, clustering=clustering)
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
# document = '''Meksiko menunjukkan ketahanan lagi. Kamis (22/06), tertinggal dari El Tori berhasil membalikkan keadaan dan akhirnya menjadi 2 - 1 di leg kedua Grup A Piala Konfederasi 2017 di Stadion Olimpiade Fisht Sochi melawan Selandia Baru. Namun, dua gol Raul Jimenez (54') dan Oribe Peralta (72') akhirnya memupus harapan All Whites untuk bangkit di turnamen tersebut. Tambahan tiga poin itu untuk sementara membawa Meksiko ke puncak klasemen Grup A dengan raihan empat poin. Poin mereka sebenarnya sama dengan Portugal, namun tim asuhan Juan Carlos Osorio memiliki kemampuan mencetak gol yang lebih baik. Selandia Baru, sebaliknya, belum mencetak satu poin pun dalam dua pertandingan, sehingga dipastikan akan tersingkir. Meksiko tidak menunjukkan sisi terbaiknya kali ini. Osorio melakukan delapan pergantian pemain dari hasil imbang melawan Portugal. Pemain seperti Marco Fabian, Diego Reyes, Giovani dos Santos dan Oribe Peralta mendapat kesempatan bermain sejak awal. Tak hanya itu, formasi dasar kali ini diubah menjadi 3-4-3, namun sebenarnya perubahan tersebut tidak berdampak besar pada keseluruhan pertandingan melawan Meksiko. Bukti bahwa mereka masih bisa mendominasi lawan, baik dalam penguasaan bola maupun dalam menciptakan peluang. Secara keseluruhan, Meksiko menguasai bola 65% dari waktu. Selain itu, mereka mampu menciptakan 16 peluang mencetak gol, 10 di antaranya tepat sasaran. Tapi mungkin karakter Meksiko tertinggal lebih dulu. Pasalnya, meski selalu dikunci, Selandia Baru justru lebih dulu tampil baik berkat Wood. Pencetak gol terbanyak divisi Championship di Leeds United itu memanfaatkan kekeliruan pertahanan Meksiko, yang mampu menaklukkan kiper Alfredo Tavarel. Begitu babak kedua dimulai, Meksiko mulai. Hanya sembilan menit setelah kick-off, Jiménez berhasil mengakhiri serangan lima pemain dengan tembakan keras dari area penalti. Gol kemenangan Meksiko dicetak pada menit ke-72 setelah tembakan brilian dari kiri oleh Javier Aquino dilanjutkan dengan jebakan yang dengan mudah dikonversi Peralta menjadi gol. Patut dicatat bahwa dalam pertandingan melawan Portugal, Meksiko tertinggal dua kali sebelum akhirnya merebut poin. Di laga ini juga Meksiko harus kebobolan terlebih dahulu untuk akhirnya mencetak gol dan menang. Selain tayangan ulang tersebut, pertandingan tersebut memiliki tiga acara menarik. Pertama, cederanya bek utama Carlos Salcedo. Pemain Eintracht Frankfurt itu diganti pada babak pertama karena cedera bahu. Kedua, pertarungan antara pemain Meksiko dan Selandia Baru setelah pertarungan sengit antara Michael Boxall dan Hector Herrera Terakhir, pidato khusus oleh Rafael Marquez. Marquez yang diperkuat Barcelona digantikan oleh Osorio dan Hector Moreno pada menit ke-68. Moreno sendiri merupakan pemain yang menggantikan Salcedo di babak pertama. Dengan menghadapi Marquez, pemain berusia 38 tahun itu resmi menjadi pemain tertua kedua dalam sejarah Piala Konfederasi. Di pertandingan final, Meksiko akan menghadapi Rusia. Hasil imbang sudah cukup untuk melaju ke semifinal El Tri, terlepas dari hasil yang didapat Portugal dalam pertandingan melawan Selandia Baru. Namun, jika ingin menghindari juara Grup B, tujuan utama Anda adalah menang.'''
# document = '''Timnas Indonesia kembali diturunkan setelah pelatih Alfred Riedl memutuskan melepas Rizuki Pera, Septian David Moran, dan Do Mingus Fakdaba. Tiga pemain tersisih setelah timnas Indonesia kembali dari uji coba di Myanmar dan Vietnam. Rizky, David dan Fakdaver jelas tak akan mengikuti pemusatan latihan di Karavaci, Tangerang. “Setelah kembali dari Vietnam, kami melepas Septian David, Lizky Pera yang masuk daftar tunggu, dan Do Mingus.-Catatan: Bagaimana Riedl membentuk Timnas Indonesia Ahli taktik asal Austria itu tidak memberikan detail apapun tentang tindakan disipliner apa yang diambil Fakdaver. Ini sesi latihan nasional di Piala AFF 2016. Riedl memanggil penjaga gawang Semyon Padang Jandia Ekputra, yang sebelumnya dipertahankan untuk pertandingan Tes di Myanmar dan Vietnam.'''
# # sentences = sent_tokenize(document)
# # print(sentences)
# start = datetime.now()
# plagiarism_cases = detailed_plagiarism(sus_text=document, n=3, threshold=.9, clustering=False)
# print(plagiarism_cases)
# end = datetime.now()
# total_time = end-start
# print(f"total time: {total_time}")








