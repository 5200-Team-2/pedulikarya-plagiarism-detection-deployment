from get_embeddings import mean_pooling_bert
from similar_documents_retrieval import get_similar_docs_by_cos_sims

if __name__ == "__main__":

   document2 = '''Versi web Google Calendar memiliki sesuatu yang baru untuk ditawarkan. Penampilannya telah sepenuhnya diperbarui, mengadopsi gaya desain modern yang sebelumnya tersedia pada aplikasi Google Calendar di perangkat mobile. Secara keseluruhan, versi Calendar baru terlihat lebih rapi dan lebih mudah untuk diarahkan. Layout responsif yang digunakan berarti penampilannya akan disesuaikan secara otomatis dengan ukuran layar dan jendela browser. Bukan hanya berubah secara visual, versi Calendar baru juga membawa beberapa fitur yang berguna. Salah satunya adalah kemampuan untuk memformat dan menambahkan tautan pada undangan, sehingga pengguna dapat membuat jadwal yang lebih detail dan memberikan akses kepada materi yang dibutuhkan oleh peserta lain sebelum rapat dimulai. Calendar sekarang juga membuat lebih mudah untuk melihat dan mengelola beberapa kalender dari akun yang berbeda dalam tampilan harian. Fitur ini tentunya akan sangat membantu pekerja yang harus menjadwalkan rapat untuk anggota tim mereka. Terakhir, saat membuat jadwal rapat baru, pengguna dapat menambahkan informasi yang lebih detail tentang ruang rapat, termasuk lokasi, kapasitas, peralatan audio dan video, dan informasi tentang aksesibilitas bagi peserta yang berkedudukan kursi roda. Bagi yang menggunakan Calendar untuk akun pribadi mereka, mereka bisa mengaktifkan tampilan baru dengan mengklik tombol "Use new Calendar" di ujung kanan atas. Administrator G Suite juga dapat mengaktifkan tampilan baru Calendar mulai hari ini. Sumber: Google. DailySocial.id adalah portal berita startup dan inovasi teknologi. Kamu bisa menjadi anggota komunitas startup dan inovasi DailySocial.id, mengunduh laporan riset dan statistik seputar teknologi secara gratis, dan mengikuti berita startup dan gadget terbaru di Indonesia.
   '''
   embeddings = mean_pooling_bert(document2)

   res = get_similar_docs_by_cos_sims(embeddings)
   print(res)


   # document = [
   #    'Ada yang baru dari Google Calendar versi web.',
   #    'Tampilannya kini telah dirombak habis, mengadopsi gaya desain modern yang sebelumnya sudah tersedia pada aplikasi Google Calendar di perangkat mobile.',
   #    'Secara keseluruhan, Calendar versi baru tampak lebih rapi dan lebih mudah dinavigasikan.',
   #    'Layout responsif yang digunakan berarti tampilannya akan disesuaikan secara otomatis dengan ukuran layar maupun jendela browser.',
   #    'Tidak hanya berubah dari segi visual, Calendar versi baru juga mengemas sejumlah fitur yang cukup bermanfaat.',
   #    'Salah satunya adalah kemampuan mengatur formatting dan menambahkan link pada undangan, sehingga pengguna dapat menciptakan agenda yang lebih merinci, dan materi- materi yang diperlukan bisa diakses oleh peserta lain sebelum rapat dimulai.',
   #    'Calendar kini juga memberikan kemudahan untuk melihat dan mengatur beberapa kalender dari akun yang berbeda dalam tampilan harian.',
   #    'Fitur ini tentunya akan sangat memudahkan pekerja yang tugasnya mengaturkan jadwal rapat anggota-anggota timnya.',
   #    'Terakhir, saat sedang membuat jadwal rapat baru, pengguna dapat menambahkan informasi yang lebih merinci soal ruang rapat yang akan digunakan, mulai dari lokasi ruangan, kapasitasnya, kelengkapan peralatan audio dan videonya, sampai informasi mengenai akses untuk peserta yang berkursi roda.',
   #    'Bagi yang menggunakan Calendar pada akun pribadinya, mereka sudah bisa mengaktifkan tampilan baru ini dengan mengklik tombol “ Use new Calendar ” di ujung kanan atas.',
   #    'Administrator G Suite juga dapat mengaktifkan tampilan baru Calendar ini mulai hari ini juga.',
   #    'Sumber: Google.',
   #    'DailySocial.id adalah portal berita startup dan inovasi teknologi.',
   #    'Kamu bisa menjadi member komunitas startup dan inovasi DailySocial.id, mengunduh laporan riset dan statistik seputar teknologi secara cuma-cuma, dan mengikuti berita startup Indonesia dan gadget terbaru.'
   # ]

   # document = " ".join(document)

   