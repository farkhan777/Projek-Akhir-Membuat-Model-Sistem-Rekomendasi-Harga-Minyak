# Laporan Proyek Machine Learning - Nama Anda

## Domain Proyek
Ekspektasi produksi minyak dunia masih mendapatkan hambatan berupa pembatasan cukup ketat Covid-19 di sejumlah negara yang memiliki tingkat permintaan konsumsi tinggi.

Sejumlah faktor memicu pergerakan harga minyak dunia seperti pembatasan mobilitas di China , badai tropis Ida di Amerika Serikat, dan proyeksi penambahan pasokan dari organisasi negara-negara pengekspor minyak bumi (OPEC) membuat 'gonjang-ganjing' harga di pasaran. Di sini saya mencoba unutk memprediksi harga minyak menggunakan data harga minyak yang ada dari tahun ke-tahun.


## Business Understanding
### Problem Statements
Bagaimana prediksi harga minyak dalam kurun waktu satu tahun kedepan ?

### Goals
Membuat prediksi harga minyak tahunan berdasarkan data yang ada.

### Solution statements
Untuk menyelesaikan masalah ini saya menggunakan Time Series. Menurut apa yang telah saya pelajari di kelas Belajar Penegmbangan Machine Learning di Dicoding.com, Time series dapat dipahami sebagai kumpulan nilai yang tersusun secara runtut dalam rentang waktu tertentu. Contohnya adalah, jumlah subscriber baru dari sebuah channel Youtube setiap harinya selama 1 tahun. Contoh lain yang paling umum adalah harga saham sebuah perusahaan dalam satu tahun. Harga minyak tahunan juga termasuk.

Di sini saya menggunakan 1 buah layer LSTM (Long Short Term Memory). [LSTM](https://mti.binus.ac.id/2019/12/02/long-short-term-memory-lstm/) merupakan salah satu jenis dari Recurrent Neural Network (RNN) dimana dilakukan modifikasi pada RNN dengan menambahkan memory cell yang dapat menyimpan informasi untuk jangka waktu yang lama (Manaswi, 2018). LSTM diusulkan sebagai solusi untuk mengatasi terjadinya vanishing gradient pada RNN saat memproses data sequential yang panjang.


## Data Understanding
Untuk dataset sendiri saya ambil dari [Brent Oil Prices](https://www.kaggle.com/mabusalah/brent-oil-prices) yang berada di platform [kaggle](https://www.kaggle.com/). Berikut adalah keterangan mengenai maksud dari variable - variable atau kolom tersebut:

* Date: Kolom ini termasuk hari-bulan-tahun dalam format tanggal waktu.
* Price: harga minyak harian dalam USD.

Di bawah ini adalah deskripsi harga minyak dari dataset yang saya gunakan.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/desc.png?token=ANXJTPNOKFESZB34454GNR3BPBM2Y)
Dari data di atas terlihat bahwa rata-rata atau mean harga minya dari tahun 1987 hingga 2021 adalah US$46,352962. Untuk harga maximum atau tertingginya dari tahun 1987 hingga 2021 adalah sebesar US$143,95. Dan untuk harga minimum atau terendahnya menyentuh harga US$9,1.

Berikut adalah grafik harga minyak dari tahun 1987 hingga 2021. Terlihat bahwa harga minyak terjadi fluktuasi di setiap tahunnya. Pada tahun 2003 hingga 2008 harga minyak meningkat drastis disetiap tahunnya hingga menyentuh harga tertinggi yaitu US$143.95. Di sekitar akhir tahun 2008 atau awal tahun 2009 harga minyak menurun drastis hingga menyentuh harga disekitar US$30 - US$40.

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/graph.png?token=ANXJTPMUYWOHLIBEMYNAFGLBPBPP2)


## Data Preparation
Untuk data preparation sendiri saya meggunakan beberapa cara. Saya menggunakan to_datetime yang ada pada library pandas untuk merubah format "Date" yang ada pada dataset. Selain itu saya menggunakan train test split untuk mengevaluasi kinerja algoritma Machine Learning. Saya juga melakukan preprocessing data menggunakan MinMaxScaler yang berfungsi untuk mengubah data berada di rentang 0 sampai 1.

Berikut adalah contoh ilustrasi data sebelum dan sesudah melakukan preprocessing menggunakan MinMaxScaler yang ada pada library sklearn menurut [sumber](https://medium.com/@uulwake/kesalahan-scaling-data-di-machine-learning-menggunakan-scikit-learn-7b88f2fbaec) yang saya baca.

![image](https://miro.medium.com/max/2000/1*IDqKtvddj8ROkH-r93qrzw.png)

Gambar kiri adalah distribusi data sebelum normalisasi. Bandingkan distribusi data gambar kiri dan tengah. Jika melakukan normalisasi, harusnya distribusi datanya sama namun dengan skala yang lebih kecil. Akan tetapi, distribusi data gambar tengah sangat berbeda dengan distribusi gambar kiri. Perbedaannya jelas terlihat. Hal ini dikarenakan saya menormalisasi training dataset dan test dataset menggunakan dua buah scaler yang berbeda satu sama lain. Jadi setelah melakukan normalisasi, sebenarnya data sudah berbeda. Maka dari itu, model yang kita buat akan sia-sia saja.
Sekarang perhatikan gambar kanan dan gambar kiri lalu bandingkan kedua gambar tersebut. Kedua gambar tersebut sama persis, hanya berbeda di skala saja. Inilah cara melakukan normalisasi yang benar yaitu dengan menggunakan sebuah scaler saja.


## Modeling
Untuk struktur model di sini saya menggunakan 1 buah layer LSTM (Long Short Term Memory). LSTM sendiri menurut [sumber](https://www.geeksforgeeks.org/deep-learning-introduction-to-long-short-term-memory/) yang saya baca adalah sejenis jaringan saraf berulang. Dalam RNN output dari langkah terakhir diumpankan sebagai input pada langkah saat ini. LSTM dirancang oleh Hochreiter & Schmidhuber. Ini mengatasi masalah ketergantungan jangka panjang RNN di mana RNN tidak dapat memprediksi kata yang disimpan dalam memori jangka panjang tetapi dapat memberikan prediksi yang lebih akurat dari informasi terbaru. Dengan bertambahnya panjang celah, RNN tidak memberikan kinerja yang efisien. LSTM dapat secara default menyimpan informasi untuk jangka waktu yang lama. Ini digunakan untuk memproses, memprediksi, dan mengklasifikasikan berdasarkan data deret waktu.

Saya juga menggunakan Dropout untuk mencega overfitting. Menurut apa yag telah saya pelajari di kelas Belajar [Pengembangan Machine Learning](https://www.dicoding.com/academies/185), Dropout adalah standar umum di industri yang dipakai untuk mencegah overfitting. Seperti yang kita ketahui, semakin kompleks sebuah model ML, maka akan semakin tinggi kemungkinan model tersebut mengalami overfitting. Dropout bekerja dengan mengurangi kompleksitas model jst tanpa merubah arsitektur model tersebut. 

Bagaimana dropout bekerja? Nama dropout mengacu pada unit/perseptron yang di-dropout (dibuang) secara temporer pada sebuah layer. Contohnya seperti di bawah di mana besaran dropout yang dipilih adalah 0.5 sehingga 50% dari persepteron hidden layer kedua dimatikan secara berkala pada saat pelatihan. 

![image](https://d17ivq9b7rppb3.cloudfront.net/original/academy/20200803125202b077a1253a77def9b9e4ae6b553bc1cc.gif)

Di bawah ini adalah grafik loss yang dihasilkan dari proses training model yang saya buat.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/loss.png?token=ANXJTPNA2WPJ2JMZSDXG3ALBPEDIM)

Di bawah ini adalah grafik mae yang dihasilkan dari proses training model yang saya buat.
![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/mae2.png?token=ANXJTPKL24KW2LUDEPJBMWDBPEDIY)

Dan di bawah ini adalah hasil dari prediksi harga minyak yang menunjukkan bahwa harga prediksi mirip dengan harga sebernarnya.

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/predic.png?token=ANXJTPIJQ2WT2C7YWK6NMN3BPEDI4)


## Evaluation
Untuk bagian Evaluasi, Saya menguji performa model ini dengan mean squared error (MSE) dan mean absolute error (MAE). Menurut sumber yang saya temukan, kedua metrik ini sangat cocok untuk mengukur performa model machine learning. Berikut adalah penjelasan dari setiap metrik :

* [Mean Absolute Error](https://medium.com/@ewuramaminka/mean-absolute-error-mae-machine-learning-ml-b9b4afc63077): Mean Absolute Error, juga dikenal sebagai MAE, adalah salah satu dari banyak metrik untuk meringkas dan menilai kualitas model pembelajaran mesin. Mengingat setiap kumpulan data pengujian, Mean Absolute Error model mengacu pada rata-rata nilai absolut dari setiap kesalahan prediksi pada semua instance dari kumpulan data pengujian. Kesalahan prediksi adalah perbedaan antara nilai aktual dan nilai prediksi untuk instance itu.

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/mae.png?token=ANXJTPKE4ZMSQFESF7AZ623BPTOKQ)
Untuk menerapkan dalam kode, Kamu hanya perlu mengetik kode dibawah ini :
metrics=["mae"]

* [Mean Squared Error](https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html): Mean Squared Error (MSE) mungkin adalah fungsi loss yang paling sederhana dan paling umum, sering diajarkan dalam kursus pengantar Machine Learning. Metode Mean Squared Error secara umum digunakan untuk mengecek estimasi berapa nilai kesalahan pada peramalan. Nilai Mean Squared Error yang rendah atau nilai mean squared error mendekati nol menunjukkan bahwa hasil peramalan sesuai dengan data aktual dan bisa dijadikan untuk perhitungan peramalan di periode mendatang. Metode Mean Squared Error biasanya digunakan untuk mengevaluasi metode pengukuran dengan model regressi atau model peramalan seperti Moving Average, Weighted Moving Average dan Analisis Trendline. Cara menghitung Mean Squared Error (MSE) adalah melakukan pengurangan nilai data aktual dengan data peramalan dan hasilnya dikuadratkan (squared) kemudian dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data yang ada.

![image](https://raw.githubusercontent.com/farkhan777/Proyek-Pertama-Kirim-Submission-dan-Review/main/mse.png?token=ANXJTPNGXGXGF6VIVUOP33TBPTOKU)
Untuk menerapkan dalam kode, Kamu hanya perlu mengetik kode dibawah ini :
loss='mean_squared_error'

