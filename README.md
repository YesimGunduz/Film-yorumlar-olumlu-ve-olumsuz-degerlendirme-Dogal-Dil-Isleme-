## Proje Tanımı

Bu proje, Bilgi Mühendisliğine Giriş dersi kapsamında bir grup projesi olarak gerçekleştirilmiştir. Temel amaç, kullanıcıların film yorumları üzerinden pozitif ya da negatif duygu durumu içeren yorumları tespit edebilecek bir makine öğrenmesi sistemi geliştirmektir. 

Proje kapsamında veri toplama, ön işleme, analiz, model eğitimi, değerlendirme ve görselleştirme adımları uygulanmıştır. Proje Python programlama dili ile gerçekleştirilmiş olup, veri bilimi ve makine öğrenmesi alanlarına giriş niteliğindedir.

## Proje İçeriği ve Yöntem

### Kullanılan Veri Kümesi

- *yorumlar_5000.csv*: İnternetten elde edilmiş ham film yorumlarından oluşan CSV dosyası.
- *proje_csv_duzgun_son.csv*: Temizlenmiş ve sınıflandırmaya uygun hale getirilmiş veri kümesi. Bu dosya model eğitimi ve testinde kullanılmıştır.

### Veri Ön İşleme

- Eksik verilerin kontrolü ve temizlenmesi
- Küfür, özel karakter ve stop word (gereksiz kelime) filtrelemesi
- Küçük harfe çevirme ve noktalama işaretlerinin kaldırılması
- Tokenization ve temel doğal dil işleme adımları

### Modelleme

Aşağıdaki sınıflandırma algoritmaları ile deneyler gerçekleştirilmiştir:

- Naive Bayes
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest

Model performansları doğruluk (% accuracy), F1 skorları ve karışıklık matrisi kullanılarak karşılaştırılmıştır.

### Görselleştirme

- Veri seti üzerindeki sınıf dağılımı
- Eğitim ve test sonuçlarının grafikle gösterimi
- Karışıklık matrislerinin görsel sunumu
