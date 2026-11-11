# 🫀 Kalp Hastalığı Tahmini (Sınıflandırma Projesi)

Bu proje, hastaların klinik verilerini ve sağlık parametrelerini kullanarak kalp hastalığı riski taşıyıp taşımadığını tahmin eden makine öğrenmesi modellerini ve detaylı veri analizlerini içerir.

---

## 📊 Veri Seti Bilgisi
Projede Kaggle platformu üzerinden erişilen **Heart Disease UCI** veri seti kullanılmıştır.
* **Veri Seti Linki:** [Kaggle - Heart Disease Data]( https://www.kaggle.com/code/zeeshanyounas001/heart-disease-uci/input )
* **Veri Özeti:** Toplam 920 hasta kaydı bulunmaktadır. Orijinal veri setindeki `num` (hastalık derecesi) değişkeni dönüştürülerek 509 hasta (`1`) ve 411 sağlıklı (`0`) kayıt elde edilmiştir. Sınıflar dengeli bir dağılıma sahiptir.

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler
* **Python 3.10+**
* **Pandas:** Veri okuma, temizleme ve ön işleme
* **NumPy:** Sayısal ve matris hesaplamaları
* **Matplotlib & Seaborn:** Keşifçi veri analizi ve grafik çizimleri
* **Scikit-Learn:** Model eğitimi, veri bölme ve başarım metriklerinin hesaplanması

---

## 🛠️ Proje Mimarisi ve Geliştirme Süreci

Proje geliştirme süreci 5 ana adımdan oluşmaktadır:

1. **Veri Ön İşleme (Data Preprocessing):**
   * **Target Oluşturma:** `num` sütunundaki değerler `0` ise `0` (sağlıklı), `0`'dan büyük ise `1` (hasta) olarak ikili (binary) sınıflandırma değişkenine dönüştürüldü.
   * **Eksik Değer Yönetimi (Imputation):** Sayısal sütunlardaki eksik değerler medyan (ortanca), kategorik sütunlardaki eksik değerler ise mod (en çok tekrar eden) yöntemiyle dolduruldu.
   * **Kategorik Dönüşüm:** Metin tabanlı kategorik değişkenler (`sex`, `cp`, `restecg` vb.) modellerin işleyebilmesi için **One-Hot Encoding** yöntemiyle dönüştürüldü.

2. **Keşifçi Veri Analizi (EDA):**
   * Veri setindeki özelliklerin hastalık üzerindeki etkisini görselleştirmek amacıyla 5 farklı analiz grafiği oluşturuldu.

3. **Model Eğitimi:**
   * Veri seti %80 eğitim, %20 test olarak ikiye ayrıldı (`train_test_split`)[cite: 1].
   * Temel sınıflandırma modeli olarak **Logistic Regression** eğitildi[cite: 1].

4. **Model Karşılaştırması (Bonus):**
   * Başarımı kıyaslamak amacıyla **Decision Tree** ve **Random Forest** algoritmaları da projeye dahil edildi[cite: 1].

5. **Değerlendirme ve Hata Analizi:**
   * Accuracy, Precision, Recall metrikleri ve Karmaşıklık Matrisi (Confusion Matrix) hesaplandı[cite: 1].

---

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerel ortamınızda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

1. Gerekli Python kütüphanelerini yükleyin:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
