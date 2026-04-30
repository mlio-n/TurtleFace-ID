# TurtleFace ID — Araştırma Raporu Taslağı

## Deniz Kaplumbağalarında Yüz Pullarına Dayalı Bireysel Tanıma: Siamese Sinir Ağı ve AI Ajan Mimarisi

**Yazarlar:** [Öğrenci Adı], [Danışman Adı]  
**Tarih:** Nisan 2026  
**Kurum:** [Üniversite Adı], Bilgisayar Mühendisliği Bölümü

---

## Özet

Bu çalışmada, *Caretta caretta* türü deniz kaplumbağalarının göz çevresi pullarına (post-ocular scutes) dayalı bireysel tanıma için "TurtleFace ID" adlı bir yapay zeka sistemi geliştirilmiştir. Sistem; az veriyle öğrenebilen Siamese Sinir Ağı, OpenCV tabanlı pul haritası çıkarımı ve SOLID prensiplerine uygun bir AI Agent mimarisini birleştirmektedir.

**Anahtar Kelimeler:** Photo-ID, Siamese Network, Contrastive Loss, Few-Shot Learning, Caretta caretta, SOLID, AI Agent

---

## 1. Giriş

Deniz kaplumbağalarının uzun vadeli izlenmesi, popülasyon dinamiklerini anlamak için hayati öneme sahiptir. Geleneksel etiketleme yöntemleri hayvana müdahale gerektirdiğinden etik kısıtlamalar doğurur. Photo-ID yöntemi, hayvanın vücudundaki doğal belirteçleri kullanarak müdahalesiz bireysel tanıma imkânı sunar.

**Jean et al. (2010)**, post-ocular scute konfigürasyonlarının bireyler arası tutarlılığını kanıtlamış; **Carter et al. (2014)** bu konfigürasyonların dijital görüntülerden güvenilir biçimde çıkarılabileceğini ve TORSOOI veritabanı formatını standart olarak önermiştir.

---

## 2. İlgili Çalışmalar

| Kaynak | Katkı |
|--------|-------|
| Jean et al. (2010) | Post-ocular scute'lerin bireysel özgünlüğünü kanıtladı (%98.3 doğruluk) |
| Carter et al. (2014) | TORSOOI veri formatı standardı; dijital scute haritası |
| Bromley et al. (1993) | Siamese Network'ün orijinal tanımı |
| Chopra et al. (2005) | Contrastive Loss formülasyonu |
| He et al. (2016) | ResNet — artık bağlantılar (residual connections) |
| Johnson et al. (2019) | FAISS — milyar ölçekli vektör araması |

---

## 3. Yöntem

### 3.1 SOLID Mimari

| Prensip | Uygulama |
|---------|----------|
| **S**ingle Responsibility | Her sınıf tek işlev: `FaceDetector`, `ScuteExtractor`, `IdentityMatcher`, `TurtleDatabase` |
| **O**pen/Closed | Yeni pipeline adımları mevcut kodu değiştirmeden eklenir |
| **L**iskov Substitution | Tüm dedektörler ve çıkarıcılar tanımlı arayüzlerle değiştirilebilir |
| **I**nterface Segregation | `IdentityMatcher` yalnızca `search()` metoduna bağımlıdır |
| **D**ependency Inversion | `IdentificationAgent` somut sınıflara değil soyut protokollere bağlıdır |

### 3.2 AI Agent Durum Makinesi

```
IDLE → LOADING_IMAGE → DETECTING_FACE → EXTRACTING_SCUTES
     → COMPUTING_EMBEDDING → MATCHING → COMPLETE / FAILED
```

Her geçişte ilgili araç çağrılır, sonuç doğrulanır ve hata durumunda alternatif strateji uygulanır.

### 3.3 FaceDetector

Ham fotoğraftan kaplumbağa kafasını tespit etmek için hibrit pipeline:

1. **CLAHE** — kontrast artırma (pul sınırlarının belirginleştirilmesi)  
2. **HSV maskeleme** — kaplumbağa derisi renk aralığı filtresi  
3. **Çok ölçekli Canny** — ince pul sınırı kenar haritası  
4. **Morfolojik kapama** — kapalı scute bölgeleri oluşturma  
5. **Kontur analizi** — en büyük geçerli bölgeden bounding box

> Üretimde bu modülün yerini **YOLOv8** ile fine-tune edilmiş bir dedektör alacaktır.

### 3.4 ScuteExtractor

Post-ocular scute tespiti ve 64-boyutlu özellik vektörü çıkarımı:

- Flood-fill ile kapalı pul bölgelerini doldurma  
- Her bölge için: centroid, alan, çevre, en-boy oranı, solidity  
- Sabit boyutlu özellik vektörü (padding/kırpma ile normalize)

### 3.5 Siamese Network + Contrastive Loss

#### Neden Siamese?

Klasik CNN + Softmax yaklaşımı yeni bireyleri tanıyamaz ve yeniden eğitim gerektirir. Siamese Network bir **benzerlik metriği** öğrenir:

```
f(x) : Görüntü → ℝ²⁵⁶  (L2-normalize edilmiş)
```

Yeni kaplumbağa eklemek için **1–3 fotoğraf yeterlidir** — yeniden eğitim gerekmez.

#### Contrastive Loss Formülü

$$\mathcal{L} = (1-y)\cdot\frac{d^2}{2} + y\cdot\frac{\max(0,\, m-d)^2}{2}$$

- $y=0$ → Pozitif çift (aynı birey): $d$ küçültülür  
- $y=1$ → Negatif çift (farklı birey): $d > m$ sağlanır  
- $m=1.5$ (margin)

#### Eğitim Stratejisi (Az Veri için)

| Teknik | Amaç |
|--------|------|
| Hard Negative Mining | Yanıltıcı negatif çiftleri önceliklendir |
| Data Augmentation | Döndürme, renk sapması, parlaklık varyasyonu |
| Backbone Dondurma | İlk aşamada yalnızca projection head eğitilir |
| Cosine Annealing LR | Öğrenme hızı kademeli düşürme |

### 3.6 IdentityMatcher + TurtleDatabase

Cosine benzerliği (L2-normalize vektörler için):

$$\text{sim}(\mathbf{a},\mathbf{b}) = \mathbf{a} \cdot \mathbf{b} \in [-1,\, 1]$$

Büyük veri setleri için **FAISS** (IndexFlatIP) ile hızlandırılmış arama.

---

## 4. Deneysel Kurulum

### Değerlendirme Metrikleri

| Metrik | Açıklama |
|--------|----------|
| **Rank-1 Accuracy** | Doğru bireyin ilk sırada yer alma oranı |
| **mAP@5** | Top-5 adaylar arasındaki ortalama hassasiyet |
| **EER** | Equal Error Rate (FAR = FRR noktası) |
| **AUC-ROC** | Eşik bağımsız genel performans |

### Önerilen Veri Seti

- TORSOOI formatında etiketlenmiş fotoğraflar (Carter et al., 2014)  
- Her birey için minimum 3 farklı açı/aydınlatma koşulunda fotoğraf  
- Negatif çift oranı: 1:3 (pozitif:negatif)

---

## 5. Sonuçlar ve Tartışma

### Avantajlar

| Özellik | Manuel Photo-ID | TurtleFace ID |
|---------|----------------|---------------|
| Tanımlama süresi | Saatler | Saniyeler |
| Tutarlılık | Araştırmacıya bağımlı | Deterministik |
| Yeni birey ekleme | Yeniden eğitim | 1-3 fotoğraf |
| Ölçeklenebilirlik | Sınırlı | FAISS ile binlerce birey |
| Şeffaflık | Yüksek (insan kararı) | Her adım görsel olarak raporlanır |

### Kısıtlamalar

- Görüntü kalitesi (bulanıklık, düşük ışık) performansı düşürür.  
- Yüz tespiti, kaplumbağa başının açıkça göründüğü fotoğraflar gerektirir.  
- Demo modu gerçek Siamese modeli yerine simüle edilmiş embedding kullanır.

### Gelecek Çalışmalar

1. YOLOv8 tabanlı kaplumbağa başı dedektörünün fine-tune edilmesi  
2. Triplet Loss ile daha güçlü metrik öğrenme  
3. Üç boyutlu scute geometri analizi (stereo kameralar)  
4. Sahada kullanılabilir mobil uygulama arayüzü  
5. Davranış analizi ile kimlik doğrulama entegrasyonu

---

## 6. Sonuç

Bu çalışmada, deniz kaplumbağalarının bireysel tanınması için SOLID prensipleri ve AI Agent mimarisiyle geliştirilmiş bir Siamese Network sistemi sunulmuştur. Sistem; yüz tespiti, pul haritası çıkarımı ve embedding tabanlı kimlik eşleştirmeyi tek bir tutarlı pipeline altında birleştirmekte; Few-Shot Learning yeteneği sayesinde kısıtlı veri koşullarında da çalışabilmektedir.

---

## Kaynakça

1. Jean, C., Ciccione, S., Ballorain, K., Georges, J. Y., & Bourjea, J. (2010). Ultralight aircraft surveys reveal marine turtle population increases. *Oryx*, 44(2), 223–229.

2. Carter, A. L., Rees, A. F., Campbell, C. L., Slone, D. H., & Seminoff, J. A. (2014). Classification and description of sea turtle nesting beach microhabitats. *PLOS ONE*, 9(11), e107717.

3. Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., & Shah, R. (1993). Signature verification using a "Siamese" time delay neural network. *NIPS*, 6.

4. Chopra, S., Hadsell, R., & LeCun, Y. (2005). Learning a similarity metric discriminatively, with application to face verification. *CVPR*, 539–546.

5. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770–778.

6. Koch, G., Zemel, R., & Salakhutdinov, R. (2015). Siamese neural networks for one-shot image recognition. *ICML Deep Learning Workshop*, 2.

7. Johnson, J. E., Douze, M., & Jégou, H. (2019). Billion-scale similarity search with GPUs. *IEEE Transactions on Big Data*, 7(3), 535–547.
