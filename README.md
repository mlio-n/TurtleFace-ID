# TurtleFace ID — Deniz Kaplumbağası Bireysel Tanıma Sistemi

## 🐢 Proje Özeti

**TurtleFace ID**, deniz kaplumbağalarını yüzlerindeki post-ocular scute (göz çevresi pul) konfigürasyonlarından bireysel olarak tanıyan bir yapay zeka sistemidir. Tıpkı insanlardaki Face ID gibi çalışır; her kaplumbağanın pul deseni parmak izi gibi benzersizdir.

## 🏗️ Mimari

```
turtlefaceid/
├── agents/          → IdentificationAgent (AI Ajan Orkestratörü)
├── detectors/       → FaceDetector (Yüz Tespiti)
├── extractors/      → ScuteExtractor (Pul Haritası)
├── models/          → SiameseNetwork + ContrastiveLoss
├── matchers/        → IdentityMatcher (Kimlik Eşleştirme)
├── database/        → TurtleDatabase (FAISS Vektör Deposu)
└── utils/           → ImageUtils, Visualizer
```

### SOLID Prensipleri

| Sınıf | Sorumluluk |
|-------|-----------|
| `FaceDetector` | Yalnızca yüz tespiti ve kırpma |
| `ScuteExtractor` | Yalnızca pul haritası çıkarımı |
| `SiameseNetwork` | Yalnızca forward pass ve embedding |
| `IdentityMatcher` | Yalnızca kimlik eşleştirme kararı |
| `TurtleDatabase` | Yalnızca kayıt depolama ve arama |
| `IdentificationAgent` | Yalnızca pipeline koordinasyonu |

## 🚀 Kurulum

```bash
# 1. Sanal ortam oluştur
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/Mac

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Demo arayüzünü başlat
streamlit run app.py
```

## 🧠 Neden Siamese Network?

```
Klasik CNN + Softmax:
  → Yeni kaplumbağa eklendi? Modeli YENIDEN EĞİT.
  → 10 kaplumbağa için yeterli veri var mı? HAYIR.

Siamese Network:
  → "Benzerlik metriği" öğrenir, sınıf değil.
  → Yeni kaplumbağa için 1-3 fotoğraf YETERLİ.
  → Buna Few-Shot Learning denir.
```

**Contrastive Loss:**
```
L = (1-y) · d²/2  +  y · max(0, m-d)²/2

y=0 → Aynı kaplumbağa: mesafeyi KÜÇÜLT
y=1 → Farklı kaplumbağa: mesafeyi BÜYÜT (> margin m)
```

## 📊 Demo Arayüzü — 4 Adım

| Adım | Görseli | Açıklama |
|------|---------|----------|
| 1 | Orijinal Fotoğraf | Ham girdi |
| 2 | Kırpılan Yüz | FaceDetector çıktısı |
| 3 | Pul Haritası | ScuteExtractor, renk kodlu konturlar |
| 4 | Eşleşme Sonucu | Benzerlik yüzdesi + kaplumbağa kimliği |

## 📚 Bilimsel Atıflar

- Jean et al. (2010) — Post-ocular scute özgünlüğü
- Carter et al. (2014) — TORSOOI veritabanı formatı
- Chopra et al. (2005) — Contrastive Loss
- He et al. (2016) — ResNet50
