"""
SiameseNetwork — Transfer Learning Tabanlı Siamese Sinir Ağı
=============================================================
NEDEN SIAMESE NETWORK?
----------------------
Klasik sınıflandırma (CNN + Softmax) şu varsayımı gerektirir:
  ► Eğitimde tüm bireyler mevcut olmalıdır.
  ► Yeni bir kaplumbağa eklendiğinde modelin YENİDEN EĞİTİLMESİ gerekir.

Elimizdeki veri kısıtı (az fotoğraf, sürekli yeni bireyler) bu
yaklaşımı imkânsız kılar.

Siamese Network ise şunu öğrenir:
  ► "Bu iki yüz AYNI kaplumbağaya mı ait?"
  ► Yani "sınıf etiketleri" değil, "benzerlik metriği" öğrenir.
  ► Yeni bir kaplumbağa veritabanına eklenmek istendiğinde
    SADECE 1-3 fotoğraf yeterlidir — yeniden eğitime gerek yoktur.
  ► Bu yaklaşım Few-Shot Learning'in özüdür.

CONTRASTIVE LOSS:
-----------------
  L = (1-y) · 0.5 · d²  +  y · 0.5 · max(0, margin - d)²
  
  y=0 → aynı birey: mesafeyi (d) küçültmeye zorla.
  y=1 → farklı birey: mesafeyi margin'den büyük yapmaya zorla.
  Bu sayede embedding uzayında aynı bireylerin yakın, farklıların
  uzak olduğu bir kümeleme kendiliğinden oluşur.

BACKBONE SEÇİMİ — ResNet50:
----------------------------
  - ImageNet üzerinde önceden eğitilmiş ağırlıklar transfer learning
    sağlar; az veriyle bile güçlü alt düzey özellikler (kenar, doku)
    hazır gelir.
  - ResNet'in artık bağlantıları (residual connections) degradasyon
    sorununu çözer; derin ağda gradyan akışını korur.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alt Sınıf — Embedding Ağı (Backbone + Projection Head)
# ---------------------------------------------------------------------------

class EmbeddingExtractor(nn.Module):
    """
    ResNet50 omurgası üzerine inşa edilmiş embedding çıkarıcı.

    Girdi  : (B, 3, 224, 224) — normalize edilmiş RGB görüntü tensörü.
    Çıktı  : (B, embedding_dim) — L2-normalize edilmiş embedding vektörü.

    L2 normalizasyon neden önemli?
        Cosine benzerliği = dot product (L2-norm sonrası), bu da
        eşleşme hesaplamalarını hem hızlı hem sayısal olarak kararlı kılar.
    """

    def __init__(self, embedding_dim: int = 256, freeze_backbone: bool = False) -> None:
        super().__init__()

        # Transfer Learning: ImageNet ağırlıklarıyla başlat
        backbone         = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        in_features      = backbone.fc.in_features   # 2048

        # Son tam bağlantılı katmanı kaldır — biz kendi projection head'imizi ekliyoruz
        self.backbone    = nn.Sequential(*list(backbone.children())[:-1])

        # İsteğe bağlı backbone dondurma (az veri varsa önerilir)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone ağırlıkları donduruldu (freeze_backbone=True).")

        # Projection Head: 2048 → 512 → embedding_dim
        self.projection  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, embedding_dim),
        )

        self._embedding_dim = embedding_dim
        logger.info("EmbeddingExtractor oluşturuldu (dim=%d).", embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 2048, 1, 1)
        embeddings = self.projection(features)  # (B, embedding_dim)
        # L2 normalizasyon — birim küreye yansıt
        return F.normalize(embeddings, p=2, dim=1)

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim


# ---------------------------------------------------------------------------
# Ana Model — Siamese Network
# ---------------------------------------------------------------------------

class SiameseNetwork(nn.Module):
    """
    İki görüntüyü karşılaştıran Siamese (Siyam) Sinir Ağı.

    Mimari:
        ┌─────────────┐      ┌─────────────┐
        │   Image A   │      │   Image B   │
        └──────┬──────┘      └──────┬──────┘
               │  Paylaşılan Ağırlıklar (Shared Weights)
        ┌──────▼──────────────────▼──────┐
        │       EmbeddingExtractor       │
        └──────┬──────────────────┬──────┘
               │ emb_A            │ emb_B
               └────────┬─────────┘
                        │
                  L2 Mesafesi
                        │
                   Similarity Score
    
    Paylaşılan ağırlıklar (weight sharing) neden kritik?
        Her iki kol da AYNI ağ olduğu için aynı "algılama mantığı"nı
        kullanır. Bu, karşılaştırmanın adil ve tutarlı olmasını sağlar.
    """

    def __init__(self, embedding_dim: int = 256, freeze_backbone: bool = False) -> None:
        super().__init__()
        # Tek bir EmbeddingExtractor nesnesi — her iki kol da bunu paylaşır
        self.embedding_extractor = EmbeddingExtractor(
            embedding_dim=embedding_dim,
            freeze_backbone=freeze_backbone,
        )
        logger.info("SiameseNetwork oluşturuldu.")

    def forward(
        self, image_a: torch.Tensor, image_b: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            image_a: (B, 3, 224, 224) — birinci görüntü tensörü.
            image_b: (B, 3, 224, 224) — ikinci görüntü tensörü.

        Returns:
            (emb_a, emb_b, distance)
              emb_a, emb_b : L2-normalize edilmiş embedding vektörleri.
              distance     : Çiftler arası Öklid mesafesi [0, 2].
        """
        emb_a    = self.embedding_extractor(image_a)
        emb_b    = self.embedding_extractor(image_b)
        distance = F.pairwise_distance(emb_a, emb_b, p=2)
        return emb_a, emb_b, distance

    def get_embedding(self, image: torch.Tensor) -> torch.Tensor:
        """Tek bir görüntünün embedding vektörünü döndürür (çıkarım için)."""
        return self.embedding_extractor(image)

    def similarity_score(self, image_a: torch.Tensor, image_b: torch.Tensor) -> float:
        """
        İki görüntü arasındaki benzerlik yüzdesini döndürür.

        Öklid mesafesi → benzerlik dönüşümü:
          sim = exp(-d) → [0, 1] aralığında yorumlanabilir bir skor.
        """
        self.eval()
        with torch.no_grad():
            _, _, distance = self.forward(image_a, image_b)
            similarity = torch.exp(-distance).item()
        return float(similarity)

    # ------------------------------------------------------------------
    # Model Kayıt / Yükleme
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> None:
        """Model ağırlıklarını kaydeder."""
        torch.save(self.state_dict(), str(path))
        logger.info("Model kaydedildi: %s", path)

    def load(self, path: Path | str, device: str = "cpu") -> "SiameseNetwork":
        """Kaydedilmiş ağırlıkları yükler."""
        state = torch.load(str(path), map_location=device)
        self.load_state_dict(state)
        logger.info("Model yüklendi: %s", path)
        return self


# ---------------------------------------------------------------------------
# Kayıp Fonksiyonu — Contrastive Loss
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Siamese Network eğitimi için Contrastive Loss (Chopra et al., 2005).

    Formül:
        L = (1 - y) * 0.5 * d²
            + y * 0.5 * max(0, margin - d)²

    Parametreler:
        y=0 → Pozitif çift (aynı kaplumbağa): mesafeyi küçült.
        y=1 → Negatif çift (farklı kaplumbağa): margin'den büyük yap.

    margin: Pozitif ve negatif çiftleri ayıran minimum mesafe.
            Genellikle 1.0 – 2.0 arasında ayarlanır.
    """

    def __init__(self, margin: float = 1.5) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, distance: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distance : (B,) — çiftler arası Öklid mesafeleri.
            label    : (B,) — 0=aynı birey, 1=farklı birey.

        Returns:
            Skaler kayıp değeri.
        """
        pos_loss = (1 - label) * 0.5 * distance.pow(2)
        neg_loss = label       * 0.5 * F.relu(self.margin - distance).pow(2)
        return (pos_loss + neg_loss).mean()


# ---------------------------------------------------------------------------
# Eğitim Yardımcısı
# ---------------------------------------------------------------------------

class SiameseTrainer:
    """
    SiameseNetwork modelini eğitmekten sorumlu sınıf.

    SRP açısından: Eğitim döngüsü model tanımından ayrılmıştır.
    Model yalnızca forward pass'i bilir; epoch, optimizer, scheduler
    gibi eğitim detayları bu sınıftadır.

    Kullanım Örneği:
        >>> trainer = SiameseTrainer(model, learning_rate=1e-4)
        >>> for epoch in range(50):
        ...     loss = trainer.train_epoch(dataloader)
    """

    def __init__(
        self,
        model       : SiameseNetwork,
        learning_rate: float = 1e-4,
        margin       : float = 1.5,
        device       : str   = "cpu",
    ) -> None:
        self.model     = model.to(device)
        self.device    = device
        self.criterion = ContrastiveLoss(margin=margin)
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Bir eğitim epoch'u çalıştırır ve ortalama kaybı döndürür."""
        self.model.train()
        total_loss = 0.0

        for img_a, img_b, labels in dataloader:
            img_a  = img_a.to(self.device)
            img_b  = img_b.to(self.device)
            labels = labels.float().to(self.device)

            self.optimizer.zero_grad()
            _, _, distance = self.model(img_a, img_b)
            loss = self.criterion(distance, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        self.scheduler.step()
        avg_loss = total_loss / max(len(dataloader), 1)
        logger.debug("Epoch tamamlandı — Ortalama Kayıp: %.4f", avg_loss)
        return avg_loss

    @torch.no_grad()
    def evaluate_epoch(self, dataloader: torch.utils.data.DataLoader) -> float:
        """Doğrulama seti üzerinde kayıp hesaplar."""
        self.model.eval()
        total_loss = 0.0

        for img_a, img_b, labels in dataloader:
            img_a  = img_a.to(self.device)
            img_b  = img_b.to(self.device)
            labels = labels.float().to(self.device)
            _, _, distance = self.model(img_a, img_b)
            loss = self.criterion(distance, labels)
            total_loss += loss.item()

        return total_loss / max(len(dataloader), 1)
