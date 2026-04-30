"""
IdentityMatcher — Kimlik Eşleştirme Motoru
==========================================
Single Responsibility Principle (SRP):
    Bu sınıfın TEK sorumluluğu: bir sorgu embedding'ini veritabanındaki
    kayıtlarla karşılaştırarak en iyi eşleşmeyi bulmaktır.
    Tespit, çıkarım veya veritabanı yönetimi bu sınıfın kapsamı DIŞINDADIR.

Dependency Inversion Principle (DIP):
    Bu sınıf, somut TurtleDatabase sınıfına değil; bir arayüze (protokol)
    bağımlıdır. Bu sayede veritabanı implementasyonu değiştirilse bile
    IdentityMatcher yeniden yazılmaz.

Eşleşme Mantığı:
    1. Cosine benzerliği ile top-K aday bulunur (TurtleDatabase.search).
    2. Confidence threshold altındaki sonuçlar "bilinmeyen" ilan edilir.
    3. Hem ham skor hem de yorumlanabilir güven seviyesi döndürülür.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from turtlefaceid.database.turtle_database import TurtleDatabase, TurtleRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Yardımcı Tipler
# ---------------------------------------------------------------------------

class ConfidenceLevel(Enum):
    """Eşleşme sonucunun güven seviyesi."""
    HIGH     = "Yüksek"      # > 0.80
    MEDIUM   = "Orta"        # 0.60 – 0.80
    LOW      = "Düşük"       # 0.40 – 0.60
    UNKNOWN  = "Bilinmeyen"  # < 0.40


@dataclass
class MatchResult:
    """
    IdentityMatcher'ın döndürdüğü eşleşme sonucu.

    Attributes:
        matched          : En az bir güvenilir eşleşme bulundu mu?
        top_match        : En yüksek skorlu kaplumbağa kaydı.
        similarity_score : Cosine benzerlik skoru [0.0 – 1.0].
        similarity_pct   : İnsan okunabilir yüzde (ör. "%85").
        confidence_level : ConfidenceLevel enum değeri.
        top_candidates   : Tüm adaylar [(TurtleRecord, score)] listesi.
        is_new_individual: Bu görüntü yeni bir birey mi?
        reasoning        : Kararın gerekçesi (demo/raporlama için).
    """

    matched         : bool
    top_match       : Optional[TurtleRecord]          = None
    similarity_score: float                           = 0.0
    similarity_pct  : str                             = "—"
    confidence_level: ConfidenceLevel                 = ConfidenceLevel.UNKNOWN
    top_candidates  : list[tuple[TurtleRecord, float]] = None
    is_new_individual: bool                           = False
    reasoning       : str                             = ""

    def __post_init__(self):
        if self.top_candidates is None:
            self.top_candidates = []


# ---------------------------------------------------------------------------
# Ana Sınıf
# ---------------------------------------------------------------------------

class IdentityMatcher:
    """
    Embedding vektörünü veritabanıyla karşılaştırarak kimlik eşleştirir.

    Bu sınıf yalnızca şu soruyu yanıtlar:
        "Bu embedding vektörü hangi kaplumbağaya ait?"

    Kullanım Örneği:
        >>> matcher = IdentityMatcher(database, threshold=0.65)
        >>> result  = matcher.match(query_embedding)
        >>> print(result.similarity_pct, result.top_match.name)
    """

    # Güven seviyesi eşikleri
    _THRESHOLD_HIGH  : float = 0.80
    _THRESHOLD_MEDIUM: float = 0.60
    _THRESHOLD_LOW   : float = 0.40

    def __init__(
        self,
        database : TurtleDatabase,
        threshold: float = 0.55,
        top_k    : int   = 5,
    ) -> None:
        """
        Args:
            database : Kaplumbağa kimlik veritabanı.
            threshold: Bu skorun altındaki eşleşmeler "bilinmeyen" sayılır.
            top_k    : Döndürülecek aday sayısı.
        """
        self._database  = database
        self._threshold = threshold
        self._top_k     = top_k
        logger.info(
            "IdentityMatcher başlatıldı (eşik=%.2f, top_k=%d).",
            threshold, top_k,
        )

    # ------------------------------------------------------------------
    # Genel Arayüz
    # ------------------------------------------------------------------

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        """
        Sorgu embedding'ini veritabanıyla karşılaştırır.

        Args:
            query_embedding: EmbeddingExtractor'dan elde edilen vektör.

        Returns:
            MatchResult — zengin eşleşme sonucu.
        """
        if self._database.turtle_count == 0:
            return MatchResult(
                matched=False,
                reasoning="Veritabanı boş — henüz kayıtlı kaplumbağa yok.",
                is_new_individual=True,
            )

        # Veritabanında en yakın adayları bul
        candidates = self._database.search(query_embedding, top_k=self._top_k)

        if not candidates:
            return MatchResult(
                matched=False,
                reasoning="Veritabanı araması sonuç döndürmedi.",
                is_new_individual=True,
            )

        best_record, best_score = candidates[0]
        conf_level = self._classify_confidence(best_score)

        matched = best_score >= self._threshold

        reasoning = self._build_reasoning(
            best_record, best_score, conf_level, matched
        )

        logger.info(
            "Eşleşme: %s | Skor: %.3f | Güven: %s",
            best_record.name if matched else "Bilinmeyen",
            best_score,
            conf_level.value,
        )

        return MatchResult(
            matched         = matched,
            top_match       = best_record if matched else None,
            similarity_score= round(best_score, 4),
            similarity_pct  = f"%{best_score * 100:.1f}",
            confidence_level= conf_level,
            top_candidates  = candidates,
            is_new_individual= not matched,
            reasoning       = reasoning,
        )

    def match_pair(
        self,
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> tuple[float, ConfidenceLevel]:
        """
        İki embedding arasındaki benzerliği doğrudan hesaplar.
        Veritabanına başvurmaksızın çift karşılaştırması için kullanılır.

        Returns:
            (similarity_score, confidence_level)
        """
        a_norm   = embedding_a / (np.linalg.norm(embedding_a) + 1e-9)
        b_norm   = embedding_b / (np.linalg.norm(embedding_b) + 1e-9)
        score    = float(np.dot(a_norm, b_norm))
        conf     = self._classify_confidence(score)
        return score, conf

    # ------------------------------------------------------------------
    # Özel Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _classify_confidence(self, score: float) -> ConfidenceLevel:
        """Skoru ConfidenceLevel enum değerine çevirir."""
        if score >= self._THRESHOLD_HIGH:
            return ConfidenceLevel.HIGH
        if score >= self._THRESHOLD_MEDIUM:
            return ConfidenceLevel.MEDIUM
        if score >= self._THRESHOLD_LOW:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNKNOWN

    @staticmethod
    def _build_reasoning(
        record    : TurtleRecord,
        score     : float,
        conf_level: ConfidenceLevel,
        matched   : bool,
    ) -> str:
        """Eşleşme kararının gerekçesini insan okunabilir metin olarak döndürür."""
        if not matched:
            return (
                f"En yüksek benzerlik skoru {score:.1%} ({conf_level.value} güven). "
                "Bu, eşleşme eşiğinin altındadır. Muhtemelen yeni bir bireydir."
            )
        return (
            f"Kaplumbağa '{record.name}' ({record.turtle_id}) ile %{score * 100:.1f} "
            f"benzerlik tespit edildi. Güven seviyesi: {conf_level.value}. "
            f"İlk gözlem: {record.first_seen or 'bilinmiyor'}, "
            f"Lokasyon: {record.location or 'bilinmiyor'}."
        )
