"""
ScuteExtractor — Post-Ocular Pul (Scute) Haritası Çıkarımı
===========================================================
Single Responsibility Principle (SRP):
    Bu sınıfın TEK sorumluluğu: kırpılmış kaplumbağa yüzündeki
    göz çevresi pullarını (post-ocular scutes) tespit edip
    haritasını çıkarmaktır.

Biyolojik Bağlam:
    Kaplumbağaların yüzündeki pulların şekli, dizilimi ve sayısı
    bireyden bireye farklılık gösterir — tıpkı insan parmak izi
    gibi. "Jean et al. (2010)" ve "Carter et al. (2014)" bu
    özelliklerin bireysel tanımada ne denli güvenilir olduğunu
    kanıtlamıştır. Bu sınıf bu biyolojik benzersizliği sayısal
    vektörlere dönüştürür.

Yaklaşım (Üretim Seviyesi Taslak):
    Gerçek bir projede derin öğrenme tabanlı instance segmentation
    (Mask R-CNN vb.) kullanılır. Bu taslak; dağıtılabilir,
    test edilebilir ve genişletilebilir bir mimari sunarken
    OpenCV tabanlı gerçekçi bir simülasyon çalıştırır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Veri Yapıları
# ---------------------------------------------------------------------------

@dataclass
class ScuteRegion:
    """Tek bir pulun geometrik ve istatistiksel bilgileri."""

    region_id   : int
    centroid    : tuple[float, float]          # (cx, cy) piksel koordinatı
    area        : float                         # piksel cinsinden alan
    perimeter   : float
    aspect_ratio: float                         # genişlik / yükseklik
    solidity    : float                         # alan / convex_hull_alanı
    contour     : np.ndarray                    # ham kontur noktaları


@dataclass
class ScuteMap:
    """
    ScuteExtractor'ın döndürdüğü zengin sonuç nesnesi.

    Attributes:
        success          : İşlemin başarılı olup olmadığı.
        regions          : Tespit edilen her pulun ScuteRegion nesneleri.
        overlay_image    : Pulların renkli konturlarla gösterildiği görüntü.
        feature_vector   : Eşleşme için kullanılacak sayısal özellik vektörü.
        scute_count      : Tespit edilen pul sayısı.
        edge_map         : Kenar haritası (Canny çıktısı).
        debug_frames     : Ara işlem görüntüleri.
        error_message    : Hata mesajı (başarısızlık durumunda).
    """

    success      : bool
    regions      : list[ScuteRegion]              = field(default_factory=list)
    overlay_image: Optional[np.ndarray]           = None
    feature_vector: Optional[np.ndarray]          = None
    scute_count  : int                            = 0
    edge_map     : Optional[np.ndarray]           = None
    debug_frames : dict[str, np.ndarray]          = field(default_factory=dict)
    error_message: str                            = ""


# ---------------------------------------------------------------------------
# Ana Sınıf
# ---------------------------------------------------------------------------

class ScuteExtractor:
    """
    Kırpılmış kaplumbağa yüzünden post-ocular scute haritasını çıkarır.

    Her scute (pul) tespit edildikten sonra geometrik özellikleri
    sayısal bir vektöre dönüştürülür. Bu vektör Siamese Network'e
    veya FAISS vektör deposuna beslenir.

    Kullanım Örneği:
        >>> extractor = ScuteExtractor(min_scute_area=150)
        >>> scute_map = extractor.extract(cropped_face_array)
        >>> print(scute_map.scute_count)
    """

    _FEATURE_DIM       : int   = 64    # sabit boyutlu çıktı vektörü
    _MAX_REGIONS       : int   = 30    # dahil edilecek max pul sayısı
    _EDGE_LOW_THRESH   : int   = 25
    _EDGE_HIGH_THRESH  : int   = 85

    def __init__(self, min_scute_area: float = 120.0) -> None:
        """
        Args:
            min_scute_area: Gürültü olarak elenecek min piksel alanı.
        """
        self._min_scute_area = min_scute_area
        logger.info("ScuteExtractor başlatıldı (min_area=%.0f)", min_scute_area)

    # ------------------------------------------------------------------
    # Genel Arayüz
    # ------------------------------------------------------------------

    def extract(self, face_image: np.ndarray) -> ScuteMap:
        """
        Kaplumbağa yüzündeki pulları tespit eder ve özelliklerini çıkarır.

        Args:
            face_image: RGB formatında kırpılmış yüz görüntüsü.

        Returns:
            ScuteMap — tespit edilen tüm pullar ve özellik vektörü.
        """
        if face_image is None or face_image.size == 0:
            return ScuteMap(success=False, error_message="Geçersiz yüz görüntüsü.")

        debug: dict[str, np.ndarray] = {}

        # 1. Gri tonlamaya dönüştür ve ön işle
        gray        = self._to_gray(face_image)
        enhanced    = self._enhance_contrast(gray)
        debug["gray_enhanced"] = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

        # 2. Kenar haritası üret (pul sınırları)
        edge_map    = self._compute_edge_map(enhanced)
        debug["edge_map"] = cv2.cvtColor(edge_map, cv2.COLOR_GRAY2RGB)

        # 3. Kontur analizi ile pul bölgelerini bul
        regions     = self._find_scute_regions(edge_map, face_image.shape)

        if not regions:
            logger.warning("Hiç pul bölgesi tespit edilemedi.")
            return ScuteMap(
                success=False,
                edge_map=edge_map,
                debug_frames=debug,
                error_message="Pul bölgesi tespit edilemedi.",
            )

        # 4. Görsel katman oluştur
        overlay     = self._build_overlay(face_image, regions)
        debug["overlay"] = overlay

        # 5. Sabit boyutlu özellik vektörü oluştur
        feature_vec = self._build_feature_vector(regions, face_image.shape)

        logger.info("Pul haritası oluşturuldu — %d bölge tespit edildi.", len(regions))

        return ScuteMap(
            success=True,
            regions=regions,
            overlay_image=overlay,
            feature_vector=feature_vec,
            scute_count=len(regions),
            edge_map=edge_map,
            debug_frames=debug,
        )

    # ------------------------------------------------------------------
    # Özel Yardımcı Metodlar
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return image
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    @staticmethod
    def _enhance_contrast(gray: np.ndarray) -> np.ndarray:
        """
        Adaptif histogram eşitleme (CLAHE) ile pul sınırlarını
        daha belirgin hale getirir.
        """
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def _compute_edge_map(self, gray: np.ndarray) -> np.ndarray:
        """
        Çok ölçekli Canny kenar haritası:
          - İnce Gauss (sigma=1) kanalı ince pul sınırlarını,
          - Kaba kanal (sigma=2) derin yapıları yakalar.
        Sonuçlar OR ile birleştirilir.
        """
        fine   = cv2.GaussianBlur(gray, (3, 3), 1.0)
        coarse = cv2.GaussianBlur(gray, (7, 7), 2.0)

        edges_fine   = cv2.Canny(fine,   self._EDGE_LOW_THRESH, self._EDGE_HIGH_THRESH)
        edges_coarse = cv2.Canny(coarse, self._EDGE_LOW_THRESH - 5, self._EDGE_HIGH_THRESH + 10)

        combined = cv2.bitwise_or(edges_fine, edges_coarse)

        # Morfolojik kapama: kapalı pul hatlarını oluştur
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed  = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        return closed

    def _find_scute_regions(
        self, edge_map: np.ndarray, image_shape: tuple
    ) -> list[ScuteRegion]:
        """Kenar haritasından konturları bulur ve ScuteRegion listesi oluşturur."""
        # Kapalı bölgeleri doldurmak için flood-fill benzeri yaklaşım
        filled = self._fill_enclosed_regions(edge_map)

        contours, _ = cv2.findContours(filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        regions: list[ScuteRegion] = []
        
        # ── MAKSİMUM ALAN FİLTRESİ ────────────────────────────
        # Bir pul, kırpılmış yüz alanının %25'inden büyük olamaz (örn. arka plan kayaları)
        max_scute_area = (image_shape[0] * image_shape[1]) * 0.25

        for idx, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < self._min_scute_area or area > max_scute_area:
                continue

            # Geometrik özellikler
            perimeter    = cv2.arcLength(cnt, True)
            x, y, w, h  = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / float(h + 1e-6)

            hull_area    = cv2.contourArea(cv2.convexHull(cnt))
            solidity     = area / (hull_area + 1e-6)

            # ── ŞEKİL FİLTRESİ (Contour Filter) ───────────────────
            # Açık uçlu büyük çizgileri veya şekilsiz lekeleri (kabuk vb.) ele.
            # Post-ocular pullar poligon/çokgen benzeri kapalı, kompakt alanlardır.
            if solidity < 0.65 or aspect_ratio < 0.25 or aspect_ratio > 4.0:
                continue

            M    = cv2.moments(cnt)
            cx   = M["m10"] / (M["m00"] + 1e-6)
            cy   = M["m01"] / (M["m00"] + 1e-6)

            regions.append(ScuteRegion(
                region_id    = idx,
                centroid     = (float(cx), float(cy)),
                area         = float(area),
                perimeter    = float(perimeter),
                aspect_ratio = float(aspect_ratio),
                solidity     = float(solidity),
                contour      = cnt,
            ))

        # En büyük _MAX_REGIONS bölgeyi al
        regions.sort(key=lambda r: r.area, reverse=True)
        return regions[: self._MAX_REGIONS]

    @staticmethod
    def _fill_enclosed_regions(edge_map: np.ndarray) -> np.ndarray:
        """
        Kenar haritasındaki kapalı bölgeleri doldurarak
        kontur analizine hazır hale getirir.
        """
        # Kenarları genişlet ve flood-fill ile kapalı bölgeleri doldur
        kernel   = np.ones((3, 3), np.uint8)
        dilated  = cv2.dilate(edge_map, kernel, iterations=1)
        h, w     = dilated.shape
        flooded  = dilated.copy()

        # Kenar dolgusunu tersine çevir ve doldur
        inv      = cv2.bitwise_not(flooded)
        mask_ff  = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inv, mask_ff, (0, 0), 255)
        filled   = cv2.bitwise_not(inv)

        return cv2.bitwise_or(edge_map, filled)

    @staticmethod
    def _build_overlay(
        face_image: np.ndarray, regions: list[ScuteRegion]
    ) -> np.ndarray:
        """
        Her pulun konturunu benzersiz bir renkle boyayarak
        görsel pul haritasını (overlay) oluşturur.
        """
        overlay = face_image.copy().astype(np.float32)
        canvas  = face_image.copy()

        # Renk paleti — bilimsel görselleştirmede yaygın kullanılan Tableau-10
        palette = [
            (76,  114, 176), (85,  168, 104), (196, 78,  82 ),
            (129, 114, 179), (204, 185, 116), (100, 181, 205),
            (255, 157, 87 ), (148, 103, 189), (23,  190, 207),
            (188, 189, 34 ),
        ]

        for i, region in enumerate(regions):
            color = palette[i % len(palette)]
            cv2.drawContours(canvas, [region.contour], -1, color, 2)
            cx, cy = int(region.centroid[0]), int(region.centroid[1])
            cv2.circle(canvas, (cx, cy), 3, color, -1)
            cv2.putText(
                canvas, str(i + 1), (cx + 4, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )

        # Yarı saydam katman birleştirme
        alpha  = 0.65
        result = cv2.addWeighted(canvas.astype(np.float32), alpha,
                                 overlay, 1 - alpha, 0).astype(np.uint8)
        return result

    def _build_feature_vector(
        self, regions: list[ScuteRegion], image_shape: tuple
    ) -> np.ndarray:
        """
        Tespit edilen pullardan sabit boyutlu (_FEATURE_DIM) bir özellik vektörü
        oluşturur. Bu vektör Siamese Network embedding'i ile birleştirilebilir
        veya bağımsız bir eşleşme sinyali olarak kullanılabilir.

        Vektör içeriği (her bölge için 5 değer, max 12 bölge → 60 + 4 global):
          - normalize centroid x, y
          - normalize alan
          - aspect_ratio
          - solidity
          + global: bölge sayısı, ort. alan, ort. perimeter, alan varyansı
        """
        h, w   = image_shape[:2]
        n_feat = 5
        n_reg  = min(len(regions), self._MAX_REGIONS)

        local_features = np.zeros(n_reg * n_feat, dtype=np.float32)

        for i, region in enumerate(regions[:n_reg]):
            offset = i * n_feat
            local_features[offset + 0] = region.centroid[0] / (w + 1e-6)
            local_features[offset + 1] = region.centroid[1] / (h + 1e-6)
            local_features[offset + 2] = region.area / (w * h + 1e-6)
            local_features[offset + 3] = min(region.aspect_ratio, 5.0) / 5.0
            local_features[offset + 4] = region.solidity

        areas         = [r.area for r in regions]
        perimeters    = [r.perimeter for r in regions]
        global_feats  = np.array([
            len(regions) / self._MAX_REGIONS,
            float(np.mean(areas))      / (w * h + 1e-6),
            float(np.mean(perimeters)) / (w + h + 1e-6),
            float(np.var(areas))       / ((w * h) ** 2 + 1e-6),
        ], dtype=np.float32)

        combined = np.concatenate([local_features, global_feats])

        # Sabit _FEATURE_DIM'e kırp veya sıfırla doldur
        if len(combined) >= self._FEATURE_DIM:
            return combined[: self._FEATURE_DIM]
        return np.pad(combined, (0, self._FEATURE_DIM - len(combined)))
