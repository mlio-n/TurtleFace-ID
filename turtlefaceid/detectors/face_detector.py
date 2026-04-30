"""
FaceDetector — Kaplumbağa Kafası Tespiti ve Kırpma
====================================================
Single Responsibility Principle (SRP):
    TEK sorumluluk: ham fotoğraftan kaplumbağanın kafa bölgesini
    tespit edip kırpmak. Özellik çıkarımı veya eşleştirme bu
    sınıfın kapsamı DIŞINDADIR.

Not: Gerçek projede YOLOv8 / Detectron2 kullanılır. Bu sınıf
     mimariyi koruyarak OpenCV tabanlı üretim kaliteli bir taslak sunar.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Yüz tespiti sonuç nesnesi."""
    success        : bool
    cropped_face   : Optional[np.ndarray] = None
    bounding_box   : Optional[tuple]      = None
    confidence     : float                = 0.0
    annotated_image: Optional[np.ndarray] = None
    debug_frames   : dict                 = field(default_factory=dict)
    error_message  : str                  = ""


class FaceDetector:
    """
    Ham fotoğrafı alıp kaplumbağanın kafa/yüz bölgesini kırpar.

    Yaklaşım: Renk segmentasyonu (HSV maskeleme) +
              Canny kenar tespiti + morfolojik kapama + kontur analizi.
    """
    _TARGET_SIZE    = (224, 224)
    _MORPH_KERNEL   = 7
    _MIN_FACE_RATIO = 0.04

    def __init__(self, confidence_threshold: float = 0.50) -> None:
        self._threshold = confidence_threshold
        logger.info("FaceDetector başlatıldı (eşik=%.2f)", confidence_threshold)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Ham görüntüden kaplumbağa kafasını tespit eder."""
        if image is None or image.size == 0:
            return DetectionResult(success=False, error_message="Geçersiz görüntü.")

        rgb   = self._ensure_rgb(image)
        debug = {"original": rgb.copy()}

        preprocessed   = self._preprocess(rgb)
        debug["preprocessed"] = preprocessed

        mask = self._segment_head_region(preprocessed)
        debug["segmentation_mask"] = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        bbox, confidence = self._extract_bounding_box(mask, rgb.shape)

        if bbox is None or confidence < self._threshold:
            return DetectionResult(
                success=False, confidence=confidence or 0.0,
                debug_frames=debug, error_message="Kaplumbağa kafası tespit edilemedi.",
            )

        x, y, w, h = bbox
        cropped   = self._crop_and_resize(rgb, x, y, w, h)
        annotated = self._draw_annotation(rgb.copy(), x, y, w, h, confidence)
        debug["annotated"] = annotated

        logger.info("Yüz tespit edildi — BBox:%s Güven:%.2f", bbox, confidence)
        return DetectionResult(
            success=True, cropped_face=cropped, bounding_box=bbox,
            confidence=confidence, annotated_image=annotated, debug_frames=debug,
        )

    def detect_from_path(self, path: Path | str) -> DetectionResult:
        p = Path(path)
        if not p.exists():
            return DetectionResult(success=False, error_message=f"Dosya bulunamadı: {p}")
        image = cv2.imread(str(p))
        return self.detect(image) if image is not None else DetectionResult(
            success=False, error_message="Görüntü okunamadı.")

    # ── Özel yardımcılar ────────────────────────────────────────────

    @staticmethod
    def _ensure_rgb(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image.copy()

    @staticmethod
    def _preprocess(rgb: np.ndarray) -> np.ndarray:
        blurred = cv2.GaussianBlur(rgb, (5, 5), 0)
        lab = cv2.cvtColor(blurred, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    def _segment_head_region(self, rgb: np.ndarray) -> np.ndarray:
        """HSV renk maskesi + Canny kenarları hibrit segmentasyon."""
        hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lower = np.array([0,  20,  40], dtype=np.uint8)
        upper = np.array([40, 200, 230], dtype=np.uint8)
        color_mask = cv2.inRange(hsv, lower, upper)

        gray  = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

        combined = cv2.bitwise_or(color_mask, edges)
        kernel   = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self._MORPH_KERNEL, self._MORPH_KERNEL))
        closed   = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=3)
        return   cv2.morphologyEx(closed,   cv2.MORPH_OPEN,  kernel, iterations=1)

    def _extract_bounding_box(self, mask, shape):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0.0
        
        # Artık kullanıcı manuel olarak kestiği için alan oranını düşürebiliriz
        img_area = shape[0] * shape[1]
        valid    = [c for c in contours if cv2.contourArea(c) > img_area * 0.01] 
        
        if not valid:
            return None, 0.0
            
        largest  = max(valid, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
            
        hull_area   = cv2.contourArea(cv2.convexHull(largest))
        confidence  = float(cv2.contourArea(largest) / (hull_area + 1e-6))
        
        # Kırpmaya hafif padding ekle (Kullanıcı zaten kırptığı için çok az)
        pad = int(min(w, h) * 0.05)
        x = max(0, x - pad); y = max(0, y - pad)
        w = min(shape[1] - x, w + 2 * pad)
        h = min(shape[0] - y, h + 2 * pad)
        return (x, y, w, h), round(confidence, 3)

    def _crop_and_resize(self, image, x, y, w, h):
        return cv2.resize(image[y:y+h, x:x+w], self._TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def _draw_annotation(image, x, y, w, h, confidence):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 220, 100), 2)
        label = f"Turtle Head  {confidence:.0%}"
        cv2.rectangle(image, (x, y-28), (x + len(label)*9, y), (0, 220, 100), -1)
        cv2.putText(image, label, (x+4, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1, cv2.LINE_AA)
        return image
