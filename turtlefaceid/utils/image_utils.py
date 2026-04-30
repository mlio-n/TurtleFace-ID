"""
ImageUtils — Görüntü Yükleme ve Dönüştürme Araçları
====================================================
SRP: Yalnızca görüntü I/O ve format dönüşümleri burada.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


class ImageUtils:
    """Görüntü okuma, yazma ve format dönüşümü statik yardımcıları."""

    @staticmethod
    def load_rgb(path: Path | str) -> Optional[np.ndarray]:
        """Dosyadan RGB NumPy dizisi olarak görüntü yükler."""
        p = Path(path)
        if not p.exists():
            return None
        img = cv2.imread(str(p))
        if img is None:
            # PIL ile tekrar dene (bazı format farklılıkları için)
            try:
                pil = Image.open(p).convert("RGB")
                return np.array(pil)
            except Exception:
                return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def pil_to_numpy(pil_image: Image.Image) -> np.ndarray:
        """PIL Image → RGB NumPy dizisi."""
        return np.array(pil_image.convert("RGB"))

    @staticmethod
    def numpy_to_pil(array: np.ndarray) -> Image.Image:
        """RGB NumPy dizisi → PIL Image."""
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array)

    @staticmethod
    def resize(
        image: np.ndarray,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray:
        """En boy oranını koruyarak yeniden boyutlandırır."""
        h, w = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            ratio = height / h
            dim   = (int(w * ratio), height)
        elif height is None:
            ratio = width / w
            dim   = (width, int(h * ratio))
        else:
            dim   = (width, height)
        return cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)

    @staticmethod
    def normalize_for_display(image: np.ndarray) -> np.ndarray:
        """Görüntüyü uint8 [0, 255] aralığına normalize eder."""
        if image.dtype == np.uint8:
            return image
        img_min, img_max = image.min(), image.max()
        if img_max == img_min:
            return np.zeros_like(image, dtype=np.uint8)
        normalized = (image - img_min) / (img_max - img_min) * 255
        return normalized.astype(np.uint8)

    @staticmethod
    def create_placeholder(
        width: int = 224,
        height: int = 224,
        text: str = "Görüntü yok",
        bg_color: tuple = (30, 30, 40),
        text_color: tuple = (120, 120, 140),
    ) -> np.ndarray:
        """Görüntü yoksa gösterilecek yer tutucu oluşturur."""
        placeholder = np.full((height, width, 3), bg_color, dtype=np.uint8)
        font_scale  = width / 300.0
        text_size   = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)[0]
        tx = (width  - text_size[0]) // 2
        ty = (height + text_size[1]) // 2
        cv2.putText(
            placeholder, text, (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA,
        )
        return placeholder
