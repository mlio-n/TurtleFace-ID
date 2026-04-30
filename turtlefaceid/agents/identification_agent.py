"""
IdentificationAgent — Üst Düzey AI Ajan Orkestratörü
=====================================================
AI Ajan (AI Agent) Mimarisi:
    Bu sınıf, sistemin beynini oluşturur. Alt sınıflara
    (FaceDetector, ScuteExtractor, IdentityMatcher) görevleri
    dağıtır ve sonuçları koordine eder.

    Klasik bir pipeline'dan farkı şudur:
        ► Durum makinesi (State Machine): Her adımın başarı/
          başarısızlık durumunu takip eder ve hata durumunda
          geri dönüş stratejisi uygular.
        ► Açıklanabilirlik (Explainability): Her adımda ne
          yapıldığını, neden yapıldığını ve ne elde edildiğini
          kayıt altına alır. Hocaya sunum için bu özellik kritiktir.
        ► Genişletilebilirlik: Yeni adımlar (örn. davranış analizi,
          GPS korelasyonu) mevcut kodu değiştirmeden eklenebilir.

Open/Closed Principle (OCP):
    Yeni bir "tool" eklemek için yalnızca yeni bir metod + AgentState
    değeri eklemek yeterlidir. Mevcut pipeline bozulmaz.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image

from turtlefaceid.detectors.face_detector   import FaceDetector, DetectionResult
from turtlefaceid.extractors.scute_extractor import ScuteExtractor, ScuteMap
from turtlefaceid.matchers.identity_matcher  import IdentityMatcher, MatchResult
from turtlefaceid.database.turtle_database  import TurtleDatabase, create_demo_database
from turtlefaceid.utils.image_utils         import ImageUtils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ajan Durum Makinesi
# ---------------------------------------------------------------------------

class AgentState(Enum):
    """Tanımlama pipeline'ının her adımı için durum."""
    IDLE            = auto()   # Bekleme
    LOADING_IMAGE   = auto()   # Görüntü yükleniyor
    DETECTING_FACE  = auto()   # Yüz tespiti
    EXTRACTING_SCUTES = auto() # Pul çıkarımı
    COMPUTING_EMBEDDING = auto() # Embedding hesaplanıyor
    MATCHING        = auto()   # Veritabanı eşleştirmesi
    COMPLETE        = auto()   # Başarıyla tamamlandı
    FAILED          = auto()   # Hata


# ---------------------------------------------------------------------------
# Sonuç Nesnesi
# ---------------------------------------------------------------------------

@dataclass
class IdentificationResult:
    """
    IdentificationAgent'ın son kullanıcıya sunduğu bütünleşik sonuç.

    Tüm ara adım verileri, hata mesajları ve nihai karar burada toplanır.
    Streamlit arayüzü bu nesneyi okuyarak görsel sunum yapar.
    """

    # Pipeline durumu
    state           : AgentState = AgentState.IDLE
    success         : bool       = False
    total_time_ms   : float      = 0.0

    # Adım 1 — Orijinal görüntü
    original_image  : Optional[np.ndarray] = None

    # Adım 2 — Yüz tespiti
    detection_result: Optional[DetectionResult] = None

    # Adım 3 — Pul haritası
    scute_map       : Optional[ScuteMap] = None

    # Adım 4 — Eşleşme
    match_result    : Optional[MatchResult] = None

    # Ajan günlüğü (her adım için zaman damgalı mesajlar)
    agent_log       : list[dict] = field(default_factory=list)

    # Hata bilgisi
    error_message   : str = ""


# ---------------------------------------------------------------------------
# Ana Ajan Sınıfı
# ---------------------------------------------------------------------------

class IdentificationAgent:
    """
    Kaplumbağa tanımlama pipeline'ını yöneten AI Ajan.

    Bu sınıf, tüm alt bileşenleri oluşturur ve koordine eder.
    Streamlit arayüzü yalnızca bu sınıfla iletişim kurar.

    Kullanım Örneği:
        >>> agent  = IdentificationAgent()
        >>> result = agent.identify(image_array)
        >>> print(result.match_result.similarity_pct)
    """

    def __init__(
        self,
        database         : Optional[TurtleDatabase] = None,
        detection_threshold: float = 0.50,
        match_threshold  : float   = 0.55,
        use_demo_db      : bool    = True,
        embedding_dim    : int     = 256,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        """
        Args:
            database           : Kaplumbağa veritabanı (None ise demo oluşturulur).
            detection_threshold: Yüz tespiti güven eşiği.
            match_threshold    : Kimlik eşleşme eşiği.
            use_demo_db        : True ise demo veritabanı oluşturulur.
            embedding_dim      : Embedding vektörü boyutu.
            progress_callback  : Streamlit için ilerleme bildirici.
        """
        self._progress_cb = progress_callback
        self._emb_dim     = embedding_dim

        # Alt bileşenler (Dependency Injection)
        self._detector  = FaceDetector(confidence_threshold=detection_threshold)
        self._extractor = ScuteExtractor(min_scute_area=120.0)

        if database is not None:
            self._database = database
        elif use_demo_db:
            self._database = create_demo_database(embedding_dim)
        else:
            self._database = TurtleDatabase()

        self._matcher = IdentityMatcher(
            database=self._database,
            threshold=match_threshold,
            top_k=5,
        )

        # Siamese model (opsiyonel — demo modunda simüle edilmiş embedding)
        self._model = None
        self._device = "cpu"
        self._try_load_model()

        logger.info(
            "IdentificationAgent hazır — DB: %d kaplumbağa.",
            self._database.turtle_count,
        )

    # ------------------------------------------------------------------
    # Ana Pipeline
    # ------------------------------------------------------------------

    def identify(self, image: np.ndarray) -> IdentificationResult:
        """
        Ham görüntüden kaplumbağa tanımlama pipeline'ını çalıştırır.

        Adımlar:
            1. Görüntü doğrulama
            2. Yüz tespiti (FaceDetector)
            3. Pul çıkarımı (ScuteExtractor)
            4. Embedding hesaplama (SiameseNetwork veya simülasyon)
            5. Veritabanı eşleştirme (IdentityMatcher)

        Args:
            image: RGB formatında NumPy dizisi.

        Returns:
            IdentificationResult — bütün adımların sonuçları.
        """
        result    = IdentificationResult(original_image=image.copy())
        t_start   = time.perf_counter()

        # ── Adım 1: Görüntü doğrulama ──────────────────────────────────
        result.state = AgentState.LOADING_IMAGE
        self._log(result, "Görüntü yüklendi", 0.10)
        self._notify("Görüntü yükleniyor…", 0.10)

        if image is None or image.size == 0:
            return self._fail(result, "Geçersiz görüntü.", t_start)

        # ── Adım 2: Yüz tespiti ────────────────────────────────────────
        result.state = AgentState.DETECTING_FACE
        self._notify("Kaplumbağa yüzü tespit ediliyor…", 0.25)

        detection = self._detector.detect(image)
        result.detection_result = detection
        self._log(
            result,
            f"Yüz tespiti {'başarılı' if detection.success else 'başarısız'} "
            f"(güven={detection.confidence:.2f})",
            0.30,
        )

        if not detection.success:
            return self._fail(
                result,
                f"Yüz tespiti başarısız: {detection.error_message}",
                t_start,
            )

        # ── Adım 3: Pul çıkarımı ───────────────────────────────────────
        result.state = AgentState.EXTRACTING_SCUTES
        self._notify("Post-ocular scuteler çıkarılıyor…", 0.50)

        scute_map = self._extractor.extract(detection.cropped_face)
        result.scute_map = scute_map
        self._log(
            result,
            f"Pul haritası: {scute_map.scute_count} bölge tespit edildi.",
            0.55,
        )

        if not scute_map.success:
            # Pul çıkarımı başarısız olsa bile devam et (yalnızca görüntü embedding'i)
            logger.warning("Pul çıkarımı başarısız; yalnızca görüntü embedding'i kullanılacak.")

        # ── Adım 4: Embedding hesaplama ────────────────────────────────
        result.state = AgentState.COMPUTING_EMBEDDING
        self._notify("Embedding vektörü hesaplanıyor…", 0.70)

        embedding = self._compute_embedding(detection.cropped_face, scute_map)
        self._log(result, f"Embedding boyutu: {embedding.shape[0]}", 0.75)

        # ── Adım 5: Veritabanı eşleştirme ─────────────────────────────
        result.state = AgentState.MATCHING
        self._notify("Veritabanında kimlik eşleştiriliyor…", 0.85)

        match = self._matcher.match(embedding)
        result.match_result = match
        self._log(
            result,
            (
                f"Eşleşme: {match.top_match.name if match.matched else 'Bilinmeyen'} "
                f"| Skor: {match.similarity_score:.3f} "
                f"| Güven: {match.confidence_level.value}"
            ),
            1.00,
        )

        # ── Tamamlandı ─────────────────────────────────────────────────
        result.state         = AgentState.COMPLETE
        result.success       = True
        result.total_time_ms = (time.perf_counter() - t_start) * 1000
        self._notify("Tanımlama tamamlandı.", 1.0)

        logger.info(
            "Pipeline tamamlandı — %.1f ms", result.total_time_ms
        )
        return result

    def identify_from_path(self, image_path: Path | str) -> IdentificationResult:
        """Dosya yolundan görüntü yükleyerek tanımlama başlatır."""
        image = ImageUtils.load_rgb(image_path)
        if image is None:
            r = IdentificationResult()
            r.error_message = f"Görüntü yüklenemedi: {image_path}"
            r.state = AgentState.FAILED
            return r
        return self.identify(image)

    @property
    def database(self) -> TurtleDatabase:
        return self._database

    # ------------------------------------------------------------------
    # Özel Yardımcı Metodlar
    # ------------------------------------------------------------------

    def _compute_embedding(
        self, face_image: np.ndarray, scute_map: ScuteMap
    ) -> np.ndarray:
        """
        Embedding vektörü hesaplar.

        Gerçek model mevcutsa → SiameseNetwork kullanılır.
        Yoksa (demo/test) → ScuteExtractor özellik vektörü + simüle
        edilmiş CNN özellikleri ile birleştirilmiş bir vektör döndürülür.
        """
        if self._model is not None:
            return self._model_embedding(face_image)
        return self._simulated_embedding(face_image, scute_map)

    def _model_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """SiameseNetwork modeli üzerinden embedding çıkarır."""
        import torch
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std =[0.229, 0.224, 0.225]),
        ])
        tensor = transform(face_image).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model.get_embedding(tensor)
        return emb.cpu().numpy().flatten()

    def _simulated_embedding(
        self, face_image: np.ndarray, scute_map: ScuteMap
    ) -> np.ndarray:
        """
        Demo modu — gerçekçi ama deterministik bir embedding simüle eder.

        Görüntünün piksel istatistiklerinden + pul özelliklerinden
        tutarlı bir vektör üretilir. Aynı görüntü her seferinde
        aynı vektörü verir (seed olarak görüntü hash'i kullanılır).
        """
        import hashlib, struct

        # Görüntü hash'inden deterministik seed
        img_bytes = face_image.tobytes()
        hash_val  = hashlib.md5(img_bytes).digest()
        seed      = struct.unpack("<I", hash_val[:4])[0]
        rng       = np.random.default_rng(seed)

        # Görüntü istatistikleri (normalize)
        img_f    = face_image.astype(np.float32) / 255.0
        img_feats = np.array([
            img_f.mean(), img_f.std(),
            img_f[:, :, 0].mean(), img_f[:, :, 1].mean(), img_f[:, :, 2].mean(),
        ], dtype=np.float32)

        # Pul özellikleri
        scute_feats = (
            scute_map.feature_vector
            if scute_map.success and scute_map.feature_vector is not None
            else np.zeros(64, dtype=np.float32)
        )

        # Rastgele bileşen (gerçek CNN'i taklit eder)
        noise = rng.standard_normal(self._emb_dim - len(img_feats) - 10).astype(np.float32)
        noise = noise * 0.3   # düşük varyans → daha tutarlı

        combined = np.concatenate([img_feats, scute_feats[:10], noise])
        combined = combined[: self._emb_dim]
        if len(combined) < self._emb_dim:
            combined = np.pad(combined, (0, self._emb_dim - len(combined)))

        # L2 normalizasyon
        norm = np.linalg.norm(combined)
        return combined / (norm + 1e-9)

    def _try_load_model(self) -> None:
        """
        Eğitilmiş model dosyası mevcutsa yükler.
        Yoksa simülasyon moduna geçer (demo için yeterli).
        """
        try:
            import torch
            from turtlefaceid.models.siamese_network import SiameseNetwork

            model_path = Path("models") / "siamese_model.pth"
            if model_path.exists():
                model = SiameseNetwork(embedding_dim=self._emb_dim)
                model.load(model_path, device=self._device)
                model.eval()
                self._model = model
                logger.info("Siamese model yüklendi: %s", model_path)
            else:
                logger.info(
                    "Model dosyası bulunamadı ('%s'). Simülasyon modu aktif.",
                    model_path,
                )
        except Exception as exc:
            logger.warning("Model yüklenemedi (%s). Simülasyon modu aktif.", exc)

    def _log(self, result: IdentificationResult, message: str, progress: float) -> None:
        result.agent_log.append({
            "state"   : result.state.name,
            "message" : message,
            "progress": progress,
            "timestamp": time.perf_counter(),
        })

    def _notify(self, message: str, progress: float) -> None:
        if self._progress_cb:
            self._progress_cb(message, progress)

    @staticmethod
    def _fail(
        result: IdentificationResult,
        message: str,
        t_start: float,
    ) -> IdentificationResult:
        result.state         = AgentState.FAILED
        result.success       = False
        result.error_message = message
        result.total_time_ms = (time.perf_counter() - t_start) * 1000
        logger.error("Pipeline başarısız: %s", message)
        return result
