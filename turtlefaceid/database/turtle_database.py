"""
TurtleDatabase — Kaplumbağa Kimlik Kayıtları ve Vektör Deposu
=============================================================
Single Responsibility Principle (SRP):
    Bu sınıfın TEK sorumluluğu: kaplumbağa kimlik kayıtlarını
    ve karşılık gelen embedding vektörlerini depolamak,
    sorgulamak ve yönetmektir. Tespit, çıkarım veya eşleşme
    mantığı bu sınıfın kapsamı DIŞINDADIR.

Vektör Arama:
    FAISS (Facebook AI Similarity Search) kullanılır. Bu,
    büyük veritabanlarında (1000+ kaplumbağa) milyonlarca
    vektör arasında milisaniyeler içinde en yakın komşu
    araması yapılmasını sağlar — TORSOOI veritabanı formatı
    ile uyumlu bir yapı sağlar.
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Veri Yapısı — Kaplumbağa Kaydı
# ---------------------------------------------------------------------------

@dataclass
class TurtleRecord:
    """
    Tek bir kaplumbağanın tüm kimlik bilgilerini tutan veri nesnesi.

    TORSOOI veritabanı formatı ile uyumlu alan isimleri kullanılmıştır
    (Carter et al., 2014).
    """

    turtle_id   : str                         # Örn: "TF-114"
    name        : str                         # Örn: "Frankie"
    species     : str = "Caretta caretta"     # Tür (Loggerhead varsayılan)
    sex         : str = "unknown"             # "male" | "female" | "unknown"
    location    : str = ""                    # Gözlem lokasyonu
    first_seen  : str = ""                    # İlk gözlem tarihi
    notes       : str = ""                    # Araştırmacı notları
    photo_paths : list[str] = field(default_factory=list)  # Referans fotoğraf yolları
    embedding   : Optional[list[float]] = None  # JSON serileştirme için liste


# ---------------------------------------------------------------------------
# Ana Veritabanı Sınıfı
# ---------------------------------------------------------------------------

class TurtleDatabase:
    """
    Kaplumbağa kimlik kayıtlarını ve embedding vektörlerini yönetir.

    Özellikler:
        - Kayıt ekleme / güncelleme / silme
        - FAISS ile hızlı vektör benzerlik araması
        - JSON formatında insan okunabilir dışa aktarma
        - Yedekleme ve geri yükleme

    Kullanım Örneği:
        >>> db = TurtleDatabase()
        >>> db.add_turtle(TurtleRecord("TF-114", "Frankie"), embedding)
        >>> results = db.search(query_embedding, top_k=5)
    """

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        """
        Args:
            db_path: Veritabanı dosyasının kaydedileceği dizin.
                     None ise yalnızca bellekte çalışır.
        """
        self._records   : dict[str, TurtleRecord] = {}
        self._embeddings: dict[str, np.ndarray]   = {}
        self._db_path   = Path(db_path) if db_path else None
        self._embedding_dim: Optional[int]        = None

        # FAISS indeksi — opsiyonel (büyük veri setleri için)
        self._faiss_index = None
        self._id_map: list[str] = []   # FAISS indeks sırası → turtle_id

        if self._db_path:
            self._db_path.mkdir(parents=True, exist_ok=True)
            self._load_if_exists()

        logger.info("TurtleDatabase başlatıldı — %d kayıt yüklendi.", len(self._records))

    # ------------------------------------------------------------------
    # CRUD İşlemleri
    # ------------------------------------------------------------------

    def add_turtle(
        self,
        record   : TurtleRecord,
        embedding: np.ndarray,
    ) -> None:
        """
        Yeni bir kaplumbağa kaydı ekler.

        Args:
            record   : Kaplumbağa kimlik bilgileri.
            embedding: Siamese Network'ten elde edilen embedding vektörü.
        """
        if self._embedding_dim is None:
            self._embedding_dim = embedding.shape[0]

        if embedding.shape[0] != self._embedding_dim:
            raise ValueError(
                f"Embedding boyutu uyumsuz: beklenen {self._embedding_dim}, "
                f"alınan {embedding.shape[0]}."
            )

        norm_emb                = embedding / (np.linalg.norm(embedding) + 1e-9)
        self._records[record.turtle_id]    = record
        self._embeddings[record.turtle_id] = norm_emb
        self._invalidate_faiss()

        logger.info("Kaplumbağa eklendi: %s (%s)", record.turtle_id, record.name)

        if self._db_path:
            self._save()

    def get_turtle(self, turtle_id: str) -> Optional[TurtleRecord]:
        """ID ile kayıt getirir."""
        return self._records.get(turtle_id)

    def remove_turtle(self, turtle_id: str) -> bool:
        """Kaydı ve embedding'i siler."""
        if turtle_id not in self._records:
            return False
        del self._records[turtle_id]
        del self._embeddings[turtle_id]
        self._invalidate_faiss()
        if self._db_path:
            self._save()
        logger.info("Kaplumbağa silindi: %s", turtle_id)
        return True

    @property
    def turtle_count(self) -> int:
        return len(self._records)

    def list_turtles(self) -> list[TurtleRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Benzerlik Araması
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[TurtleRecord, float]]:
        """
        Sorgu embedding'ine en benzer kaplumbağaları bulur.

        Cosine benzerliği (L2-normalize edilmiş vektörler için dot product):
            similarity = emb_query · emb_db  ∈ [-1, 1]

        Args:
            query_embedding: Sorgu vektörü.
            top_k          : Döndürülecek sonuç sayısı.

        Returns:
            [(TurtleRecord, similarity_score)] listesi, skora göre azalan sırada.
        """
        if not self._embeddings:
            return []

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)

        scores: list[tuple[str, float]] = []
        for tid, emb in self._embeddings.items():
            similarity = float(np.dot(query_norm, emb))
            scores.append((tid, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        top = scores[:top_k]

        return [(self._records[tid], sim) for tid, sim in top]

    def search_faiss(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> list[tuple[TurtleRecord, float]]:
        """
        FAISS ile hızlandırılmış benzerlik araması.
        Büyük veritabanları için önerilir (100+ kaplumbağa).
        """
        try:
            import faiss  # type: ignore
        except ImportError:
            logger.warning("FAISS yüklü değil, brute-force aramaya geçiliyor.")
            return self.search(query_embedding, top_k)

        if self._faiss_index is None:
            self._build_faiss_index(faiss)

        query = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        distances, indices = self._faiss_index.search(query, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            tid = self._id_map[idx]
            # FAISS Inner Product → benzerlik skoru
            results.append((self._records[tid], float(dist)))
        return results

    # ------------------------------------------------------------------
    # Dışa / İçe Aktarma
    # ------------------------------------------------------------------

    def export_json(self, output_path: Path | str) -> None:
        """Tüm kayıtları JSON formatında dışa aktarır."""
        data = {
            "metadata": {
                "version"    : "1.0",
                "turtle_count": self.turtle_count,
                "exported_at": datetime.utcnow().isoformat(),
                "embedding_dim": self._embedding_dim,
            },
            "turtles": [asdict(r) for r in self._records.values()],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Veritabanı JSON olarak dışa aktarıldı: %s", output_path)

    # ------------------------------------------------------------------
    # Depolama (Private)
    # ------------------------------------------------------------------

    def _save(self) -> None:
        if not self._db_path:
            return
        payload = {
            "records"      : self._records,
            "embeddings"   : self._embeddings,
            "embedding_dim": self._embedding_dim,
        }
        with open(self._db_path / "database.pkl", "wb") as f:
            pickle.dump(payload, f)

    def _load_if_exists(self) -> None:
        db_file = self._db_path / "database.pkl"
        if not db_file.exists():
            return
        with open(db_file, "rb") as f:
            payload = pickle.load(f)
        self._records       = payload.get("records", {})
        self._embeddings    = payload.get("embeddings", {})
        self._embedding_dim = payload.get("embedding_dim")
        logger.info("Veritabanı diskten yüklendi: %d kayıt.", len(self._records))

    def _invalidate_faiss(self) -> None:
        """FAISS indeksini geçersiz kıl — yeni kayıt/silme sonrası."""
        self._faiss_index = None
        self._id_map      = []

    def _build_faiss_index(self, faiss_module) -> None:
        """Tüm embedding'lerden FAISS inner-product indeksi oluşturur."""
        dim    = self._embedding_dim
        index  = faiss_module.IndexFlatIP(dim)   # Inner Product (cosine için normalize edilmiş)
        ids    = list(self._embeddings.keys())
        matrix = np.stack([self._embeddings[i] for i in ids]).astype(np.float32)
        faiss_module.normalize_L2(matrix)
        index.add(matrix)
        self._faiss_index = index
        self._id_map      = ids
        logger.info("FAISS indeksi oluşturuldu — %d vektör.", len(ids))


# ---------------------------------------------------------------------------
# Demo Verisi Üreteci
# ---------------------------------------------------------------------------

def create_demo_database(embedding_dim: int = 256) -> TurtleDatabase:
    """
    Demo amaçlı sahte kaplumbağa kayıtları ve rastgele embedding vektörleri
    oluşturur. Gerçek veri olmadığında arayüz testi için kullanılır.
    """
    rng = np.random.default_rng(seed=42)
    db  = TurtleDatabase()

    demo_turtles = [
        ("TF-114", "Frankie",  "female", "Antalya Beach"),
        ("TF-227", "Poseidon", "male",   "Dalyan Delta"),
        ("TF-038", "Luna",     "female", "Patara Shore"),
        ("TF-301", "Achilles", "male",   "Iztuzu Beach"),
        ("TF-189", "Coral",    "female", "Çıralı Cove"),
        ("TF-055", "Atlas",    "male",   "Belek Station"),
        ("TF-412", "Nereid",   "female", "Side Beach"),
        ("TF-076", "Titan",    "male",   "Akyatan Lagoon"),
    ]

    for tid, name, sex, loc in demo_turtles:
        record = TurtleRecord(
            turtle_id  = tid,
            name       = name,
            species    = "Caretta caretta",
            sex        = sex,
            location   = loc,
            first_seen = "2023-06-15",
            notes      = f"Demo kaydı — {name}",
        )
        # Deterministik ama ayırt edici embedding
        emb = rng.standard_normal(embedding_dim).astype(np.float32)
        db.add_turtle(record, emb)

    logger.info("Demo veritabanı oluşturuldu — %d kaplumbağa.", db.turtle_count)
    return db
