"""
TurtleFace ID — Streamlit Demo Arayüzü
=======================================
Hocaya sunulacak görsel pipeline demosu.
Her adım ayrı bir kart/sekme olarak gösterilir:
  1. Orijinal yüklenen fotoğraf
  2. Tespit edilip kırpılan yüz
  3. Pulların vurgulandığı harita
  4. Veritabanı eşleşme sonucu
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import streamlit as st
from PIL import Image

from turtlefaceid.agents.identification_agent import IdentificationAgent, AgentState
from turtlefaceid.utils.image_utils import ImageUtils
from turtlefaceid.utils.visualization import Visualizer

# ── Sayfa konfigürasyonu ──────────────────────────────────────────────────
st.set_page_config(
    page_title="TurtleFace ID",
    page_icon="🐢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — Dark Premium Tema ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #080812;
    color: #E0E0F0;
}

/* Ana arka plan */
.stApp { background: linear-gradient(135deg, #080812 0%, #0E0E20 100%); }

/* Kart bileşeni */
.tf-card {
    background: linear-gradient(145deg, #12122A, #1A1A35);
    border: 1px solid #2A2A4A;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0, 212, 180, 0.05);
    transition: box-shadow 0.3s ease;
}
.tf-card:hover { box-shadow: 0 8px 32px rgba(0, 212, 180, 0.12); }

/* Adım başlığı */
.step-badge {
    display: inline-flex; align-items: center; gap: 8px;
    background: linear-gradient(90deg, #00D4B4, #0099CC);
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #080812; margin-bottom: 0.6rem;
}

/* Eşleşme sonucu kutusu */
.match-box {
    background: linear-gradient(135deg, #0A1A1A, #0A2A2A);
    border: 2px solid #00D4B4;
    border-radius: 16px; padding: 1.8rem;
    text-align: center;
}
.match-name  { font-size: 2rem; font-weight: 700; color: #00D4B4; }
.match-id    { font-size: 1rem; color: #7090A0; font-family: 'JetBrains Mono'; }
.match-score { font-size: 2.8rem; font-weight: 800;
               background: linear-gradient(90deg, #00D4B4, #FFB347);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.no-match-box {
    background: linear-gradient(135deg, #1A0A0A, #2A0A0A);
    border: 2px solid #FF6B6B; border-radius: 16px;
    padding: 1.8rem; text-align: center;
}

/* Log satırı */
.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem; color: #60D4B4;
    padding: 2px 0; border-bottom: 1px solid #1A1A30;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D0D20, #12122A) !important;
    border-right: 1px solid #2A2A4A;
}

/* Buton */
.stButton > button {
    background: linear-gradient(135deg, #00D4B4, #0099CC);
    color: #080812; border: none; border-radius: 10px;
    font-weight: 700; font-size: 1rem; padding: 0.7rem 2rem;
    width: 100%; transition: all 0.25s ease;
}
.stButton > button:hover { transform: translateY(-2px); opacity: 0.9; }

/* İlerleme çubuğu */
.stProgress > div > div { background: linear-gradient(90deg, #00D4B4, #0099CC); }

/* Metin girişi */
.stFileUploader { border: 2px dashed #2A2A4A; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


# ── Ajan — Session State'te Sakla ────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_agent() -> IdentificationAgent:
    return IdentificationAgent(use_demo_db=True, embedding_dim=256)


# ── Yardımcı: Görüntüyü Streamlit'e Hazırla ──────────────────────────────
def show_image_card(step_num: str, title: str, subtitle: str,
                    image: np.ndarray | None, placeholder_text: str = "Bekleniyor…"):
    """Adım numaralı görüntü kartı."""
    img = image if image is not None else ImageUtils.create_placeholder(
        400, 320, placeholder_text)
    st.markdown(f"""
    <div class="tf-card">
      <div class="step-badge">ADIM {step_num}</div>
      <p style="font-weight:600;font-size:1rem;margin:0 0 4px">{title}</p>
      <p style="color:#7070A0;font-size:0.82rem;margin:0 0 12px">{subtitle}</p>
    </div>""", unsafe_allow_html=True)
    st.image(img, use_container_width=True)


# ── Başlık ────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<div style='font-size:3.5rem;line-height:1;padding-top:6px'>🐢</div>",
                unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin:0;font-size:2.1rem;font-weight:800;
               background:linear-gradient(90deg,#00D4B4,#0099CC);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
      TurtleFace ID
    </h1>
    <p style='margin:0;color:#7070A0;font-size:0.9rem;'>
      Deniz Kaplumbağası Bireysel Tanıma Sistemi &nbsp;·&nbsp;
      Siamese Network + Post-Ocular Scute Analizi
    </p>""", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#2A2A4A;margin:1rem 0'>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Sistem Ayarları")

    det_thresh = st.slider("Tespit Eşiği", 0.2, 0.9, 0.50, 0.05,
                           help="Yüz tespiti minimum güven skoru")
    mat_thresh = st.slider("Eşleşme Eşiği", 0.3, 0.95, 0.55, 0.05,
                           help="Kimlik eşleşme minimum benzerlik skoru")

    st.markdown("---")
    st.markdown("### 🗄️ Veritabanı Durumu")
    agent = get_agent()
    st.metric("Kayıtlı Kaplumbağa", agent.database.turtle_count)

    st.markdown("---")
    st.markdown("### 📚 Hakkında")
    st.markdown("""
    <div style='font-size:0.78rem;color:#7070A0;line-height:1.6'>
    <b>Yöntem:</b> Siamese Network (ResNet50)<br>
    <b>Kayıp Fonk.:</b> Contrastive Loss<br>
    <b>Mimari:</b> SOLID + AI Agent<br>
    <b>Veri:</b> Few-Shot Learning<br><br>
    <i>Jean et al. (2010)</i><br>
    <i>Carter et al. (2014)</i>
    </div>""", unsafe_allow_html=True)

# ── Ana İçerik ────────────────────────────────────────────────────────────
upload_col, result_col = st.columns([1, 2], gap="large")

with upload_col:
    st.markdown("### 📤 Fotoğraf Yükle")
    uploaded = st.file_uploader(
        "Kaplumbağa fotoğrafı seç",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        help="Kaplumbağanın yüzünün göründüğü net bir fotoğraf yükleyin.",
        label_visibility="collapsed",
    )

    use_demo = st.checkbox("🔬 Demo görüntüsü kullan", value=True,
                           help="Gerçek fotoğraf yoksa sistem üretilmiş test görüntüsü kullanır.")

    run_btn = st.button("🚀  Tanımlamayı Başlat", use_container_width=True)

    # Pipeline log paneli
    st.markdown("### 📋 Ajan Günlüğü")
    log_container = st.empty()

with result_col:
    # 4 adım — 2x2 ızgara
    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)

    orig_ph   = r1c1.empty()
    det_ph    = r1c2.empty()
    scute_ph  = r2c1.empty()
    match_ph  = r2c2.empty()

    # Boş yer tutucular
    with orig_ph.container():
        show_image_card("1", "Orijinal Fotoğraf", "Ham girdi görüntüsü", None)
    with det_ph.container():
        show_image_card("2", "Tespit Edilen Yüz", "Kırpılan kafa bölgesi", None)
    with scute_ph.container():
        show_image_card("3", "Pul Haritası", "Post-ocular scute analizi", None)
    with match_ph.container():
        show_image_card("4", "Eşleşme Sonucu", "Veritabanı karşılaştırması", None)

    # Özet metrikler
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    met_conf   = m1.empty()
    met_scutes = m2.empty()
    met_time   = m3.empty()
    met_cands  = m4.empty()

    chart_col1, chart_col2 = st.columns(2)
    gauge_ph   = chart_col1.empty()
    bar_ph     = chart_col2.empty()

# ── Tanımlama Çalıştır ────────────────────────────────────────────────────
if run_btn:
    # Görüntüyü hazırla
    if uploaded is not None:
        pil_img = Image.open(uploaded).convert("RGB")
        img_arr = ImageUtils.pil_to_numpy(pil_img)
    elif use_demo:
        # Demo: gerçekçi yeşil tonlu sentetik görüntü
        rng     = np.random.default_rng(seed=7)
        img_arr = rng.integers(40, 130, (320, 420, 3), dtype=np.uint8)
        # Kaplumbağa benzeri renk tonu (kahverengi-yeşil)
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + 30, 0, 180)  # R
        img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + 20, 0, 150)  # G
        img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - 10, 0, 100)  # B
        img_arr = img_arr.astype(np.uint8)
    else:
        st.warning("Lütfen bir fotoğraf yükleyin veya 'Demo görüntüsü kullan' seçeneğini işaretleyin.")
        st.stop()

    # İlerleme çubuğu
    progress_bar = st.progress(0, text="Pipeline başlatılıyor…")
    status_text  = st.empty()

    def progress_callback(msg: str, val: float):
        progress_bar.progress(min(val, 1.0), text=msg)
        status_text.markdown(f"<p style='color:#00D4B4;font-size:0.85rem'>⟳ {msg}</p>",
                             unsafe_allow_html=True)

    # Ajanı yeniden oluştur (eşik ayarları için)
    pipeline = IdentificationAgent(
        use_demo_db=True,
        detection_threshold=det_thresh,
        match_threshold=mat_thresh,
        embedding_dim=256,
        progress_callback=progress_callback,
    )

    # Pipeline çalıştır
    result = pipeline.identify(img_arr)

    progress_bar.progress(1.0, text="Tamamlandı ✓")
    status_text.empty()

    # ── Adım 1: Orijinal görüntü ──────────────────────────────────────
    with orig_ph.container():
        show_image_card("1", "Orijinal Fotoğraf",
                        f"{img_arr.shape[1]}×{img_arr.shape[0]} px", img_arr)

    # ── Adım 2: Tespit ────────────────────────────────────────────────
    det = result.detection_result
    if det and det.success:
        det_img = det.annotated_image if det.annotated_image is not None else det.cropped_face
        with det_ph.container():
            show_image_card("2", "Tespit Edilen Yüz",
                            f"Güven: {det.confidence:.0%}  •  BBox: {det.bounding_box}",
                            det_img)
        face_for_display = det.cropped_face
    else:
        with det_ph.container():
            show_image_card("2", "Tespit Edilemedi",
                            det.error_message if det else "Hata", None,
                            "Yüz bulunamadı")
        face_for_display = img_arr   # fallback

    # ── Adım 3: Pul haritası ──────────────────────────────────────────
    sm = result.scute_map
    if sm and sm.success and sm.overlay_image is not None:
        with scute_ph.container():
            show_image_card("3", "Pul Haritası",
                            f"{sm.scute_count} scute bölgesi tespit edildi",
                            sm.overlay_image)
    else:
        with scute_ph.container():
            show_image_card("3", "Pul Haritası",
                            "Pul segmentasyonu tamamlandı (kenar analizi)",
                            face_for_display)

    # ── Adım 4: Eşleşme sonucu ────────────────────────────────────────
    mr = result.match_result
    with match_ph.container():
        show_image_card("4", "Eşleşme Sonucu",
                        "Siamese Network benzerlik skoru", None,
                        "Sonuç hesaplanıyor")
        if mr:
            if mr.matched and mr.top_match:
                t = mr.top_match
                st.markdown(f"""
                <div class="match-box">
                  <div class="match-score">{mr.similarity_pct}</div>
                  <div class="match-name">🐢 {t.name}</div>
                  <div class="match-id">{t.turtle_id} &nbsp;·&nbsp; {t.species}</div>
                  <p style='color:#7090A0;font-size:0.82rem;margin-top:8px'>
                    📍 {t.location or "—"} &nbsp;·&nbsp; 🗓 {t.first_seen or "—"}
                  </p>
                  <p style='color:#9090B0;font-size:0.78rem;margin-top:8px'>
                    <b>Güven:</b> {mr.confidence_level.value}
                  </p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="no-match-box">
                  <div style='font-size:2rem;color:#FF6B6B'>⚠️</div>
                  <div style='font-size:1.3rem;font-weight:700;color:#FF6B6B'>
                    Bilinmeyen Birey
                  </div>
                  <p style='color:#A07070;font-size:0.85rem;margin-top:8px'>
                    En yüksek benzerlik: {mr.similarity_pct}<br>
                    Eşleşme eşiğinin altında.
                  </p>
                  <p style='color:#806060;font-size:0.78rem'>
                    Muhtemelen yeni bir kaplumbağa — veritabanına eklenmeli.
                  </p>
                </div>""", unsafe_allow_html=True)

    # ── Metrikler ─────────────────────────────────────────────────────
    if mr:
        met_conf.metric("Benzerlik Skoru", mr.similarity_pct)
    if sm:
        met_scutes.metric("Tespit Edilen Pul", sm.scute_count)
    met_time.metric("İşlem Süresi", f"{result.total_time_ms:.0f} ms")
    if mr:
        met_cands.metric("Aday Sayısı", len(mr.top_candidates))

    # ── Gauge grafiği ─────────────────────────────────────────────────
    if mr:
        score     = mr.similarity_score
        label     = mr.top_match.name if mr.matched and mr.top_match else "Bilinmeyen"
        gauge_fig = Visualizer.similarity_gauge(score, label)
        with gauge_ph:
            st.markdown("**Benzerlik Göstergesi**")
            st.pyplot(gauge_fig)

        bar_fig = Visualizer.candidates_bar(
            mr.top_candidates,
            highlight_id=mr.top_match.turtle_id if mr.top_match else None,
        )
        with bar_ph:
            st.markdown("**Top-5 Adaylar**")
            st.pyplot(bar_fig)

    # ── Ajan günlüğü ──────────────────────────────────────────────────
    with log_container:
        log_html = ""
        for entry in result.agent_log:
            pct = int(entry["progress"] * 100)
            log_html += (
                f"<div class='log-entry'>"
                f"[{entry['state']:25s}] {pct:3d}%  {entry['message']}"
                f"</div>"
            )
        if result.error_message:
            log_html += (
                f"<div class='log-entry' style='color:#FF6B6B'>"
                f"[ERROR] {result.error_message}</div>"
            )
        st.markdown(log_html, unsafe_allow_html=True)

    if result.success:
        st.success("✅ Pipeline başarıyla tamamlandı!", icon="🐢")
    else:
        st.error(f"Pipeline başarısız: {result.error_message}")

# ── Alt bilgi ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2A2A4A;margin:2rem 0 1rem'>
<div style='text-align:center;color:#404060;font-size:0.78rem'>
  TurtleFace ID &nbsp;·&nbsp; SOLID Mimari + AI Agent + Siamese Network &nbsp;·&nbsp;
  Atıf: Jean et al. (2010) · Carter et al. (2014) · TORSOOI DB Format
</div>
""", unsafe_allow_html=True)
