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
from streamlit_cropper import st_cropper

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
    uploaded_files = st.file_uploader(
        "Kaplumbağa fotoğrafı seç",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="Kaplumbağanın yüzünün göründüğü net bir fotoğraf yükleyin. Birden fazla fotoğraf seçebilirsiniz.",
        label_visibility="collapsed",
    )

    use_demo = st.checkbox("🔬 Demo görüntüsü kullan", value=True,
                           help="Gerçek fotoğraf yoksa sistem üretilmiş test görüntüsü kullanır.")

    cropped_images = []
    if uploaded_files:
        st.markdown("<div style='margin-top: 1rem; color: #00D4B4;'><b>✂️ Kafa Bölgesini Seçin (Human-in-the-Loop)</b></div>", unsafe_allow_html=True)
        for idx, f in enumerate(uploaded_files):
            pil_img = Image.open(f).convert("RGB")
            st.markdown(f"<div style='font-size: 0.85rem; color: #A0A0C0; margin-bottom: 5px;'>{f.name}</div>", unsafe_allow_html=True)
            # Etkileşimli cropper
            cropped_pil = st_cropper(
                pil_img, 
                realtime_update=True, 
                box_color='#00D4B4',
                aspect_ratio=None,
                key=f"cropper_{idx}"
            )
            img_arr = ImageUtils.pil_to_numpy(cropped_pil)
            cropped_images.append({"name": f.name, "array": img_arr})
            st.markdown("<hr style='border-color: #2A2A4A; margin: 10px 0;'>", unsafe_allow_html=True)

    run_btn = st.button("🚀  Tanımlamayı Başlat", use_container_width=True)

    # Pipeline log paneli
    st.markdown("### 📋 Ajan Günlüğü")
    log_container = st.empty()

with result_col:
    result_container = st.container()

# ── Tanımlama Çalıştır ────────────────────────────────────────────────────
if run_btn:
    # Görüntüleri hazırla
    images_to_process = []
    if uploaded_files:
        images_to_process = cropped_images
    elif use_demo:
        rng = np.random.default_rng(seed=7)
        img_arr = rng.integers(40, 130, (320, 420, 3), dtype=np.uint8)
        img_arr[:, :, 0] = np.clip(img_arr[:, :, 0] + 30, 0, 180)
        img_arr[:, :, 1] = np.clip(img_arr[:, :, 1] + 20, 0, 150)
        img_arr[:, :, 2] = np.clip(img_arr[:, :, 2] - 10, 0, 100)
        img_arr = img_arr.astype(np.uint8)
        images_to_process.append({"name": "Demo Görüntüsü", "array": img_arr})
    else:
        st.warning("Lütfen en az bir fotoğraf yükleyin veya 'Demo görüntüsü kullan' seçeneğini işaretleyin.")
        st.stop()

    with result_container:
        st.markdown(f"<h3 style='color:#00D4B4; margin-bottom: 2rem;'>🧪 {len(images_to_process)} Fotoğraf İşleniyor...</h3>", unsafe_allow_html=True)
        
        # Ajanı oluştur
        pipeline = IdentificationAgent(
            use_demo_db=True,
            detection_threshold=det_thresh,
            match_threshold=mat_thresh,
            embedding_dim=256,
        )

        all_logs = []

        for idx, item in enumerate(images_to_process):
            img_name = item["name"]
            img_arr = item["array"]
            
            st.markdown(f"<h4 style='color:#E0E0F0; margin-top: 1rem;'>📷 Fotoğraf {idx+1}: {img_name}</h4>", unsafe_allow_html=True)
            
            # Progress bar
            progress_bar = st.progress(0, text=f"{img_name} işleniyor…")
            status_text  = st.empty()
            
            def progress_callback(msg: str, val: float):
                progress_bar.progress(min(val, 1.0), text=msg)
                status_text.markdown(f"<p style='color:#00D4B4;font-size:0.85rem'>⟳ {msg}</p>", unsafe_allow_html=True)
            
            pipeline._progress_cb = progress_callback
            
            # Pipeline çalıştır
            result = pipeline.identify(img_arr)
            
            progress_bar.progress(1.0, text="Tamamlandı ✓")
            status_text.empty()
            
            # Sonuçları göster (Gereksiz paneller kaldırıldı, sadece Pul Haritası ve Kimlik Gösteriliyor)
            img_col, info_col = st.columns([1.2, 1])
            
            sm = result.scute_map
            with img_col:
                if sm and sm.success and sm.overlay_image is not None:
                    show_image_card("SONUÇ", "Pul Haritası (Scute Map)", f"{sm.scute_count} scute bölgesi tespit edildi", sm.overlay_image)
                else:
                    show_image_card("SONUÇ", "Pul Haritası", "Pul segmentasyonu tamamlandı", img_arr)
                    
            mr = result.match_result
            with info_col:
                if mr:
                    if mr.matched and mr.top_match:
                        t = mr.top_match
                        st.markdown(f"""
                        <div class="match-box">
                          <div class="match-score">{mr.similarity_pct}</div>
                          <div class="match-name">🐢 {t.name}</div>
                          <div class="match-id">{t.turtle_id} &nbsp;·&nbsp; {t.species}</div>
                        </div>""", unsafe_allow_html=True)
                        st.success(f"**Açıklama:** En yakın aday '{t.name}' ({mr.similarity_pct} benzerlik). **%{mat_thresh*100:.0f} olan Eşleşme Eşiği (Threshold)** aşıldığı için kimlik DOĞRULANDI.")
                    else:
                        st.markdown(f"""
                        <div class="no-match-box">
                          <div style='font-size:2rem;color:#FF6B6B'>⚠️</div>
                          <div style='font-size:1.3rem;font-weight:700;color:#FF6B6B'>
                            Bilinmeyen Birey
                          </div>
                        </div>""", unsafe_allow_html=True)
                        
                        top_pct = mr.similarity_pct if mr.top_match else "%0"
                        top_name = mr.top_match.name if mr.top_match else "Bilinmiyor"
                        st.info(f"**Açıklama:** Sistemdeki en yakın aday '{top_name}' ({top_pct} benzerlik). Ancak bu oran **%{mat_thresh*100:.0f} olan Eşleşme Eşiğinin (Threshold)** altında kaldığı için bu kaplumbağa BİLİNMEYEN BİREY olarak işaretlendi.")
                else:
                    show_image_card("HATA", "Eşleşme Sonucu", "Eşleştirme yapılamadı", None, "Hata")
                    
            # Metrikler ve Basit Gösterge
            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            if sm: m1.metric("Tespit Edilen Pul", sm.scute_count)
            m2.metric("İşlem Süresi", f"{result.total_time_ms:.0f} ms")
            if mr: m3.metric("En Yakın Benzerlik", mr.similarity_pct)
            
            if mr:
                # Sadece Gauge Chart gösteriliyor, karmaşık bar grafiği kaldırıldı
                score = mr.similarity_score
                label = mr.top_match.name if mr.matched and mr.top_match else "Bilinmeyen"
                st.markdown("**Benzerlik Göstergesi**")
                st.pyplot(Visualizer.similarity_gauge(score, label))
            
            st.markdown("<hr style='border-color:#2A2A4A;margin:2rem 0'>", unsafe_allow_html=True)
            
            # Logları biriktir
            all_logs.append(f"<div style='margin-top: 10px; color: #E0E0F0;'><b>{img_name} Logları:</b></div>")
            for entry in result.agent_log:
                pct = int(entry["progress"] * 100)
                all_logs.append(f"<div class='log-entry'>[{entry['state']:25s}] {pct:3d}%  {entry['message']}</div>")
            if result.error_message:
                all_logs.append(f"<div class='log-entry' style='color:#FF6B6B'>[ERROR] {result.error_message}</div>")

        st.success("✅ Tüm fotoğrafların analizi başarıyla tamamlandı!", icon="🐢")
        
        with log_container:
            st.markdown("".join(all_logs), unsafe_allow_html=True)

else:
    # Varsayılan boş ekran (Sadeleştirilmiş)
    with result_container:
        img_col, info_col = st.columns([1.2, 1])
        with img_col: show_image_card("SONUÇ", "Pul Haritası (Scute Map)", "Post-ocular scute analizi sonucu", None)
        with info_col: show_image_card("KİMLİK", "Eşleşme Sonucu", "Veritabanı karşılaştırması", None)

# ── Alt bilgi ─────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#2A2A4A;margin:2rem 0 1rem'>
<div style='text-align:center;color:#404060;font-size:0.78rem'>
  TurtleFace ID &nbsp;·&nbsp; SOLID Mimari + AI Agent + Siamese Network &nbsp;·&nbsp;
  Atıf: Jean et al. (2010) · Carter et al. (2014) · TORSOOI DB Format
</div>
""", unsafe_allow_html=True)
