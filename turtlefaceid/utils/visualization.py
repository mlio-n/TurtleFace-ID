"""
Visualizer — Demo Arayüzü için Görsel Araçlar
============================================
SRP: Yalnızca görselleştirme ve grafik oluşturma burada.
"""

from __future__ import annotations

from typing import Optional

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

matplotlib.use("Agg")   # Sunucu ortamında GUI olmaksızın render


class Visualizer:
    """Streamlit arayüzü için görsel bileşenler üretir."""

    # Renk paleti (BGR — OpenCV için)
    TEAL   = (0,  210, 180)
    AMBER  = (0,  190, 255)
    CORAL  = (80,  90, 240)
    WHITE  = (240, 240, 245)
    DARK   = (20,  20,  30)

    # ------------------------------------------------------------------
    # Skor Göstergesi (Gauge Chart)
    # ------------------------------------------------------------------

    @staticmethod
    def similarity_gauge(score: float, label: str = "") -> Figure:
        """
        Benzerlik skorunu dairesel gösterge (gauge) olarak çizer.

        Args:
            score: [0.0 – 1.0] arasında benzerlik skoru.
            label: Altında gösterilecek etiket.
        """
        fig, ax = plt.subplots(figsize=(3.5, 2.5), facecolor="#0E0E1A")
        ax.set_facecolor("#0E0E1A")

        # Arka plan yayı
        theta = np.linspace(0, np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="#2A2A3E", linewidth=18)

        # Skor yayı
        theta_score = np.linspace(0, np.pi * score, 200)
        if score >= 0.80:
            color = "#00D4B4"
        elif score >= 0.60:
            color = "#FFB347"
        else:
            color = "#FF6B6B"

        ax.plot(np.cos(theta_score), np.sin(theta_score), color=color, linewidth=18)

        # Merkez metin
        ax.text(0, 0.15, f"{score:.0%}", ha="center", va="center",
                fontsize=22, fontweight="bold", color="white",
                fontfamily="monospace")
        if label:
            ax.text(0, -0.25, label, ha="center", va="center",
                    fontsize=8, color="#9090B0")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.5, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout(pad=0.2)
        return fig

    # ------------------------------------------------------------------
    # Adım Görüntüsü Çerçevesi
    # ------------------------------------------------------------------

    @staticmethod
    def step_frame(
        image: np.ndarray,
        title: str,
        subtitle: str = "",
        border_color: tuple = (0, 210, 180),
    ) -> np.ndarray:
        """
        Görüntünün üstüne başlık ve renkli çerçeve ekleyerek
        sunuma hazır bir kare oluşturur.
        """
        h, w = image.shape[:2]
        canvas = np.zeros((h + 50, w, 3), dtype=np.uint8)
        canvas[:] = Visualizer.DARK

        # Üst başlık alanı
        cv2.rectangle(canvas, (0, 0), (w, 45), (15, 15, 25), -1)
        cv2.line(canvas, (0, 45), (w, 45), border_color, 2)

        cv2.putText(canvas, title, (10, 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, Visualizer.WHITE, 1, cv2.LINE_AA)
        if subtitle:
            cv2.putText(canvas, subtitle, (10, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 170), 1, cv2.LINE_AA)

        canvas[50:, :] = image
        cv2.rectangle(canvas, (0, 0), (w - 1, h + 49), border_color, 1)
        return canvas

    # ------------------------------------------------------------------
    # Adaylar Bar Grafiği
    # ------------------------------------------------------------------

    @staticmethod
    def candidates_bar(
        candidates: list[tuple],   # [(TurtleRecord, float)]
        highlight_id: Optional[str] = None,
    ) -> Figure:
        """
        Top-K adayları yatay bar grafiği olarak gösterir.
        """
        if not candidates:
            fig, ax = plt.subplots(figsize=(5, 2), facecolor="#0E0E1A")
            ax.text(0.5, 0.5, "Aday yok", ha="center", color="white", fontsize=12)
            ax.axis("off")
            return fig

        names  = [f"{r.name}\n({r.turtle_id})" for r, _ in candidates]
        scores = [s for _, s in candidates]
        colors = []
        for r, _ in candidates:
            if highlight_id and r.turtle_id == highlight_id:
                colors.append("#00D4B4")
            else:
                colors.append("#3A3A5E")

        fig, ax = plt.subplots(
            figsize=(5, max(2.5, len(candidates) * 0.7)),
            facecolor="#0E0E1A",
        )
        ax.set_facecolor("#0E0E1A")

        bars = ax.barh(names, scores, color=colors, height=0.6, edgecolor="none")

        # Değer etiketleri
        for bar, score in zip(bars, scores):
            ax.text(
                score + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.0%}",
                va="center", ha="left", fontsize=9, color="white",
            )

        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Benzerlik Skoru", color="#9090B0", fontsize=9)
        ax.tick_params(colors="#9090B0")
        for spine in ax.spines.values():
            spine.set_color("#2A2A3E")
        ax.invert_yaxis()
        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Pipeline Akış Diyagramı
    # ------------------------------------------------------------------

    @staticmethod
    def pipeline_diagram(active_step: int = -1) -> Figure:
        """
        Pipeline adımlarını görsel olarak gösteren statik diyagram.

        active_step: Vurgulanacak adım (0=Yükleme, 1=Tespit, 2=Pul, 3=Eşleşme)
        """
        steps = [
            ("📷", "Görüntü\nYükleme"),
            ("🐢", "Yüz\nTespiti"),
            ("🔬", "Pul\nHaritası"),
            ("🔗", "Kimlik\nEşleştirme"),
        ]

        fig, ax = plt.subplots(figsize=(7, 1.8), facecolor="#0E0E1A")
        ax.set_facecolor("#0E0E1A")
        ax.set_xlim(-0.5, len(steps) - 0.5)
        ax.set_ylim(-0.5, 1.5)

        for i, (icon, label) in enumerate(steps):
            is_active = (i == active_step)
            box_color = "#00D4B4" if is_active else "#2A2A3E"
            txt_color = "#0E0E1A" if is_active else "#9090B0"

            circle = plt.Circle(
                (i, 0.7), 0.32,
                color=box_color, zorder=3,
            )
            ax.add_patch(circle)
            ax.text(i, 0.72, icon, ha="center", va="center",
                    fontsize=14, zorder=4)
            ax.text(i, 0.15, label, ha="center", va="top",
                    fontsize=7.5, color="#9090B0" if not is_active else "#00D4B4",
                    multialignment="center")

            if i < len(steps) - 1:
                ax.annotate(
                    "", xy=(i + 0.38, 0.7), xytext=(i + 0.62, 0.7),
                    arrowprops=dict(arrowstyle="<-", color="#3A3A5E", lw=1.5),
                )

        ax.axis("off")
        plt.tight_layout(pad=0.1)
        return fig
