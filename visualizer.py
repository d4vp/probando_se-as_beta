"""
visualizer.py - Motor gráfico del Karaoke
Características:
  • Pantalla de karaoke con scroll suave
  • Efecto "Bola de Karaoke" (resaltado palabra a palabra)
  • Modo Disco Light (fondo con cambios de color sincronizados)
  • Overlay de estado de gestos en la ventana de cámara
"""

import time
import random
import colorsys
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config


# ──────────────────────────────────────────────────────────────
#  HELPERS DE FUENTE
# ──────────────────────────────────────────────────────────────
def _cargar_fuente(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


# ──────────────────────────────────────────────────────────────
#  VISUALIZADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────
class Visualizer:

    # Colores base (modo normal)
    C_BG_NORMAL    = (12, 12, 20)
    C_ACTIVE       = (255, 235, 59)    # amarillo karaoke
    C_SYNC_ACCENT  = (0, 229, 255)     # cian (LRC)
    C_UNSYNC_ACCT  = (29, 185, 84)     # verde Spotify (texto plano)
    C_DIM          = (100, 100, 110)
    C_LINE         = (50, 50, 60)
    C_WHITE        = (240, 240, 240)

    # Colores modo despecho
    C_BG_ESPECIAL  = (40, 5, 5)
    C_ESPECIAL_ACC = (255, 50, 50)

    def __init__(self):
        self.w = config.KARAOKE_W
        self.h = config.KARAOKE_H

        self.font_titulo  = _cargar_fuente(config.FONT_PATH, config.FONT_SIZE_TITLE)
        self.font_artista = _cargar_fuente(config.FONT_PATH, config.FONT_SIZE_ARTIST)
        self.font_normal  = _cargar_fuente(config.FONT_PATH, config.FONT_SIZE_NORMAL)
        self.font_activa  = _cargar_fuente(config.FONT_PATH, config.FONT_SIZE_ACTIVE)

        # Disco Light
        self._disco_activo = False
        self._disco_hue    = random.random()
        self._disco_ts     = time.time()
        self._disco_bg     = (12, 12, 20)

        # Suavizado de scroll
        self._scroll_y_actual = 0.0

        # Crear ventanas
        cv2.namedWindow("🎤 Karaoke", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("🎤 Karaoke", self.w, self.h)
        cv2.namedWindow("🖐 Gestos", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("🖐 Gestos", config.CAMERA_W, config.CAMERA_H)

    # ──────────────────────────────────────────
    #  DISCO LIGHT
    # ──────────────────────────────────────────
    def toggle_disco(self):
        self._disco_activo = not self._disco_activo
        print(f"[Visual] Disco Light: {'ON' if self._disco_activo else 'OFF'}")

    def _actualizar_disco(self):
        if not self._disco_activo:
            return
        now = time.time()
        if now - self._disco_ts >= config.DISCO_INTERVAL_S:
            self._disco_ts = now
            self._disco_hue = (self._disco_hue + random.uniform(0.08, 0.25)) % 1.0
            r, g, b = colorsys.hsv_to_rgb(self._disco_hue, 0.7, 0.18)
            self._disco_bg = (int(r * 255), int(g * 255), int(b * 255))

    # ──────────────────────────────────────────
    #  RENDER KARAOKE
    # ──────────────────────────────────────────
    def render_karaoke(
        self,
        titulo: str,
        artista: str,
        lineas: list,
        indice_activo: int,
        progreso_palabra: float,
        es_sincronizado: bool,
        modo_especial: bool,
        cargando: bool,
        volumen: int,
        comando_activo: str,
    ) -> np.ndarray:
        """
        Genera el frame PIL y lo convierte a numpy para imshow.
        """
        self._actualizar_disco()

        # Fondo
        if self._disco_activo:
            bg = self._disco_bg
        elif modo_especial:
            bg = self.C_BG_ESPECIAL
        else:
            bg = self.C_BG_NORMAL

        img = Image.new("RGB", (self.w, self.h), bg)
        draw = ImageDraw.Draw(img)

        accent = (self.C_ESPECIAL_ACC if modo_especial
                  else self.C_SYNC_ACCENT if es_sincronizado
                  else self.C_UNSYNC_ACCT)

        # ── Cabecera ───────────────────────────────────────────
        draw.text((40, 22), titulo.upper()[:50], font=self.font_titulo, fill=accent)
        modo_txt = ("💔 MODO ESPECIAL" if modo_especial
                    else "⚡ LRC SYNC" if es_sincronizado
                    else "📜 TEXTO PLANO")
        draw.text((40, 65),
                  f"{artista}   │   {modo_txt}   │   🔊 {volumen}%",
                  font=self.font_artista, fill=self.C_DIM)
        draw.line([(40, 100), (self.w - 40, 100)], fill=self.C_LINE, width=1)

        # ── Letras ─────────────────────────────────────────────
        if cargando or len(lineas) < 4:
            txt = lineas[1] if len(lineas) > 1 else "Cargando…"
            w_t = draw.textlength(txt, font=self.font_normal)
            draw.text(((self.w - w_t) // 2, self.h // 2 - 20),
                      txt, font=self.font_normal, fill=self.C_DIM)
        else:
            centro_y = 360
            for offset in range(-4, 5):
                idx = indice_activo + offset
                if not (0 <= idx < len(lineas)):
                    continue

                texto = lineas[idx]
                pos_y = centro_y + offset * 52

                if offset == 0:
                    # ── Línea activa + efecto bouncing ball ────
                    self._render_linea_activa(
                        draw, texto, pos_y, accent,
                        progreso_palabra, modo_especial
                    )
                else:
                    # Líneas vecinas: fade por distancia
                    op = max(200 - abs(offset) * 45, 20)
                    color = (op, op, op)
                    w_t = draw.textlength(texto, font=self.font_normal)
                    draw.text(((self.w - w_t) // 2, pos_y),
                              texto, font=self.font_normal, fill=color)

        # ── Barra de progreso (si LRC) ─────────────────────────
        if es_sincronizado and not cargando:
            self._render_progreso_bar(draw, progreso_palabra, accent)

        # ── Comando activo overlay ─────────────────────────────
        if comando_activo:
            self._render_badge(draw, comando_activo, accent)

        # ── Disco light indicator ──────────────────────────────
        if self._disco_activo:
            draw.text((self.w - 160, 22), "🪩 DISCO ON",
                      font=self.font_artista, fill=(255, 180, 255))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # ──────────────────────────────────────────
    #  RENDER LÍNEA ACTIVA (BOUNCING BALL)
    # ──────────────────────────────────────────
    def _render_linea_activa(
        self,
        draw: ImageDraw.ImageDraw,
        texto: str,
        pos_y: int,
        accent: Tuple[int, int, int],
        progreso: float,
        modo_especial: bool,
    ):
        if not texto.strip():
            return

        palabras = texto.split()
        if not palabras:
            return

        total_palabras = len(palabras)
        # Índice de la palabra activa según progreso (0.0–1.0)
        idx_pal_f = progreso * total_palabras
        idx_pal = int(idx_pal_f)
        frac_intra = idx_pal_f - idx_pal  # qué tan avanzada está esa palabra

        # Calcular ancho total para centrar
        anchos = [draw.textlength(p + " ", font=self.font_activa) for p in palabras]
        ancho_total = sum(anchos)
        x_ini = (self.w - ancho_total) // 2

        x = x_ini
        for i, (pal, ancho) in enumerate(zip(palabras, anchos)):
            if i < idx_pal:
                # Palabras ya cantadas: color acento sólido
                draw.text((x, pos_y - 12), pal, font=self.font_activa, fill=accent)
            elif i == idx_pal:
                # Palabra actual: mezcla acento → blanco según fracción
                r = int(accent[0] + (self.C_WHITE[0] - accent[0]) * (1 - frac_intra))
                g = int(accent[1] + (self.C_WHITE[1] - accent[1]) * (1 - frac_intra))
                b = int(accent[2] + (self.C_WHITE[2] - accent[2]) * (1 - frac_intra))
                draw.text((x, pos_y - 12), pal, font=self.font_activa,
                          fill=(r, g, b))
                # Punto de la bola bajo la palabra
                ball_x = int(x + ancho * frac_intra)
                ball_y = pos_y + 28
                draw.ellipse(
                    [(ball_x - 5, ball_y - 5), (ball_x + 5, ball_y + 5)],
                    fill=accent
                )
            else:
                # Palabras futuras: gris claro
                draw.text((x, pos_y - 12), pal, font=self.font_activa,
                          fill=(160, 160, 160))
            x += ancho

        # Icono lateral
        icono = "💔" if modo_especial else "🎤"
        draw.text((30, pos_y - 12), icono, font=self.font_artista, fill=accent)

    # ──────────────────────────────────────────
    #  BARRA DE PROGRESO DE LÍNEA
    # ──────────────────────────────────────────
    def _render_progreso_bar(self, draw, progreso: float, color: tuple):
        bar_x0, bar_y = 40, self.h - 30
        bar_w = self.w - 80
        draw.rectangle([(bar_x0, bar_y), (bar_x0 + bar_w, bar_y + 4)],
                        fill=(40, 40, 50))
        fill_w = int(bar_w * max(0.0, min(progreso, 1.0)))
        if fill_w > 0:
            draw.rectangle([(bar_x0, bar_y),
                             (bar_x0 + fill_w, bar_y + 4)],
                            fill=color)

    # ──────────────────────────────────────────
    #  BADGE DE COMANDO
    # ──────────────────────────────────────────
    def _render_badge(self, draw, texto: str, color: tuple):
        tw = draw.textlength(texto, font=self.font_artista)
        x0, y0 = (self.w - tw) // 2 - 12, self.h - 75
        draw.rounded_rectangle(
            [(x0, y0), (x0 + tw + 24, y0 + 34)],
            radius=8, fill=(20, 20, 30), outline=color, width=2
        )
        draw.text((x0 + 12, y0 + 8), texto,
                  font=self.font_artista, fill=color)

    # ──────────────────────────────────────────
    #  RENDER CÁMARA
    # ──────────────────────────────────────────
    def render_camara(
        self,
        frame: np.ndarray,
        dedos: int,
        gesto_nombre: str,
        volumen: int,
        comando_display: str,
    ) -> np.ndarray:
        """Dibuja el HUD sobre el frame de la cámara."""
        # Contador de dedos
        cv2.rectangle(frame, (15, 15), (135, 130), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, str(dedos), (35, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.2, (0, 255, 255), 5)

        # Barra de volumen lateral
        h_frame = frame.shape[0]
        bh = int(h_frame * 0.6)
        by0 = (h_frame - bh) // 2
        cv2.rectangle(frame, (frame.shape[1] - 25, by0),
                      (frame.shape[1] - 10, by0 + bh), (40, 40, 40), cv2.FILLED)
        fill_h = int(bh * volumen / 100)
        cv2.rectangle(frame, (frame.shape[1] - 25, by0 + bh - fill_h),
                      (frame.shape[1] - 10, by0 + bh), (0, 229, 255), cv2.FILLED)
        cv2.putText(frame, f"{volumen}%",
                    (frame.shape[1] - 50, by0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

        # Gesto activo
        if gesto_nombre:
            cv2.putText(frame, gesto_nombre, (150, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        if comando_display:
            cv2.putText(frame, f">> {comando_display}", (150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        return frame

    # ──────────────────────────────────────────
    #  MOSTRAR EN VENTANAS
    # ──────────────────────────────────────────
    def mostrar_karaoke(self, frame: np.ndarray):
        cv2.imshow("🎤 Karaoke", frame)

    def mostrar_camara(self, frame: np.ndarray):
        cv2.imshow("🖐 Gestos", frame)

    def cerrar(self):
        cv2.destroyAllWindows()