"""
gesture_engine.py - Motor de detección de gestos con MediaPipe
Mejoras sobre el original:
  • EMA (Exponential Moving Average) para suavizado agresivo de landmarks
  • Historial de frames para confirmar gesto (evita falsas detecciones)
  • Lógica de puño mejorada
  • Devuelve GestureResult en lugar de efectos secundarios
"""

import math
import time
import threading
import collections
from dataclasses import dataclass, field
from typing import Optional

import mediapipe as mp
import cv2
import numpy as np

import config


# ──────────────────────────────────────────────────────────────
#  RESULTADO DE GESTO
# ──────────────────────────────────────────────────────────────
@dataclass
class GestureResult:
    dedos: int = 0
    rotacion_muneca: float = 0.0    # grados, útil para DJ-knob
    mano_detectada: bool = False
    gesto_nombre: str = ""          # "SIGUIENTE", "PLAY_PAUSE", etc.
    confianza: float = 0.0


# ──────────────────────────────────────────────────────────────
#  GESTOS RECONOCIDOS
# ──────────────────────────────────────────────────────────────
GESTOS = {
    1: "SIGUIENTE",
    2: "PLAYLIST DESPECHO",
    3: "DISCO LIGHT",
    4: "PLAY_PAUSE",
    5: "PLAY_PAUSE",
}


# ──────────────────────────────────────────────────────────────
#  MOTOR
# ──────────────────────────────────────────────────────────────
class GestureEngine:
    """
    Detecta gestos de mano en tiempo real usando MediaPipe LIVE_STREAM.
    Thread-safe: consume frames desde el loop principal y expone el
    resultado más reciente a través de `resultado_actual`.
    """

    CONEXIONES = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(0,17),(17,18),(18,19),(19,20),
    ]
    PUNTAS = [4, 8, 12, 16, 20]  # pulgar, índice, medio, anular, meñique

    def __init__(self):
        self._raw_result = None
        self._lock = threading.Lock()

        # Suavizado EMA por landmark (21 puntos × {x,y,z})
        self._smooth: Optional[list] = None
        self._alpha = config.SMOOTH_ALPHA

        # Historial de conteos para confirmar gesto (ventana de N frames)
        self._historial: collections.deque = collections.deque(maxlen=8)

        self.resultado_actual = GestureResult()
        self._ultimo_ts = 0

        # Inicializar MediaPipe
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=config.HAND_LANDMARKER_MODEL
            ),
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._callback_mp,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        print("[Gestos] Motor MediaPipe iniciado.")

    # ──────────────────────────────────────────
    #  CALLBACK INTERNO (hilo de MediaPipe)
    # ──────────────────────────────────────────
    def _callback_mp(self, result, output_image, timestamp_ms):
        with self._lock:
            self._raw_result = result

    # ──────────────────────────────────────────
    #  PROCESAR FRAME
    # ──────────────────────────────────────────
    def procesar_frame(self, frame_bgr: np.ndarray) -> GestureResult:
        """
        Envía el frame a MediaPipe y actualiza `resultado_actual`.
        Llama este método una vez por frame desde el loop principal.
        """
        ts_ms = int(time.time() * 1000)

        # Evitar timestamps duplicados (MediaPipe lo requiere estrictamente creciente)
        if ts_ms <= self._ultimo_ts:
            ts_ms = self._ultimo_ts + 1
        self._ultimo_ts = ts_ms

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._landmarker.detect_async(mp_image, ts_ms)

        # Leer último resultado disponible
        with self._lock:
            raw = self._raw_result

        resultado = GestureResult()

        if raw and raw.hand_landmarks:
            landmarks_raw = raw.hand_landmarks[0]

            # ── Suavizado EMA ──────────────────────────────
            if self._smooth is None:
                self._smooth = [(lm.x, lm.y, lm.z) for lm in landmarks_raw]
            else:
                a = self._alpha
                self._smooth = [
                    (
                        a * lm.x + (1 - a) * sx,
                        a * lm.y + (1 - a) * sy,
                        a * lm.z + (1 - a) * sz,
                    )
                    for lm, (sx, sy, sz) in zip(landmarks_raw, self._smooth)
                ]

            lm = self._smooth  # alias cómodo

            # ── Geometría base ─────────────────────────────
            x0, y0 = lm[0][0] * w, lm[0][1] * h
            x9, y9 = lm[9][0] * w, lm[9][1] * h
            dist_palma = math.hypot(x9 - x0, y9 - y0)

            # ── Rotación de muñeca (grados) ────────────────
            rotacion = math.degrees(math.atan2(y9 - y0, x9 - x0)) + 90
            resultado.rotacion_muneca = rotacion

            # ── Conteo de dedos ────────────────────────────
            dedos = 0
            for i, punta in enumerate(self.PUNTAS):
                xp, yp = lm[punta][0] * w, lm[punta][1] * h
                dist = math.hypot(xp - x0, yp - y0)
                ratio = config.THUMB_RATIO if i == 0 else config.FINGER_RATIO
                if dist > dist_palma * ratio:
                    dedos += 1

            # ── Verificación de puño (override) ───────────
            y_puntas = [lm[p][1] for p in [8, 12, 16, 20]]
            y_nudill = [lm[p][1] for p in [6, 10, 14, 18]]
            if all(p > n for p, n in zip(y_puntas, y_nudill)):
                dedos = 0

            # ── Historial para confirmar gesto ─────────────
            self._historial.append(dedos)
            # Solo confirmar si la mayoría del historial coincide
            mas_comun = max(set(self._historial), key=self._historial.count)
            votos = self._historial.count(mas_comun)
            confianza = votos / len(self._historial)

            resultado.dedos = mas_comun
            resultado.mano_detectada = True
            resultado.confianza = confianza
            resultado.gesto_nombre = GESTOS.get(mas_comun, "")

            # ── Dibujar esqueleto en el frame ──────────────
            for ini, fin in self.CONEXIONES:
                p1 = (int(lm[ini][0] * w), int(lm[ini][1] * h))
                p2 = (int(lm[fin][0] * w), int(lm[fin][1] * h))
                cv2.line(frame_bgr, p1, p2, (255, 80, 80), 2)

            # Puntos de referencia clave
            for idx in self.PUNTAS + [0]:
                px = int(lm[idx][0] * w)
                py = int(lm[idx][1] * h)
                cv2.circle(frame_bgr, (px, py), 5, (0, 255, 255), -1)

        else:
            self._smooth = None
            self._historial.clear()

        self.resultado_actual = resultado
        return resultado

    def cerrar(self):
        self._landmarker.close()
        print("[Gestos] Motor cerrado.")