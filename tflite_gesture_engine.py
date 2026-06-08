"""
tflite_gesture_engine.py  –  Motor de reconocimiento de gestos por clasificador TFLite
────────────────────────────────────────────────────────────────────────────────────────
REGLA DE ORO: Este módulo es 100 % ADITIVO.
  • No importa ni modifica gesture_engine.py, main.py, visualizer.py ni audio_manager.py.
  • Se instancia opcionalmente en main.py con 3 líneas de código.
  • Toda la inferencia ocurre en un hilo daemon independiente (no bloquea el loop principal).

Clases reconocidas (labels.txt):
  0  tomar       → activa playlist de despecho
  1  amor        → activa una pista de amor
  2  sin_gestos  → ignorado (filtro de ruido)

Parámetros de activación:
  • Confianza  > CONFIDENCE_THRESHOLD  (85 %)
  • Estabilidad: misma clase durante STABILITY_FRAMES frames consecutivos
                 dentro de una ventana deslizante de BUFFER_SIZE frames
"""

from __future__ import annotations

import threading
import collections
import time
import logging
from pathlib import Path
from typing import Optional, Callable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
#  CONSTANTES AJUSTABLES
# ──────────────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.85   # Umbral mínimo de confianza
BUFFER_SIZE:          int   = 5      # Ventana de frames para el voto de estabilidad
STABILITY_FRAMES:     int   = 5      # Frames consecutivos iguales para activar
INPUT_SIZE:           int   = 224    # Dimensión cuadrada esperada por el modelo
COOLDOWN_S:           float = 10.0    # Segundos entre activaciones del mismo gesto

# Clases que SÍ disparan acciones (la lista blanca)
ACTIVE_CLASSES = {"tomar", "amor"}

# ──────────────────────────────────────────────────────────────
#  RESULTADO PÚBLICO
# ──────────────────────────────────────────────────────────────
class TFLiteGestureResult:
    """Snapshot inmutable del estado del motor en un instante dado."""

    __slots__ = ("label", "confidence", "activated", "timestamp")

    def __init__(
        self,
        label: str       = "sin_gestos",
        confidence: float = 0.0,
        activated: bool  = False,
        timestamp: float = 0.0,
    ):
        self.label      = label
        self.confidence = confidence
        self.activated  = activated   # True sólo el frame en que se dispara la acción
        self.timestamp  = timestamp

    def __repr__(self) -> str:
        return (
            f"TFLiteGestureResult("
            f"label={self.label!r}, conf={self.confidence:.2f}, "
            f"activated={self.activated})"
        )


# ──────────────────────────────────────────────────────────────
#  MOTOR PRINCIPAL
# ──────────────────────────────────────────────────────────────
class TFLiteGestureEngine:
    """
    Carga un modelo .tflite cuantizado, procesa frames de OpenCV en un
    hilo demonio y expone el resultado más reciente a través de
    `get_result()`.

    Uso mínimo
    ----------
    >>> engine = TFLiteGestureEngine(
    ...     model_path="model_quant.tflite",
    ...     labels_path="labels.txt",
    ...     on_activate=lambda label: print(f"Gesto detectado: {label}")
    ... )
    >>> engine.start()
    >>> # … dentro del loop:
    >>> result = engine.process_frame(frame)
    >>> engine.stop()
    """

    def __init__(
        self,
        model_path:  str | Path,
        labels_path: str | Path,
        on_activate: Optional[Callable[[str], None]] = None,
    ):
        """
        Parámetros
        ----------
        model_path  : ruta al archivo model_quant.tflite
        labels_path : ruta al archivo labels.txt  (formato "idx etiqueta")
        on_activate : callback(label: str) que se llama en el hilo de inferencia
                      cuando un gesto estable se activa.  Puede ser None.
        """
        self._model_path  = Path(model_path)
        self._labels_path = Path(labels_path)
        self._on_activate = on_activate

        # Carga diferida del intérprete (se hace en start())
        self._interpreter = None
        self._input_details  = None
        self._output_details = None
        self._labels: list[str] = []

        # Cola de frames para el hilo de inferencia (tamaño 1 = último frame)
        self._frame_queue: collections.deque[np.ndarray] = collections.deque(maxlen=1)
        self._lock = threading.Lock()

        # Buffer deslizante de predicciones para el voto de estabilidad
        self._buffer: collections.deque[str] = collections.deque(maxlen=BUFFER_SIZE)

        # Control del hilo
        self._running   = False
        self._thread: Optional[threading.Thread] = None

        # Resultado público (actualizado desde el hilo de inferencia)
        self._result = TFLiteGestureResult()

        # Anti-spam: timestamp de la última activación por clase
        self._last_activation: dict[str, float] = {}

    # ──────────────────────────────────────────────────────────
    #  CICLO DE VIDA
    # ──────────────────────────────────────────────────────────
    def start(self) -> "TFLiteGestureEngine":
        """Carga el modelo e inicia el hilo de inferencia. Retorna self (fluent)."""
        self._load_model()
        self._running = True
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="TFLiteGestureEngine",
            daemon=True,          # muere con el proceso principal
        )
        self._thread.start()
        logger.info("[TFLiteGesture] Motor iniciado. Modelo: %s", self._model_path.name)
        print(f"[TFLiteGesture] ✅ Motor iniciado. Clases: {self._labels}")
        return self

    def stop(self):
        """Detiene el hilo de inferencia limpiamente."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[TFLiteGesture] Motor detenido.")

    # ──────────────────────────────────────────────────────────
    #  API PÚBLICA (hilo principal)
    # ──────────────────────────────────────────────────────────
    def process_frame(self, frame_bgr: np.ndarray) -> TFLiteGestureResult:
        """
        Encola el frame para inferencia asíncrona y devuelve el último
        resultado disponible.  NO bloquea el hilo principal.

        Parámetros
        ----------
        frame_bgr : frame de OpenCV en formato BGR (uint8)

        Retorna
        -------
        TFLiteGestureResult con el resultado más reciente.
        """
        if frame_bgr is not None:
            self._frame_queue.append(frame_bgr)   # deque(maxlen=1) descarta el anterior

        with self._lock:
            return self._result                   # snapshot atómico

    def get_result(self) -> TFLiteGestureResult:
        """Alias de process_frame sin encolar frame (sólo lectura del estado)."""
        with self._lock:
            return self._result

    # ──────────────────────────────────────────────────────────
    #  CARGA DEL MODELO  (llamada desde start())
    # ──────────────────────────────────────────────────────────
    def _load_model(self):
        """Carga el intérprete TFLite usando el framework compatible."""
        if not self._model_path.exists():
            raise FileNotFoundError(f"[TFLiteGesture] Modelo no encontrado: {self._model_path}")
        if not self._labels_path.exists():
            raise FileNotFoundError(f"[TFLiteGesture] Etiquetas no encontradas: {self._labels_path}")

        # Importación segura para evitar dependencias pesadas
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                # Fallback final utilizando las herramientas internas ya instaladas
                from mediapipe.python._framework_bindings import interpreter as tflite_interpreter
                Interpreter = tflite_interpreter.Interpreter

        # Inicialización estándar
        self._interpreter = Interpreter(str(self._model_path))
        self._interpreter.allocate_tensors()
        self._input_details  = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Parsear labels.txt de forma limpia
        raw_labels = self._labels_path.read_text(encoding="utf-8").strip().splitlines()
        self._labels = [line.strip().split(None, 1)[-1] for line in raw_labels]

        logger.info("[TFLiteGesture] Modelo cargado correctamente.")

        # Parsear labels.txt  →  ["tomar", "amor", "sin_gestos"]
        raw_labels = self._labels_path.read_text(encoding="utf-8").strip().splitlines()
        self._labels = []
        for line in raw_labels:
            parts = line.strip().split(None, 1)   # "0 tomar" → ["0", "tomar"]
            self._labels.append(parts[-1] if parts else line.strip())

        logger.debug(
            "[TFLiteGesture] Modelo cargado. Input shape: %s  Dtype: %s",
            self._input_details[0]["shape"],
            self._input_details[0]["dtype"],
        )

    # ──────────────────────────────────────────────────────────
    #  HILO DE INFERENCIA
    # ──────────────────────────────────────────────────────────
    def _inference_loop(self):
        """Corre en el hilo daemon.  Procesa frames de la cola."""
        while self._running:
            if not self._frame_queue:
                time.sleep(0.005)   # ~200 Hz de polling sin consumir CPU
                continue

            frame = self._frame_queue.pop()        # extrae el frame más reciente

            try:
                label, confidence = self._run_inference(frame)
            except Exception as exc:               # robustez: nunca matar el hilo
                logger.warning("[TFLiteGesture] Error en inferencia: %s", exc)
                continue

            # ── Buffer de estabilidad  ─────────────────────
            self._buffer.append(label)

            # Clase ganadora por mayoría en los últimos BUFFER_SIZE frames
            if len(self._buffer) == BUFFER_SIZE:
                winner = max(set(self._buffer), key=self._buffer.count)
                winner_votes = self._buffer.count(winner)
            else:
                winner = label
                winner_votes = 1

            activated = False

            if winner == "sin_gestos":
                activated = False
                self._buffer.clear()
                continue

            # ── Lógica de activación ───────────────────────
            if (
                winner in ACTIVE_CLASSES
                and confidence >= CONFIDENCE_THRESHOLD
                and winner_votes >= STABILITY_FRAMES
            ):
                now = time.time()
                last = self._last_activation.get(winner, 0.0)

                if now - last >= COOLDOWN_S:
                    self._last_activation[winner] = now
                    activated = True
                    logger.info(
                        "[TFLiteGesture] 🎯 Activación: %s  conf=%.2f  votos=%d/%d",
                        winner, confidence, winner_votes, BUFFER_SIZE
                    )
                    print(
                        f"[TFLiteGesture] 🎯 Gesto confirmado: '{winner}' "
                        f"(confianza={confidence:.0%}, votos={winner_votes}/{BUFFER_SIZE})"
                    )

                    # Callback → se ejecuta aquí, en el hilo de inferencia
                    if self._on_activate:
                        try:
                            self._on_activate(winner)
                        except Exception as cb_exc:
                            logger.error(
                                "[TFLiteGesture] Error en callback: %s", cb_exc
                            )

            # ── Publicar resultado ─────────────────────────
            new_result = TFLiteGestureResult(
                label=winner,
                confidence=confidence,
                activated=activated,
                timestamp=time.time(),
            )
            with self._lock:
                self._result = new_result

    # ──────────────────────────────────────────────────────────
    #  PREPROCESADO + INFERENCIA
    # ──────────────────────────────────────────────────────────
    def _run_inference(self, frame_bgr: np.ndarray) -> tuple[str, float]:
        """
        Preprocesa el frame y ejecuta una pasada de inferencia.

        Retorna (label: str, confidence: float).
        """
        # 1. Redimensionar al tamaño de entrada del modelo
        resized = cv2.resize(frame_bgr, (INPUT_SIZE, INPUT_SIZE),
                             interpolation=cv2.INTER_LINEAR)

        # 2. BGR → RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # 3. Normalización y tipo según el modelo
        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.uint8:
            # Modelo cuantizado INT8/UINT8: mantener [0, 255]
            input_tensor = rgb.astype(np.uint8)
        else:
            # Modelo float: normalizar a [0.0, 1.0]
            input_tensor = rgb.astype(np.float32) / 255.0

        # 4. Añadir dimensión de batch → (1, H, W, C)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # 5. Inferencia
        self._interpreter.set_tensor(
            self._input_details[0]["index"], input_tensor
        )
        self._interpreter.invoke()

        # 6. Leer salidas
        output = self._interpreter.get_tensor(
            self._output_details[0]["index"]
        )[0]   # shape: (num_classes,)

        # 7. Si el modelo está cuantizado uint8, desescalar con los parámetros de quant
        out_dtype = self._output_details[0]["dtype"]
        if out_dtype == np.uint8:
            scale, zero_point = self._output_details[0]["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale

        # 8. Softmax por si el modelo devuelve logits
        if output.max() > 1.01 or output.min() < -0.01:
            exp_o = np.exp(output - output.max())
            output = exp_o / exp_o.sum()

        # 9. Clase con mayor probabilidad
        idx = int(np.argmax(output))
        label      = self._labels[idx] if idx < len(self._labels) else f"clase_{idx}"
        confidence = float(output[idx])

        return label, confidence
