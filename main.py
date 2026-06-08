"""
main.py - Punto de entrada del sistema Karaoke Spotify
Orquesta la comunicación entre todos los módulos.

Gestos disponibles:
  1 dedo   → Siguiente pista
  2 dedos  → Toggle Disco Light 🪩
  3 dedos  → Pista anterior
  4/5 ded. → Play / Pause
  Puño + rotación muñeca → Control de volumen (DJ knob)

Teclado:
  →/D  : scroll de letra adelante
  ←/A  : scroll de letra atrás
  Enter: reset scroll
  Q/Esc: salir
"""

import sys
import time
import cv2

import config
from audio_manager import AudioManager
from gesture_engine import GestureEngine
from tflite_gesture_engine import TFLiteGestureEngine
from lyrics_engine import LyricsEngine
from transcription_service import TranscriptionService
from visualizer import Visualizer


def main():
    print("=" * 55)
    print("   🎤 KARAOKE SPOTIFY  –  Control por Gestos v2.0")
    print("=" * 55)

    # ── Inicializar módulos ────────────────────────────────
    audio    = AudioManager()
    gestos   = GestureEngine()

    def _on_tflite_activate(label: str):
        if label == "tomar":
            audio.iniciar_contexto(config.URI_PLAYLIST_ESPECIAL)
        elif label == "amor":
            audio._sp.start_playback(uris=["spotify:track:3HHqVJHqwgkxWhOQ4MhLB6"])

    tflite_motor = TFLiteGestureEngine(
        model_path="model.tflite",
        labels_path="labels.txt",
        on_activate=_on_tflite_activate,
    ).start()

    letras   = LyricsEngine()
    trans    = TranscriptionService(letras)
    visual   = Visualizer()

    if not audio.conectar():
        print("[Main] No se pudo conectar a Spotify. Revisa config.py")
        sys.exit(1)

    # Callback: cuando cambia la pista, pedir letras al servicio
    def on_track_change(titulo, artista, duracion_s):
        trans.cargar_letras_async(titulo, artista, duracion_s)

    audio.on_track_change(on_track_change)

    # Cargar estado inicial
    audio.actualizar_estado()

    # ── Cámara ────────────────────────────────────────────
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[Main] No se pudo abrir la cámara.")
        gestos.cerrar()
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.CAMERA_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_H)

    # ── Estado del bucle ──────────────────────────────────
    ultimo_poll_spotify = 0.0
    ultimo_comando_ts   = 0.0
    ajuste_lineas       = 0
    modo_especial       = False
    comando_activo      = ""
    comando_ts          = 0.0

    print("\n[Main] Sistema listo. ¡Mueve tus manos! 🤙\n")

    # ── BUCLE PRINCIPAL ───────────────────────────────────
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        now   = time.time()

        # ── Poll Spotify ───────────────────────────────────
        if now - ultimo_poll_spotify >= config.SPOTIFY_POLL_S:
            audio.actualizar_estado()
            ultimo_poll_spotify = now

        # ── Detección de gestos ────────────────────────────
        gesto = gestos.procesar_frame(frame)
        tflite_motor.process_frame(frame)

        # ── Despacho de comandos ───────────────────────────
        if (gesto.mano_detectada
                and gesto.confianza >= 0.62
                and now - ultimo_comando_ts >= config.GESTURE_COOLDOWN_S):

            comando_nuevo = None

            if gesto.dedos == 1:
                audio.siguiente()
                ajuste_lineas = 0
                comando_nuevo = "⏭ SIGUIENTE"

            elif gesto.dedos == 2:
                modo_especial = True
                audio.iniciar_contexto(config.URI_PLAYLIST_ESPECIAL)
                ajuste_lineas = 0
                comando_nuevo = "💔 PLAYLIST DESPECHO"

            elif gesto.dedos == 3:
                visual.toggle_disco()
                comando_nuevo = "🪩 DISCO LIGHT"

            elif gesto.dedos in (4, 5):
                audio.play_pause()
                comando_nuevo = "⏯ PLAY/PAUSE"

            elif gesto.dedos == 0:
                # DJ knob – rotación de muñeca
                rot = gesto.rotacion_muneca
                if rot > config.WRIST_ROT_DEADZONE:
                    audio.subir_volumen()
                    comando_nuevo = f"🔊 VOL+ {audio.volumen_actual}%"
                elif rot < -config.WRIST_ROT_DEADZONE:
                    audio.bajar_volumen()
                    comando_nuevo = f"🔉 VOL- {audio.volumen_actual}%"

            if comando_nuevo:
                ultimo_comando_ts = now
                comando_activo    = comando_nuevo
                comando_ts        = now

        # Limpiar badge tras 1.8 s
        if now - comando_ts > 1.8:
            comando_activo = ""

        # ── Cálculo de índice de letra ─────────────────────
        progreso_ms = audio.progreso_fresco_ms()
        indice = letras.calcular_indice(
            progreso_ms, audio.duracion_ms, ajuste_lineas
        )

        # ── Render Karaoke ─────────────────────────────────
        frame_karaoke = visual.render_karaoke(
            titulo=audio.titulo,
            artista=audio.artista,
            lineas=letras.state.lineas,
            indice_activo=indice,
            progreso_palabra=letras.state.progreso_palabra,
            es_sincronizado=letras.state.es_sincronizado,
            modo_especial=modo_especial,
            cargando=letras.state.cargando,
            volumen=audio.volumen_actual,
            comando_activo=comando_activo,
        )
        visual.mostrar_karaoke(frame_karaoke)

        # ── Render Cámara ──────────────────────────────────
        frame_cam = visual.render_camara(
            frame=frame,
            dedos=gesto.dedos,
            gesto_nombre=gesto.gesto_nombre if gesto.confianza > 0.5 else "",
            volumen=audio.volumen_actual,
            comando_display=comando_activo,
        )
        visual.mostrar_camara(frame_cam)

        # ── Teclado ────────────────────────────────────────
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord('d'), 83):   # → / D
            ajuste_lineas += 1
        elif tecla in (ord('a'), 81): # ← / A
            ajuste_lineas -= 1
        elif tecla == 13:             # Enter → reset
            ajuste_lineas = 0
            modo_especial = False
        elif tecla in (ord('q'), 27): # Q / Esc → salir
            break

    # ── Limpieza ───────────────────────────────────────────
    cap.release()
    gestos.cerrar()
    tflite_motor.stop()
    visual.cerrar()
    print("\n[Main] Sistema cerrado correctamente. 👋")


if __name__ == "__main__":
    main()