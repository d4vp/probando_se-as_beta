"""
audio_manager.py - Gestión completa de Spotify
Responsabilidades: conexión OAuth, estado de reproducción, volumen, comandos.
"""

import time
import threading
import spotipy
from spotipy.oauth2 import SpotifyOAuth

import config


class AudioManager:
    """Wrapper sobre spotipy que expone sólo lo que el sistema necesita."""

    def __init__(self):
        self._sp: spotipy.Spotify | None = None
        self._lock = threading.Lock()

        # Estado cacheado para no golpear la API en cada frame
        self._playback_cache: dict = {}
        self._cache_ts: float = 0.0
        self._cache_ttl: float = config.SPOTIFY_POLL_S

        self.volumen_actual: int = 50
        self.track_id: str = ""
        self.titulo: str = "Sin reproducción"
        self.artista: str = ""
        self.duracion_ms: int = 0
        self.progreso_ms: int = 0
        self.reproduciendo: bool = False

        self._on_track_change_cb = None  # callback(titulo, artista, duracion_s)

    # ──────────────────────────────────────────
    #  CONEXIÓN
    # ──────────────────────────────────────────
    def conectar(self) -> bool:
        """Autentica con Spotify. Devuelve True si tuvo éxito."""
        try:
            auth = SpotifyOAuth(
                client_id=config.SPOTIPY_CLIENT_ID,
                client_secret=config.SPOTIPY_CLIENT_SECRET,
                redirect_uri=config.SPOTIPY_REDIRECT_URI,
                scope=config.SPOTIFY_SCOPE,
            )
            self._sp = spotipy.Spotify(auth_manager=auth)
            usuario = self._sp.current_user()["display_name"]
            print(f"[Spotify] Conectado como: {usuario}")
            return True
        except Exception as e:
            print(f"[Spotify] ERROR de conexión: {e}")
            return False

    # ──────────────────────────────────────────
    #  POLLING DE ESTADO
    # ──────────────────────────────────────────
    def actualizar_estado(self) -> bool:
        """
        Consulta el estado actual de Spotify.
        Llama al callback si detecta un cambio de pista.
        Devuelve True si hay algo reproduciéndose.
        """
        if not self._sp:
            return False

        now = time.time()
        if now - self._cache_ts < self._cache_ttl:
            return self.reproduciendo

        try:
            with self._lock:
                pb = self._sp.current_playback()
                self._playback_cache = pb or {}
                self._cache_ts = now

            if pb and pb.get("item"):
                track = pb["item"]
                nuevo_id = track["id"]

                self.progreso_ms   = pb.get("progress_ms", 0)
                self.reproduciendo = pb.get("is_playing", False)
                self.duracion_ms   = track.get("duration_ms", 1)

                if nuevo_id != self.track_id:
                    self.track_id = nuevo_id
                    self.titulo   = track["name"]
                    self.artista  = track["artists"][0]["name"]
                    print(f"[Spotify] Nueva pista: {self.titulo} – {self.artista}")
                    if self._on_track_change_cb:
                        self._on_track_change_cb(
                            self.titulo, self.artista,
                            int(self.duracion_ms / 1000)
                        )
                return True

        except Exception as e:
            print(f"[Spotify] Error al actualizar estado: {e}")
        return False

    # ──────────────────────────────────────────
    #  PROPIEDADES DE LECTURA RÁPIDA
    # ──────────────────────────────────────────
    @property
    def progreso_s(self) -> float:
        return self.progreso_ms / 1000.0

    def progreso_fresco_ms(self) -> int:
        """Retorna el progreso interpolando desde el último poll para mayor fluidez."""
        if self.reproduciendo:
            delta = time.time() - self._cache_ts
            return int(self.progreso_ms + delta * 1000)
        return self.progreso_ms

    # ──────────────────────────────────────────
    #  COMANDOS
    # ──────────────────────────────────────────
    def _ejecutar(self, fn, *args, **kwargs):
        """Envuelve llamadas a la API con manejo de errores."""
        try:
            with self._lock:
                fn(*args, **kwargs)
            self._cache_ts = 0  # invalida cache para refrescar rápido
        except Exception as e:
            print(f"[Spotify] Error en comando: {e}")

    def siguiente(self):
        self._ejecutar(self._sp.next_track)
        print("[Spotify] → Siguiente pista")

    def anterior(self):
        self._ejecutar(self._sp.previous_track)
        print("[Spotify] ← Pista anterior")

    def play_pause(self):
        try:
            pb = self._sp.current_playback()
            if pb and pb["is_playing"]:
                self._ejecutar(self._sp.pause_playback)
                print("[Spotify] ⏸ Pausado")
            else:
                self._ejecutar(self._sp.start_playback)
                print("[Spotify] ▶ Play")
        except Exception as e:
            print(f"[Spotify] Error play/pause: {e}")

    def iniciar_contexto(self, uri: str):
        self._ejecutar(self._sp.start_playback, context_uri=uri)
        print(f"[Spotify] ▶ Contexto iniciado: {uri}")

    def set_volumen(self, vol: int):
        vol = max(0, min(100, vol))
        if vol != self.volumen_actual:
            self.volumen_actual = vol
            self._ejecutar(self._sp.volume, vol)
            print(f"[Spotify] 🔊 Volumen: {vol}%")

    def subir_volumen(self, paso: int = 10):
        self.set_volumen(self.volumen_actual + paso)

    def bajar_volumen(self, paso: int = 10):
        self.set_volumen(self.volumen_actual - paso)

    # ──────────────────────────────────────────
    #  CALLBACK
    # ──────────────────────────────────────────
    def on_track_change(self, callback):
        """Registra una función callback(titulo, artista, duracion_s)."""
        self._on_track_change_cb = callback
