"""
transcription_service.py - Integración con Google Gemini para generar .lrc
Solo se invoca cuando la canción NO está en caché.
"""

import threading
import requests

import config
from lyrics_engine import LyricsEngine


LRCLIB_UA = "KaraokeSpotify-Python/2.0"

GEMINI_PROMPT = """Eres un experto en letras de canciones y formato LRC.
Tu tarea es generar el archivo .lrc con timestamps para la siguiente canción.

CANCIÓN: "{titulo}"
ARTISTA: "{artista}"
DURACIÓN APROXIMADA: {duracion_s} segundos

Genera ÚNICAMENTE el contenido del archivo .lrc con el formato estándar:
[mm:ss.xx] Línea de letra

Reglas:
- Cada línea empieza con su timestamp en formato [mm:ss.xx]
- No incluyas metadatos como [ti:], [ar:], etc.
- Distribuye las líneas uniformemente a lo largo de la duración
- Si conoces la letra real de la canción, úsala
- Si no la conoces con certeza, indica [Letra no disponible]
- Responde SOLO con el contenido LRC, sin explicaciones ni markdown

Ejemplo:
[00:05.00] Primera línea de la canción
[00:10.50] Segunda línea de la canción
"""


class TranscriptionService:
    """
    Orquesta la obtención de letras:
      1. Caché en disco (LyricsEngine)
      2. LRCLIB (letras sincronizadas gratis)
      3. Google Gemini (generación IA con timestamps)
    """

    def __init__(self, lyrics_engine: LyricsEngine):
        self.le = lyrics_engine

    # ──────────────────────────────────────────
    #  API PÚBLICA
    # ──────────────────────────────────────────
    def cargar_letras_async(self, titulo: str, artista: str, duracion_s: int):
        """Lanza la búsqueda en un hilo demonio y retorna inmediatamente."""
        self.le.set_cargando()
        hilo = threading.Thread(
            target=self._pipeline,
            args=(titulo, artista, duracion_s),
            daemon=True,
        )
        hilo.start()

    # ──────────────────────────────────────────
    #  PIPELINE INTERNO
    # ──────────────────────────────────────────
    def _pipeline(self, titulo: str, artista: str, duracion_s: int):
        # 1. Caché en disco
        if self.le.cargar_desde_cache(titulo, artista):
            return

        # 2. LRCLIB
        if self._buscar_lrclib(titulo, artista, duracion_s):
            return

        # 3. Gemini
        if self._buscar_gemini(titulo, artista, duracion_s):
            return

        # 4. Sin resultados
        self.le.set_error("No se encontraron letras para esta canción.")

    # ──────────────────────────────────────────
    #  FUENTE 1: LRCLIB
    # ──────────────────────────────────────────
    def _buscar_lrclib(self, titulo: str, artista: str, duracion_s: int) -> bool:
        """Busca letras sincronizadas en lrclib.net (gratuito, sin API key)."""
        try:
            url = (
                "https://lrclib.net/api/get"
                f"?track_name={requests.utils.quote(titulo)}"
                f"&artist_name={requests.utils.quote(artista)}"
                f"&duration={duracion_s}"
            )
            resp = requests.get(url,
                                headers={"User-Agent": LRCLIB_UA},
                                timeout=6)
            if resp.status_code == 200:
                data = resp.json()
                contenido = data.get("syncedLyrics") or data.get("plainLyrics")
                if contenido:
                    self.le.aplicar_lrc_crudo(titulo, artista, contenido)
                    print(f"[Letras] 🌐 LRCLIB OK: {titulo}")
                    return True
        except Exception as e:
            print(f"[Letras] LRCLIB error: {e}")
        return False

    # ──────────────────────────────────────────
    #  FUENTE 2: GOOGLE GEMINI
    # ──────────────────────────────────────────
    def _buscar_gemini(self, titulo: str, artista: str, duracion_s: int) -> bool:
        """Genera letras con timestamps usando Gemini (consume tokens)."""
        if not config.GEMINI_API_KEY or config.GEMINI_API_KEY.startswith("TU_"):
            print("[Letras] Gemini no configurado, saltando.")
            return False
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{config.GEMINI_MODEL}:generateContent"
                f"?key={config.GEMINI_API_KEY}"
            )
            prompt = GEMINI_PROMPT.format(
                titulo=titulo, artista=artista, duracion_s=duracion_s
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.3,
                    "maxOutputTokens": 2048,
                },
            }
            resp = requests.post(url, json=payload, timeout=20)

            # Cuota agotada u otros errores HTTP → degradar silenciosamente
            if resp.status_code == 429:
                print("[Letras] ⚠️ Gemini: cuota agotada, continuando sin letras.")
                return False
            if resp.status_code == 403:
                print("[Letras] ⚠️ Gemini: API key inválida o sin permisos.")
                return False
            if resp.status_code != 200:
                print(f"[Letras] Gemini HTTP {resp.status_code} – saltando.")
                return False

            data = resp.json()

            # Verificar que la respuesta tenga la estructura esperada
            candidates = data.get("candidates", [])
            if not candidates:
                print("[Letras] Gemini: respuesta vacía.")
                return False

            # Puede venir bloqueado por safety filters
            finish_reason = candidates[0].get("finishReason", "")
            if finish_reason in ("SAFETY", "RECITATION", "BLOCKED"):
                print(f"[Letras] Gemini bloqueado ({finish_reason}).")
                return False

            texto = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            if texto and "[Letra no disponible]" not in texto:
                self.le.aplicar_lrc_crudo(titulo, artista, texto)
                print(f"[Letras] 🤖 Gemini OK: {titulo}")
                return True
            else:
                print(f"[Letras] Gemini: letra no disponible para {titulo}")

        except requests.exceptions.Timeout:
            print("[Letras] Gemini: timeout, continuando sin letras.")
        except Exception as e:
            print(f"[Letras] Gemini error (no crítico): {e}")
        return False