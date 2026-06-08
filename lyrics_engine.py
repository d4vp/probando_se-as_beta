"""
lyrics_engine.py - Motor de letras sincronizadas (.lrc)
Responsabilidades:
  • Parsear archivos .lrc con timestamps
  • Guardar/cargar caché en disco (lyrics_cache/)
  • Calcular línea activa en función del progreso de reproducción
  • Exponer índice de "bola de karaoke" (bouncing ball)
"""

import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Tuple

import config


# ──────────────────────────────────────────────────────────────
#  ESTRUCTURAS DE DATOS
# ──────────────────────────────────────────────────────────────
@dataclass
class LyricsState:
    lineas: List[str] = field(default_factory=list)
    tiempos_ms: List[int] = field(default_factory=list)
    es_sincronizado: bool = False
    cargando: bool = False

    # Bouncing ball: índice de la palabra activa dentro de la línea activa
    indice_linea: int = 0
    indice_palabra: int = 0      # para efecto bola
    progreso_palabra: float = 0.0  # 0.0–1.0 interpolación entre palabras


# ──────────────────────────────────────────────────────────────
#  MOTOR
# ──────────────────────────────────────────────────────────────
class LyricsEngine:
    """
    Gestiona la carga, caché y sincronización de letras.
    Thread-safe: el estado interno se actualiza desde hilos de carga
    pero `calcular_indice` se llama desde el hilo gráfico.
    """

    _LRC_TAG = re.compile(r'\[(\d+):(\d+)[\.:](\d+)\](.*)')
    _SAFE_CHARS = re.compile(r'[\\/*?:"<>|]')

    PAD = 3  # líneas de relleno al inicio y al final

    def __init__(self):
        self.state = LyricsState(cargando=False)
        os.makedirs(config.LYRICS_CACHE_DIR, exist_ok=True)

    # ──────────────────────────────────────────
    #  UTILIDADES
    # ──────────────────────────────────────────
    def _nombre_seguro(self, titulo: str, artista: str) -> str:
        base = f"{titulo} - {artista}"
        return self._SAFE_CHARS.sub("", base).strip()

    def ruta_lrc(self, titulo: str, artista: str) -> str:
        return os.path.join(config.LYRICS_CACHE_DIR,
                            f"{self._nombre_seguro(titulo, artista)}.lrc")

    def ruta_txt(self, titulo: str, artista: str) -> str:
        return os.path.join(config.LYRICS_CACHE_DIR,
                            f"{self._nombre_seguro(titulo, artista)}.txt")

    # ──────────────────────────────────────────
    #  PARSING LRC
    # ──────────────────────────────────────────
    def parsear_lrc(self, contenido: str) -> Tuple[List[str], List[int]]:
        """Devuelve (lineas, tiempos_ms) parseados desde un string .lrc"""
        lineas, tiempos = [], []
        for raw in contenido.splitlines():
            m = self._LRC_TAG.match(raw)
            if m:
                mins = int(m.group(1))
                secs = int(m.group(2))
                frac = int(m.group(3))
                texto = m.group(4).strip()
                # Fracciones pueden ser centésimas (xx) o milésimas (xxx)
                if len(m.group(3)) == 2:
                    ms = (mins * 60_000) + (secs * 1000) + (frac * 10)
                else:
                    ms = (mins * 60_000) + (secs * 1000) + frac
                tiempos.append(ms)
                lineas.append(texto)
        return lineas, tiempos

    def _aplicar_letras(self, lineas: List[str],
                        tiempos: List[int], sincronizado: bool):
        PAD = self.PAD
        pad_l = [""] * PAD
        pad_t_ini = [0] * PAD
        pad_t_fin = (
            [tiempos[-1] + 5_000, tiempos[-1] + 10_000, tiempos[-1] + 15_000]
            if tiempos else [0] * PAD
        )
        self.state = LyricsState(
            lineas=pad_l + lineas + pad_l,
            tiempos_ms=pad_t_ini + tiempos + pad_t_fin if tiempos else [],
            es_sincronizado=sincronizado,
            cargando=False,
        )

    # ──────────────────────────────────────────
    #  CACHÉ
    # ──────────────────────────────────────────
    def en_cache(self, titulo: str, artista: str) -> bool:
        return (os.path.exists(self.ruta_lrc(titulo, artista)) or
                os.path.exists(self.ruta_txt(titulo, artista)))

    def cargar_desde_cache(self, titulo: str, artista: str) -> bool:
        """Intenta cargar desde disco. Devuelve True si tuvo éxito."""
        ruta = self.ruta_lrc(titulo, artista)
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                lineas, tiempos = self.parsear_lrc(f.read())
            if lineas:
                self._aplicar_letras(lineas, tiempos, True)
                print(f"[Letras] 💾 Caché LRC: {titulo}")
                return True

        ruta = self.ruta_txt(titulo, artista)
        if os.path.exists(ruta):
            with open(ruta, "r", encoding="utf-8") as f:
                lineas = [l for l in f.read().splitlines() if l.strip()]
            self._aplicar_letras(lineas, [], False)
            print(f"[Letras] 💾 Caché TXT: {titulo}")
            return True

        return False

    def guardar_lrc(self, titulo: str, artista: str, contenido: str):
        with open(self.ruta_lrc(titulo, artista), "w", encoding="utf-8") as f:
            f.write(contenido)

    def guardar_txt(self, titulo: str, artista: str, lineas: List[str]):
        with open(self.ruta_txt(titulo, artista), "w", encoding="utf-8") as f:
            f.write("\n".join(lineas))

    def aplicar_lrc_crudo(self, titulo: str, artista: str, contenido: str):
        """Parsea, guarda y aplica un LRC obtenido de la red o de Gemini."""
        lineas, tiempos = self.parsear_lrc(contenido)
        if lineas and tiempos:
            self.guardar_lrc(titulo, artista, contenido)
            self._aplicar_letras(lineas, tiempos, True)
            print(f"[Letras] ✅ LRC sincronizado aplicado: {titulo}")
        else:
            # Gemini devolvió texto plano sin timestamps
            lineas_planas = [l for l in contenido.splitlines() if l.strip()
                             and not l.startswith("[")]
            if lineas_planas:
                self.guardar_txt(titulo, artista, lineas_planas)
                self._aplicar_letras(lineas_planas, [], False)
                print(f"[Letras] ℹ️ Texto plano aplicado: {titulo}")

    def set_cargando(self):
        self.state = LyricsState(
            lineas=["", "📥 Cargando letras…", "Buscando sincronización…", ""],
            cargando=True,
        )

    def set_error(self, msg: str):
        self.state = LyricsState(
            lineas=["", f"⚠️ {msg}", ""],
            cargando=False,
        )

    # ──────────────────────────────────────────
    #  CÁLCULO DE ÍNDICE ACTIVO
    # ──────────────────────────────────────────
    def calcular_indice(self, progreso_ms: int,
                         duracion_ms: int,
                         ajuste: int = 0) -> int:
        """
        Devuelve el índice de la línea activa.
        Actualiza también state.indice_linea para la bola.
        """
        st = self.state
        total = len(st.lineas)
        if total < 6:
            return self.PAD

        if st.es_sincronizado and st.tiempos_ms:
            idx = self.PAD
            for i, t in enumerate(st.tiempos_ms):
                if progreso_ms >= t:
                    idx = i
            idx = max(self.PAD, min(idx + ajuste, total - self.PAD - 1))
        else:
            if duracion_ms <= 0:
                duracion_ms = 1
            pct = min(progreso_ms / duracion_ms, 1.0)
            idx = int(pct * (total - self.PAD * 2)) + self.PAD + ajuste
            idx = max(self.PAD, min(idx, total - self.PAD - 1))

        st.indice_linea = idx

        # ── Bouncing ball: calcular progreso dentro de la línea ──
        if st.es_sincronizado and st.tiempos_ms and idx + 1 < len(st.tiempos_ms):
            t_ini = st.tiempos_ms[idx]
            t_fin = st.tiempos_ms[min(idx + 1, len(st.tiempos_ms) - 1)]
            duracion_linea = max(t_fin - t_ini, 1)
            st.progreso_palabra = min(
                (progreso_ms - t_ini) / duracion_linea, 1.0
            )
        else:
            st.progreso_palabra = 0.0

        return idx
