# 🎤 Karaoke Spotify — Control por Gestos v2.0

Control tu Spotify con las manos mientras ves las letras sincronizadas en pantalla.

---

## 📁 Estructura del proyecto

```
karaoke_spotify/
├── config.py               ← ⚙️  CREDENCIALES y parámetros (editar aquí)
├── main.py                 ← 🚀  Punto de entrada
├── audio_manager.py        ← 🎵  Gestión de Spotify
├── gesture_engine.py       ← 🖐  Motor MediaPipe + suavizado EMA
├── lyrics_engine.py        ← 📜  Carga, caché y sync de .lrc
├── transcription_service.py← 🤖  LRCLIB + Gemini para generar letras
├── visualizer.py           ← 🖥️  OpenCV/PIL – pantalla karaoke
├── requirements.txt
├── hand_landmarker.task    ← ⬇️  descargar (ver abajo)
└── lyrics_cache/           ← creado automáticamente
```

---

## ⚙️ Instalación

### 1. Dependencias
```bash
pip install -r requirements.txt
```

### 2. Modelo MediaPipe
Descarga `hand_landmarker.task` y colócalo en la raíz del proyecto:
```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. Credenciales en `config.py`

| Variable | Dónde obtenerla |
|---|---|
| `SPOTIPY_CLIENT_ID` / `SECRET` | [developer.spotify.com](https://developer.spotify.com/dashboard) → crear app, Redirect URI: `http://127.0.0.1:8888/callback` |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com/app/apikey) (gratis) |

> **Gemini es opcional.** Si no lo configuras, el sistema usará LRCLIB (gratuito, sin límites). Gemini solo se invoca si LRCLIB no encuentra la canción.

---

## 🖐 Gestos

| Dedos | Acción |
|-------|--------|
| ☝️ 1  | ⏭ Siguiente pista |
| ✌️ 2  | 🪩 Toggle Disco Light |
| 🤟 3  | ⏮ Pista anterior |
| 🖐 4/5 | ⏯ Play / Pause |
| ✊ 0 + rotar muñeca derecha | 🔊 Subir volumen |
| ✊ 0 + rotar muñeca izquierda | 🔉 Bajar volumen |

**Cooldown:** 2.5 segundos entre comandos para evitar activaciones accidentales.

---

## ⌨️ Teclado (ventana karaoke)

| Tecla | Acción |
|-------|--------|
| `→` / `D` | Avanzar línea manualmente |
| `←` / `A` | Retroceder línea manualmente |
| `Enter` | Reset de ajuste manual |
| `Q` / `Esc` | Salir |

---

## ✨ Características principales

- **Bola de Karaoke**: resaltado palabra a palabra con una bolita animada
- **Disco Light**: fondo que cambia de color al ritmo de un timer configurable (`DISCO_INTERVAL_S` en config.py)
- **Suavizado EMA agresivo**: los landmarks de la mano se suavizan para evitar gestos fantasma
- **Confirmación por historial**: el gesto debe mantenerse ~8 frames antes de ejecutarse
- **Caché persistente**: los `.lrc` se guardan en `lyrics_cache/` y no consumen tokens de Gemini en recargas futuras
- **Sin Genius**: eliminado completamente; se usa LRCLIB (gratuito) + Gemini como respaldo

---

## 🚀 Ejecución
```bash
python main.py
```

La primera vez Spotify abrirá el navegador para autorizar la app. Acepta y se cerrará solo.

---

## 🔧 Ajuste fino (config.py)

```python
SMOOTH_ALPHA = 0.25      # menor = más suave (0.1 = muy suave, 0.5 = más reactivo)
GESTURE_COOLDOWN_S = 2.5 # segundos entre comandos
DISCO_INTERVAL_S = 0.35  # velocidad del cambio de color disco
FINGER_RATIO = 1.55      # aumentar si detecta dedos fantasma
```
