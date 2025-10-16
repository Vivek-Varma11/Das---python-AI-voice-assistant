import os
import sys
import cv2
import time
import threading
import tempfile
import logging
import warnings
from datetime import datetime
import contextlib
import difflib
from shutil import which

# --- Suppress ALL noisy logs at the environment BEFORE any import ---
os.environ["ABSL_LOG_TO_STDERR"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_CPP_ENABLE_LOG"] = "0"
os.environ["GRPC_CXX_LOG"] = "ERROR"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore")

def find_ffmpeg():
    ffmpeg = which("ffmpeg") or which("ffmpeg.exe")
    ffprobe = which("ffprobe") or which("ffprobe.exe")
    return ffmpeg, ffprobe

class LogToFile(object):
    def __init__(self, filename): self.file = open(filename, 'a')
    def write(self, data): self.file.write(data)
    def flush(self): self.file.flush()
    def close(self): self.file.close()

ERROR_LOG_PATH = "assistanterror.logs"
def error_logging_context():
    return contextlib.redirect_stderr(LogToFile(ERROR_LOG_PATH))

# --- Clear previous error log on each start ---
if os.path.exists(ERROR_LOG_PATH):
    try: os.remove(ERROR_LOG_PATH)
    except: pass

# --- Redirect ALL error logging in Python as well, so errors never print to stdout ---
class SilentErrorHandler(logging.Handler):
    def emit(self, record):
        with open(ERROR_LOG_PATH, 'a') as f:
            f.write(self.format(record) + "\n")

root_logger = logging.getLogger()
root_logger.setLevel(logging.ERROR)
if not any(isinstance(h, SilentErrorHandler) for h in root_logger.handlers):
    silent_handler = SilentErrorHandler()
    silent_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    silent_handler.setFormatter(formatter)
    root_logger.handlers = [silent_handler]

with error_logging_context():
    try: import tkinter as tk; TKINTER_AVAILABLE = True
    except: TKINTER_AVAILABLE = False
    try: import pygame; pygame.mixer.init(); PYGAME_AVAILABLE = True
    except: PYGAME_AVAILABLE = False
    try: from PIL import Image, ImageGrab; PIL_AVAILABLE = True
    except: PIL_AVAILABLE = False
    try: import speech_recognition as sr; SR_AVAILABLE = True
    except: SR_AVAILABLE = False
    try: import pyaudio; PYAUDIO_AVAILABLE = True
    except: PYAUDIO_AVAILABLE = False
    try: from gtts import gTTS; GTTS_AVAILABLE = True
    except: GTTS_AVAILABLE = False
    try: from pydub import AudioSegment; PYDUB_AVAILABLE = True
    except: PYDUB_AVAILABLE = False
    try:
        import google.generativeai as genai
        from google.generativeai.types import GenerationConfig
    except: pass

def setup_logging():
    logger = logging.getLogger("assistant")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler)
    error_handler = SilentErrorHandler()
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(error_handler)
    return logger

logger = setup_logging()
def log_user(msg): logger.info(f"USER: {msg}")
def log_tool(tool): logger.info(f"TOOL USED: {tool}")
def log_assistant(msg): logger.info(f"ASSISTANT: {msg}")

stop_assistant_event = threading.Event()
GOOGLE_GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
EXIT_PHRASES = ["goodbye", "exit", "quit", "stop listening", "shutdown", "that's all", "bye"]
DAS_VARIANTS = {"das", "daas", "dos", "dhass", "daasu", "dus"}
SELECTED_MICROPHONE_INDEX = None
MANUAL_ENERGY_THRESHOLD = 3000
CHUNK_SIZE = 180

FFMPEG_PATH, FFPROBE_PATH = find_ffmpeg()

def take_screenshot():
    log_tool('take_screenshot')
    if not PIL_AVAILABLE: return "Error: Pillow not installed."
    try:
        path = 'speech_temp.jpg'
        ImageGrab.grab().convert('RGB').save(path, quality=85)
        return path
    except Exception as e:
        logger.error(f"Screenshot error: {e}")
        return f"Error: {e}"

def web_cam_capture():
    log_tool('web_cam_capture')
    try:
        cam = cv2.VideoCapture(0); time.sleep(1); ret, frame = cam.read()
        if ret:
            path = 'speech_temp.jpg'
            cv2.imwrite(path, frame)
            return path
        logger.error("Webcam error: cam not ready")
        return "Error: Webcam capture failed."
    except Exception as e:
        logger.error(f"Webcam error: {e}")
        return f"Error: {e}"
    finally:
        try: cam.release()
        except: pass

def get_clipboard_text():
    log_tool('get_clipboard_text')
    if not TKINTER_AVAILABLE: return "Error: tkinter not installed."
    try:
        root = tk.Tk(); root.withdraw()
        content = root.clipboard_get(); root.destroy()
        return content.strip() if isinstance(content, str) else "Clipboard empty."
    except Exception as e:
        logger.error(f"Clipboard error: {e}")
        return "Clipboard error."

def get_current_time():
    return datetime.now().strftime("%I:%M %p")

# --- Redesigned system prompt ---
DAS_SYSTEM_PROMPT = (
    "You are Das, an AI assistant and a genuine friend. It’s {date}. "
    "Be warm, friendly, and natural in every answer. Speak exactly as a best friend helping me with a question or a daily task—never sound robotic. "
    "If I ask what you can do, say: I can take a screenshot, take a webcam photo, read your clipboard, or tell you the current time. "
    "Infer my intention thoughtfully. If I say something like 'Can you see me?' or 'Look at me', use the webcam and tell me what you honestly observe, like a real friend. "
    "If I ask 'What's on my screen?', 'What am I doing?', or something similar, take a screenshot and naturally say what you notice. "
    "If you need extra info, ask me in friendly language and no. "
    "Never list steps. Only reply with conversational, easy language. "
    "Use your tool-calling only if my intention really needs it."
)

class GeminiClient:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name='gemini-2.5-flash',
            system_instruction=DAS_SYSTEM_PROMPT.format(date=datetime.now().strftime('%A, %B %d, %Y')))
        self.conversation = self.model.start_chat(history=[])
        self.tools = {
            'take_screenshot': take_screenshot,
            'web_cam_capture': web_cam_capture,
            'get_clipboard_text': get_clipboard_text,
            'get_current_time': get_current_time,
        }
    def send_message(self, user_prompt, image_path=None):
        parts = [user_prompt]
        if image_path and PIL_AVAILABLE:
            try: img = Image.open(image_path); parts.append(img)
            except Exception as e: logger.error(f"Image open error: {e}"); return f"Error opening {image_path}."
        try:
            response = self.conversation.send_message(
                parts, tools=list(self.tools.values()), generation_config=GenerationConfig(temperature=0.7))
            for part in response.parts:
                if getattr(part, "function_call", None):
                    fc = part.function_call
                    name = fc.get('name') if isinstance(fc, dict) else getattr(fc, 'name', None)
                    if name and name in self.tools:
                        output = self.tools[name]()
                        resp2 = self.conversation.send_message({
                            "function_response": {
                                "name": name,
                                "response": {"output": output}
                            }
                        })
                        return resp2.text.strip()
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"AI error: {e}"

def speak_text(text, lang='en', speed=1.25):
    if stop_assistant_event.is_set() or not (GTTS_AVAILABLE and PYGAME_AVAILABLE) or not text:
        return
    try:
        sentences, chunk = [], ''
        for word in text.split():
            if len(chunk) + len(word) + 1 > CHUNK_SIZE:
                sentences.append(chunk.strip())
                chunk = ''
            chunk += (' ' if chunk else '') + word
        if chunk: sentences.append(chunk.strip())
        for chunk in sentences:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tf:
                with error_logging_context():
                    gTTS(text=chunk, lang=lang, slow=False).save(tf.name)
                mp3_path = tf.name
            if PYDUB_AVAILABLE and FFMPEG_PATH and FFPROBE_PATH:
                try:
                    with error_logging_context():
                        AudioSegment.converter = FFMPEG_PATH
                        AudioSegment.ffprobe = FFPROBE_PATH
                        song = AudioSegment.from_mp3(mp3_path)
                        fast_song = song.speedup(playback_speed=speed)
                        fast_path = mp3_path.replace(".mp3", "_fast.mp3")
                        fast_song.export(fast_path, format="mp3")
                except Exception as e:
                    logger.error(f"pydub error: {e}")
                    fast_path = mp3_path
            else:
                fast_path = mp3_path
            try:
                pygame.mixer.music.load(fast_path)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy() and not stop_assistant_event.is_set():
                    pygame.time.Clock().tick(20)
                pygame.mixer.music.stop(); pygame.mixer.music.unload()
            except Exception as e:
                logger.error(f"TTS playback error: {e}")
            for f in [mp3_path, fast_path]:
                if os.path.exists(f): os.remove(f)
            if stop_assistant_event.is_set(): break
    except Exception as e:
        logger.error(f"TTS error: {e}")

def intention_from_message(text):
    text = text.lower().strip()
    if any(phrase in text for phrase in [
        "look at me", "can you see me", "photo of me", "see me", "my webcam", "take my photo"
    ]):
        return "webcam"
    if any(phrase in text for phrase in [
        "what am i doing", "screen now", "current screen", "can you see my screen", "what's on my screen",
        "show my screen", "what's visible", "screen look like", "tell me about my screen"
    ]):
        return "screenshot"
    if "clipboard" in text:
        return "clipboard"
    if "time" in text:
        return "time"
    if "screenshot" in text or "screen" in text:
        return "screenshot"
    if "webcam" in text or "camera" in text or "photo" in text:
        return "webcam"
    return None

def should_process_command(command):
    cmd_lower = (command or "").strip().lower()
    if not cmd_lower or len(cmd_lower) < 4 or len(cmd_lower.split()) < 2:
        if cmd_lower in EXIT_PHRASES:
            return "exit"
        return None
    return "cmd"

def process_command(text, client):
    log_user(text)
    if any(x in text.lower() for x in EXIT_PHRASES):
        stop_assistant_event.set()
        speak_text("Alright, shutting down. Talk to you soon.")
        return
    image_path = None
    intention = intention_from_message(text)
    if intention == "webcam":
        image_path = web_cam_capture()
    elif intention == "screenshot":
        image_path = take_screenshot()
    if image_path and image_path.lower().startswith("error"):
        speak_text(image_path)
        return
    response = client.send_message(text, image_path=image_path if intention in ("webcam", "screenshot") else None)
    log_assistant(response)
    speak_text(response)

def start_listening(client):
    if not (SR_AVAILABLE and PYAUDIO_AVAILABLE): return None
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = MANUAL_ENERGY_THRESHOLD
    try: mic = sr.Microphone(device_index=SELECTED_MICROPHONE_INDEX)
    except: mic = sr.Microphone()
    with mic as source: recognizer.adjust_for_ambient_noise(source, duration=1)
    def background_callback(r, audio_data):
        if stop_assistant_event.is_set(): return
        try:
            text = r.recognize_google(audio_data).strip().lower()
            if any(v in text for v in DAS_VARIANTS):  # Wake word
                filtered = " ".join(w for w in text.split() if w not in DAS_VARIANTS).strip()
                action_type = should_process_command(filtered)
                if action_type == "exit":
                    log_user(filtered)
                    stop_assistant_event.set()
                    speak_text("Alright, shutting down. Talk to you soon.")
                elif action_type == "cmd":
                    process_command(filtered, client)
                else:
                    speak_text("Listening.")
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            pass
    return recognizer.listen_in_background(
        mic, background_callback, phrase_time_limit=7
    )

def main():
    if not all([SR_AVAILABLE, PYAUDIO_AVAILABLE, GTTS_AVAILABLE, PYGAME_AVAILABLE, PIL_AVAILABLE]):
        logger.error("Missing libraries.")
        return
    client = GeminiClient(GOOGLE_GEMINI_API_KEY)
    speak_text("Hey! Das is online and ready as your friend. Just say 'Das' when you need me.")
    stop_listening_func = start_listening(client)
    try:
        while not stop_assistant_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_assistant_event.set()
    finally:
        if callable(stop_listening_func): stop_listening_func(wait_for_stop=False)

def cleanup():
    if PYGAME_AVAILABLE and pygame.mixer.get_init(): pygame.mixer.quit()
    for f in ['speech_temp.mp3', 'speech_temp.jpg']:
        if os.path.exists(f): os.remove(f)

if __name__ == "__main__":
    try: main()
    except Exception as e: logger.error(f"FATAL ERROR: {e}")
    finally: cleanup()
