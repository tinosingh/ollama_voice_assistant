#!/usr/bin/env python3
"""
Roy voice-assistant.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from collections import deque

import gradio as gr
import psutil

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
CFG_FILE = Path("config.json")
REQ_FILE = Path("requirements.txt")
MODEL_DIR = Path("models")
VENV_DIR = Path(".venv")

DEFAULT_OLLAMA = "gemma:2b"
DEFAULT_WHISP = "base"
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a concise, helpful voice assistant. Keep answers brief.",
}

WHISPER_VARIANT = "mlx-whisper" if platform.system() == "Darwin" else "openai-whisper"


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Config:
    ollama_model: str = DEFAULT_OLLAMA
    whisper_model: str = DEFAULT_WHISP

    @staticmethod
    def load() -> "Config":
        if CFG_FILE.exists():
            return Config(**json.loads(CFG_FILE.read_text()))
        return Config()

    def save(self) -> None:
        CFG_FILE.write_text(json.dumps(asdict(self), indent=2))


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #
class SetupManager:
    """Handles first-time installation and model pulls."""

    BASE_DEPS = ["gradio", "ollama", "gTTS", "psutil", WHISPER_VARIANT]

    def __init__(self) -> None:
        self.cfg = Config.load()

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _run(cmd: List[str], desc: str) -> None:
        print(f"[*] {desc} …")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[+] {desc} – done")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            print(f"[!] {desc} failed: {exc}")
            sys.exit(1)

    @staticmethod
    def _hw_report() -> Dict[str, Any]:
        total_ram = psutil.virtual_memory().total // (1024**3)
        cores = psutil.cpu_count(logical=False)
        vram = 0
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            vram = sum(int(line) for line in out.strip().splitlines()) // 1024
        except Exception:
            pass
        print(
            f"[+] RAM: {total_ram} GB | CPU: {cores} cores | GPU-VRAM: {vram or 'N/A'} GB"
        )
        return {"ram": total_ram, "vram": vram}

    # --------------------------------------------------------------------- #
    # model choice
    # --------------------------------------------------------------------- #
    def _ask_models(self) -> None:
        hw = self._hw_report()
        if hw["vram"] >= 10 and hw["ram"] >= 16:
            rec_o, rec_w = "llama3:8b", "small"
        elif hw["vram"] >= 6 or hw["ram"] >= 16:
            rec_o, rec_w = "gemma:7b", "base"
        else:
            rec_o, rec_w = "gemma:2b", "base"

        print(f"\nSuggested → Ollama: {rec_o}  Whisper: {rec_w}")
        self.cfg.ollama_model = input(f"Ollama model [{rec_o}]: ").strip() or rec_o
        while True:
            w = input(f"Whisper model [{rec_w}]: ").strip() or rec_w
            if w in {
                "tiny",
                "base",
                "small",
                "medium",
                "large",
                "large-v2",
                "large-v3",
            }:
                self.cfg.whisper_model = w
                break
            print("! invalid choice")

    # --------------------------------------------------------------------- #
    # venv + pip
    # --------------------------------------------------------------------- #
    def _ensure_venv(self) -> Path:
        py = VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not py.exists():
            print("[*] creating venv …")
            self._run([sys.executable, "-m", "venv", str(VENV_DIR)], "create venv")
        return py

    def _ensure_deps(self, py: Path) -> None:
        if not REQ_FILE.exists():
            REQ_FILE.write_text("\n".join(self.BASE_DEPS) + "\n")
        self._run(
            [str(py), "-m", "pip", "install", "-r", str(REQ_FILE)], "install deps"
        )

    # --------------------------------------------------------------------- #
    # model pulls
    # --------------------------------------------------------------------- #
    def _pull_ollama(self) -> None:
        try:
            listed = subprocess.check_output(["ollama", "list"], text=True)
            if self.cfg.ollama_model in listed:
                print(f"[+] Ollama model {self.cfg.ollama_model} already present")
                return
        except Exception:
            print("[!] Ollama daemon not reachable – be sure it is running")
            sys.exit(1)
        self._run(
            ["ollama", "pull", self.cfg.ollama_model], f"pull {self.cfg.ollama_model}"
        )

    def _pull_whisper(self) -> None:
        # whisper-mlx downloads on first use – nothing to do
        if WHISPER_VARIANT == "openai-whisper":
            MODEL_DIR.mkdir(exist_ok=True)
            # import happens *after* setup – use subprocess so we do not need the import yet
            self._run(
                [
                    str(self._ensure_venv()),
                    "-c",
                    f"import whisper,os,sys; whisper.load_model('{self.cfg.whisper_model}', download_root='{MODEL_DIR}')",
                ],
                f"download Whisper {self.cfg.whisper_model}",
            )

    # --------------------------------------------------------------------- #
    # main entry
    # --------------------------------------------------------------------- #
    def run(self) -> Config:
        if CFG_FILE.exists():
            print("[+] config found – skipping setup")
            return self.cfg

        print("\n==== First-time setup ====")
        self._ask_models()
        py = self._ensure_venv()
        self._ensure_deps(py)
        self._pull_ollama()
        self._pull_whisper()
        self.cfg.save()
        print("\n✅ setup complete\n")
        return self.cfg


# --------------------------------------------------------------------------- #
# Runtime imports (guaranteed to be available after setup)
# --------------------------------------------------------------------------- #
def import_runtime() -> Tuple[Any, Any, Any, bool]:
    # We rely on a successful previous setup run inside the VENV
    # which guarantees the correct package (whisper-mlx or openai-whisper)
    # is installed.
    # Note: pyttsx3 is not mandatory and is not imported here to simplify.
    import ollama
    from gtts import gTTS

    is_mlx = False
    if platform.system() == "Darwin":
        try:
            import mlx_whisper as whisper

            is_mlx = True
        except ImportError:
            # Fallback if MLX was supposed to be installed but failed
            import whisper
    else:
        import whisper
    # Suppress FP16 warning on CPU for openai-whisper
    if not is_mlx:
        warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")
    return ollama, gTTS, whisper, is_mlx


# --------------------------------------------------------------------------- #
# Services
# --------------------------------------------------------------------------- #
class OllamaSvc:
    def __init__(self, model: str) -> None:
        self.model = model

    def stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        messages = (
            [SYSTEM_PROMPT] + messages
            if (not messages or messages[0]["role"] != "system")
            else messages
        )
        try:
            for chunk in ollama.chat(model=self.model, messages=messages, stream=True):
                yield chunk.get("message", {}).get("content", "")
        except Exception as exc:
            yield f"Error: cannot reach Ollama – {exc}"


class TTSSvc:
    def __init__(self, lang: str = "en") -> None:
        self.lang = lang

    def synthesise(self, text: str, idx: int) -> Optional[str]:
        if not text or text.startswith("Error:"):
            return None
        dest = Path(f"temp_resp_{idx}_{int(time.time())}.mp3")
        try:
            gTTS(text=text, lang=self.lang, slow=False).save(dest)
            return str(dest)
        except Exception as exc:
            logging.error("gTTS failed: %s", exc)
            return None

    def synthesise_bytes(self, text: str) -> Optional[bytes]:
        """Return mp3 bytes for the given text (used if you want in-memory audio)."""
        try:
            from io import BytesIO

            fp = BytesIO()
            gTTS(text=text, lang=self.lang, slow=False).write_to_fp(fp)
            return fp.getvalue()
        except Exception as exc:
            logging.debug("synthesise_bytes failed: %s", exc)
            return None


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #
class VoiceAssistantApp:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.llm = OllamaSvc(cfg.ollama_model)
        self.tts = TTSSvc()
        self.transcriber: Optional[Any] = None
        self.counter = 0
        # Single background thread for file deletions to avoid O(n) threads
        self.deletion_queue: deque = deque()
        self.deletion_thread = threading.Thread(target=self._deletion_reaper, daemon=True)
        self.deletion_thread.start()
        threading.Thread(target=self._load_whisper, daemon=True).start()

    # --------------------------------------------------------------------- #
    # whisper lazy load
    # --------------------------------------------------------------------- #
    def _load_whisper(self) -> None:
        global whisper, is_mlx
        try:
            self.transcriber = whisper.load_model(
                self.cfg.whisper_model,
                # Use download_root only for openai-whisper (non-MLX mode)
                download_root=str(MODEL_DIR) if not is_mlx else None,
            )
            logging.info("Whisper model loaded")
        except Exception as exc:
            logging.error("Whisper load failed: %s", exc)

    # --------------------------------------------------------------------- #
    # transcription
    # --------------------------------------------------------------------- #
    def transcribe(self, audio: Optional[str]) -> Optional[str]:
        if not audio:
            return None
        if self.transcriber is None:
            return "Error: Whisper not ready yet"
        try:
            # Both MLX and standard Whisper return a dict with a "text" key
            return self.transcriber.transcribe(audio)["text"]
        except Exception as exc:
            return f"Transcription failed: {exc}"

    # --------------------------------------------------------------------- #
    # chat core
    # --------------------------------------------------------------------- #
    def _stream(
        self, prompt: str, hist: List[Dict[str, str]]
    ) -> Iterator[List[Dict[str, str]]]:
        def _escape_md(s: str) -> str:
            # Escape common markdown characters so UI shows literals (avoid * interpreted as emphasis)
            if not s:
                return s
            # backslash first
            s = s.replace("\\", "\\\\")
            # Only escape * and _ for italics/bold
            s = s.replace("*", "\\*").replace("_", "\\_")
            return s

        if not prompt or not prompt.strip():
            if hist and hist[-1]["role"] == "user":
                # For voice input, user message is already in history, add assistant and stream
                hist.append({"role": "assistant", "content": "", "_raw": ""})
            else:
                yield hist
                return

        # store both raw and display content in the history entries
        hist.append({"role": "user", "content": _escape_md(prompt), "_raw": prompt})
        hist.append({"role": "assistant", "content": "", "_raw": ""})

        # Build messages for the model from the raw content, not the escaped display content
        def _messages_for_model():
            return [
                {"role": item["role"], "content": item.get("_raw", item["content"])}
                for item in hist
            ]

        for token in self.llm.stream(_messages_for_model()):
            # append token to raw assistant content, then update escaped display field
            hist[-1]["_raw"] += token
            hist[-1]["content"] = _escape_md(hist[-1]["_raw"])
            # Sanitize history for the Chatbot UI (only include role and content)
            display_hist = [
                {"role": it["role"], "content": it["content"]} for it in hist
            ]
            yield display_hist

    # --------------------------------------------------------------------- #
    # tts + file cleanup
    # --------------------------------------------------------------------- #
    def _speak(self, text: str) -> Optional[str]:
        self.counter += 1
        return self.tts.synthesise(text, self.counter)

    # --------------------------------------------------------------------- #
    # Helpers: sentence-splitting and scheduled deletion
    # --------------------------------------------------------------------- #
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Very small heuristic-based splitter; for better results, use nltk/punkt
        import re

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return sentences or [text]

    @staticmethod
    def _estimate_duration_seconds(mp3_path: Path) -> float:
        # Crude heuristic (bytes / bitrate) since mutagen is not installed
        try:
            size = mp3_path.stat().st_size
            return max(0.5, size / 8192)  # assume average 64kbps bitrate -> 8KB/s
        except Exception:
            return 1.0

    def _schedule_delete(self, path: Path, delay: float) -> None:
        delete_time = time.time() + delay
        self.deletion_queue.append((path, delete_time))

    def _deletion_reaper(self) -> None:
        while True:
            now = time.time()
            while self.deletion_queue and self.deletion_queue[0][1] <= now:
                path, _ = self.deletion_queue.popleft()
                try:
                    path.unlink()
                    logging.debug("Deleted audio file: %s", path)
                except Exception as e:
                    logging.warning("Failed to delete %s: %s", path, e)
            time.sleep(0.1)  # brief sleep to avoid busy loop

    def speak_per_sentence(self, text: str) -> Iterator[Optional[str]]:
        """Yield audio file paths for each sentence. Each file is scheduled for deletion
        shortly after its estimated playback duration to keep RAM/disk usage low."""
        sentences = self._split_sentences(text)
        for s in sentences:
            if not s:
                continue
            idx = self.counter + 1
            audio_path = self.tts.synthesise(s, idx)
            if not audio_path:
                yield None
                continue
            p = Path(audio_path)
            # estimate duration and schedule deletion slightly after
            dur = self._estimate_duration_seconds(p)
            # schedule delete a bit after end to account for playback buffering
            safe_delay = max(2.0, dur + 0.8)
            self._schedule_delete(p, safe_delay)
            # increment the counter now that an audio file is created
            self.counter += 1
            yield str(p)

    def _stream_audio(
        self, history: List[Dict[str, str]], full_text: str
    ) -> Iterator[Tuple[List[Dict[str, str]], Optional[str]]]:
        """Given the conversation history and the full assistant text (raw), yield tuples
        of (history, audio_path) for each sentence audio. Final yield returns (history, None).
        """
        for audio_path in self.speak_per_sentence(full_text):
            yield history, audio_path
            if audio_path:
                # Wait for this audio to finish playing before yielding the next
                dur = self._estimate_duration_seconds(Path(audio_path))
                time.sleep(dur + 0.5)  # Add buffer
        # final signal: no more audio
        yield history, None

    @staticmethod
    def _cleanup() -> None:
        for f in Path(".").glob("temp_resp_*.mp3"):
            f.unlink(missing_ok=True)

    # --------------------------------------------------------------------- #
    # gradio UI
    # --------------------------------------------------------------------- #
    def launch(self) -> None:
        self._cleanup()
        with gr.Blocks(theme=gr.themes.Soft(), title="Ollama Voice Assistant") as demo:
            gr.Markdown(
                f"# 🎤 Ollama Voice Assistant  \n**LLM:** `{self.cfg.ollama_model}`  "
                f"**STT:** `Whisper ({self.cfg.whisper_model}){' – MLX' if is_mlx else ''}`"
            )
            chat = gr.Chatbot(label="Chat", height=400, type="messages")
            audio_out = gr.Audio(label="Reply", autoplay=True, interactive=False)
            # Visualizer for audio waveform
            gr.HTML("""
<div id="waveform-container" style="width: 100%; height: 100px; background: #f0f0f0; border: 1px solid #ccc; margin-top: 10px;">
    <canvas id="waveform-canvas" width="800" height="100"></canvas>
</div>
<script>
function drawWaveform(audioSrc) {
    if (!audioSrc) return;
    const canvas = document.getElementById('waveform-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // Simple placeholder waveform - in a real implementation, you'd decode the audio and draw the waveform
    ctx.fillStyle = '#007bff';
    for (let i = 0; i < canvas.width; i += 10) {
        const height = Math.random() * 50 + 10;
        ctx.fillRect(i, canvas.height / 2 - height / 2, 8, height);
    }
}
// Attach to audio changes
const audioElement = document.querySelector('audio');
if (audioElement) {
    audioElement.addEventListener('loadstart', () => drawWaveform(audioElement.src));
}
</script>
""")
            # Hidden field used by client JS to notify server when a playback completes
            playback_marker = gr.Textbox(value=None, visible=False)

            with gr.Row():
                mic = gr.Audio(sources=["microphone"], type="filepath", label="🎙️")
                txt = gr.Textbox(
                    show_label=False, placeholder="Type or use microphone …"
                )

            gr.ClearButton([chat, txt, mic, audio_out])

            # Server callback: client posts the audio path here when playback finishes
            def playback_ack(audio_path: Optional[str]):
                if not audio_path:
                    return ""
                try:
                    p = Path(audio_path)
                    if p.exists():
                        p.unlink()
                        logging.info("Deleted audio after client ack: %s", audio_path)
                except Exception as e:
                    logging.warning("Failed to delete audio %s: %s", audio_path, e)
                return ""

            playback_marker.change(
                playback_ack, inputs=[playback_marker], outputs=[playback_marker]
            )

            # ----- unified submit chain ----- #
            # This function yields the streamed history and clears the text input
            def submit(
                new_text: str, history: List[Dict[str, str]]
            ) -> Iterator[Tuple[List[Dict[str, str]], str]]:
                for h in self._stream(new_text, history):
                    yield h, new_text  # Keep the last full text for TTS step
                # Final yield: return the full assistant response text (raw) for the TTS step
                final_response_text = (
                    history[-1].get("_raw", "")
                    if history and history[-1]["role"] == "assistant"
                    else ""
                )
                yield history, final_response_text

            # This function takes the transcribed audio and returns the text to the textbox
            def attach_audio(audio_path, history):
                # Transcribe audio and put the text into the textbox
                transcribed = self.transcribe(audio_path) or ""
                # Update chat with the transcribed user message
                if transcribed:
                    escaped = transcribed.replace("\\", "\\\\").replace("*", "\\*").replace("_", "\\_")
                    history = history + [{"role": "user", "content": escaped, "_raw": transcribed}]
                return "", gr.update(value=None), history  # Clear txt, clear mic input, update chat

            # text-based submission chain
            stream_text_pipe = txt.submit(
                submit,
                [txt, chat],
                [chat, txt],  # txt now holds the full response text temporarily
            )

            # audio generation and final cleanup for text-based chain
            stream_text_pipe.then(
                self._stream_audio,
                [chat, txt],
                [chat, audio_out],
            ).then(
                lambda: "",
                outputs=txt,  # Clear the textbox for next user input
            )

            # voice-based submission chain
            transcribe_pipe = mic.change(
                attach_audio,
                [mic, chat],
                [txt, mic, chat],  # Transcribed text goes to txt, mic input is cleared, chat updated
            )

            # Chain from transcription result to LLM stream and TTS/cleanup
            transcribe_pipe.then(submit, [txt, chat], [chat, txt]).then(
                self._stream_audio, [chat, txt], [chat, audio_out]
            ).then(
                lambda: "",
                outputs=txt,  # Clear the textbox for next user input
            )

            # Add client-side script that listens for audio end events and sets playback_marker
            js_reporter = """
<script>
// Find the Gradio audio element and attach ended listeners.
function attachAudioEndedReporter() {
    const observer = new MutationObserver(() => {
        const audios = document.querySelectorAll('audio');
        audios.forEach(a => {
            if (!a.dataset._ended_attached) {
                a.addEventListener('ended', (ev) => {
                    try {
                        // find the hidden input generated by Gradio (playback_marker)
                        const inputs = document.querySelectorAll('input[type="text"][style*="display: none"]');
                        if (inputs.length > 0) {
                            // set the value to the current src to notify the server
                            inputs[0].value = a.currentSrc || a.src;
                            inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                        }
                    } catch (e) { console.warn(e); }
                });
                a.dataset._ended_attached = '1';
            }
        });
    });
    observer.observe(document.body, { childList: true, subtree: true });
}
window.addEventListener('load', attachAudioEndedReporter);
</script>
"""
            gr.HTML(js_reporter)

        demo.launch(inbrowser=True, debug=True)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    # Setup logic is run first
    cfg = SetupManager().run()
    # Runtime imports happen after setup ensures dependencies are available
    ollama, gTTS, whisper, is_mlx = import_runtime()
    # Start the app
    VoiceAssistantApp(cfg).launch()
