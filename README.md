# YOUR PERSONAL AI SIDEKICK: Ollama Voice Assistant

A cross-platform voice assistant built with Python and Gradio that runs local brains (Ollama and Whisper) on your machine.

This project is optimized to be fast and clean:

- The setup is self-contained (thanks to the included setup script).
- Temporary audio files are cleaned up automatically after playback.

---

## 1. Prerequisites

Before you start, make sure you have:

- Python 3.8 or newer.
- Ollama installed and its server running. (Follow the official Ollama documentation to install and start the Ollama server — make sure it is running before you begin setup.)
- Git (to clone the repository).

---

## 2. Easy setup

We use a Python virtual environment (.venv) and a simple script (`setup.sh`) to install dependencies and perform the initial configuration.

Getting the code:

```bash
git clone https://github.com/tinosingh/ollama_voice_assistant
cd ollama_voice_assistant
```

Make scripts executable (Unix/macOS):

```bash
chmod +x setup.sh run.sh
```

Run the setup script:

```bash
./setup.sh
```

What `setup.sh` does:

- Creates and activates a Python virtual environment in `.venv`.
- Installs Python dependencies from `requirements.txt`.
- Runs the app's first-time setup routine, which includes interactively choosing and pulling models.

Model configuration is interactive:

- The first time the app runs, it will prompt you to choose an Ollama model (for example, `gemma:2b` or `llama3:8b`) and a Whisper model (for example, `base` or `small`).
- Your selections are saved to `config.json`.
- The script can automatically pull the chosen models for you if desired.

Notes for different platforms:

- To manually activate the created virtual environment:
  - On macOS / Linux (bash/zsh): `source .venv/bin/activate`
  - On Windows (PowerShell): `.venv\Scripts\Activate.ps1`
  - On Windows (cmd.exe): `.venv\Scripts\activate.bat`

`setup.sh` attempts to handle activation on Unix-like systems, but if you're on Windows or your shell differs, activate the venv manually as shown above.

---

## 3. Usage

Start the application:

```bash
./run.sh
```

This launches the Gradio interface. By default, the app is available in your browser at:

http://127.0.0.1:7860

Interacting with the assistant:

- Text input: Type your message into the box and press Enter or click Submit.
- Voice input: Click the microphone icon, speak, and stop recording. The app transcribes your speech locally with Whisper and sends the text to the LLM.

If the default port is already in use, Gradio will either fail to start or will choose another free port — check the console output for the actual URL.

---

## 4. Cleanup

The app generates temporary `.mp3` files for audio playback. Cleanup behavior:

- Each audio file is deleted automatically after the browser reports that playback finished.
- If the app crashes or the system shuts down unexpectedly, any leftover temporary files are cleaned up the next time the application starts.

---

## 5. Under the hood

Key files:

- `app.py` — The main application. Handles the Gradio UI, Ollama streaming, speech-to-text (Whisper), text-to-speech, and core logic.
- `setup.sh` — Creates the virtual environment and runs the first-time installation and model configuration.
- `run.sh` — Launches the application after setup (this is the script you use to run the assistant).
- `config.json` — Stores chosen model names and other configuration values.
- `requirements.txt` — Python dependencies required by the app.

---

That's it — a cleaner, clearer README with corrected grammar and tightened instructions. If you want, I can also:

- Add a short troubleshooting section (common errors and their fixes).
- Put explicit example commands for pulling Ollama models (if you'd like me to add exact Ollama CLI commands — tell me which Ollama version you expect to target).
- Open a PR updating the README in your repo.
