** YOUR PERSONAL AI SIDEKICK: Ollama Voice Assistant **

A cross-platform voice assistant built with Python and Gradio, which means it runs local brains (Ollama and Whisper) right on your machine.

It's optimized to be super fast and clean:
- It shows your question in the chat immediately—no waiting around for the answer to start!
- The setup is neat and self-contained (thanks, SetupManager).
- It automatically cleans up its temporary audio files after you finish listening, so no mess!

1. STUFF YOU NEED (Prerequisites)

Before firing up your assistant, you just need a couple of things ready:
- Python 3.8+: Gotta have Python to run the show.
- Ollama: This is the brain of the operation! Download and install it from the official Ollama website. Make absolutely sure the Ollama server is running in the background before you jump into the setup. 

2. EASY SETUP
We use a virtual environment (.venv) and a simple script (setup.sh) to handle all the tricky dependency stuff for you.

The Quick Setup (First Time Only!)

a) Make Scripts Executable:

chmod +x setup.sh run.sh

b) Run the Setup Script:

./setup.sh

This script will take care of creating your Python environment, getting it activated, and starting the app's internal setup routine.

Model Configuration is Interactive):
The first time it runs, the app will propose and ask you to choose your Ollama Model (like gemma:2b or llama3:8b) and your Whisper Model (like base or small).

Pro Tip: Your choices get saved in config.json, and the script automatically pulls those models for you.

3. Let's Get Talking (Usage)

Running the Application
Once the initial setup is done, you can launch your assistant anytime with a quick command:

./run.sh

The app will pop up in your browser, usually at http://127.0.0.1:7860.

Interacting with the Assistant
You have two ways to chat:

Text Input: Just type your question in the box and hit Enter or click submit.

Voice Input: Click the microphone icon, say your piece, and stop recording. The app instantly transcribes your voice and sends it off to the LLM.

**Audio File ManagementÄÄ
- The app creates temporary .mp3 files for its answers. Don't worry about them, they're automatically taken care of:
- The file is deleted instantly once your browser says playback has finished.
- Any leftover files (if your computer had a meltdown or the app crashed) are automatically cleaned up every time the application starts up.

**What's Under the Hood? (File Structure)**
File            Description

app.py            The heart of the app! It handles the Gradio interface, Ollama streaming, text-to-speech, and all the core logic.

setup.sh          The handy script that creates your environment (.venv) and kicks off the entire first-time installation process.

run.sh            The shortcut script you'll use every day to launch the app after setup.

config.json       Stores the names of the LLM and Whisper models you chose.

requirements.txt  The shopping list for all the Python packages the app needs.


That's all folks!
