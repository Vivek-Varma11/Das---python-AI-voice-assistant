# Das AI 🤖

Your friendly desktop AI sidekick, built with Python and powered by Google Gemini.

Das is a voice-activated assistant that listens for its name, understands natural language, and can interact with your computer. Ask it to take a screenshot, see what's on your webcam, or just have a chat. It's designed to be a simple, fun, and helpful companion for your desktop.

## Features

*   **Voice Activated:** Listens for the wake word "Das" to start a conversation.
*   **Sees Your Screen:** Can take a screenshot when you ask things like, "What am I looking at?"
*   **Uses Your Webcam:** Accesses your camera to take a photo if you say, "Can you see me?"
*   **Clipboard Access:** Can read the text you've copied to your clipboard.
*   **Natural Conversation:** Powered by Google's Gemini-1.5-Flash for fluid, human-like responses.
*   **Text-to-Speech:** Responds out loud with a clear voice.

## How's the Vibe?

The whole idea is to make interaction feel natural.

1.  **Wake Word:** You say "Das," followed by your question or command.
2.  **AI Brain:** The audio is sent to Google's Speech-to-Text, then your command is processed by the Gemini model.
3.  **Tool Use:** Das intelligently decides if it needs to use a tool (like the screenshot or webcam function) based on what you said.
4.  **Response:** The final answer from Gemini is converted back into speech using gTTS, so Das can talk back to you.

## Getting Started 🚀

Follow these steps to get Das running on your machine.

### Prerequisites

You'll need a few things installed first:

*   **Python 3.8+**
*   **FFmpeg:** This is required for speeding up the audio playback.
    *   **Windows:** Download from [here](https://ffmpeg.org/download.html) and add it to your system's PATH.
    *   **macOS:** `brew install ffmpeg`
    *   **Linux:** `sudo apt update && sudo apt install ffmpeg`

### Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/das-ai-assistant.git
    cd das-ai-assistant
    ```

2.  **Install the required Python packages:**
    ```sh
    pip install -r requirements.txt
    ```
    *(Note: You might need to install `PyAudio` separately if you run into issues. Check its documentation for your OS.)*

3.  **Add Your API Key (Important!)**
    *   Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Open the Python script (`main.py`) and replace the placeholder text with your actual key:
        ```python
        # Find this line in the script
        GOOGLE_GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
        ```

### Running the Assistant

Once everything is set up, just run the script from your terminal:

```sh
python main.py
