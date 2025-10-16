# Das AI: Desktop Voice Assistant 🤖

Das is a voice-activated desktop assistant designed for Windows, bringing the power of Google's Gemini model to your local machine. It listens for a wake word and processes natural language commands to perform tasks, answer questions, and interact with your desktop environment.

This application provides a seamless, hands-free way to get information, capture your screen, and access your webcam through a conversational interface.

### Platform Compatibility

*   **OS:** This application is developed and tested primarily for **Windows 10 and 11**.
*   While the core Python logic is cross-platform, tool-specific functions like screen and clipboard capture are optimized for the Windows environment.

## Key Features

*   **Voice-Activated Control:** Uses the wake word "Das" to begin listening for commands, ensuring it only activates when you need it.
*   **Screen Analysis via Screenshot:** Can capture your screen to answer context-aware questions like, "What's on my screen?"
*   **Webcam Integration:** Accesses your webcam to provide visual feedback for prompts such as, "Can you see me?"
*   **Conversational AI:** Leverages the Gemini 2.5 Flash model for fluid, intelligent, and context-aware responses.
*   **Text-to-Speech Output:** Communicates back to you with a clear, audible voice for a true assistant experience.
*   **Clipboard Access:** Can read and report the current text content from your system clipboard.

## How It Works

The assistant operates through a streamlined, multi-step process:

1.  **Wake Word Detection:** The script continuously listens in the background for the wake word "Das."
2.  **Speech-to-Text Conversion:** Once activated, your spoken command is captured and transcribed into text using the `SpeechRecognition` library.
3.  **AI Processing with Gemini:** The transcribed text is sent to the Google Gemini API, which interprets your intent.
4.  **Intelligent Tool Selection:** Based on your command, the AI decides whether to use a local tool (like taking a screenshot or capturing a webcam image) to gather more context.
5.  **Text-to-Speech Response:** The final, generated response from Gemini is converted back into audio using `gTTS` and played aloud.

## Getting Started

Follow these instructions to set up and run Das on your machine.

### Prerequisites

*   **Python 3.8+**
*   **FFmpeg:** This is required by the `pydub` library for audio processing.
    *   **Windows:** Download the FFmpeg binaries from the [official site](https://ffmpeg.org/download.html). Unzip the file and add the `bin` folder to your system's PATH environment variable.

### Installation

1.  **Clone the Repository**
    ```sh
    git clone https://github.com/Vivek-Varma11/Das-python-AI-voice-assistant.git
    cd das-ai-assistant
    # Commands to set up virtual environment (these should be run in terminal, not in requirements.txt)
    python -m venv venv  
    pip install -r requirements.txt     
    python.exe -m pip install --upgrade pip

   .\venv\Scripts\activate 
    ```

2.  **Install Dependencies**
    Create a virtual environment (recommended) and install the required packages from the `requirements.txt` file.
    ```sh
    pip install -r requirements.txt
    ```
    *Note: `PyAudio` can sometimes have complex installation steps. If you encounter issues, please consult the official `PyAudio` documentation for platform-specific instructions.*

3.  **Configure API Key**
    You must add your Google Gemini API key to the script.
    *   Generate your free key at [Google AI Studio](https://aistudio.google.com/app/apikey).
    *   Open the `main.py` file and replace the placeholder string with your key:
        ```python
        # Locate this line in the script
        GOOGLE_GEMINI_API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY_HERE"
        ```

### Running the Application

Execute the main script from your terminal:

```sh
python main.py

Once everything is set up, just run the script from your terminal:

```sh
python main.py
