### README.md

# Extensive Listening Trainer

Extensive Listening Trainer is a Python-based application that converts Wikipedia articles into audio files using the WhisperSpeech model. The app uses a simple GUI to let users input a Wikipedia article title and generate an audio version of the entire article.

## Features
- Converts any English Wikipedia article to audio using the WhisperSpeech model.
- Supports GPU (CUDA) for faster audio generation.
- Simple GUI for easy usage.
- User-Agent and contact information configurable via environment variables.

## Requirements
- Python 3.10+
- CUDA-enabled GPU (for WhisperSpeech)
- Python libraries:
  - `wikipedia-api`
  - `whisperspeech`
  - `torch`
  - `python-dotenv`

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/extensive-listening-trainer.git
cd extensive-listening-trainer
```

### 2. Install dependencies

Use `pip` to install the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory and specify your contact information:

```
CONTACT_EMAIL=your-email@example.com
```

This contact email will be used in the User-Agent sent to Wikipedia API as required by the [Wikipedia User-Agent policy](https://meta.wikimedia.org/wiki/User-Agent_policy).

### 4. Run the application

You can run the application using Python:

```bash
python whisper.py
```

### 5. CUDA Setup

This application requires a CUDA-enabled GPU for WhisperSpeech to work. Ensure that:
1. NVIDIA drivers are installed.
2. CUDA Toolkit is installed.
3. PyTorch with CUDA support is installed.

To install PyTorch with CUDA, run:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Adjust the CUDA version based on your installed CUDA Toolkit.

## Usage

1. Launch the application by running `python whisper.py`.
2. Enter the title of a Wikipedia article (in English).
3. Click the "Convert Article to Audio" button.
4. The application will fetch the full article text and convert it to an audio file using WhisperSpeech.
5. The resulting audio file will be saved as `output.wav` in the current directory.

## Dependencies

The following Python libraries are used in this project:

- **wikipedia-api**: To fetch Wikipedia article data.
- **whisperspeech**: To convert text to audio using the WhisperSpeech model.
- **torch**: Required for WhisperSpeech (with CUDA support).
- **python-dotenv**: To manage environment variables like contact information.
- **tkinter**: For creating the GUI.

To install the dependencies, use:

```bash
pip install wikipedia-api whisperspeech torch python-dotenv ipython
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.