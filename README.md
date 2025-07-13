# Speech Tuner

A professional, modern web app for AI-powered voice and text emotion/sentiment analysis.

## Features
- Upload or record voice, or enter text for analysis
- Detects emotion and sentiment from speech and text
- **Numerical results only:**
  - Emotion confidence (percentage)
  - Sentiment score (percentage)
- Professional, minimal, and responsive UI
- Robust backend with responsible handling of sensitive/crisis input
- No charts or graphs—just clear, accurate numbers
- Built with ❤️ by Tooba Jatoi

## How It Works
- **Voice or text input** is analyzed using state-of-the-art models
- **Emotion** (e.g., happy, sad, angry, frustrated, etc.) and **sentiment** (positive/neutral/negative) are detected
- **Confidence** and **sentiment score** are shown as large, clear percentages
- If a crisis phrase is detected, a warning and helpline are shown

## Running the App
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Open your browser to [http://localhost:5000](http://localhost:5000)

## Output Example
- **Detected Emotion:** frustrated
- **Emotion Confidence:** 72.5%
- **Sentiment:** 😞 Negative
- **Sentiment Score:** -85.0%
- **Speech Tuning:** empathetic

## Project Structure
- `app.py` — Main Flask backend
- `templates/index.html` — Modern web UI
- `emotion_detector/`, `sentiment_checker/`, `speech_tuner/` — Core analysis modules
- `requirements.txt` — Python dependencies

## Responsible AI
- Explicit crisis/self-harm phrase detection
- Clear warnings and helpline links for sensitive input

## Credits
- UI/UX, backend, and all code by Tooba Jatoi

---

**Note:** All legacy files, charts, and unused features have been removed for clarity and performance. 