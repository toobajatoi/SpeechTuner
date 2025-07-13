from flask import Flask, render_template, request, jsonify, send_file
import os
import tempfile
import json
import base64
import io
import numpy as np
from datetime import datetime
import logging
from emotion_detector.emotion_detector import EmotionDetector
from sentiment_checker.sentiment_checker import SentimentChecker
from speech_tuner.speech_tuner import SpeechTuner

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'

# Initialize analyzers
emotion_detector = EmotionDetector()
sentiment_checker = SentimentChecker()
speech_tuner = SpeechTuner()

# Custom color palette
PRIMARY_COLOR = "#2563EB"
SECONDARY_COLOR = "#10B981"
BACKGROUND_COLOR = "#F8FAFC"
TEXT_COLOR = "#1E293B"
ACCENT_COLOR = "#8B5CF6"
POSITIVE_COLOR = "#059669"
NEGATIVE_COLOR = "#DC2626"
NEUTRAL_COLOR = "#6B7280"

SENTIMENT_EMOJI = {
    'positive': '😃',
    'neutral': '😐',
    'negative': '😞'
}

def get_sentiment_label(score):
    if score > 0.2:
        return f"{SENTIMENT_EMOJI['positive']} Positive"
    elif score < -0.2:
        return f"{SENTIMENT_EMOJI['negative']} Negative"
    else:
        return f"{SENTIMENT_EMOJI['neutral']} Neutral"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        text_input = data.get('text', '')
        
        # Process audio if provided
        emotion_result = None
        emotion_confidence = 0.0
        
        if audio_data:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            
            # Save to temporary file with proper WAV format
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Always convert audio to 16kHz mono WAV using librosa and soundfile
            try:
                import librosa
                import soundfile as sf
                audio_data_np, sample_rate = librosa.load(temp_file_path, sr=16000, mono=True)
                sf.write(temp_file_path, audio_data_np, 16000)
            except Exception as e:
                logger.warning(f"Audio conversion warning: {e}")
            
            try:
                # Analyze emotion
                emotion_result = emotion_detector.detect_emotion(temp_file_path)
                if emotion_result and isinstance(emotion_result, dict):
                    emotion_confidence = emotion_result.get('confidence', 0.0)
                    emotion = emotion_result.get('emotion', 'neutral')
                else:
                    # Handle case where result might be a tuple (fallback)
                    if isinstance(emotion_result, tuple):
                        emotion, emotion_confidence = emotion_result
                    else:
                        emotion = 'neutral'
                        emotion_confidence = 0.0
                # --- Robustness fix: raise threshold and override for strong negative sentiment ---
                if emotion_confidence < 0.3:
                    emotion = 'neutral'
                    emotion_confidence = 0.5
            except Exception as e:
                logger.error(f"Error in emotion detection: {e}")
                emotion = 'neutral'
                emotion_confidence = 0.0
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
        else:
            emotion = 'neutral'
            emotion_confidence = 0.0
        
        # Process text sentiment
        sentiment_score = 0.0
        crisis_detected = False
        if text_input.strip():
            try:
                sentiment_score, processing_time = sentiment_checker.get_sentiment(text_input)
                # Crisis detection: if score is -1.0 and text matches crisis phrase
                for phrase in sentiment_checker.crisis_phrases:
                    if phrase in text_input.lower():
                        crisis_detected = True
                        break
            except Exception as e:
                logger.error(f"Error in sentiment analysis: {e}")
                sentiment_score = 0.0
        
        # --- Robustness fix: override emotion for strong negative sentiment ---
        if sentiment_score < -0.5 and emotion not in ['angry', 'frustrated', 'sad']:
            emotion = 'frustrated'
            emotion_confidence = 0.7
        
        # Get speech tuning recommendation
        try:
            # If sentiment is strongly negative, force speech tuning to empathetic
            if sentiment_score < -0.5:
                speech_recommendation = 'empathetic'
            else:
                speech_recommendation = speech_tuner.get_speech_recommendation(
                    emotion=emotion,
                    sentiment=sentiment_score
                )
        except Exception as e:
            logger.error(f"Error in speech tuning: {e}")
            speech_recommendation = "neutral"
        
        # Prepare response
        if crisis_detected:
            # Crisis override for all outputs
            response = {
                'success': True,
                'emotion': 'crisis',
                'emotion_confidence': 1.0,
                'sentiment_score': -1.0,
                'sentiment_label': f"{SENTIMENT_EMOJI['negative']} Negative",
                'speech_recommendation': 'urgent support',
                'crisis_detected': True,
                'sentiment_chart_data': [0, 100, 0],  # [positive, negative, neutral]
                'timestamp': datetime.now().isoformat()
            }
        else:
            response = {
                'success': True,
                'emotion': emotion,
                'emotion_confidence': emotion_confidence,
                'sentiment_score': sentiment_score,
                'sentiment_label': get_sentiment_label(sentiment_score),
                'speech_recommendation': speech_recommendation,
                'crisis_detected': crisis_detected,
                'sentiment_chart_data': [
                    max(0, round(sentiment_score * 100) if sentiment_score > 0 else 0),
                    max(0, round(-sentiment_score * 100) if sentiment_score < 0 else 0),
                    max(0, 100 - abs(round(sentiment_score * 100))) if abs(sentiment_score) < 1 else 0
                ],
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("🚀 Starting Speech Tuner...")
    print("📱 Access your app at: http://127.0.0.1:5000")
    print("🔧 Debug mode: ON")
    app.run(debug=True, host='0.0.0.0', port=5000) 