"""
Speech Tuner - Voice & Text Analysis Tool
A professional voice and text analysis tool that detects emotions from voice recordings 
and analyzes sentiment from text input.

Built with ❤️ by Tooba Jatoi
"""

import gradio as gr
import time
import threading
import queue
import numpy as np
import plotly.graph_objects as go
import sounddevice as sd
import soundfile as sf
from datetime import datetime
import logging
import os
import tempfile
import shutil
from emotion_detector.emotion_detector import EmotionDetector
from sentiment_checker.sentiment_checker import SentimentChecker
from speech_tuner.speech_tuner import SpeechTuner
from pydantic import BaseModel, ConfigDict

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Custom color palette
PRIMARY_COLOR = "#2563EB"  # Modern blue
SECONDARY_COLOR = "#10B981"  # Emerald green
BACKGROUND_COLOR = "#F8FAFC"  # Light gray background
TEXT_COLOR = "#1E293B"  # Slate gray
ACCENT_COLOR = "#8B5CF6"  # Purple
POSITIVE_COLOR = "#059669"  # Green
NEGATIVE_COLOR = "#DC2626"  # Red
NEUTRAL_COLOR = "#6B7280"  # Gray
CARD_BACKGROUND = "#FFFFFF"  # White
BORDER_COLOR = "#E2E8F0"  # Light gray border

SENTIMENT_EMOJI = {
    'positive': '😃',
    'neutral': '😐',
    'negative': '😞'
}

# Custom CSS for professional styling
CUSTOM_CSS = """
.gradio-container {
    font-family: 'Inter', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header {
    text-align: center;
    margin-bottom: 2rem;
}

.header h1 {
    color: #2563EB;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.header p {
    color: #6B7280;
    font-size: 1.1rem;
    max-width: 600px;
    margin: 0 auto;
}

.card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    margin-bottom: 1.5rem;
}

.card-title {
    color: #1E293B;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.button-primary {
    background: #2563EB !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.button-primary:hover {
    background: #1D4ED8 !important;
    transform: translateY(-1px) !important;
}

.button-stop {
    background: #DC2626 !important;
    color: white !important;
    border: none !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
}

.button-stop:hover {
    background: #B91C1C !important;
    transform: translateY(-1px) !important;
}

.textbox {
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 0.75rem !important;
    font-size: 1rem !important;
}

.textbox:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
}

.status-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 0.75rem;
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #6B7280;
}

.results-container {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}

.results-title {
    color: #1E293B;
    font-size: 1.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 1.5rem;
}

.metric-card {
    background: #F8FAFC;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2563EB;
    margin: 0.5rem 0;
}

.metric-label {
    color: #6B7280;
    font-size: 0.9rem;
}

.analysis-section {
    margin-bottom: 2rem;
    padding: 1.5rem;
    background: #FFFFFF;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.section-title {
    color: #1E293B;
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #E2E8F0;
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
    margin-top: 1rem;
}

.metric-card {
    background: #F8FAFC;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s;
}

.metric-card:hover {
    transform: translateY(-2px);
}

.metric-value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #2563EB;
    margin: 0.75rem 0;
}

.metric-label {
    color: #6B7280;
    font-size: 0.95rem;
    font-weight: 500;
}
"""

# Pydantic models for type safety
class AnalysisResult(BaseModel):
    emotion: str
    emotion_conf: float
    sentiment: float
    tone: str

    class Config:
        arbitrary_types_allowed = True

class AudioAnalyzer:
    def __init__(self):
        self.emotion_detector = EmotionDetector()
        self.sentiment_checker = SentimentChecker()
        self.speech_tuner = SpeechTuner()
        self.sample_rate = 44100
        self.recording_buffer = []
        self.is_recording = False
        self._lock = threading.Lock()
        self.temp_dir = tempfile.mkdtemp()
        # Start the emotion detector
        self.emotion_detector.start_listening()
    
    def __del__(self):
        """Cleanup temporary files and directory"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
            if hasattr(self, 'emotion_detector'):
                self.emotion_detector.stop_listening()
        except Exception as e:
            logger.error(f"Error cleaning up temporary directory: {e}")
    
    def get_temp_file(self):
        """Get a temporary file path in our temp directory"""
        return os.path.join(self.temp_dir, f"temp_{int(time.time() * 1000)}.wav")
    
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.recording_buffer = []
        
        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            with self._lock:
                if self.is_recording:
                    self.recording_buffer.append(indata.copy())
                    
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=callback
            )
            self.stream.start()
            logger.info("Recording started")
            return "Recording in progress...", None
        except Exception as e:
            logger.error(f"Error starting recording: {e}")
            self.is_recording = False
            raise
    
    def stop_recording(self):
        """Stop recording and save to temporary file"""
        if not self.is_recording:
            return "No recording in progress", None
            
        self.is_recording = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        if not self.recording_buffer:
            return "No audio data to save", None
            
        try:
            # Concatenate all recorded chunks
            audio_data = np.concatenate(self.recording_buffer, axis=0)
            
            # Save to temporary file
            temp_file = self.get_temp_file()
            sf.write(temp_file, audio_data, self.sample_rate)
            
            logger.info(f"Recording saved as {temp_file}")
            return "Recording completed successfully!", temp_file
        except Exception as e:
            logger.error(f"Error saving recording: {e}")
            return f"Error: {str(e)}", None
    
    def analyze(self, audio_file, text):
        """Analyze audio and text"""
        try:
            if not audio_file:
                logger.error("No audio file provided")
                return "<b>Error:</b> No audio file provided.", None, None, None
                
            if not os.path.exists(audio_file):
                logger.error(f"Audio file not found: {audio_file}")
                return "<b>Error:</b> Audio file not found. Please try recording again.", None, None, None
                
            # Process audio
            try:
                start_time = time.time()
                emotion, emotion_conf = self.emotion_detector.detect_emotion(audio_file)
                emotion_time = time.time() - start_time
                logger.info(f"Emotion detection took: {emotion_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
                return f"<b>Error processing audio:</b> {str(e)}", None, None, None
            
            # Process text
            try:
                start_time = time.time()
                sentiment_score, _ = self.sentiment_checker.get_sentiment(text)
                sentiment_time = time.time() - start_time
                logger.info(f"Sentiment analysis took: {sentiment_time:.2f} seconds")
                sentiment_label = self.get_sentiment_label(sentiment_score)
            except Exception as e:
                logger.error(f"Error processing text: {e}")
                return f"<b>Error processing text:</b> {str(e)}", None, None, None
            
            # Get speech tuning suggestion
            try:
                start_time = time.time()
                speech_recommendation = self.speech_tuner.get_speech_recommendation(emotion, sentiment_score)
                speech_time = time.time() - start_time
                logger.info(f"Speech tuning took: {speech_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error getting speech suggestion: {e}")
                speech_recommendation = "neutral"  # Default to neutral if speech selection fails
            
            # Create plots
            try:
                start_time = time.time()
                emotion_plot = self.create_emotion_plot(emotion, emotion_conf)
                sentiment_plot = self.create_sentiment_plot(sentiment_score)
                confidence_gauge = self.create_confidence_gauge(emotion_conf)
                plot_time = time.time() - start_time
                logger.info(f"Plot creation took: {plot_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error creating plots: {e}")
                emotion_plot = None
                sentiment_plot = None
                confidence_gauge = None
            
            # Clean up audio file
            try:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
            except Exception as e:
                logger.warning(f"Error removing temporary audio file: {e}")
            
            # Create HTML summary with enhanced styling and timing information
            html = f"""
            <div class='results-container'>
                <div class='results-title'>Analysis Results</div>
                <div class='results-grid'>
                    <div class='metric-card'>
                        <div class='metric-label'>Detected Emotion</div>
                        <div class='metric-value'>{emotion.title()}</div>
                        <div class='metric-subtext'>Confidence: {emotion_conf:.1%}</div>
                        <div class='metric-subtext'>Time: {emotion_time:.2f}s</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-label'>Sentiment</div>
                        <div class='metric-value'>{sentiment_label}</div>
                        <div class='metric-subtext'>Score: {sentiment_score:.2f}</div>
                        <div class='metric-subtext'>Time: {sentiment_time:.2f}s</div>
                    </div>
                    <div class='metric-card'>
                        <div class='metric-label'>Speech Tuning</div>
                        <div class='metric-value'>{speech_recommendation.title()}</div>
                        <div class='metric-subtext'>Time: {speech_time:.2f}s</div>
                    </div>
                </div>
                <div class='metric-card' style='margin-top: 1rem;'>
                    <div class='metric-label'>Total Processing Time</div>
                    <div class='metric-value'>{emotion_time + sentiment_time + speech_time + plot_time:.2f}s</div>
                </div>
            </div>
            """
            return html, emotion_plot, sentiment_plot, confidence_gauge
            
        except Exception as e:
            logger.error(f"Error in analysis: {e}")
            return f"<b>Error in analysis:</b> {str(e)}", None, None, None
    
    def get_sentiment_label(self, score):
        if score > 0.2:
            return f"{SENTIMENT_EMOJI['positive']} Positive"
        elif score < -0.2:
            return f"{SENTIMENT_EMOJI['negative']} Negative"
        else:
            return f"{SENTIMENT_EMOJI['neutral']} Neutral"
    
    def create_emotion_plot(self, emotion, confidence):
        """Create emotion confidence visualization"""
        emotions = [
            'happy', 'neutral', 'sad', 'angry', 'excited', 'frustrated', 'fearful', 'disgusted', 'surprised'
        ]
        values = [0.0] * len(emotions)
        if emotion in emotions:
            values[emotions.index(emotion)] = confidence

        # Create a more visually appealing bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=emotions,
                y=values,
                marker_color=[SECONDARY_COLOR, PRIMARY_COLOR, ACCENT_COLOR, '#e74c3c', '#f59e42', '#a832a6', '#42f5e6', '#8B5CF6', '#FFD700'],
                text=[f'{v:.2%}' if v > 0 else '' for v in values],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Confidence: %{y:.2%}<extra></extra>'
            )
        ])
        
        # Update layout for better visualization
        fig.update_layout(
            plot_bgcolor=BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            title={
                'text': 'Emotion Detection Confidence',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            yaxis=dict(
                title='Confidence Level',
                tickformat='.0%',
                range=[0, 1]
            ),
            height=400,
            margin=dict(l=20, r=20, t=60, b=20),
            showlegend=False
        )
        return fig

    def create_confidence_gauge(self, confidence):
        """Create a gauge chart for confidence level"""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            title={'text': "Detection Confidence", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickformat': '.0%'},
                'bar': {'color': ACCENT_COLOR},
                'steps': [
                    {'range': [0, 30], 'color': NEGATIVE_COLOR},
                    {'range': [30, 70], 'color': NEUTRAL_COLOR},
                    {'range': [70, 100], 'color': POSITIVE_COLOR}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            plot_bgcolor=BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

    def create_sentiment_plot(self, sentiment):
        """Create sentiment visualization"""
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=sentiment,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [-1, 1]},
                'bar': {'color': ACCENT_COLOR},
                'steps': [
                    {'range': [-1, -0.2], 'color': NEGATIVE_COLOR},
                    {'range': [-0.2, 0.2], 'color': NEUTRAL_COLOR},
                    {'range': [0.2, 1], 'color': POSITIVE_COLOR}
                ]
            }
        ))
        fig.update_layout(
            plot_bgcolor=BACKGROUND_COLOR,
            paper_bgcolor=BACKGROUND_COLOR,
            font=dict(color=TEXT_COLOR),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        return fig

def main():
    analyzer = AudioAnalyzer()
    
    with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as interface:
        with gr.Column(elem_classes="container"):
            # Header
            with gr.Column(elem_classes="header"):
                gr.Markdown("""
                # 🎙️ Speech Tuner
                Analyze emotions from voice recordings and sentiment from text input
                
                **Built with ❤️ by Tooba Jatoi**
                """)
            
            # Main Content
            with gr.Row():
                # Left Column - Audio Recording
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 🎙️ Voice Recording", elem_classes="card-title")
                        with gr.Row():
                            record_btn = gr.Button("Start Recording", variant="primary", elem_classes="button-primary")
                            stop_record_btn = gr.Button("Stop Recording", variant="stop", elem_classes="button-stop")
                        recording_status = gr.Textbox(label="Status", interactive=False, elem_classes="status-box")
                        audio_output = gr.Audio(label="Recorded Audio", type="filepath")
                
                # Right Column - Text Input
                with gr.Column(scale=1):
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 📝 Text Analysis", elem_classes="card-title")
                        text_input = gr.Textbox(
                            label="Enter text for sentiment analysis",
                            placeholder="Type your text here...",
                            lines=3,
                            elem_classes="textbox"
                        )
            
            # Analysis Button
            analyze_btn = gr.Button("Analyze", variant="primary", size="large", elem_classes="button-primary")
            
            # Results Section
            results_html = gr.HTML(label="Analysis Results")
            
            # Visualizations in separate sections
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 🎙️ Voice Analysis", elem_classes="card-title")
                        emotion_plot = gr.Plot(label="Emotion Detection")
                with gr.Column():
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 📊 Confidence Level", elem_classes="card-title")
                        confidence_gauge = gr.Plot(label="Detection Confidence")
            
            with gr.Row():
                with gr.Column():
                    with gr.Column(elem_classes="card"):
                        gr.Markdown("### 📝 Text Analysis", elem_classes="card-title")
                        sentiment_plot = gr.Plot(label="Sentiment Analysis")
        
        # Set up event handlers
        record_btn.click(
            analyzer.start_recording,
            outputs=[recording_status, audio_output]
        )
        
        stop_record_btn.click(
            analyzer.stop_recording,
            outputs=[recording_status, audio_output]
        )
        
        analyze_btn.click(
            analyzer.analyze,
            inputs=[audio_output, text_input],
            outputs=[results_html, emotion_plot, sentiment_plot, confidence_gauge]
        )
    
    # Launch the interface
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=7863,
            share=False,
            debug=False
        )
    except Exception as e:
        logger.error(f"Error launching interface: {str(e)}")
        raise

if __name__ == "__main__":
    main() 