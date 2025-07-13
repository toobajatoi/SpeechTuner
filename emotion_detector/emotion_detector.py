import numpy as np
import sounddevice as sd
import soundfile as sf
from pyAudioAnalysis import audioSegmentation as aS
from pyAudioAnalysis import audioBasicIO as aIO
import threading
import queue
import time
from transformers import pipeline, AutoFeatureExtractor, AutoModelForAudioClassification
import torch
import os
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class EmotionDetector:
    def __init__(self, sample_rate=16000, chunk_duration=2.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.emotion_history = []
        
        logger.info("Initializing emotion detection model...")
        # Initialize the emotion classification model with a more suitable model
        self.model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.classifier = pipeline("audio-classification", model=self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        # Define emotion mapping
        self.emotion_map = {
            'ang': 'angry',
            'hap': 'happy',
            'neu': 'neutral',
            'sad': 'sad',
            'exc': 'excited',
            'fru': 'frustrated',
            'fea': 'fearful',
            'dis': 'disgusted',
            'sur': 'surprised'
        }
        
        # Lower the confidence threshold
        self.confidence_threshold = 0.1  # Adjusted from 0.4
        
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio stream"""
        if status:
            logger.warning(f"Audio callback status: {status}")
        self.audio_queue.put(indata.copy())
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio and return emotion using the Hugging Face model"""
        temp_file = None
        try:
            # Convert to mono if stereo
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Normalize audio
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
            
            # Create a temporary file with delete=False to ensure it's not deleted prematurely
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_filename = temp_file.name
            
            # Write audio data to the temporary file
            sf.write(temp_filename, audio_chunk, self.sample_rate)
            
            # Get emotion predictions
            results = self.classifier(temp_filename)
            
            # Process results
            emotions = []
            for result in results:
                label = result['label']
                score = result['score']
                # Map the label to our emotion categories
                mapped_emotion = self.emotion_map.get(label, label)
                emotions.append((mapped_emotion, score))
            
            # Get the top emotion with confidence threshold
            top_emotion, confidence = emotions[0]
            if confidence < self.confidence_threshold:
                return "neutral", 0.5
            
            return top_emotion, confidence
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return "neutral", 0.5
        finally:
            # Ensure temporary file is closed and deleted after processing
            if temp_file is not None:
                temp_file.close()
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
    
    def start_listening(self):
        """Start listening to microphone input"""
        self.is_running = True
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=self.chunk_size
            )
            self.stream.start()
            logger.info("Started listening to microphone")
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            raise
        
        def process_audio():
            while self.is_running:
                if not self.audio_queue.empty():
                    audio_chunk = self.audio_queue.get()
                    emotion, confidence = self.process_audio_chunk(audio_chunk)
                    self.emotion_history.append((emotion, confidence, time.time()))
                    # Keep only last 10 seconds of history
                    self.emotion_history = [x for x in self.emotion_history 
                                          if time.time() - x[2] < 10]
                time.sleep(0.1)
        
        self.process_thread = threading.Thread(target=process_audio)
        self.process_thread.start()
    
    def stop_listening(self):
        """Stop listening to microphone input"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        if hasattr(self, 'process_thread'):
            self.process_thread.join()
        logger.info("Stopped listening to microphone")
    
    def get_current_emotion(self):
        """Get the most recent emotion and confidence"""
        if not self.emotion_history:
            return "neutral", 0.5
            
        # Get the most recent emotions
        recent_emotions = [x for x in self.emotion_history if time.time() - x[2] < 3]
        if not recent_emotions:
            return "neutral", 0.5
            
        # Count occurrences of each emotion
        emotion_counts = {}
        for emotion, conf, _ in recent_emotions:
            if conf > 0.4:  # Lowered confidence threshold
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        if not emotion_counts:
            return "neutral", 0.5
            
        # Get the most frequent emotion
        most_common = max(emotion_counts.items(), key=lambda x: x[1])
        return most_common[0], most_common[1] / len(recent_emotions)
    
    def get_emotion_history(self):
        """Get the emotion history for the last 10 seconds"""
        return self.emotion_history

    def detect_emotion(self, audio_file):
        """Process an audio file and return the detected emotion"""
        try:
            # Read the audio file
            audio_data, sample_rate = sf.read(audio_file)
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Normalize audio
            audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Get emotion predictions
            results = self.classifier(audio_file)
            
            # Process results
            emotions = []
            for result in results:
                label = result['label']
                score = result['score']
                # Map the label to our emotion categories
                mapped_emotion = self.emotion_map.get(label, label)
                emotions.append((mapped_emotion, score))
            
            # Get the top emotion with confidence threshold
            top_emotion, confidence = emotions[0]
            if confidence < self.confidence_threshold:  # Lowered confidence threshold
                return {"emotion": "neutral", "confidence": 0.5}
            
            return {"emotion": top_emotion, "confidence": confidence}
                
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return {"emotion": "neutral", "confidence": 0.5}

if __name__ == "__main__":
    # Test the emotion detector
    detector = EmotionDetector()
    try:
        print("Starting emotion detection...")
        detector.start_listening()
        time.sleep(5)  # Listen for 5 seconds
        emotion, confidence = detector.get_current_emotion()
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
    finally:
        detector.stop_listening() 