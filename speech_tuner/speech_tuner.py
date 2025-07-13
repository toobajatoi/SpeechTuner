"""
Speech Tuner Module
A professional voice and text analysis tool for emotion detection and sentiment analysis.

Built with ❤️ by Tooba Jatoi
"""

import time
from enum import Enum

class VoiceTone(Enum):
    CALM = "calm"
    BRIGHT = "bright"
    NEUTRAL = "neutral"
    EMPATHETIC = "empathetic"

class SpeechTuner:
    def __init__(self):
        self.current_tone = VoiceTone.NEUTRAL
        self.tone_history = []
        self.last_switch_time = time.time()
        self.min_switch_interval = 2.0  # Minimum seconds between tone switches
        
    def get_speech_recommendation(self, emotion, sentiment):
        """Get speech tuning recommendation based on emotion and sentiment"""
        current_time = time.time()
        
        # Don't switch too frequently
        if current_time - self.last_switch_time < self.min_switch_interval:
            return self.current_tone.value
        
        # Weight the decision based on emotion
        if emotion == "angry":
            new_tone = VoiceTone.CALM
        elif emotion == "happy":
            new_tone = VoiceTone.BRIGHT
        elif emotion == "sad":
            new_tone = VoiceTone.EMPATHETIC
        elif emotion == "fearful":
            new_tone = VoiceTone.EMPATHETIC
        elif emotion == "excited":
            new_tone = VoiceTone.BRIGHT
        elif emotion == "frustrated":
            new_tone = VoiceTone.CALM
        else:
            # For neutral emotions, use sentiment as guide
            if sentiment < -0.3:
                new_tone = VoiceTone.EMPATHETIC
            elif sentiment > 0.3:
                new_tone = VoiceTone.BRIGHT
            else:
                new_tone = VoiceTone.NEUTRAL
        
        # Only switch if tone is different
        if new_tone != self.current_tone:
            self.current_tone = new_tone
            self.last_switch_time = current_time
            self.tone_history.append((new_tone.value, current_time))
            
            # Keep only last 10 seconds of history
            self.tone_history = [x for x in self.tone_history 
                               if current_time - x[1] < 10]
        
        return self.current_tone.value
    
    def get_tone_history(self):
        """Get tone history for the last 10 seconds"""
        return self.tone_history
    
    def get_current_tone(self):
        """Get current tone"""
        return self.current_tone.value

if __name__ == "__main__":
    # Test the speech tuner
    tuner = SpeechTuner()
    
    test_cases = [
        ("angry", -0.8),
        ("happy", 0.7),
        ("sad", -0.3),
        ("neutral", 0.1),
        ("excited", 0.9),
        ("fearful", -0.5)
    ]
    
    for emotion, sentiment in test_cases:
        recommendation = tuner.get_speech_recommendation(emotion, sentiment)
        print(f"Emotion: {emotion}, Sentiment: {sentiment:.2f}")
        print(f"Speech recommendation: {recommendation}")
        print("---") 