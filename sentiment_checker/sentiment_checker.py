import os
from openai import OpenAI
from dotenv import load_dotenv
import time
from langdetect import detect
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

class SentimentChecker:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        # Define sentiment words with weights
        self.positive_words = {
            'happy': 1.0, 'good': 0.8, 'great': 1.0, 'excellent': 1.2, 'wonderful': 1.2,
            'amazing': 1.2, 'fantastic': 1.2, 'love': 1.5, 'like': 0.8, 'best': 1.0,
            'perfect': 1.2, 'awesome': 1.2, 'brilliant': 1.2, 'outstanding': 1.2,
            'pleased': 0.8, 'satisfied': 0.8, 'delighted': 1.2, 'enjoy': 0.8,
            'enjoyed': 0.8, 'enjoying': 0.8, 'beautiful': 1.0, 'favorite': 1.0,
            'glad': 0.8, 'grateful': 1.0, 'joy': 1.2, 'pleasure': 0.8, 'proud': 0.8,
            'success': 1.0, 'successful': 1.0, 'win': 1.0, 'won': 1.0, 'winning': 1.0,
            'خوش': 1.0, 'اچھا': 0.8, 'بہترین': 1.2, 'عمدہ': 1.0, 'پسند': 0.8,
            'محبت': 1.5, 'شکریہ': 0.8, 'مبارک': 1.0, 'کامیاب': 1.0
        }
        
        self.negative_words = {
            'bad': -0.8, 'terrible': -1.2, 'awful': -1.2, 'horrible': -1.2,
            'worst': -1.5, 'poor': -0.8, 'disappointing': -1.0, 'hate': -1.5,
            'dislike': -1.0, 'angry': -1.2, 'upset': -1.0, 'frustrated': -1.0,
            'annoyed': -0.8, 'sad': -1.0, 'unhappy': -1.0, 'miserable': -1.2,
            'depressed': -1.2, 'furious': -1.5, 'outraged': -1.5, 'disgusting': -1.2,
            'hateful': -1.5, 'horrible': -1.2, 'terrible': -1.2, 'awful': -1.2,
            'disappointed': -1.0, 'failing': -1.0, 'failed': -1.0, 'failure': -1.0,
            'lose': -1.0, 'lost': -1.0, 'losing': -1.0, 'wrong': -0.8,
            'برا': -0.8, 'خراب': -1.0, 'ناپسند': -1.0, 'غصہ': -1.2, 'ناراض': -1.0,
            'مایوس': -1.0, 'دکھی': -1.0, 'تکلیف': -1.0, 'نفرت': -1.5
        }
        
        # Negation words
        self.negation_words = {
            'not', 'no', 'never', 'none', 'neither', 'nor', 'nothing', 'nowhere',
            'hardly', 'barely', 'scarcely', 'doesnt', 'isnt', 'wasnt', 'shouldnt',
            'wouldnt', 'couldnt', 'wont', 'cant', 'dont', 'نہیں', 'مت', 'کبھی نہیں'
        }
        
        # Intensifier words
        self.intensifier_words = {
            'very': 1.5, 'really': 1.5, 'extremely': 2.0, 'absolutely': 2.0,
            'completely': 1.5, 'totally': 1.5, 'incredibly': 1.5, 'especially': 1.2,
            'particularly': 1.2, 'highly': 1.2, 'بہت': 1.5, 'انتہائی': 2.0,
            'نہایت': 1.5, 'خاص طور پر': 1.2
        }
        
        self.sentiment_history = []
        self.stop_words = set(stopwords.words('english'))
        # Crisis phrase detection
        self.crisis_phrases = [
            "kill myself", "end my life", "suicide", "want to die", "can't go on", "give up on life",
            "hurt myself", "self harm", "take my life", "no reason to live", "don't want to live",
            "wish I was dead", "wish to die", "life is not worth living", "i'm done with life"
        ]
    
    def preprocess_text(self, text):
        """Preprocess the text for sentiment analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^ -]+', ' ', text)  # Remove non-ASCII for crisis detection
        
        # Tokenize
        tokens = word_tokenize(text)
        
        return tokens
        
    def get_sentiment(self, text):
        """Get sentiment score for text (-1 to +1) using enhanced word matching"""
        start_time = time.time()
        
        try:
            # Preprocess text
            tokens = self.preprocess_text(text)
            
            # Initialize sentiment score
            score = 0.0
            negation = False
            intensifier = 1.0
            
            # Process each token
            for i, token in enumerate(tokens):
                # Check for negation
                if token in self.negation_words:
                    negation = True
                    continue
                    
                # Check for intensifiers
                if token in self.intensifier_words:
                    intensifier = self.intensifier_words[token]
                    continue
                    
                # Check for sentiment words
                if token in self.positive_words:
                    word_score = self.positive_words[token]
                    if negation:
                        word_score = -word_score
                    score += word_score * intensifier
                    negation = False
                    intensifier = 1.0
                elif token in self.negative_words:
                    word_score = self.negative_words[token]
                    if negation:
                        word_score = -word_score
                    score += word_score * intensifier
                    negation = False
                    intensifier = 1.0
            
            # Normalize score between -1 and 1
            if len(tokens) > 0:
                score = score / len(tokens)
                score = max(min(score, 1.0), -1.0)
            
            # Record timing
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            self.sentiment_history.append((score, processing_time, time.time()))
            
            # Keep only last 10 seconds of history
            self.sentiment_history = [x for x in self.sentiment_history 
                                    if time.time() - x[2] < 10]
            
            return score, processing_time
            
        except Exception as e:
            print(f"Error in sentiment analysis: {e}")
            return 0.0, 0.0
    
    def get_sentiment_history(self):
        """Get sentiment history for the last 10 seconds"""
        return self.sentiment_history
    
    def get_average_sentiment(self):
        """Get average sentiment over the last 10 seconds"""
        if not self.sentiment_history:
            return 0.0
        return sum(x[0] for x in self.sentiment_history) / len(self.sentiment_history)
    
    def get_sentiment_label(self, score):
        """Convert sentiment score to label"""
        if score > 0.3:
            return "positive"
        elif score < -0.3:
            return "negative"
        else:
            return "neutral"

if __name__ == "__main__":
    # Test the sentiment checker
    checker = SentimentChecker()
    
    test_texts = [
        "I am so happy with the service!",
        "This is terrible, I want a refund.",
        "The product is okay, nothing special.",
        "I absolutely love this amazing product!",
        "This is not bad at all, I'm quite satisfied.",
        "میں بہت خوش ہوں"  # Urdu text: "I am very happy"
    ]
    
    for text in test_texts:
        score, processing_time = checker.get_sentiment(text)
        label = checker.get_sentiment_label(score)
        print(f"Text: {text}")
        print(f"Sentiment: {label} (score: {score:.2f})")
        print(f"Processing time: {processing_time:.2f}ms")
        print("---") 