import numpy as np
from scipy import stats

class EmotionDetector:
    def __init__(self, model, feature_scaler=None, label_encoder=None, 
                 smoothing_window=5, confidence_threshold=0.6):
        self.model = model
        self.feature_scaler = feature_scaler
        self.label_encoder = label_encoder
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.prediction_history = []
        self.probability_history = []
        
    def predict_emotion(self, features):
        """Predict emotion with smoothing and confidence thresholding"""
        # Scale features if scaler is available
        if self.feature_scaler:
            features = self.feature_scaler.transform([features])
        
        # Get prediction probabilities
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(features)[0]
        else:
            # Fallback for models without predict_proba
            decision_scores = self.model.decision_function(features)
            probabilities = self._softmax(decision_scores[0])
        
        # Get predicted class
        predicted_idx = np.argmax(probabilities)
        
        # Apply smoothing using exponential moving average
        smoothed_probs = self._apply_ema_smoothing(probabilities)
        smoothed_idx = np.argmax(smoothed_probs)
        confidence = smoothed_probs[smoothed_idx]
        
        # Decode label if encoder is available
        if self.label_encoder:
            emotion = self.label_encoder.classes_[smoothed_idx]
        else:
            emotion = str(smoothed_idx)
        
        # Apply confidence threshold - revert to neutral if low confidence
        if confidence < self.confidence_threshold:
            if self.label_encoder and "neutral" in self.label_encoder.classes_:
                neutral_idx = np.where(self.label_encoder.classes_ == "neutral")[0][0]
                emotion = "neutral"
                confidence = smoothed_probs[neutral_idx]
        
        return emotion, confidence, smoothed_probs
    
    def _apply_ema_smoothing(self, new_probs, alpha=0.3):
        """Apply exponential moving average smoothing to probabilities"""
        if not self.probability_history:
            self.probability_history.append(new_probs)
            return new_probs
        
        # Calculate EMA
        smoothed = alpha * new_probs + (1 - alpha) * self.probability_history[-1]
        self.probability_history.append(smoothed)
        
        # Keep history limited to window size
        if len(self.probability_history) > self.smoothing_window:
            self.probability_history.pop(0)
            
        return smoothed
    
    def _softmax(self, x):
        """Compute softmax values for each sets of scores in x"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history = []
        self.probability_history = []