# modules/emotion_classifier.py
# ============================================
# Emotion Classification Wrapper
# Handles model inference with confidence scoring
# ============================================

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmotionClassifier:
    """
    Wrapper for emotion classification with confidence scoring
    and prediction stabilization.
    """
    
    def __init__(self, model_bundle: Dict):
        self.model = model_bundle['model']
        self.scaler = model_bundle['feature_scaler']
        self.feature_selector = model_bundle.get('feature_selector', None)  # For dimensionality reduction
        self.feature_names = model_bundle.get('feature_names', [])
        self.label_encoder = model_bundle.get('label_encoder', None)
        
        # Prediction stabilization - DISABLED
        self.prediction_history = []
        self.history_size = 1  # Minimal history
        self.stability_threshold = 0.0  # DISABLED - no stabilization
        
        logger.info(f"EmotionClassifier initialized (feature_selector: {self.feature_selector is not None})")
    
    def predict(self, features: np.ndarray, already_scaled: bool = True) -> Tuple[str, float, Dict]:
        """
        Predict emotion from features with confidence scoring.
        
        Args:
            features: Feature vector (already scaled if from app)
            already_scaled: If True, skip scaling (default: True for app compatibility)
            
        Returns:
            tuple: (emotion_label, confidence, probability_dict)
        """
        try:
            # Validate input
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Scale features only if not already scaled
            if already_scaled:
                features_scaled = features  # Already scaled from X_test_processed.csv
            else:
                features_scaled = self.scaler.transform(features)
            
            # Apply feature selection if available (to reduce overfitting)
            if self.feature_selector is not None:
                features_scaled = self.feature_selector.transform(features_scaled)
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
            else:
                # For models without predict_proba, use decision function
                if hasattr(self.model, 'decision_function'):
                    decision_scores = self.model.decision_function(features_scaled)[0]
                    probabilities = self._decision_to_probability(decision_scores)
                else:
                    # Fallback: uniform distribution
                    n_classes = len(self.model.classes_)
                    probabilities = np.ones(n_classes) / n_classes
            
            # Get predicted class
            predicted_idx = np.argmax(probabilities)
            confidence = probabilities[predicted_idx]
            
            # Get emotion label
            if self.label_encoder:
                emotion = self.label_encoder.inverse_transform([predicted_idx])[0]
            else:
                emotion = self.model.classes_[predicted_idx]
            
            # Create probability dictionary
            prob_dict = {}
            for i, prob in enumerate(probabilities):
                if self.label_encoder:
                    label = self.label_encoder.inverse_transform([i])[0]
                else:
                    label = self.model.classes_[i]
                prob_dict[label] = float(prob)
            
            # EMERGENCY: DISABLE all stabilization - return raw prediction
            logger.debug(f"RAW PREDICTION: {emotion} (confidence: {confidence:.3f})")
            
            return emotion, confidence, prob_dict
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "neutral", 0.0, {"neutral": 1.0}
    
    def _decision_to_probability(self, decision_scores: np.ndarray) -> np.ndarray:
        """Convert decision function scores to probabilities"""
        # Simple softmax conversion
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        return exp_scores / np.sum(exp_scores)
    
    def _apply_stabilization(self, emotion: str, confidence: float, 
                           probabilities: Dict) -> Tuple[str, float]:
        """
        Apply prediction stabilization to reduce jitter.
        Uses moving average and confidence thresholding.
        """
        # Add current prediction to history
        self.prediction_history.append({
            'emotion': emotion,
            'confidence': confidence,
            'probabilities': probabilities
        })
        
        # Keep only recent history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # If confidence is high, return immediately
        if confidence >= self.stability_threshold:
            return emotion, confidence
        
        # If we don't have enough history, return current prediction
        if len(self.prediction_history) < 3:
            return emotion, confidence
        
        # Calculate moving average of probabilities
        avg_probabilities = {}
        for pred in self.prediction_history:
            for emotion_key, prob in pred['probabilities'].items():
                if emotion_key not in avg_probabilities:
                    avg_probabilities[emotion_key] = []
                avg_probabilities[emotion_key].append(prob)
        
        # Get average probabilities
        for emotion_key in avg_probabilities:
            avg_probabilities[emotion_key] = np.mean(avg_probabilities[emotion_key])
        
        # Find emotion with highest average probability
        stabilized_emotion = max(avg_probabilities.items(), key=lambda x: x[1])
        stabilized_confidence = stabilized_emotion[1]
        stabilized_emotion = stabilized_emotion[0]
        
        # Only use stabilized prediction if it's significantly better
        confidence_gain = stabilized_confidence - confidence
        if confidence_gain > 0.1:  # 10% improvement threshold
            logger.debug(f"Stabilized: {emotion} -> {stabilized_emotion} "
                        f"(gain: {confidence_gain:.3f})")
            return stabilized_emotion, stabilized_confidence
        
        return emotion, confidence
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_names, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), 
                             key=lambda x: x[1], reverse=True))
        else:
            return {}
    
    def validate_features(self, features: np.ndarray) -> bool:
        """Validate feature vector against expected shape and distribution"""
        expected_features = self.scaler.n_features_in_
        actual_features = features.shape[-1] if len(features.shape) > 1 else len(features)
        
        if actual_features != expected_features:
            logger.warning(f"Feature mismatch: expected {expected_features}, "
                          f"got {actual_features}")
            return False
        
        # Check for NaN or infinite values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            logger.warning("Features contain NaN or infinite values")
            return False
        
        # Basic statistical validation for EEG data
        feature_max = np.abs(features).max()
        if feature_max > 100:  # Arbitrary threshold for normalized EEG data
            logger.warning(f"Feature values too extreme: max={feature_max}")
            return False
        
        return True
    
    def reset_history(self):
        """Reset prediction history"""
        self.prediction_history = []
        logger.debug("Prediction history reset")