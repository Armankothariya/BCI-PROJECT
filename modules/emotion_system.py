# modules/emotion_system.py
# ============================================
# 🎯 INTEGRATED EMOTION PROCESSING SYSTEM
# Combines Classifier (ML) + Detector (Temporal)
# ============================================

import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import logging

from modules.emotion_classifier import EmotionClassifier
from modules.emotion_detector import EmotionDetector

logger = logging.getLogger(__name__)


class EmotionSystem:
    """
    Complete emotion processing system that integrates:
    - EmotionClassifier: Pure ML prediction (features -> emotion probabilities)
    - EmotionDetector: Temporal stabilization (raw predictions -> stable decisions)
    
    This is the primary interface for the application.
    """
    
    def __init__(
        self,
        model_bundle: Dict,
        detector_config: Optional[Dict] = None
    ):
        """
        Initialize complete emotion system.
        
        Args:
            model_bundle: Model bundle with model, scaler, etc.
            detector_config: Configuration for emotion detector
        """
        # Initialize classifier (STATELESS)
        self.classifier = EmotionClassifier(model_bundle)
        logger.info("EmotionClassifier initialized")
        
        # Initialize detector (STATEFUL)
        if detector_config is None:
            detector_config = {
                'smoothing_window': 5,
                'confidence_threshold': 0.6,
                'min_state_duration': 3.0,
                'require_consistent_predictions': 2
            }
        
        self.detector = EmotionDetector(**detector_config)
        logger.info("EmotionDetector initialized with config")
        
        # System statistics
        self.total_processed = 0
        self.system_start_time = datetime.now()
        
    def process(
        self,
        features: np.ndarray,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Process EEG features through complete emotion pipeline.
        
        Args:
            features: EEG feature vector
            timestamp: Time of prediction
        
        Returns:
            dict: Complete processing result with classifier & detector outputs
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.total_processed += 1
        
        # Step 1: Classify emotion (PURE ML)
        raw_emotion, raw_confidence, raw_probabilities = self.classifier.predict(features)
        
        # Step 2: Apply temporal detection
        stable_emotion, detection_metadata = self.detector.detect(
            raw_emotion, raw_confidence, raw_probabilities, timestamp
        )
        
        # Step 3: Combine results
        result = {
            # Classifier outputs (raw ML predictions)
            'classifier': {
                'emotion': raw_emotion,
                'confidence': raw_confidence,
                'probabilities': raw_probabilities
            },
            
            # Detector outputs (temporally stable)
            'detector': {
                'emotion': stable_emotion,
                'confidence': detection_metadata['stable_confidence'],
                'is_transition': detection_metadata['is_transition'],
                'transition_reason': detection_metadata['transition_reason'],
                'time_in_state': detection_metadata['time_in_state'],
                'consistency_score': detection_metadata['consistency_score']
            },
            
            # Final output (what the system recommends)
            'emotion': stable_emotion,
            'confidence': detection_metadata['stable_confidence'],
            
            # Metadata
            'timestamp': timestamp.isoformat(),
            'processing_number': self.total_processed,
            'full_metadata': detection_metadata
        }
        
        logger.debug(f"Processed #{self.total_processed}: "
                    f"{raw_emotion}({raw_confidence:.2f}) -> "
                    f"{stable_emotion}({detection_metadata['stable_confidence']:.2f})")
        
        return result
    
    def process_batch(
        self,
        features_batch: np.ndarray,
        timestamps: Optional[list] = None
    ) -> list:
        """
        Process multiple samples through the system.
        
        Args:
            features_batch: Array of feature vectors (samples x features)
            timestamps: Optional list of timestamps
        
        Returns:
            list: Results for each sample
        """
        if timestamps is None:
            timestamps = [None] * len(features_batch)
        
        results = []
        for features, timestamp in zip(features_batch, timestamps):
            result = self.process(features, timestamp)
            results.append(result)
        
        return results
    
    def get_current_state(self) -> Dict:
        """Get current system state"""
        detector_stats = self.detector.get_statistics()
        
        return {
            'current_emotion': detector_stats['current_emotion'],
            'current_confidence': detector_stats['current_confidence'],
            'time_in_state': detector_stats['time_in_state'],
            'total_processed': self.total_processed,
            'uptime': (datetime.now() - self.system_start_time).total_seconds(),
            'detector_stats': detector_stats,
            'classifier_info': {
                'feature_count': self.classifier.scaler.n_features_in_,
                'model_type': type(self.classifier.model).__name__
            }
        }
    
    def get_statistics(self) -> Dict:
        """Get comprehensive system statistics"""
        detector_stats = self.detector.get_statistics()
        
        return {
            'system': {
                'total_processed': self.total_processed,
                'uptime_seconds': (datetime.now() - self.system_start_time).total_seconds(),
                'processing_rate': self.total_processed / max((datetime.now() - self.system_start_time).total_seconds(), 1)
            },
            'detector': detector_stats,
            'classifier': {
                'model_type': type(self.classifier.model).__name__,
                'feature_count': self.classifier.scaler.n_features_in_,
                'feature_names': self.classifier.feature_names[:5] if self.classifier.feature_names else []
            }
        }
    
    def reset(self):
        """Reset the entire system"""
        self.detector.reset()
        self.classifier.reset_history()
        self.total_processed = 0
        self.system_start_time = datetime.now()
        logger.info("EmotionSystem reset")
    
    def export_session(self) -> Dict:
        """Export complete session data"""
        return {
            'statistics': self.get_statistics(),
            'detector_state': self.detector.export_state(),
            'buffer_info': self.detector.get_buffer_info(),
            'timestamp': datetime.now().isoformat()
        }
    
    def validate_features(self, features: np.ndarray) -> Tuple[bool, str]:
        """
        Validate feature vector.
        
        Returns:
            tuple: (is_valid, message)
        """
        is_valid = self.classifier.validate_features(features)
        
        if not is_valid:
            expected = self.classifier.scaler.n_features_in_
            actual = features.shape[-1] if len(features.shape) > 1 else len(features)
            message = f"Feature dimension mismatch: expected {expected}, got {actual}"
            return False, message
        
        return True, "Features valid"
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from classifier"""
        return self.classifier.get_feature_importance()
