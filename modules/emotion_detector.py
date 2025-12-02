# modules/emotion_detector.py
# ============================================
# 🔬 TEMPORAL EMOTION DETECTION - STATEFUL
# Separate from EmotionClassifier (Pure ML)
# Handles: Temporal smoothing, state transitions, stability
# ============================================

import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Temporal emotion detector with state management.
    
    Separates concerns:
    - EmotionClassifier: "What emotion is this?" (STATELESS)
    - EmotionDetector: "When does emotion change?" (STATEFUL)
    
    Applies:
    - Moving average smoothing
    - Confidence thresholding
    - Minimum state duration
    - Transition validation
    """
    
    def __init__(
        self,
        smoothing_window: int = 3,
        confidence_threshold: float = 0.4,
        min_state_duration: float = 1.0,
        require_consistent_predictions: int = 1
    ):
        """
        Initialize emotion detector.
        
        Args:
            smoothing_window: Number of predictions to average
            confidence_threshold: Minimum confidence to accept prediction
            min_state_duration: Minimum seconds before allowing emotion change
            require_consistent_predictions: Consecutive predictions needed for change
        """
        self.smoothing_window = smoothing_window
        self.confidence_threshold = confidence_threshold
        self.min_state_duration = min_state_duration
        self.require_consistent = require_consistent_predictions
        
        # State tracking
        self.current_emotion = None
        self.current_confidence = 0.0
        self.emotion_start_time = None
        self.last_update_time = None
        
        # History buffers
        self.prediction_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)
        self.probability_buffer = deque(maxlen=smoothing_window)
        
        # Statistics
        self.total_predictions = 0
        self.transitions_made = 0
        self.rejected_transitions = 0
        
        logger.info(f"EmotionDetector initialized: window={smoothing_window}, "
                   f"threshold={confidence_threshold}, min_duration={min_state_duration}s")
    
    def detect(
        self,
        raw_emotion: str,
        raw_confidence: float,
        raw_probabilities: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> Tuple[str, Dict[str, any]]:
        """
        Process raw emotion prediction and return stable emotion.
        
        Args:
            raw_emotion: Raw emotion from classifier
            raw_confidence: Raw confidence from classifier
            raw_probabilities: Full probability distribution
            timestamp: Time of prediction (defaults to now)
        
        Returns:
            tuple: (stable_emotion, metadata_dict)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.total_predictions += 1
        self.last_update_time = timestamp
        
        # Add to buffers
        self.prediction_buffer.append(raw_emotion)
        self.confidence_buffer.append(raw_confidence)
        self.probability_buffer.append(raw_probabilities)
        
        # Apply temporal detection logic
        stable_emotion, metadata = self._apply_temporal_logic(
            raw_emotion, raw_confidence, raw_probabilities, timestamp
        )
        
        # Log detection
        logger.debug(f"Detection: {raw_emotion}({raw_confidence:.2f}) -> "
                    f"{stable_emotion}({metadata['stable_confidence']:.2f}) "
                    f"[transition={metadata['is_transition']}]")
        
        return stable_emotion, metadata
    
    def _apply_temporal_logic(
        self,
        raw_emotion: str,
        raw_confidence: float,
        raw_probabilities: Dict[str, float],
        timestamp: datetime
    ) -> Tuple[str, Dict]:
        """Apply temporal detection logic"""
        
        # Initialize if first prediction
        if self.current_emotion is None:
            self.current_emotion = raw_emotion
            self.current_confidence = raw_confidence
            self.emotion_start_time = timestamp
            
            return self.current_emotion, {
                'stable_emotion': self.current_emotion,
                'stable_confidence': self.current_confidence,
                'raw_emotion': raw_emotion,
                'raw_confidence': raw_confidence,
                'is_transition': False,
                'transition_reason': 'initialization',
                'time_in_state': 0.0,
                'buffer_size': len(self.prediction_buffer),
                'consistency_score': 1.0
            }
        
        # Calculate smoothed probabilities
        smoothed_probs = self._smooth_probabilities()
        smoothed_emotion = max(smoothed_probs.items(), key=lambda x: x[1])[0]
        smoothed_confidence = smoothed_probs[smoothed_emotion]
        
        # Calculate consistency score
        consistency = self._calculate_consistency(raw_emotion)
        
        # Time in current state
        time_in_state = (timestamp - self.emotion_start_time).total_seconds()
        
        # Determine if transition should occur
        should_transition, reason = self._should_transition(
            raw_emotion, raw_confidence, smoothed_emotion, 
            smoothed_confidence, consistency, time_in_state
        )
        
        # Execute transition if needed
        if should_transition:
            self.current_emotion = smoothed_emotion
            self.current_confidence = smoothed_confidence
            self.emotion_start_time = timestamp
            self.transitions_made += 1
            is_transition = True
        else:
            is_transition = False
            # Update confidence even when not transitioning (reflect current model confidence)
            self.current_confidence = smoothed_probs.get(self.current_emotion, self.current_confidence)
            if raw_emotion != self.current_emotion:
                self.rejected_transitions += 1
        
        return self.current_emotion, {
            'stable_emotion': self.current_emotion,
            'stable_confidence': self.current_confidence,
            'raw_emotion': raw_emotion,
            'raw_confidence': raw_confidence,
            'smoothed_emotion': smoothed_emotion,
            'smoothed_confidence': smoothed_confidence,
            'is_transition': is_transition,
            'transition_reason': reason if is_transition else 'no_transition',
            'time_in_state': time_in_state,
            'buffer_size': len(self.prediction_buffer),
            'consistency_score': consistency,
            'smoothed_probabilities': smoothed_probs
        }
    
    def _smooth_probabilities(self) -> Dict[str, float]:
        """Apply moving average to probability distributions"""
        if not self.probability_buffer:
            return {}
        
        # Get all unique emotions across buffer
        all_emotions = set()
        for prob_dict in self.probability_buffer:
            all_emotions.update(prob_dict.keys())
        
        # Calculate average probability for each emotion
        smoothed = {}
        for emotion in all_emotions:
            probs = [
                prob_dict.get(emotion, 0.0) 
                for prob_dict in self.probability_buffer
            ]
            smoothed[emotion] = np.mean(probs)
        
        # Normalize to ensure sum=1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v/total for k, v in smoothed.items()}
        
        return smoothed
    
    def _calculate_consistency(self, current_emotion: str) -> float:
        """Calculate consistency score (what % of recent predictions agree)"""
        if not self.prediction_buffer:
            return 0.0
        
        matches = sum(1 for e in self.prediction_buffer if e == current_emotion)
        return matches / len(self.prediction_buffer)
    
    def _should_transition(
        self,
        raw_emotion: str,
        raw_confidence: float,
        smoothed_emotion: str,
        smoothed_confidence: float,
        consistency: float,
        time_in_state: float
    ) -> Tuple[bool, str]:
        """
        Determine if emotion should transition.
        
        Returns:
            tuple: (should_transition, reason)
        """
        # If same as current, no transition needed
        if smoothed_emotion == self.current_emotion:
            return False, "same_emotion"
        
        # Check minimum state duration
        if time_in_state < self.min_state_duration:
            return False, f"min_duration_not_met ({time_in_state:.1f}s < {self.min_state_duration}s)"
        
        # Check confidence threshold
        if smoothed_confidence < self.confidence_threshold:
            return False, f"confidence_too_low ({smoothed_confidence:.2f} < {self.confidence_threshold})"
        
        # Check consistency (recent predictions should agree)
        required_consistency = self.require_consistent / self.smoothing_window
        if consistency < required_consistency:
            return False, f"insufficient_consistency ({consistency:.2f} < {required_consistency:.2f})"
        
        # All checks passed - allow transition
        return True, f"valid_transition (conf={smoothed_confidence:.2f}, consist={consistency:.2f}, time={time_in_state:.1f}s)"
    
    def get_buffer_info(self) -> List[Dict]:
        """Get detailed buffer information for debugging"""
        buffer_info = []
        for i, (emotion, confidence, probs) in enumerate(zip(
            self.prediction_buffer,
            self.confidence_buffer,
            self.probability_buffer
        )):
            buffer_info.append({
                'index': i,
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probs
            })
        return buffer_info
    
    def get_statistics(self) -> Dict:
        """Get detector statistics"""
        return {
            'current_emotion': self.current_emotion,
            'current_confidence': self.current_confidence,
            'time_in_state': (
                (datetime.now() - self.emotion_start_time).total_seconds()
                if self.emotion_start_time else 0.0
            ),
            'total_predictions': self.total_predictions,
            'transitions_made': self.transitions_made,
            'rejected_transitions': self.rejected_transitions,
            'transition_rate': (
                self.transitions_made / self.total_predictions
                if self.total_predictions > 0 else 0.0
            ),
            'buffer_size': len(self.prediction_buffer),
            'buffer_fullness': len(self.prediction_buffer) / self.smoothing_window
        }
    
    def reset(self):
        """Reset detector state"""
        self.current_emotion = None
        self.current_confidence = 0.0
        self.emotion_start_time = None
        self.last_update_time = None
        
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.probability_buffer.clear()
        
        self.total_predictions = 0
        self.transitions_made = 0
        self.rejected_transitions = 0
        
        logger.info("EmotionDetector reset")
    
    def export_state(self) -> Dict:
        """Export detector state for persistence"""
        return {
            'current_emotion': self.current_emotion,
            'current_confidence': self.current_confidence,
            'emotion_start_time': self.emotion_start_time.isoformat() if self.emotion_start_time else None,
            'statistics': self.get_statistics(),
            'config': {
                'smoothing_window': self.smoothing_window,
                'confidence_threshold': self.confidence_threshold,
                'min_state_duration': self.min_state_duration,
                'require_consistent': self.require_consistent
            }
        }
