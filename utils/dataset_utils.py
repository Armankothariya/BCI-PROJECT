# ============================================
# 🧠 Dataset Utilities for BCI Project
# ============================================

import numpy as np
import random

# -------------------------
# Synthetic Feature Generation
# -------------------------
def generate_mock_features(n_samples=2548, emotion=None):
    """Generate synthetic EEG features"""
    features = np.random.randn(n_samples)
    emotion_biases = {
        "happy": [0.5] * n_samples,
        "sad": [-0.5] * n_samples,
        "neutral": [0.0] * n_samples,
        "excited": [0.7] * n_samples,
        "relaxed": [-0.2] * n_samples,
        "angry": [0.8] * n_samples
    }
    if emotion in emotion_biases:
        features += emotion_biases[emotion]
    return features

# -------------------------
# Synthetic EEG Signal Generation
# -------------------------
def generate_mock_eeg(n_samples=256, emotion=None):
    """Generate synthetic raw EEG data"""
    t = np.linspace(0, 2, n_samples)
    emotion_freqs = {
        "happy": [10, 12, 30],
        "sad": [4, 6, 8],
        "neutral": [8, 10, 12],
        "excited": [15, 20, 35],
        "relaxed": [8, 10, 12],
        "angry": [18, 25, 40]
    }
    freqs = emotion_freqs.get(emotion, [8 + random.uniform(-2, 2) for _ in range(3)])
    signal = np.zeros(n_samples)
    for freq in freqs:
        amplitude = random.uniform(0.3, 1.0)
        phase = random.uniform(0, 2 * np.pi)
        signal += amplitude * np.sin(2 * np.pi * freq * t + phase)
    signal += 0.1 * np.random.randn(n_samples)
    return signal

# -------------------------
# Sample Selector from Dataset
# -------------------------
def get_test_sample(test_data, test_labels, emotion=None):
    """Get a sample from the test dataset"""
    if test_data is None or test_labels is None or len(test_data) == 0:
        return generate_mock_features(), "unknown"
    if emotion:
        emotion_indices = np.where(test_labels == emotion)[0]
        if len(emotion_indices) > 0:
            sample_idx = random.choice(emotion_indices)
            return test_data[sample_idx], emotion
    sample_idx = random.randint(0, len(test_data) - 1)
    return test_data[sample_idx], test_labels[sample_idx] if sample_idx < len(test_labels) else "unknown"

