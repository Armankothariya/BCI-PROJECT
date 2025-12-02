# modules/research_utils.py
# ============================================
# Scholarship-Grade Research Utilities
# Statistical validation, reproducibility, and research integrity
# ============================================

import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils import resample
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import json
import hashlib

logger = logging.getLogger(__name__)

class ResearchValidator:
    """Comprehensive validation for scholarship-grade research"""
    
    def __init__(self):
        self.validation_history = []
    
    def validate_features_for_research(self, features: np.ndarray, model_bundle: Dict) -> Dict:
        """Comprehensive feature validation with alignment"""
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': features.shape,
            'expected_features': model_bundle['feature_scaler'].n_features_in_,
            'issues': [],
            'warnings': [],
            'actions_taken': [],
            'is_valid': True
        }
        
        try:
            # Check feature dimensions
            if len(features) != validation_report['expected_features']:
                validation_report['issues'].append(
                    f"Feature dimension mismatch: expected {validation_report['expected_features']}, got {len(features)}"
                )
                
                # Auto-alignment with logging
                if len(features) > validation_report['expected_features']:
                    features = features[:validation_report['expected_features']]
                    validation_report['actions_taken'].append(
                        f"Truncated features from {len(features)} to {validation_report['expected_features']}"
                    )
                else:
                    features = np.pad(features, (0, validation_report['expected_features'] - len(features)))
                    validation_report['actions_taken'].append(
                        f"Padded features from {len(features)} to {validation_report['expected_features']}"
                    )
            
            # Data quality checks
            quality_checks = self._check_feature_quality(features)
            validation_report['quality_metrics'] = quality_checks
            
            if quality_checks['has_nans']:
                validation_report['warnings'].append("Features contain NaN values")
                features = np.nan_to_num(features)
                validation_report['actions_taken'].append("Replaced NaN values with zeros")
            
            if quality_checks['is_constant']:
                validation_report['issues'].append("Features are constant (no variance)")
                validation_report['is_valid'] = False
            
            if quality_checks['outlier_ratio'] > 0.3:
                validation_report['warnings'].append(f"High outlier ratio: {quality_checks['outlier_ratio']:.2f}")
            
            validation_report['final_features'] = features.tolist()
            validation_report['feature_hash'] = self._hash_features(features)
            
        except Exception as e:
            validation_report['issues'].append(f"Validation error: {str(e)}")
            validation_report['is_valid'] = False
        
        self.validation_history.append(validation_report)
        return validation_report
    
    def _check_feature_quality(self, features: np.ndarray) -> Dict:
        """Comprehensive feature quality assessment"""
        features = np.array(features, dtype=np.float64)
        
        return {
            'has_nans': np.any(np.isnan(features)),
            'has_infs': np.any(np.isinf(features)),
            'is_constant': np.std(features) < 1e-10,
            'mean': float(np.mean(features)),
            'std': float(np.std(features)),
            'min': float(np.min(features)),
            'max': float(np.max(features)),
            'outlier_ratio': self._calculate_outlier_ratio(features),
            'skewness': float(stats.skew(features)),
            'kurtosis': float(stats.kurtosis(features))
        }
    
    def _calculate_outlier_ratio(self, features: np.ndarray) -> float:
        """Calculate proportion of outliers using IQR method"""
        if len(features) < 4:
            return 0.0
        
        Q1 = np.percentile(features, 25)
        Q3 = np.percentile(features, 75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            return 0.0
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = np.sum((features < lower_bound) | (features > upper_bound))
        return outliers / len(features)
    
    def _hash_features(self, features: np.ndarray) -> str:
        """Create hash for feature reproducibility"""
        return hashlib.md5(features.tobytes()).hexdigest()[:16]
    
    def calculate_research_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_proba: Optional[np.ndarray] = None) -> Dict:
        """Comprehensive research-grade metrics with statistical validation"""
        
        metrics = {
            'primary_metrics': {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'balanced_accuracy': float(self._balanced_accuracy(y_true, y_pred)),
                'f1_macro': float(f1_score(y_true, y_pred, average='macro')),
                'f1_weighted': float(f1_score(y_true, y_pred, average='weighted')),
                'precision_macro': float(precision_score(y_true, y_pred, average='macro')),
                'recall_macro': float(recall_score(y_true, y_pred, average='macro'))
            },
            'confidence_intervals': self._bootstrap_confidence_intervals(y_true, y_pred),
            'effect_sizes': self._calculate_effect_sizes(y_true, y_pred),
            'statistical_tests': self._run_statistical_tests(y_true, y_pred),
            'class_metrics': self._calculate_per_class_metrics(y_true, y_pred)
        }
        
        return metrics
    
    def _balanced_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy"""
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y_true, y_pred))
    
    def _bootstrap_confidence_intervals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      n_bootstraps: int = 1000, confidence_level: float = 0.95) -> Dict:
        """Calculate bootstrap confidence intervals for metrics"""
        
        def accuracy_func(y_t, y_p):
            return accuracy_score(y_t, y_p)
        
        def f1_func(y_t, y_p):
            return f1_score(y_t, y_p, average='macro')
        
        metrics_to_bootstrap = {
            'accuracy': accuracy_func,
            'f1_macro': f1_func
        }
        
        ci_results = {}
        
        for metric_name, metric_func in metrics_to_bootstrap.items():
            bootstrapped_scores = []
            
            for _ in range(n_bootstraps):
                indices = np.random.randint(0, len(y_true), len(y_true))
                if len(np.unique(y_true[indices])) < 2:
                    continue
                
                score = metric_func(y_true[indices], y_pred[indices])
                bootstrapped_scores.append(score)
            
            if bootstrapped_scores:
                alpha = (1 - confidence_level) / 2
                lower = np.percentile(bootstrapped_scores, 100 * alpha)
                upper = np.percentile(bootstrapped_scores, 100 * (1 - alpha))
                
                ci_results[metric_name] = {
                    'mean': float(np.mean(bootstrapped_scores)),
                    'std': float(np.std(bootstrapped_scores)),
                    'confidence_interval': [float(lower), float(upper)],
                    'confidence_level': confidence_level,
                    'n_bootstraps': len(bootstrapped_scores)
                }
        
        return ci_results
    
    def _calculate_effect_sizes(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate various effect sizes"""
        from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
        
        return {
            'cohens_kappa': float(cohen_kappa_score(y_true, y_pred)),
            'matthews_correlation': float(matthews_corrcoef(y_true, y_pred)),
            'cramers_v': self._cramers_v(y_true, y_pred)
        }
    
    def _cramers_v(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cramér's V for categorical association"""
        confusion_matrix = pd.crosstab(y_true, y_pred)
        chi2 = stats.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2 / n
        r, k = confusion_matrix.shape
        phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        rcorr = r - ((r-1)**2)/(n-1)
        kcorr = k - ((k-1)**2)/(n-1)
        
        return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))
    
    def _run_statistical_tests(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Run statistical significance tests"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Chi-square test for independence
        try:
            chi2, chi2_p, _, _ = stats.chi2_contingency(cm)
        except:
            chi2, chi2_p = 0, 1
        
        return {
            'chi_square_test': {
                'statistic': float(chi2),
                'p_value': float(chi2_p),
                'significant': chi2_p < 0.05
            },
            'baseline_comparison': self._compare_to_baseline(y_true, y_pred)
        }
    
    def _compare_to_baseline(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compare model performance to baseline (majority class)"""
        from collections import Counter
        from sklearn.dummy import DummyClassifier
        
        # Majority class baseline
        majority_class = Counter(y_true).most_common(1)[0][0]
        baseline_pred = np.full_like(y_true, majority_class)
        baseline_accuracy = accuracy_score(y_true, baseline_pred)
        model_accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'baseline_accuracy': float(baseline_accuracy),
            'model_accuracy': float(model_accuracy),
            'improvement': float(model_accuracy - baseline_accuracy),
            'relative_improvement': float((model_accuracy - baseline_accuracy) / (1 - baseline_accuracy)) if baseline_accuracy < 1 else 0
        }
    
    def _calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate detailed per-class metrics"""
        from sklearn.metrics import classification_report
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        # Add support ratios
        for class_name, metrics in report.items():
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            if 'support' in metrics:
                metrics['support_ratio'] = metrics['support'] / len(y_true)
        
        return report

class ResearchSessionManager:
    """Manage research-grade session data with full reproducibility"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.predictions = []
        self.metadata = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'system_info': self._capture_system_info(),
            'random_seeds': self._capture_random_seeds()
        }
        self.feature_hashes = set()
    
    def _capture_system_info(self) -> Dict:
        """Capture comprehensive system information"""
        import platform
        import sys
        
        return {
            'platform': platform.platform(),
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'pandas_version': pd.__version__,
            'scikit_learn_version': self._get_sklearn_version(),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_sklearn_version(self) -> str:
        """Get scikit-learn version safely"""
        try:
            import sklearn
            return sklearn.__version__
        except:
            return "unknown"
    
    def _capture_random_seeds(self) -> Dict:
        """Capture random seeds for reproducibility"""
        return {
            'numpy_seed': np.random.get_state()[1][0],
            'python_seed': random.getstate()[1][0],
            'timestamp': datetime.now().isoformat()
        }
    
    def add_prediction(self, features: np.ndarray, raw_prediction: Dict, 
                      final_prediction: Dict, processing_metadata: Dict) -> str:
        """Add a prediction with full research context"""
        
        # Create feature hash for deduplication
        feature_hash = hashlib.md5(features.tobytes()).hexdigest()[:16]
        
        prediction_record = {
            'prediction_id': f"pred_{len(self.predictions)}_{feature_hash}",
            'timestamp': datetime.now().isoformat(),
            'feature_hash': feature_hash,
            'features_shape': features.shape,
            'feature_statistics': {
                'mean': float(np.mean(features)),
                'std': float(np.std(features)),
                'min': float(np.min(features)),
                'max': float(np.max(features))
            },
            'raw_prediction': raw_prediction,
            'final_prediction': final_prediction,
            'processing_metadata': processing_metadata,
            'system_state': self._capture_system_state()
        }
        
        self.predictions.append(prediction_record)
        self.feature_hashes.add(feature_hash)
        
        return prediction_record['prediction_id']
    
    def _capture_system_state(self) -> Dict:
        """Capture current system state"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        return {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_research_report(self) -> Dict:
        """Generate comprehensive research report"""
        if not self.predictions:
            return {'error': 'No predictions recorded'}
        
        # Calculate session statistics
        predictions_df = pd.DataFrame([p['final_prediction'] for p in self.predictions])
        
        return {
            'session_metadata': self.metadata,
            'prediction_statistics': {
                'total_predictions': len(self.predictions),
                'unique_feature_hashes': len(self.feature_hashes),
                'emotion_distribution': predictions_df['emotion'].value_counts().to_dict(),
                'confidence_statistics': {
                    'mean': float(predictions_df['confidence'].mean()),
                    'std': float(predictions_df['confidence'].std()),
                    'min': float(predictions_df['confidence'].min()),
                    'max': float(predictions_df['confidence'].max())
                }
            },
            'quality_metrics': {
                'duplicate_predictions': len(self.predictions) - len(self.feature_hashes),
                'average_processing_time': self._calculate_avg_processing_time(),
                'system_stability': self._assess_system_stability()
            },
            'reproducibility_info': {
                'feature_hashes': list(self.feature_hashes),
                'prediction_ids': [p['prediction_id'] for p in self.predictions],
                'session_duration': self._calculate_session_duration()
            }
        }
    
    def _calculate_avg_processing_time(self) -> float:
        """Calculate average processing time per prediction"""
        if len(self.predictions) < 2:
            return 0.0
        
        timestamps = [datetime.fromisoformat(p['timestamp']) for p in self.predictions]
        durations = [(timestamps[i] - timestamps[i-1]).total_seconds() for i in range(1, len(timestamps))]
        
        return float(np.mean(durations)) if durations else 0.0
    
    def _assess_system_stability(self) -> Dict:
        """Assess system stability throughout session"""
        memory_usage = [p['system_state']['memory_usage_mb'] for p in self.predictions]
        
        return {
            'memory_stability': float(np.std(memory_usage) / np.mean(memory_usage)) if memory_usage else 0,
            'memory_leak_detected': self._check_memory_leak(memory_usage),
            'system_crashes': 0  # Would be tracked in real implementation
        }
    
    def _check_memory_leak(self, memory_usage: List[float]) -> bool:
        """Check for potential memory leaks"""
        if len(memory_usage) < 10:
            return False
        
        # Simple linear regression to detect increasing trend
        x = np.arange(len(memory_usage))
        slope, _, _, p_value, _ = stats.linregress(x, memory_usage)
        
        return slope > 0.1 and p_value < 0.05
    
    def _calculate_session_duration(self) -> float:
        """Calculate total session duration"""
        if len(self.predictions) < 2:
            return 0.0
        
        start_time = datetime.fromisoformat(self.predictions[0]['timestamp'])
        end_time = datetime.fromisoformat(self.predictions[-1]['timestamp'])
        
        return (end_time - start_time).total_seconds()
    
    def save_session(self, filepath: str):
        """Save session data to file"""
        session_data = {
            'metadata': self.metadata,
            'predictions': self.predictions,
            'research_report': self.generate_research_report()
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def load_session(self, filepath: str):
        """Load session data from file"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self.metadata = session_data['metadata']
        self.predictions = session_data['predictions']
        self.feature_hashes = set(p['feature_hash'] for p in self.predictions)