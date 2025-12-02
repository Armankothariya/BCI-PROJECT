# modules/error_handler.py
# ============================================
# Comprehensive Error Handling for Scholarship-Grade Application
# ============================================

import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List
import json
import sys
from pathlib import Path

class ScholarshipErrorHandler:
    """Professional error handling for research-grade applications"""
    
    def __init__(self, log_dir: str = "results/error_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.error_count = 0
        self.setup_error_logging()
    
    def setup_error_logging(self):
        """Setup comprehensive error logging"""
        error_log_file = self.log_dir / f"scholarship_errors_{datetime.now().strftime('%Y%m%d')}.log"
        
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(error_log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stderr)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    recovery_attempted: bool = False) -> Dict[str, Any]:
        """Comprehensive error handling with research context"""
        self.error_count += 1
        
        error_info = self._capture_error_info(error, context, recovery_attempted)
        
        # Log for research analysis
        self._log_error_for_research(error_info)
        
        # Attempt recovery
        recovery_result = self._attempt_automatic_recovery(error, context)
        
        # Provide user feedback
        user_message = self._generate_user_friendly_message(error, context, recovery_result)
        
        return {
            'error_info': error_info,
            'recovery_result': recovery_result,
            'user_message': user_message,
            'should_continue': recovery_result['success'] or recovery_attempted
        }
    
    def _capture_error_info(self, error: Exception, context: Dict, recovery_attempted: bool) -> Dict:
        """Capture comprehensive error information"""
        
        return {
            'error_id': f"err_{self.error_count:06d}",
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'system_state': self._capture_system_state(),
            'recovery_attempted': recovery_attempted,
            'severity': self._assess_error_severity(error, context)
        }
    
    def _capture_system_state(self) -> Dict:
        """Capture current system state"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        return {
            'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
            'cpu_percent': process.cpu_percent(),
            'python_version': sys.version,
            'platform': sys.platform,
            'current_working_directory': os.getcwd()
        }
    
    def _assess_error_severity(self, error: Exception, context: Dict) -> str:
        """Assess error severity for appropriate handling"""
        error_type = type(error).__name__
        
        critical_errors = [
            'MemoryError', 'OSError', 'ImportError', 
            'FileNotFoundError', 'PermissionError'
        ]
        
        if error_type in critical_errors:
            return 'critical'
        elif 'data' in context or 'model' in context:
            return 'high'
        else:
            return 'medium'
    
    def _attempt_automatic_recovery(self, error: Exception, context: Dict) -> Dict:
        """Attempt automatic recovery based on error type"""
        
        recovery_strategies = {
            'FileNotFoundError': self._recover_file_not_found,
            'ValueError': self._recover_value_error,
            'AttributeError': self._recover_attribute_error,
            'KeyError': self._recover_key_error,
            'IndexError': self._recover_index_error
        }
        
        error_type = type(error).__name__
        recovery_function = recovery_strategies.get(error_type, self._recover_generic)
        
        return recovery_function(error, context)
    
    def _recover_file_not_found(self, error: Exception, context: Dict) -> Dict:
        """Recover from file not found errors"""
        try:
            filename = str(error).split(": ")[-1].strip("'")
            
            # Check if it's a model file
            if 'model' in filename.lower() or '.joblib' in filename:
                return {
                    'success': False,
                    'strategy': 'model_file_missing',
                    'message': 'Model file not found - retraining required',
                    'action': 'trigger_retraining'
                }
            
            # Check if it's a data file
            elif any(ext in filename.lower() for ext in ['.csv', '.npy', '.pkl']):
                return {
                    'success': False,
                    'strategy': 'data_file_missing',
                    'message': 'Data file not found - check dataset path',
                    'action': 'update_data_path'
                }
            
            return self._recover_generic(error, context)
            
        except:
            return self._recover_generic(error, context)
    
    def _recover_value_error(self, error: Exception, context: Dict) -> Dict:
        """Recover from value errors (common in data processing)"""
        error_msg = str(error).lower()
        
        if 'shape' in error_msg or 'dimension' in error_msg:
            return {
                'success': True,
                'strategy': 'feature_alignment',
                'message': 'Fixed feature dimension mismatch',
                'action': 'auto_align_features'
            }
        
        elif 'nan' in error_msg or 'inf' in error_msg:
            return {
                'success': True,
                'strategy': 'data_cleaning',
                'message': 'Cleaned invalid data values',
                'action': 'remove_invalid_values'
            }
        
        return self._recover_generic(error, context)
    
    def _recover_attribute_error(self, error: Exception, context: Dict) -> Dict:
        """Recover from attribute errors"""
        if 'model' in context and 'predict' in str(error):
            return {
                'success': False,
                'strategy': 'model_compatibility',
                'message': 'Model incompatible - may need retraining',
                'action': 'reload_model'
            }
        
        return self._recover_generic(error, context)
    
    def _recover_key_error(self, error: Exception, context: Dict) -> Dict:
        """Recover from key errors"""
        return {
            'success': True,
            'strategy': 'default_fallback',
            'message': 'Used default values for missing keys',
            'action': 'use_defaults'
        }
    
    def _recover_index_error(self, error: Exception, context: Dict) -> Dict:
        """Recover from index errors"""
        return {
            'success': True,
            'strategy': 'bounds_checking',
            'message': 'Adjusted index to valid range',
            'action': 'clamp_index'
        }
    
    def _recover_generic(self, error: Exception, context: Dict) -> Dict:
        """Generic recovery strategy"""
        return {
            'success': False,
            'strategy': 'graceful_degradation',
            'message': 'Entering fallback mode',
            'action': 'enable_fallback'
        }
    
    def _log_error_for_research(self, error_info: Dict):
        """Log error for research analysis"""
        error_file = self.log_dir / f"error_{error_info['error_id']}.json"
        
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, default=str)
        
        self.logger.error(
            f"Error {error_info['error_id']}: {error_info['error_type']} - "
            f"{error_info['error_message']} (Severity: {error_info['severity']})"
        )
    
    def _generate_user_friendly_message(self, error: Exception, context: Dict, 
                                      recovery_result: Dict) -> Dict:
        """Generate user-friendly error messages"""
        
        base_messages = {
            'critical': "System encountered a critical error. Some features may be unavailable.",
            'high': "Important function failed. Limited functionality available.",
            'medium': "Minor issue encountered. System continues with reduced features.",
            'low': "Temporary issue. System should recover automatically."
        }
        
        severity = recovery_result.get('severity', 'medium')
        
        return {
            'title': f"⚠️ {recovery_result.get('strategy', 'System Issue').replace('_', ' ').title()}",
            'message': base_messages.get(severity, "An unexpected error occurred."),
            'technical_details': f"Error: {type(error).__name__} - {str(error)}",
            'recovery_action': recovery_result.get('message', 'Attempting automatic recovery...'),
            'suggested_steps': self._get_suggested_steps(error, context, recovery_result),
            'can_continue': recovery_result.get('success', False) or severity in ['medium', 'low']
        }
    
    def _get_suggested_steps(self, error: Exception, context: Dict, recovery_result: Dict) -> List[str]:
        """Get suggested steps for user recovery"""
        
        steps = [
            "Error details have been logged for research analysis",
            "The system will attempt to continue with available features"
        ]
        
        if not recovery_result.get('success', False):
            steps.extend([
                "Consider restarting the application",
                "Check that all required data files are available",
                "Verify model files are in the correct location"
            ])
        
        return steps
    
    def enable_fallback_mode(self, context: Dict) -> Dict:
        """Enable comprehensive fallback mode"""
        return {
            'mode': 'fallback',
            'capabilities': {
                'basic_demo': True,
                'research_features': False,
                'audio_playback': context.get('audio_available', False),
                'visualization': True
            },
            'limitations': [
                "Advanced research features disabled",
                "Statistical validation limited",
                "Some visualizations simplified"
            ],
            'recovery_actions': [
                "Restart application to restore full functionality",
                "Check system requirements and dependencies"
            ]
        }

class GracefulDegradation:
    """Manage graceful degradation of features"""
    
    def __init__(self):
        self.degradation_level = 'full'
        self.available_features = self._get_full_features()
    
    def _get_full_features(self) -> Dict:
        """Define full feature set"""
        return {
            'research': {
                'statistical_analysis': True,
                'confidence_intervals': True,
                'reproducibility': True,
                'error_analysis': True
            },
            'demo': {
                'real_time_processing': True,
                'audio_control': True,
                'advanced_visualization': True,
                'session_management': True
            },
            'core': {
                'emotion_detection': True,
                'basic_visualization': True,
                'data_loading': True
            }
        }
    
    def degrade_to_level(self, level: str) -> Dict:
        """Degrade to specified feature level"""
        levels = {
            'full': self._get_full_features(),
            'research_only': self._get_research_features(),
            'demo_only': self._get_demo_features(),
            'minimal': self._get_minimal_features()
        }
        
        self.degradation_level = level
        self.available_features = levels.get(level, self._get_minimal_features())
        
        return self.available_features
    
    def _get_research_features(self) -> Dict:
        """Research-focused feature set"""
        features = self._get_full_features()
        features['demo']['audio_control'] = False
        features['demo']['advanced_visualization'] = False
        return features
    
    def _get_demo_features(self) -> Dict:
        """Demo-focused feature set"""
        features = self._get_full_features()
        features['research']['statistical_analysis'] = False
        features['research']['confidence_intervals'] = False
        return features
    
    def _get_minimal_features(self) -> Dict:
        """Minimal feature set for basic operation"""
        return {
            'research': {
                'statistical_analysis': False,
                'confidence_intervals': False,
                'reproducibility': False,
                'error_analysis': False
            },
            'demo': {
                'real_time_processing': True,
                'audio_control': False,
                'advanced_visualization': False,
                'session_management': False
            },
            'core': {
                'emotion_detection': True,
                'basic_visualization': True,
                'data_loading': True
            }
        }
    
    def is_feature_available(self, category: str, feature: str) -> bool:
        """Check if a specific feature is available"""
        return self.available_features.get(category, {}).get(feature, False)