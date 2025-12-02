import streamlit as st
from pathlib import Path
import json
import joblib
import pandas as pd
import sys

class PreflightChecker:
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.checks = []
    
    def check_model_bundle(self):
        """Check if model bundle exists and is valid"""
        try:
            model_path = self.root_dir / "models" / "production_model.joblib"
            if not model_path.exists():
                return False, "Model bundle not found", "Run pipeline to generate production_model.joblib"
            
            model_bundle = joblib.load(model_path)
            required_keys = ['model', 'feature_scaler', 'label_encoder']
            missing_keys = [k for k in required_keys if k not in model_bundle]
            
            if missing_keys:
                return False, f"Model missing keys: {missing_keys}", "Retrain model with complete bundle"
            
            # Check feature scaler dimensions
            if hasattr(model_bundle['feature_scaler'], 'n_features_in_'):
                expected_features = model_bundle['feature_scaler'].n_features_in_
                return True, f"Model OK ({expected_features} features)", None
            else:
                return True, "Model OK (legacy format)", None
                
        except Exception as e:
            return False, f"Model load error: {e}", "Check model file integrity"
    
    def check_processed_features(self):
        """Check if processed features exist"""
        try:
            features_path = self.root_dir / "results" / "X_test_processed.csv"
            labels_path = self.root_dir / "results" / "y_test.csv"
            feature_names_path = self.root_dir / "results" / "feature_names.csv"
            
            checks = []
            if features_path.exists():
                checks.append(("Features CSV", True, f"{len(pd.read_csv(features_path))} samples"))
            else:
                checks.append(("Features CSV", False, "Run pipeline to generate"))
            
            if labels_path.exists():
                checks.append(("Labels CSV", True, "Available"))
            else:
                checks.append(("Labels CSV", False, "Run pipeline to generate"))
            
            if feature_names_path.exists():
                checks.append(("Feature names", True, "Available"))
            else:
                checks.append(("Feature names", False, "Run pipeline to generate"))
            
            return checks
            
        except Exception as e:
            return [("Feature check", False, f"Error: {e}")]
    
    def check_pipeline_artifacts(self):
        """Check for pipeline output artifacts"""
        artifacts = [
            ("Model summary JSON", self.root_dir / "results" / "model_summary*.json"),
            ("Confusion matrices", self.root_dir / "results" / "*confusion_matrix*.png"),
            ("Accuracy charts", self.root_dir / "results" / "*accuracy_comparison*.png"),
        ]
        
        results = []
        for name, pattern in artifacts:
            files = list(self.root_dir.glob(str(pattern)))
            if files:
                results.append((name, True, f"{len(files)} files"))
            else:
                results.append((name, False, "Run pipeline to generate"))
        
        return results
    
    def check_audio_mode(self, mode='simulate'):
        """Check audio system availability"""
        if mode == 'pygame':
            try:
                import pygame
                pygame.mixer.init()
                pygame.mixer.quit()
                return True, "Pygame audio available", None
            except Exception as e:
                return False, "Pygame audio failed", f"Use simulate mode: {e}"
        elif mode == 'browser':
            return True, "Browser audio ready", "Audio files will play in browser"
        else:
            return True, "Simulate mode ready", "Audio playback simulated"
    
    def run_checks(self, audio_mode='simulate'):
        """Run all preflight checks"""
        self.checks = []
        
        # Model bundle check
        model_ok, model_msg, model_fix = self.check_model_bundle()
        self.checks.append(("Model Bundle", model_ok, model_msg, model_fix))
        
        # Processed features check
        feature_checks = self.check_processed_features()
        for name, ok, msg in feature_checks:
            self.checks.append((f"Features - {name}", ok, msg, "Run pipeline"))
        
        # Pipeline artifacts
        artifact_checks = self.check_pipeline_artifacts()
        for name, ok, msg in artifact_checks:
            self.checks.append((f"Artifacts - {name}", ok, msg, "Run pipeline"))
        
        # Audio system
        audio_ok, audio_msg, audio_fix = self.check_audio_mode(audio_mode)
        self.checks.append(("Audio System", audio_ok, audio_msg, audio_fix))
        
        return all(ok for (_, ok, _, _) in self.checks)
    
    def get_failed_checks(self):
        """Get list of failed checks with remediation"""
        return [(name, msg, fix) for (name, ok, msg, fix) in self.checks if not ok]