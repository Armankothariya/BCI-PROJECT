"""
Ablation Study Module
Systematic feature and model component ablation for research validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import json
from datetime import datetime


class AblationStudy:
    """
    Conduct ablation studies to understand feature and model component importance
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
    
    def feature_ablation(self, X_train, y_train, X_test, y_test, 
                        feature_names: List[str], 
                        model=None) -> Dict:
        """
        Ablation study by removing features one at a time
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        
        # Baseline: all features
        model.fit(X_train, y_train)
        baseline_acc = accuracy_score(y_test, model.predict(X_test))
        baseline_f1 = f1_score(y_test, model.predict(X_test), average='macro')
        
        ablation_results = {
            'baseline': {
                'accuracy': float(baseline_acc),
                'f1_score': float(baseline_f1),
                'n_features': X_train.shape[1]
            },
            'feature_importance': []
        }
        
        # Remove each feature and measure impact
        for i, feat_name in enumerate(feature_names):
            # Create dataset without this feature
            X_train_ablated = np.delete(X_train, i, axis=1)
            X_test_ablated = np.delete(X_test, i, axis=1)
            
            # Train and evaluate
            model_ablated = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            model_ablated.fit(X_train_ablated, y_train)
            
            acc_ablated = accuracy_score(y_test, model_ablated.predict(X_test_ablated))
            f1_ablated = f1_score(y_test, model_ablated.predict(X_test_ablated), average='macro')
            
            impact = {
                'feature_name': feat_name,
                'feature_index': i,
                'accuracy_without': float(acc_ablated),
                'f1_without': float(f1_ablated),
                'accuracy_drop': float(baseline_acc - acc_ablated),
                'f1_drop': float(baseline_f1 - f1_ablated),
                'importance_score': float(baseline_acc - acc_ablated)  # Higher drop = more important
            }
            
            ablation_results['feature_importance'].append(impact)
        
        # Sort by importance
        ablation_results['feature_importance'].sort(
            key=lambda x: x['importance_score'], reverse=True
        )
        
        return ablation_results
    
    def model_component_ablation(self, X_train, y_train, X_test, y_test,
                                model_configs: Dict[str, Dict]) -> Dict:
        """
        Ablation study by modifying model components (e.g., removing regularization)
        """
        results = {}
        
        for config_name, config in model_configs.items():
            model = RandomForestClassifier(**config, random_state=self.random_state)
            model.fit(X_train, y_train)
            
            acc = accuracy_score(y_test, model.predict(X_test))
            f1 = f1_score(y_test, model.predict(X_test), average='macro')
            
            results[config_name] = {
                'accuracy': float(acc),
                'f1_score': float(f1),
                'config': config
            }
        
        return results
    
    def cross_validation_ablation(self, X, y, feature_sets: Dict[str, List[int]]) -> Dict:
        """
        Cross-validation ablation study with different feature sets
        """
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        results = {}
        
        for set_name, feature_indices in feature_sets.items():
            X_subset = X[:, feature_indices]
            
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_subset, y, cv=cv, scoring='accuracy')
            
            results[set_name] = {
                'mean_accuracy': float(cv_scores.mean()),
                'std_accuracy': float(cv_scores.std()),
                'n_features': len(feature_indices),
                'feature_indices': feature_indices
            }
        
        return results
    
    def save_results(self, results: Dict, filepath: str):
        """Save ablation study results"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Ablation study results saved to: {filepath}")

