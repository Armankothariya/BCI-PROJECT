"""
User Profile System
Track multiple users over time with session history
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


class UserProfileManager:
    """
    Manage user profiles and session history
    """
    
    def __init__(self, profiles_dir='user_profiles'):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
    
    def create_user_profile(self, user_id, name, age=None, gender=None, 
                           baseline_emotion='neutral', notes=''):
        """
        Create a new user profile
        
        Args:
            user_id: Unique user identifier
            name: User's name
            age: Optional age
            gender: Optional gender
            baseline_emotion: Baseline emotional state
            notes: Additional notes
        
        Returns:
            User profile dictionary
        """
        profile = {
            'user_id': user_id,
            'name': name,
            'age': age,
            'gender': gender,
            'baseline_emotion': baseline_emotion,
            'notes': notes,
            'created_at': datetime.now().isoformat(),
            'sessions': [],
            'total_predictions': 0,
            'avg_confidence': 0,
            'dominant_emotion': None
        }
        
        # Save profile
        self._save_profile(profile)
        
        return profile
    
    def load_user_profile(self, user_id):
        """Load user profile from disk"""
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        if not profile_path.exists():
            return None
        
        with open(profile_path, 'r') as f:
            return json.load(f)
    
    def _save_profile(self, profile):
        """Save user profile to disk"""
        profile_path = self.profiles_dir / f"{profile['user_id']}.json"
        
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
    
    def add_session(self, user_id, session_data):
        """
        Add a session to user's history
        
        Args:
            user_id: User identifier
            session_data: Dictionary containing session information
        """
        profile = self.load_user_profile(user_id)
        
        if profile is None:
            raise ValueError(f"User {user_id} not found")
        
        # Create session entry
        session = {
            'session_id': len(profile['sessions']) + 1,
            'timestamp': datetime.now().isoformat(),
            'predictions': session_data.get('predictions', []),
            'duration': session_data.get('duration', 0),
            'avg_confidence': session_data.get('avg_confidence', 0),
            'dominant_emotion': session_data.get('dominant_emotion', 'neutral'),
            'emotion_distribution': session_data.get('emotion_distribution', {}),
            'notes': session_data.get('notes', '')
        }
        
        # Add to profile
        profile['sessions'].append(session)
        
        # Update profile statistics
        self._update_profile_stats(profile)
        
        # Save
        self._save_profile(profile)
        
        return session
    
    def _update_profile_stats(self, profile):
        """Update aggregate statistics for profile"""
        if not profile['sessions']:
            return
        
        # Total predictions
        profile['total_predictions'] = sum(
            len(s['predictions']) for s in profile['sessions']
        )
        
        # Average confidence across all sessions
        all_confidences = []
        all_emotions = []
        
        for session in profile['sessions']:
            for pred in session['predictions']:
                all_confidences.append(pred.get('confidence', 0))
                all_emotions.append(pred.get('emotion', 'neutral'))
        
        if all_confidences:
            profile['avg_confidence'] = np.mean(all_confidences)
        
        # Dominant emotion
        if all_emotions:
            emotion_counts = pd.Series(all_emotions).value_counts()
            profile['dominant_emotion'] = emotion_counts.index[0]
    
    def get_all_users(self):
        """Get list of all user profiles"""
        users = []
        
        for profile_path in self.profiles_dir.glob("*.json"):
            with open(profile_path, 'r') as f:
                users.append(json.load(f))
        
        return users
    
    def compare_users(self, user_ids):
        """
        Compare multiple users
        
        Args:
            user_ids: List of user IDs to compare
        
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for user_id in user_ids:
            profile = self.load_user_profile(user_id)
            
            if profile:
                comparison_data.append({
                    'User ID': user_id,
                    'Name': profile['name'],
                    'Sessions': len(profile['sessions']),
                    'Total Predictions': profile['total_predictions'],
                    'Avg Confidence': profile['avg_confidence'],
                    'Dominant Emotion': profile['dominant_emotion'],
                    'Baseline': profile['baseline_emotion']
                })
        
        return pd.DataFrame(comparison_data)
    
    def plot_user_progress(self, user_id):
        """
        Plot user's progress over sessions
        
        Args:
            user_id: User identifier
        
        Returns:
            plotly figure
        """
        profile = self.load_user_profile(user_id)
        
        if not profile or not profile['sessions']:
            return None
        
        # Extract session data
        sessions = []
        for session in profile['sessions']:
            sessions.append({
                'Session': session['session_id'],
                'Date': datetime.fromisoformat(session['timestamp']).strftime('%Y-%m-%d'),
                'Avg Confidence': session['avg_confidence'],
                'Predictions': len(session['predictions']),
                'Dominant Emotion': session['dominant_emotion']
            })
        
        df = pd.DataFrame(sessions)
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Confidence over time
        fig.add_trace(go.Scatter(
            x=df['Session'],
            y=df['Avg Confidence'],
            mode='lines+markers',
            name='Avg Confidence',
            line=dict(color='#3498db', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title=f"Progress for {profile['name']}",
            xaxis_title='Session Number',
            yaxis_title='Average Confidence',
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_emotion_distribution_over_time(self, user_id):
        """
        Plot how emotion distribution changes over sessions
        
        Args:
            user_id: User identifier
        
        Returns:
            plotly figure
        """
        profile = self.load_user_profile(user_id)
        
        if not profile or not profile['sessions']:
            return None
        
        # Extract emotion distributions
        data = []
        
        for session in profile['sessions']:
            session_date = datetime.fromisoformat(session['timestamp']).strftime('%Y-%m-%d')
            
            for emotion, count in session.get('emotion_distribution', {}).items():
                data.append({
                    'Session': session['session_id'],
                    'Date': session_date,
                    'Emotion': emotion.upper(),
                    'Count': count
                })
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        # Create stacked area chart
        fig = px.area(
            df,
            x='Session',
            y='Count',
            color='Emotion',
            title=f"Emotion Distribution Over Time - {profile['name']}",
            labels={'Count': 'Number of Predictions'},
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#95a5a6'
            }
        )
        
        fig.update_layout(height=400)
        
        return fig


class CrossSubjectAnalyzer:
    """
    Analyze model performance across different subjects/users
    """
    
    def __init__(self):
        self.results = {}
    
    def add_subject_results(self, subject_id, y_true, y_pred, y_proba=None):
        """
        Add results for a subject
        
        Args:
            subject_id: Subject identifier
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
        """
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        self.results[subject_id] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n_samples': len(y_true)
        }
    
    def get_generalization_metrics(self):
        """
        Calculate cross-subject generalization metrics
        
        Returns:
            Dictionary with generalization statistics
        """
        if not self.results:
            return None
        
        accuracies = [r['accuracy'] for r in self.results.values()]
        f1_scores = [r['f1'] for r in self.results.values()]
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'mean_f1': np.mean(f1_scores),
            'std_f1': np.std(f1_scores),
            'n_subjects': len(self.results)
        }
    
    def plot_subject_comparison(self):
        """
        Plot performance comparison across subjects
        
        Returns:
            plotly figure
        """
        if not self.results:
            return None
        
        # Prepare data
        subjects = list(self.results.keys())
        metrics_data = []
        
        for subject, results in self.results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                metrics_data.append({
                    'Subject': subject,
                    'Metric': metric.upper(),
                    'Score': results[metric]
                })
        
        df = pd.DataFrame(metrics_data)
        
        # Create grouped bar chart
        fig = px.bar(
            df,
            x='Subject',
            y='Score',
            color='Metric',
            barmode='group',
            title='Cross-Subject Performance Comparison',
            labels={'Score': 'Score Value'},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            height=500,
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        return fig
    
    def plot_generalization_box(self):
        """
        Box plot showing generalization across subjects
        
        Returns:
            plotly figure
        """
        if not self.results:
            return None
        
        data = []
        
        for subject, results in self.results.items():
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                data.append({
                    'Metric': metric.upper(),
                    'Score': results[metric]
                })
        
        df = pd.DataFrame(data)
        
        fig = px.box(
            df,
            x='Metric',
            y='Score',
            title='Model Generalization Across Subjects',
            labels={'Score': 'Score Value'},
            color='Metric',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        
        fig.update_layout(
            height=500,
            yaxis=dict(range=[0, 1]),
            showlegend=False
        )
        
        return fig
