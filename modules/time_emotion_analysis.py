"""
Time-Emotion Correlation Analysis
Analyze emotional changes over time vs EEG activity
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.signal import correlate
from scipy.stats import pearsonr, spearmanr
from datetime import datetime, timedelta


class TimeEmotionAnalyzer:
    """
    Analyze correlation between time and emotional states
    """
    
    def __init__(self):
        self.emotion_colors = {
            'positive': '#2ecc71',
            'negative': '#e74c3c',
            'neutral': '#95a5a6'
        }
    
    def plot_emotion_timeline(self, timestamps, emotions, confidences, 
                             eeg_features=None):
        """
        Plot emotional changes over time with EEG activity
        
        Args:
            timestamps: List of timestamps
            emotions: List of emotions
            confidences: List of confidence scores
            eeg_features: Optional EEG feature values
        
        Returns:
            plotly figure
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'emotion': emotions,
            'confidence': confidences
        })
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add confidence line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#3498db', width=2),
            marker=dict(size=8, color=[self.emotion_colors.get(e, 'gray') for e in emotions]),
            yaxis='y1',
            hovertemplate='<b>%{text}</b><br>Confidence: %{y:.2%}<extra></extra>',
            text=[e.upper() for e in emotions]
        ))
        
        # Add EEG features if provided
        if eeg_features is not None:
            # Normalize EEG features to 0-1 range
            eeg_normalized = (eeg_features - np.min(eeg_features)) / (np.max(eeg_features) - np.min(eeg_features))
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=eeg_normalized,
                mode='lines',
                name='EEG Activity',
                line=dict(color='rgba(155, 89, 182, 0.5)', width=1),
                yaxis='y2',
                hovertemplate='EEG Activity: %{y:.3f}<extra></extra>'
            ))
        
        # Add emotion transition markers
        for i in range(1, len(emotions)):
            if emotions[i] != emotions[i-1]:
                fig.add_vline(
                    x=df['timestamp'].iloc[i],
                    line_dash="dash",
                    line_color=self.emotion_colors.get(emotions[i], 'gray'),
                    opacity=0.3,
                    annotation_text=f"→ {emotions[i].upper()}",
                    annotation_position="top"
                )
        
        # Update layout
        fig.update_layout(
            title='Emotion Timeline with EEG Activity',
            xaxis_title='Time',
            yaxis=dict(
                title='Confidence',
                titlefont=dict(color='#3498db'),
                tickfont=dict(color='#3498db'),
                range=[0, 1]
            ),
            yaxis2=dict(
                title='EEG Activity (Normalized)',
                titlefont=dict(color='#9b59b6'),
                tickfont=dict(color='#9b59b6'),
                overlaying='y',
                side='right',
                range=[0, 1]
            ) if eeg_features is not None else None,
            height=500,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def calculate_emotion_duration_stats(self, timestamps, emotions):
        """
        Calculate statistics about emotion durations
        
        Args:
            timestamps: List of timestamps
            emotions: List of emotions
        
        Returns:
            Dictionary with duration statistics
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'emotion': emotions
        })
        
        # Identify emotion segments
        df['emotion_change'] = (df['emotion'] != df['emotion'].shift()).cumsum()
        
        # Calculate duration for each segment
        durations_by_emotion = {emotion: [] for emotion in df['emotion'].unique()}
        
        for _, group in df.groupby('emotion_change'):
            emotion = group['emotion'].iloc[0]
            duration = (group['timestamp'].iloc[-1] - group['timestamp'].iloc[0]).total_seconds()
            durations_by_emotion[emotion].append(duration)
        
        # Calculate statistics
        stats = {}
        for emotion, durations in durations_by_emotion.items():
            if durations:
                stats[emotion] = {
                    'mean_duration': np.mean(durations),
                    'median_duration': np.median(durations),
                    'std_duration': np.std(durations),
                    'min_duration': np.min(durations),
                    'max_duration': np.max(durations),
                    'total_time': np.sum(durations),
                    'occurrences': len(durations)
                }
        
        return stats
    
    def plot_emotion_duration_distribution(self, duration_stats):
        """
        Plot distribution of emotion durations
        
        Args:
            duration_stats: Output from calculate_emotion_duration_stats
        
        Returns:
            plotly figure
        """
        data = []
        
        for emotion, stats in duration_stats.items():
            data.append({
                'Emotion': emotion.upper(),
                'Mean Duration (s)': stats['mean_duration'],
                'Median Duration (s)': stats['median_duration'],
                'Std Duration (s)': stats['std_duration'],
                'Total Time (s)': stats['total_time'],
                'Occurrences': stats['occurrences']
            })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Emotion'],
            y=df['Mean Duration (s)'],
            name='Mean Duration',
            marker_color=[self.emotion_colors.get(e.lower(), 'gray') for e in df['Emotion']],
            error_y=dict(
                type='data',
                array=df['Std Duration (s)']
            )
        ))
        
        fig.update_layout(
            title='Average Emotion Duration',
            xaxis_title='Emotion',
            yaxis_title='Duration (seconds)',
            height=400,
            showlegend=False
        )
        
        return fig
    
    def calculate_emotion_transition_matrix(self, emotions):
        """
        Calculate transition probability matrix
        
        Args:
            emotions: List of emotions
        
        Returns:
            Transition matrix as DataFrame
        """
        unique_emotions = sorted(set(emotions))
        
        # Initialize matrix
        matrix = pd.DataFrame(
            0,
            index=unique_emotions,
            columns=unique_emotions
        )
        
        # Count transitions
        for i in range(len(emotions) - 1):
            from_emotion = emotions[i]
            to_emotion = emotions[i + 1]
            matrix.loc[from_emotion, to_emotion] += 1
        
        # Convert to probabilities
        row_sums = matrix.sum(axis=1)
        matrix = matrix.div(row_sums, axis=0).fillna(0)
        
        return matrix
    
    def plot_transition_heatmap(self, transition_matrix):
        """
        Plot emotion transition probability heatmap
        
        Args:
            transition_matrix: Output from calculate_emotion_transition_matrix
        
        Returns:
            plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=transition_matrix.values,
            x=[e.upper() for e in transition_matrix.columns],
            y=[e.upper() for e in transition_matrix.index],
            colorscale='Blues',
            text=transition_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 12},
            colorbar=dict(title='Probability')
        ))
        
        fig.update_layout(
            title='Emotion Transition Probability Matrix',
            xaxis_title='To Emotion',
            yaxis_title='From Emotion',
            height=500
        )
        
        return fig
    
    def correlate_emotion_with_time_of_day(self, timestamps, emotions):
        """
        Analyze if certain emotions occur more at specific times of day
        
        Args:
            timestamps: List of timestamps
            emotions: List of emotions
        
        Returns:
            Dictionary with time-of-day analysis
        """
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(timestamps),
            'emotion': emotions
        })
        
        # Extract hour of day
        df['hour'] = df['timestamp'].dt.hour
        
        # Count emotions by hour
        emotion_by_hour = df.groupby(['hour', 'emotion']).size().unstack(fill_value=0)
        
        # Calculate percentages
        emotion_by_hour_pct = emotion_by_hour.div(emotion_by_hour.sum(axis=1), axis=0)
        
        return {
            'counts': emotion_by_hour,
            'percentages': emotion_by_hour_pct
        }
    
    def plot_emotion_by_time_of_day(self, time_analysis):
        """
        Plot emotion distribution by time of day
        
        Args:
            time_analysis: Output from correlate_emotion_with_time_of_day
        
        Returns:
            plotly figure
        """
        df = time_analysis['percentages'].reset_index()
        df_melted = df.melt(id_vars='hour', var_name='emotion', value_name='percentage')
        
        fig = px.area(
            df_melted,
            x='hour',
            y='percentage',
            color='emotion',
            title='Emotion Distribution by Time of Day',
            labels={'hour': 'Hour of Day', 'percentage': 'Percentage'},
            color_discrete_map={
                'positive': '#2ecc71',
                'negative': '#e74c3c',
                'neutral': '#95a5a6'
            }
        )
        
        fig.update_layout(
            height=400,
            xaxis=dict(tickmode='linear', tick0=0, dtick=2),
            yaxis=dict(tickformat='.0%')
        )
        
        return fig
    
    def calculate_emotion_autocorrelation(self, emotions, max_lag=20):
        """
        Calculate autocorrelation of emotions (how predictable are transitions)
        
        Args:
            emotions: List of emotions
            max_lag: Maximum lag to calculate
        
        Returns:
            Dictionary with autocorrelation values
        """
        # Convert emotions to numeric
        emotion_to_num = {e: i for i, e in enumerate(sorted(set(emotions)))}
        emotion_numeric = [emotion_to_num[e] for e in emotions]
        
        # Calculate autocorrelation
        autocorr = []
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr, _ = pearsonr(
                    emotion_numeric[:-lag],
                    emotion_numeric[lag:]
                )
                autocorr.append(corr)
        
        return {
            'lags': list(range(max_lag + 1)),
            'autocorrelation': autocorr
        }
    
    def plot_autocorrelation(self, autocorr_results):
        """
        Plot emotion autocorrelation
        
        Args:
            autocorr_results: Output from calculate_emotion_autocorrelation
        
        Returns:
            plotly figure
        """
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=autocorr_results['lags'],
            y=autocorr_results['autocorrelation'],
            marker_color='#3498db'
        ))
        
        # Add significance lines
        n = len(autocorr_results['lags'])
        significance = 1.96 / np.sqrt(n)  # 95% confidence interval
        
        fig.add_hline(y=significance, line_dash="dash", line_color="red", 
                     annotation_text="95% CI")
        fig.add_hline(y=-significance, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Emotion Autocorrelation (Predictability)',
            xaxis_title='Lag (samples)',
            yaxis_title='Autocorrelation',
            height=400
        )
        
        return fig
