# modules/visualization.py
# ============================================
# Enhanced Visualization for Scholarship-Grade Application
# ============================================

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import streamlit as st

def create_emotion_orb(emotion: str, confidence: float, size: int = 200) -> str:
    """Create enhanced emotion visualization orb with research context"""
    colors = {
        'positive': '#45B7D1', 'negative': '#FF6B6B', 'neutral': '#4ECDC4',
        'happy': '#45B7D1', 'sad': '#FF6B6B', 'angry': '#FF8E53', 'relaxed': '#4ECDC4'
    }
    
    icons = {
        'positive': '😊', 'negative': '😔', 'neutral': '😐',
        'happy': '😊', 'sad': '😢', 'angry': '😠', 'relaxed': '😌'
    }
    
    color = colors.get(emotion, '#666666')
    icon = icons.get(emotion, '🧠')
    
    # Confidence-based styling
    confidence_level = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
    pulse_intensity = "2s" if confidence_level == "high" else "3s" if confidence_level == "medium" else "4s"
    
    return f"""
    <div style='
        width: {size}px;
        height: {size}px;
        border-radius: 50%;
        background: radial-gradient(circle, {color}80, {color}20);
        border: 6px solid {color};
        margin: 20px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 4rem;
        color: {color};
        box-shadow: 0 0 40px {color}60;
        animation: pulse {pulse_intensity} infinite;
        position: relative;
    '>
        {icon}
        <div style='
            position: absolute;
            bottom: -30px;
            left: 0;
            right: 0;
            text-align: center;
            font-size: 1rem;
            color: {color};
            font-weight: bold;
        '>
            {confidence:.1%}
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); box-shadow: 0 0 40px {color}60; }}
        50% {{ transform: scale(1.05); box-shadow: 0 0 60px {color}80; }}
        100% {{ transform: scale(1); box-shadow: 0 0 40px {color}60; }}
    }}
    </style>
    """

def plot_feature_values(features: np.ndarray, title: str = "Feature Values") -> go.Figure:
    """Enhanced feature visualization with research context"""
    fig = go.Figure()
    
    # Color based on feature values
    colors = ['#2E86AB' if x >= 0 else '#FF6B6B' for x in features]
    
    fig.add_trace(go.Bar(
        y=features,
        x=[f'F{i+1}' for i in range(len(features))],
        marker_color=colors,
        hovertemplate=(
            "<b>Feature %{x}</b><br>" +
            "Value: %{y:.3f}<br>" +
            "<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="Feature Index",
        yaxis_title="Feature Value",
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif")
    )
    
    # Add zero line for reference
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def plot_confidence_timeline(history_df: pd.DataFrame) -> Optional[go.Figure]:
    """Enhanced confidence timeline with research context"""
    if len(history_df) < 2 or 'timestamp' not in history_df.columns:
        return None
    
    try:
        # Ensure timestamp is datetime
        history_df = history_df.copy()
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        fig = go.Figure()
        
        # Plot confidence with emotion coloring
        for emotion in history_df['emotion'].unique():
            emotion_data = history_df[history_df['emotion'] == emotion]
            
            fig.add_trace(go.Scatter(
                x=emotion_data['timestamp'],
                y=emotion_data['confidence'],
                mode='lines+markers',
                name=emotion.title(),
                line=dict(width=3),
                marker=dict(size=6),
                hovertemplate=(
                    "<b>%{x}</b><br>" +
                    "Emotion: " + emotion.title() + "<br>" +
                    "Confidence: %{y:.1%}<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title=dict(
                text="Confidence Timeline with Emotion Transitions",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Time",
            yaxis_title="Confidence",
            yaxis_tickformat=".0%",
            height=400,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating confidence timeline: {e}")
        return None

def plot_emotion_distribution(history_df: pd.DataFrame) -> Optional[go.Figure]:
    """Plot emotion distribution with enhanced research styling"""
    if len(history_df) == 0 or 'emotion' not in history_df.columns:
        return None
    
    try:
        emotion_counts = history_df['emotion'].value_counts()
        
        # Color mapping for emotions
        color_map = {
            'positive': '#45B7D1', 'negative': '#FF6B6B', 'neutral': '#4ECDC4',
            'happy': '#45B7D1', 'sad': '#FF6B6B', 'angry': '#FF8E53', 'relaxed': '#4ECDC4'
        }
        
        colors = [color_map.get(emotion, '#666666') for emotion in emotion_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=emotion_counts.index,
            values=emotion_counts.values,
            hole=0.4,
            marker=dict(colors=colors),
            hovertemplate=(
                "<b>%{label}</b><br>" +
                "Count: %{value}<br>" +
                "Percentage: %{percent}<br>" +
                "<extra></extra>"
            )
        )])
        
        fig.update_layout(
            title=dict(
                text="Emotion Distribution Analysis",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=400,
            showlegend=True,
            annotations=[dict(
                text=f"Total: {len(history_df)}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )]
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating emotion distribution: {e}")
        return None

def plot_emotion_radar(probabilities: Dict[str, float], title: str = "Emotion Probability Radar") -> go.Figure:
    """Create radar chart for emotion probabilities with research styling"""
    if not probabilities:
        # Return empty radar chart
        fig = go.Figure()
        fig.update_layout(
            title=title,
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=False,
            height=300
        )
        return fig
    
    emotions = list(probabilities.keys())
    values = list(probabilities.values())
    
    # Close the radar chart
    emotions.append(emotions[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=emotions,
        fill='toself',
        fillcolor='rgba(46, 134, 171, 0.6)',
        line=dict(color='#2E86AB', width=2),
        name='Current Emotion'
    ))
    
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=14)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickformat='.0%',
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray',
                linecolor='lightgray'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_session_summary_card(history_df: pd.DataFrame, title: str = "Session Summary") -> go.Figure:
    """Create a comprehensive session summary visualization"""
    if len(history_df) == 0:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No session data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=title,
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        return fig
    
    try:
        # Calculate session metrics
        total_predictions = len(history_df)
        avg_confidence = history_df['confidence'].mean() if 'confidence' in history_df.columns else 0
        emotion_changes = sum(1 for i in range(1, len(history_df)) 
                          if history_df.iloc[i]['emotion'] != history_df.iloc[i-1]['emotion'])
        
        # Create subplot figure
        fig = go.Figure()
        
        # Add metrics as annotations
        annotations = [
            dict(
                x=0.1, y=0.9,
                xref='paper', yref='paper',
                text=f"Total Predictions: {total_predictions}",
                showarrow=False,
                font=dict(size=14, color='#2E86AB')
            ),
            dict(
                x=0.1, y=0.7,
                xref='paper', yref='paper',
                text=f"Avg Confidence: {avg_confidence:.1%}",
                showarrow=False,
                font=dict(size=14, color='#45B7D1')
            ),
            dict(
                x=0.1, y=0.5,
                xref='paper', yref='paper',
                text=f"Emotion Changes: {emotion_changes}",
                showarrow=False,
                font=dict(size=14, color='#FF6B6B')
            ),
            dict(
                x=0.1, y=0.3,
                xref='paper', yref='paper',
                text=f"Session Duration: {calculate_session_duration(history_df)}",
                showarrow=False,
                font=dict(size=14, color='#4ECDC4')
            )
        ]
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            height=250,
            paper_bgcolor='rgba(248,249,250,0.8)',
            plot_bgcolor='rgba(248,249,250,0.8)',
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=annotations,
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        )
        
        # Add a subtle border
        fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=0, y0=0, x1=1, y1=1,
            line=dict(color="#dee2e6", width=2),
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating session summary: {e}")
        # Fallback simple summary
        return create_simple_session_summary(history_df, title)

def calculate_session_duration(history_df: pd.DataFrame) -> str:
    """Calculate and format session duration"""
    if 'timestamp' not in history_df.columns or len(history_df) < 2:
        return "N/A"
    
    try:
        history_df = history_df.copy()
        history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
        
        start_time = history_df['timestamp'].min()
        end_time = history_df['timestamp'].max()
        duration = end_time - start_time
        
        total_seconds = duration.total_seconds()
        
        if total_seconds < 60:
            return f"{int(total_seconds)}s"
        elif total_seconds < 3600:
            return f"{int(total_seconds // 60)}m {int(total_seconds % 60)}s"
        else:
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
            
    except:
        return "N/A"

def create_simple_session_summary(history_df: pd.DataFrame, title: str) -> go.Figure:
    """Create a simple session summary as fallback"""
    fig = go.Figure()
    
    total_predictions = len(history_df)
    
    fig.add_annotation(
        text=f"Session Summary<br>Total Predictions: {total_predictions}",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14),
        align="center"
    )
    
    fig.update_layout(
        title=title,
        height=200,
        paper_bgcolor='rgba(248,249,250,0.8)',
        plot_bgcolor='rgba(248,249,250,0.8)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    
    return fig

def plot_research_metrics(metrics: Dict, title: str = "Research Metrics") -> go.Figure:
    """Create comprehensive research metrics visualization"""
    if not metrics:
        return create_empty_metrics_plot(title)
    
    try:
        # Extract primary metrics
        primary_metrics = metrics.get('primary_metrics', {})
        confidence_intervals = metrics.get('confidence_intervals', {})
        
        # Create subplots
        fig = go.Figure()
        
        # Primary metrics bars
        metric_names = list(primary_metrics.keys())
        metric_values = list(primary_metrics.values())
        
        fig.add_trace(go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=['#2E86AB', '#45B7D1', '#4ECDC4', '#FFD93D'],
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Value: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add confidence intervals if available
        if 'accuracy' in confidence_intervals:
            ci = confidence_intervals['accuracy']
            fig.add_annotation(
                x=metric_names.index('accuracy'),
                y=ci['mean'],
                text=f"CI: [{ci['confidence_interval'][0]:.3f}, {ci['confidence_interval'][1]:.3f}]",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#FF6B6B',
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#FF6B6B',
                borderwidth=1
            )
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Metrics",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating research metrics plot: {e}")
        return create_empty_metrics_plot(title)

def create_empty_metrics_plot(title: str) -> go.Figure:
    """Create empty metrics plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No research metrics available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title=title,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    return fig

def plot_error_analysis(error_data: Dict, title: str = "Error Analysis") -> go.Figure:
    """Create error analysis visualization"""
    if not error_data:
        return create_empty_error_plot(title)
    
    try:
        fig = go.Figure()
        
        # Extract error types and counts
        error_types = list(error_data.keys())
        error_counts = [error_data[error_type].get('count', 0) for error_type in error_types]
        
        fig.add_trace(go.Bar(
            x=error_types,
            y=error_counts,
            marker_color='#FF6B6B',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Count: %{y}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Error Type",
            yaxis_title="Count",
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating error analysis plot: {e}")
        return create_empty_error_plot(title)

def create_empty_error_plot(title: str) -> go.Figure:
    """Create empty error plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No error data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title=title,
        height=200,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    return fig

def create_system_status_visualization(system_status: Dict) -> go.Figure:
    """Create system status visualization"""
    fig = go.Figure()
    
    # Create gauge-like indicators for system components
    components = list(system_status.keys())
    status_values = [system_status[comp].get('status', 0) for comp in components]
    
    # Convert status to numerical values (0-1)
    status_numeric = []
    for status in status_values:
        if status == 'healthy': status_numeric.append(1.0)
        elif status == 'degraded': status_numeric.append(0.5)
        else: status_numeric.append(0.1)  # error
    
    fig.add_trace(go.Bar(
        x=components,
        y=status_numeric,
        marker_color=['#4ECDC4' if val > 0.7 else '#FFD93D' if val > 0.3 else '#FF6B6B' 
                     for val in status_numeric],
        hovertemplate=(
            "<b>%{x}</b><br>" +
            "Status: %{customdata}<br>" +
            "<extra></extra>"
        ),
        customdata=status_values
    ))
    
    fig.update_layout(
        title=dict(
            text="System Status Overview",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        xaxis_title="System Components",
        yaxis_title="Status Level",
        yaxis_range=[0, 1],
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_realtime_eeg_visualization(eeg_data: np.ndarray, channels: List[str] = None) -> go.Figure:
    """Create real-time EEG visualization (simulated for demo)"""
    if eeg_data is None or len(eeg_data) == 0:
        return create_empty_eeg_plot()
    
    try:
        fig = go.Figure()
        
        # Simulate EEG channels if not provided
        if channels is None:
            channels = [f'Channel {i+1}' for i in range(min(8, len(eeg_data)))]
        
        # Plot first few channels for clarity
        num_channels = min(4, len(eeg_data), len(channels))
        
        for i in range(num_channels):
            # Add some simulated time variation for demo
            time_points = np.linspace(0, 2 * np.pi, len(eeg_data))
            signal = eeg_data[i] * np.sin(time_points + i * 0.5) + np.random.normal(0, 0.1, len(eeg_data))
            
            fig.add_trace(go.Scatter(
                x=list(range(len(signal))),
                y=signal,
                mode='lines',
                name=channels[i],
                line=dict(width=2),
                hovertemplate=(
                    f"<b>{channels[i]}</b><br>" +
                    "Time: %{x}<br>" +
                    "Amplitude: %{y:.3f} µV<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title=dict(
                text="EEG Signal Visualization (Simulated)",
                x=0.5,
                xanchor='center',
                font=dict(size=14)
            ),
            xaxis_title="Time (samples)",
            yaxis_title="Amplitude (µV)",
            height=350,
            showlegend=True,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating EEG visualization: {e}")
        return create_empty_eeg_plot()

def create_empty_eeg_plot() -> go.Figure:
    """Create empty EEG plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No EEG data available for visualization",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title="EEG Signal Visualization",
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    return fig

def create_comparison_visualization(metrics_dict: Dict[str, Dict], title: str = "Model Comparison") -> go.Figure:
    """Create model comparison visualization"""
    if not metrics_dict:
        return create_empty_comparison_plot(title)
    
    try:
        fig = go.Figure()
        
        models = list(metrics_dict.keys())
        accuracy_scores = [metrics_dict[model].get('accuracy', 0) for model in models]
        f1_scores = [metrics_dict[model].get('f1_score', 0) for model in models]
        
        # Add accuracy bars
        fig.add_trace(go.Bar(
            name='Accuracy',
            x=models,
            y=accuracy_scores,
            marker_color='#2E86AB',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "Accuracy: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))
        
        # Add F1-score bars
        fig.add_trace(go.Bar(
            name='F1-Score',
            x=models,
            y=f1_scores,
            marker_color='#4ECDC4',
            hovertemplate=(
                "<b>%{x}</b><br>" +
                "F1-Score: %{y:.3f}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="Models",
            yaxis_title="Score",
            yaxis_range=[0, 1],
            height=400,
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating comparison visualization: {e}")
        return create_empty_comparison_plot(title)

def create_empty_comparison_plot(title: str) -> go.Figure:
    """Create empty comparison plot with message"""
    fig = go.Figure()
    fig.add_annotation(
        text="No comparison data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14)
    )
    fig.update_layout(
        title=title,
        height=300,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )
    return fig

# Utility function for Streamlit display
def display_visualization(fig: go.Figure, use_container_width: bool = True):
    """Display Plotly figure in Streamlit with error handling"""
    try:
        if fig is None:
            st.warning("No visualization data available")
            return
        
        st.plotly_chart(fig, use_container_width=use_container_width)
        
    except Exception as e:
        st.error(f"Error displaying visualization: {e}")
        st.info("Please check the data format and try again.")

# Color scheme constants for consistent styling
COLOR_SCHEME = {
    'primary': '#2E86AB',
    'secondary': '#45B7D1', 
    'success': '#4ECDC4',
    'warning': '#FFD93D',
    'error': '#FF6B6B',
    'neutral': '#A0A0A0'
}

# Emotion color mapping
EMOTION_COLORS = {
    'positive': '#45B7D1',
    'negative': '#FF6B6B', 
    'neutral': '#4ECDC4',
    'happy': '#45B7D1',
    'sad': '#FF6B6B',
    'angry': '#FF8E53',
    'relaxed': '#4ECDC4'
}