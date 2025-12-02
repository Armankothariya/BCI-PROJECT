"""
Brain Topography and Advanced EEG Visualization
Publication-ready brain mapping and frequency analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.interpolate import griddata
import mne
from mne.channels import make_standard_montage
from pathlib import Path


class BrainTopographyVisualizer:
    """
    Create publication-ready brain topography maps
    """
    
    def __init__(self, channel_names=None):
        """
        Initialize with standard 10-20 electrode positions
        """
        if channel_names is None:
            # Standard 10-20 system channels (18 channels - reduced to match typical datasets)
            self.channel_names = [
                'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                'T7', 'C3', 'Cz', 'C4', 'T8',
                'P7', 'P3', 'Pz', 'P4', 'P8',
                'O1', 'O2'
            ]
        else:
            self.channel_names = channel_names
        
        # Get standard montage
        self.montage = make_standard_montage('standard_1020')
        
        # Filter channel names to only include those available in the montage
        available_ch_names = self.montage.ch_names
        # Map old channel names to new standard names if needed
        name_mapping = {
            'T3': 'T7',  # T3 is old nomenclature, T7 is standard
            'T4': 'T8',  # T4 is old nomenclature, T8 is standard
            'T5': 'P7',  # T5 is old nomenclature, P7 is standard
            'T6': 'P8'   # T6 is old nomenclature, P8 is standard
        }
        
        # Apply name mapping
        mapped_channels = [name_mapping.get(ch, ch) for ch in self.channel_names]
        
        # Only keep channels that exist in the montage
        valid_channels = [ch for ch in mapped_channels if ch in available_ch_names]
        
        if len(valid_channels) != len(self.channel_names):
            print(f"[WARNING] Some channels not found in montage. Using {len(valid_channels)}/{len(self.channel_names)} channels.")
            self.channel_names = valid_channels
        
        # Create info object with valid channels only
        self.info = mne.create_info(
            ch_names=self.channel_names,
            sfreq=250,  # Standard EEG sampling rate
            ch_types='eeg'
        )
        
        # Set montage - this should now work without errors
        try:
            self.info.set_montage(self.montage)
        except Exception as e:
            print(f"[WARNING] Could not set montage: {e}. Trying with subset selection.")
            # Try picking channels from montage
            self.info.set_montage(self.montage, match_case=False, on_missing='warn')
    
    def plot_emotion_topography(self, feature_values, emotion, title=None):
        """
        Plot brain topography for a specific emotion
        
        Args:
            feature_values: Array of values for each channel
            emotion: Emotion label (for coloring)
            title: Plot title
        
        Returns:
            matplotlib figure
        """
        # Convert to numpy array
        feature_values = np.asarray(feature_values).flatten()
        
        # Ensure info and data have same number of channels
        n_channels_available = len(self.channel_names)
        n_features = len(feature_values)
        
        # Handle mismatch: pad with zeros or truncate
        if n_features < n_channels_available:
            # Pad with zeros if we have fewer features
            vals = np.zeros(n_channels_available)
            vals[:n_features] = feature_values
            print(f"[INFO] Padded {n_features} features to {n_channels_available} channels for topography")
        elif n_features > n_channels_available:
            # Truncate if we have more features
            vals = feature_values[:n_channels_available]
            print(f"[INFO] Truncated {n_features} features to {n_channels_available} channels for topography")
        else:
            vals = feature_values
        
        # Ensure vals matches exactly the number of channels in info
        assert len(vals) == len(self.info.ch_names), \
            f"Mismatch: {len(vals)} values vs {len(self.info.ch_names)} channels"
        
        # Color scheme based on emotion
        emotion_cmaps = {
            'positive': 'RdYlGn',
            'negative': 'RdYlBu_r',
            'neutral': 'viridis'
        }
        cmap = emotion_cmaps.get(emotion.lower() if emotion else 'neutral', 'RdBu_r')
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot topography - use info directly, no show_names parameter
        try:
            im, cn = mne.viz.plot_topomap(
                vals,
                self.info,
                axes=ax,
                show=False,
                cmap=cmap,
                contours=6,
                sensors=True
            )
        except Exception as e:
            # Fallback: try with fewer parameters
            print(f"[WARNING] Topography plotting failed: {e}. Trying alternative method.")
            try:
                im, cn = mne.viz.plot_topomap(
                    vals,
                    self.info,
                    show=False,
                    cmap=cmap,
                    contours=6
                )
                ax = plt.gca()
            except Exception as e2:
                print(f"[ERROR] Could not plot topography: {e2}")
                # Create a simple placeholder plot
                ax.text(0.5, 0.5, f"Topography unavailable\n({e2})", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
        
        # Add colorbar
        if 'im' in locals():
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Activation Level', rotation=270, labelpad=20)
        
        # Set title
        if title is None:
            title = f"Brain Topography - {emotion.upper() if emotion else 'Unknown'} Emotion"
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_frequency_band_topography(self, eeg_data, sfreq=250):
        """
        Plot topography for each frequency band (δ, θ, α, β, γ)
        
        Args:
            eeg_data: EEG data (channels x time)
            sfreq: Sampling frequency
        
        Returns:
            matplotlib figure with subplots
        """
        # Define frequency bands
        bands = {
            'Delta (δ)': (0.5, 4),
            'Theta (θ)': (4, 8),
            'Alpha (α)': (8, 13),
            'Beta (β)': (13, 30),
            'Gamma (γ)': (30, 50)
        }
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Calculate power for each band
        for idx, (band_name, (low_freq, high_freq)) in enumerate(bands.items()):
            # Bandpass filter
            filtered = self._bandpass_filter(eeg_data, low_freq, high_freq, sfreq)
            
            # Calculate power (mean squared amplitude)
            power = np.mean(filtered ** 2, axis=1)
            
            # Normalize
            power = (power - power.min()) / (power.max() - power.min() + 1e-10)
            
            # Plot topography
            im, cn = mne.viz.plot_topomap(
                power,
                self.info,
                axes=axes[idx],
                show=False,
                cmap='hot',
                contours=6,
                sensors=True
            )
            
            axes[idx].set_title(band_name, fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[idx])
            cbar.set_label('Power', rotation=270, labelpad=15)
        
        # Remove extra subplot
        fig.delaxes(axes[-1])
        
        fig.suptitle('Frequency Band Power Topography', fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def _bandpass_filter(self, data, low_freq, high_freq, sfreq):
        """Apply bandpass filter to data"""
        nyquist = sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)
        
        return filtered
    
    def plot_feature_importance_brain_map(self, feature_importance, feature_names):
        """
        Map feature importance to brain regions
        
        Args:
            feature_importance: Array of importance values
            feature_names: List of feature names
        
        Returns:
            matplotlib figure
        """
        # Convert to numpy arrays
        feature_importance = np.asarray(feature_importance).flatten()
        feature_names = list(feature_names) if feature_names else [f"feat_{i}" for i in range(len(feature_importance))]
        
        # Extract channel-based features
        channel_importance = {}
        
        # Standard channel name mappings (handle various naming conventions)
        channel_aliases = {
            'Fp1': ['fp1', 'f_p1', 'f1'],
            'Fp2': ['fp2', 'f_p2', 'f2'],
            'F7': ['f7'],
            'F3': ['f3'],
            'Fz': ['fz', 'f_z'],
            'F4': ['f4'],
            'F8': ['f8'],
            'T7': ['t7', 't3'],
            'C3': ['c3'],
            'Cz': ['cz', 'c_z'],
            'C4': ['c4'],
            'T8': ['t8', 't4'],
            'P7': ['p7', 't5'],
            'P3': ['p3'],
            'Pz': ['pz', 'p_z'],
            'P4': ['p4'],
            'P8': ['p8', 't6'],
            'O1': ['o1'],
            'O2': ['o2']
        }
        
        for channel in self.channel_names:
            # Find all features related to this channel
            channel_features = []
            aliases = channel_aliases.get(channel, [channel.lower()])
            
            for name, imp in zip(feature_names, feature_importance):
                name_lower = str(name).lower()
                # Check if any alias matches
                if any(alias in name_lower for alias in aliases):
                    channel_features.append(imp)
            
            if channel_features:
                # Average importance for this channel
                avg_importance = np.mean(channel_features)
                channel_importance[channel] = avg_importance
            else:
                channel_importance[channel] = 0
        
        # Create importance array matching channel order
        importance_values = np.array([
            channel_importance.get(ch, 0) for ch in self.channel_names
        ])
        
        # Normalize
        if importance_values.max() > 0:
            importance_values = importance_values / importance_values.max()
        
        # Ensure length matches
        if len(importance_values) != len(self.info.ch_names):
            # Pad or truncate
            if len(importance_values) < len(self.info.ch_names):
                padded = np.zeros(len(self.info.ch_names))
                padded[:len(importance_values)] = importance_values
                importance_values = padded
            else:
                importance_values = importance_values[:len(self.info.ch_names)]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot topography - REMOVED show_names parameter (doesn't exist in MNE)
        try:
            im, cn = mne.viz.plot_topomap(
                importance_values,
                self.info,
                axes=ax,
                show=False,
                cmap='YlOrRd',
                contours=8,
                sensors=True
            )
        except Exception as e:
            print(f"[WARNING] Feature importance topography failed: {e}")
            # Fallback
            try:
                im, cn = mne.viz.plot_topomap(
                    importance_values,
                    self.info,
                    show=False,
                    cmap='YlOrRd',
                    contours=8
                )
                ax = plt.gca()
            except Exception as e2:
                print(f"[ERROR] Could not plot feature importance topography: {e2}")
                ax.text(0.5, 0.5, f"Feature importance map unavailable\n({e2})", 
                       ha='center', va='center', transform=ax.transAxes)
                return fig
        
        # Add colorbar
        if 'im' in locals():
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Feature Importance', rotation=270, labelpad=20)
        
        ax.set_title('Feature Importance Mapped to Brain Regions', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        return fig


class FrequencyBandAnalyzer:
    """
    Analyze EEG frequency bands over time and per emotion
    """
    
    def __init__(self, sfreq=250):
        self.sfreq = sfreq
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 50)
        }
    
    def calculate_band_powers(self, eeg_data):
        """
        Calculate power in each frequency band
        
        Args:
            eeg_data: EEG data (channels x time)
        
        Returns:
            Dictionary of band powers
        """
        band_powers = {}
        
        for band_name, (low_freq, high_freq) in self.bands.items():
            # Bandpass filter
            filtered = self._bandpass_filter(eeg_data, low_freq, high_freq)
            
            # Calculate power (mean squared amplitude)
            power = np.mean(filtered ** 2)
            band_powers[band_name] = power
        
        return band_powers
    
    def _bandpass_filter(self, data, low_freq, high_freq):
        """Apply bandpass filter"""
        nyquist = self.sfreq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=1)
        
        return filtered
    
    def plot_band_powers_over_time(self, eeg_segments, timestamps, emotions=None):
        """
        Plot frequency band powers over time
        
        Args:
            eeg_segments: List of EEG data segments
            timestamps: Corresponding timestamps
            emotions: Optional emotion labels
        
        Returns:
            plotly figure
        """
        # Calculate band powers for each segment
        band_power_timeline = {band: [] for band in self.bands.keys()}
        
        for segment in eeg_segments:
            powers = self.calculate_band_powers(segment)
            for band, power in powers.items():
                band_power_timeline[band].append(power)
        
        # Create figure
        fig = go.Figure()
        
        # Add trace for each band
        colors = {
            'Delta': '#1f77b4',
            'Theta': '#ff7f0e',
            'Alpha': '#2ca02c',
            'Beta': '#d62728',
            'Gamma': '#9467bd'
        }
        
        for band, powers in band_power_timeline.items():
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=powers,
                mode='lines+markers',
                name=f'{band} ({self.bands[band][0]}-{self.bands[band][1]} Hz)',
                line=dict(color=colors[band], width=2),
                marker=dict(size=6)
            ))
        
        # Add emotion markers if provided
        if emotions is not None:
            emotion_colors = {
                'positive': 'green',
                'negative': 'red',
                'neutral': 'gray'
            }
            
            for i, emotion in enumerate(emotions):
                if i > 0 and emotion != emotions[i-1]:
                    fig.add_vline(
                        x=timestamps[i],
                        line_dash="dash",
                        line_color=emotion_colors.get(emotion, 'black'),
                        annotation_text=emotion.upper(),
                        annotation_position="top"
                    )
        
        fig.update_layout(
            title='Frequency Band Powers Over Time',
            xaxis_title='Time',
            yaxis_title='Power (μV²)',
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        return fig
    
    def plot_band_powers_by_emotion(self, band_powers_by_emotion):
        """
        Compare band powers across emotions
        
        Args:
            band_powers_by_emotion: Dict with structure {emotion: {band: [powers]}}
        
        Returns:
            plotly figure
        """
        # Prepare data for plotting
        data = []
        
        for emotion, band_powers in band_powers_by_emotion.items():
            for band, powers in band_powers.items():
                for power in powers:
                    data.append({
                        'Emotion': emotion.upper(),
                        'Band': band,
                        'Power': power
                    })
        
        df = pd.DataFrame(data)
        
        # Create grouped bar chart
        fig = px.box(
            df,
            x='Band',
            y='Power',
            color='Emotion',
            title='Frequency Band Powers by Emotion',
            labels={'Power': 'Power (μV²)'},
            color_discrete_map={
                'POSITIVE': '#2ecc71',
                'NEGATIVE': '#e74c3c',
                'NEUTRAL': '#95a5a6'
            }
        )
        
        fig.update_layout(
            height=500,
            xaxis_title='Frequency Band',
            yaxis_title='Power (μV²)',
            legend_title='Emotion'
        )
        
        return fig


class LiveEEGDisplay:
    """
    Real-time EEG waveform visualization
    """
    
    def __init__(self, channel_names, sfreq=250, window_size=5):
        """
        Args:
            channel_names: List of channel names
            sfreq: Sampling frequency
            window_size: Display window in seconds
        """
        self.channel_names = channel_names
        self.sfreq = sfreq
        self.window_size = window_size
        self.n_samples = int(sfreq * window_size)
    
    def plot_live_eeg(self, eeg_data, highlight_channels=None):
        """
        Create live EEG display with multiple channels
        
        Args:
            eeg_data: EEG data (channels x time)
            highlight_channels: List of channels to highlight
        
        Returns:
            plotly figure
        """
        n_channels = min(len(self.channel_names), eeg_data.shape[0])
        time_axis = np.arange(eeg_data.shape[1]) / self.sfreq
        
        # Create figure with subplots
        fig = go.Figure()
        
        # Plot each channel with offset
        offset_scale = 100  # Vertical spacing between channels
        
        for i in range(n_channels):
            channel_data = eeg_data[i, :] + (n_channels - i - 1) * offset_scale
            
            # Determine if this channel should be highlighted
            is_highlighted = (highlight_channels is not None and 
                            self.channel_names[i] in highlight_channels)
            
            fig.add_trace(go.Scatter(
                x=time_axis,
                y=channel_data,
                mode='lines',
                name=self.channel_names[i],
                line=dict(
                    width=2 if is_highlighted else 1,
                    color='red' if is_highlighted else None
                ),
                hovertemplate=f'<b>{self.channel_names[i]}</b><br>' +
                             'Time: %{x:.3f}s<br>' +
                             'Amplitude: %{y:.2f}μV<extra></extra>'
            ))
        
        # Update layout
        fig.update_layout(
            title='Live EEG Signals - Multi-Channel Display',
            xaxis_title='Time (seconds)',
            yaxis_title='Channels',
            height=800,
            hovermode='x unified',
            yaxis=dict(
                ticktext=self.channel_names[:n_channels][::-1],
                tickvals=[i * offset_scale for i in range(n_channels)],
                showgrid=True,
                gridcolor='lightgray'
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor='lightgray'
            ),
            plot_bgcolor='white'
        )
        
        return fig
    
    def plot_spectrogram(self, eeg_data, channel_idx=0):
        """
        Create spectrogram for a single channel
        
        Args:
            eeg_data: EEG data (channels x time)
            channel_idx: Channel index to plot
        
        Returns:
            plotly figure
        """
        # Calculate spectrogram
        f, t, Sxx = signal.spectrogram(
            eeg_data[channel_idx, :],
            fs=self.sfreq,
            nperseg=256,
            noverlap=128
        )
        
        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=Sxx_db,
            x=t,
            y=f,
            colorscale='Jet',
            colorbar=dict(title='Power (dB)')
        ))
        
        fig.update_layout(
            title=f'Spectrogram - {self.channel_names[channel_idx]}',
            xaxis_title='Time (seconds)',
            yaxis_title='Frequency (Hz)',
            height=400
        )
        
        # Limit frequency range to 0-50 Hz (relevant for EEG)
        fig.update_yaxes(range=[0, 50])
        
        return fig
