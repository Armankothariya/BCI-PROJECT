# modules/music_controller.py
import os
import random
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pygame
import yaml

class MusicController:
    """
    Enhanced Music Controller for emotion-based music playback
    Integrates with music_map.yaml for perfect emotion-music mapping
    """
    
    def __init__(self, music_base_dir: Path, music_config: Dict, audio_mode: str = 'pygame'):
        self.music_base_dir = music_base_dir
        self.music_config = music_config
        self.audio_mode = audio_mode
        self.current_track = None
        self.is_playing_flag = False
        self.volume = 0.7
        self.current_emotion = None
        
        # Initialize pygame mixer if using pygame
        if audio_mode == 'pygame':
            try:
                pygame.mixer.init()
                pygame.mixer.music.set_volume(self.volume)
                self.initialized = True
            except Exception as e:
                logging.error(f"Failed to initialize pygame mixer: {e}")
                self.initialized = False
        else:
            self.initialized = True
    
    def initialize(self) -> bool:
        """Initialize the music controller and verify music files"""
        try:
            # Verify music base directory exists
            if not self.music_base_dir.exists():
                logging.error(f"Music directory not found: {self.music_base_dir}")
                return False
            
            # Verify we have music tracks for each emotion
            required_emotions = ['positive', 'negative', 'neutral']
            for emotion in required_emotions:
                if emotion not in self.music_config:
                    logging.warning(f"Missing music configuration for emotion: {emotion}")
                    continue
                
                tracks = self.music_config[emotion]
                if not isinstance(tracks, list) or len(tracks) == 0:
                    logging.warning(f"No tracks configured for emotion: {emotion}")
                    continue
                
                # Check if at least one track exists
                available_tracks = []
                for track in tracks:
                    if isinstance(track, dict) and 'file' in track:
                        track_path = self.music_base_dir / track['file']
                        if track_path.exists():
                            available_tracks.append(track)
                        else:
                            logging.warning(f"Music file not found: {track_path}")
                
                if len(available_tracks) == 0:
                    logging.error(f"No available music tracks for emotion: {emotion}")
                    return False
            
            logging.info("Music controller initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Music controller initialization failed: {e}")
            return False
    
    def play_emotion(self, emotion: str) -> bool:
        """
        Play music based on detected emotion
        
        Args:
            emotion: One of 'positive', 'negative', 'neutral'
            
        Returns:
            bool: Success status
        """
        try:
            emotion = emotion.lower()
            self.current_emotion = emotion
            
            # Get tracks for this emotion
            if emotion not in self.music_config:
                logging.warning(f"No music configuration for emotion: {emotion}")
                # Try default as fallback
                if 'default' in self.music_config:
                    emotion = 'default'
                else:
                    return False
            
            tracks = self.music_config[emotion]
            if not isinstance(tracks, list) or len(tracks) == 0:
                logging.warning(f"No tracks available for emotion: {emotion}")
                return False
            
            # Filter available tracks
            available_tracks = []
            for track in tracks:
                if isinstance(track, dict) and 'file' in track:
                    track_path = self.music_base_dir / track['file']
                    if track_path.exists():
                        available_tracks.append(track)
            
            if len(available_tracks) == 0:
                logging.error(f"No available music files for emotion: {emotion}")
                return False
            
            # Select a random track (using time-based randomization to avoid fixed seed)
            # Use SystemRandom which is based on os.urandom() and not affected by random.seed()
            secure_random = random.SystemRandom()
            selected_track = secure_random.choice(available_tracks)
            track_path = self.music_base_dir / selected_track['file']
            
            # Stop current playback
            self.stop()
            
            # Play new track
            if self.audio_mode == 'pygame' and self.initialized:
                try:
                    pygame.mixer.music.load(str(track_path))
                    pygame.mixer.music.play()
                    self.is_playing_flag = True
                    self.current_track = selected_track
                    logging.info(f"Playing {emotion} music: {selected_track['file']}")
                    return True
                except Exception as e:
                    logging.error(f"Failed to play music file: {e}")
                    return False
            else:
                # Fallback for other audio modes
                self.current_track = selected_track
                logging.info(f"Selected {emotion} music: {selected_track['file']} (audio mode: {self.audio_mode})")
                return True
                
        except Exception as e:
            logging.error(f"Error in play_emotion: {e}")
            return False
    
    def stop(self) -> None:
        """Stop music playback"""
        try:
            if self.audio_mode == 'pygame' and self.initialized:
                pygame.mixer.music.stop()
            self.is_playing_flag = False
            self.current_track = None
        except Exception as e:
            logging.error(f"Error stopping music: {e}")
    
    def pause(self) -> None:
        """Pause music playback"""
        try:
            if self.audio_mode == 'pygame' and self.initialized:
                pygame.mixer.music.pause()
            self.is_playing_flag = False
        except Exception as e:
            logging.error(f"Error pausing music: {e}")
    
    def resume(self) -> None:
        """Resume music playback"""
        try:
            if self.audio_mode == 'pygame' and self.initialized:
                pygame.mixer.music.unpause()
            self.is_playing_flag = True
        except Exception as e:
            logging.error(f"Error resuming music: {e}")
    
    def skip(self) -> bool:
        """Skip to next track for current emotion"""
        if self.current_emotion:
            return self.play_emotion(self.current_emotion)
        return False
    
    def set_volume(self, volume: float) -> None:
        """Set playback volume (0.0 to 1.0)"""
        try:
            self.volume = max(0.0, min(1.0, volume))  # Clamp to valid range
            if self.audio_mode == 'pygame' and self.initialized:
                pygame.mixer.music.set_volume(self.volume)
        except Exception as e:
            logging.error(f"Error setting volume: {e}")
    
    def get_volume(self) -> float:
        """Get current volume"""
        return self.volume
    
    def is_playing(self) -> bool:
        """Check if music is currently playing"""
        if self.audio_mode == 'pygame' and self.initialized:
            return pygame.mixer.music.get_busy()
        return self.is_playing_flag
    
    def get_current_track_info(self) -> Optional[Dict]:
        """Get information about currently playing track"""
        if self.current_track:
            return {
                'file': self.current_track.get('file'),
                'mood': self.current_track.get('mood', 'unknown'),
                'tempo': self.current_track.get('tempo', 'unknown'),
                'intensity': self.current_track.get('intensity', 0.5),
                'emotion': self.current_emotion
            }
        return None
    
    def get_available_tracks(self, emotion: str) -> List[Dict]:
        """Get list of available tracks for an emotion"""
        emotion = emotion.lower()
        if emotion in self.music_config:
            tracks = self.music_config[emotion]
            if isinstance(tracks, list):
                available_tracks = []
                for track in tracks:
                    if isinstance(track, dict) and 'file' in track:
                        track_path = self.music_base_dir / track['file']
                        if track_path.exists():
                            available_tracks.append(track)
                return available_tracks
        return []
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            if self.audio_mode == 'pygame':
                pygame.mixer.quit()
        except:
            pass 