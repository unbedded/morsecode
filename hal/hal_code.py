#  TOKENS: 1992 (of:8000) = 990 + 1002(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: hal/hal_code.py 
# dest: hal/hal_code.py 
"""
Hardware Abstraction Layer (HAL) for reading chunks of audio data.

This module provides functionality to read and process audio data from a WAV file.
It supports configuration management and logging for debugging and error handling.

Code shall be saved in a file named: hal/hal_code.py
Date: 2025-02-04

Example usage:
```
from hal_code import HardwareAbstractionLayer

cfg = {
    'wav_filename': '/path/to/audio.wav',
    'audio_rate_hz': 48000
}

hal = HardwareAbstractionLayer(cfg_dict=cfg)
audio_chunk = hal.get_next_chunk(update_interval_ms=100)
```
"""

import logging
from typing import Dict
import numpy as np
from scipy.io import wavfile

class HardwareAbstractionLayer:
    """
    A class to handle audio data processing from a WAV file.
    
    Attributes:
        cfg_dict (Dict): Configuration dictionary for the HAL.
        audio_data (np.ndarray): The audio data loaded from the WAV file.
        audio_rate_hz (int): The sampling rate of the audio data.
    """

    DEFAULT_WAV_FILENAME = '/home/preact/sw/morsecode/wav/NightOfNights2015-various-12Jul2015.wav'
    DEFAULT_AUDIO_RATE_HZ = 44100

    def __init__(self, cfg_dict: Dict = {}):
        """
        Initializes the HardwareAbstractionLayer with configuration parameters.
        
        STEP_1: Set up logging configuration.
        """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            filename='hal_debug.log',
            level=logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger.setLevel(logging.WARNING)

        self.cfg_dict = cfg_dict
        self.audio_data = np.array([])
        self.audio_rate_hz = self._init_param('audio_rate_hz', self.DEFAULT_AUDIO_RATE_HZ)
        self.wav_filename = self._init_param('wav_filename', self.DEFAULT_WAV_FILENAME)

        self.load_audio_file()

    def _init_param(self, key: str, default):
        """
        Helper method to initialize a parameter with a default value if the key is missing.
        
        STEP_2: Initialize configuration parameter.
        """
        value = self.cfg_dict.get(key, default)
        if key not in self.cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %s", key, default)
        return value

    def get_params(self) -> Dict:
        """
        Returns the current configuration parameters.
        
        STEP_3: Return configuration parameters.
        """
        return {
            'wav_filename': self.wav_filename,
            'audio_rate_hz': self.audio_rate_hz
        }

    def load_audio_file(self):
        """
        Reads in a WAV format file and extracts audio data.
        
        STEP_4: Load audio file and extract data.
        """
        try:
            self.audio_rate_hz, data = wavfile.read(self.wav_filename)
            if data.ndim > 1:
                self.audio_data = data[:, 0]  # Extract the first channel if stereo
            else:
                self.audio_data = data
            self.logger.info("Audio file '%s' loaded successfully.", self.wav_filename)
        except FileNotFoundError:
            self.logger.exception("Audio file '%s' not found.", self.wav_filename)
            raise
        except Exception as e:
            self.logger.exception("Error loading audio file: %s", e)
            raise

    def get_next_chunk(self, update_interval_ms: int) -> np.ndarray:
        """
        Retrieves the next chunk of audio data.
        
        STEP_5: Retrieve next audio data chunk.
        """
        try:
            samples_per_chunk = int((update_interval_ms / 1000) * self.audio_rate_hz)
            if len(self.audio_data) == 0:
                self.logger.warning("No audio data available. Returning zeros.")
                return np.zeros(samples_per_chunk)
            
            chunk = self.audio_data[:samples_per_chunk]
            self.audio_data = self.audio_data[samples_per_chunk:]
            return chunk
        except Exception as e:
            self.logger.exception("Error retrieving audio chunk: %s", e)
            raise

    def get_audio_rate_hz(self) -> int:
        """
        Returns the sampling rate of the audio signal in Hz.
        
        STEP_6: Return audio sampling rate.
        """
        return self.audio_rate_hz

    def get_cfg(self) -> Dict:
        """
        Returns the updated configuration dictionary.
        
        STEP_7: Return updated configuration dictionary.
        """
        return self.get_params()
