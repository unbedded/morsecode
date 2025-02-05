#  TOKENS: 2132 (of:8000) = 1087 + 1045(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_fft/fft/fft_code.py 
# dest: model_fft/fft/fft_code.py 
"""
This module implements the Streaming FFT processing class, designed to handle real-time
signal processing using Fast Fourier Transform (FFT). The class is configurable and
supports different window functions for signal processing.

Code shall be saved in a file named: model_fft/fft/fft_code.py

Example usage:
from model_fft.fft.fft_code import Streaming_fft
fft_processor = Streaming_fft(cfg_dict={'window': 'hamming'})
frequency_bins, time_bins, fft_magnitude = fft_processor.fft(audio_freq_hz=44100, current_segment=signal_array)

<DATE>: 2025-02-04
"""

import numpy as np
from scipy.fft import fft
import logging
from typing import Dict, Tuple

# Constants
DEFAULT_WINDOW = 'none'
VALID_WINDOWS = ['none', 'hamming', 'blackman']
NPERSEG = 256
NOVERLAP = 128

class Streaming_fft:
    """
    A class to perform streaming FFT on audio signals with configurable window functions.

    Attributes:
        window (str): The window function to apply. Options are 'none', 'hamming', 'blackman'.
        previous_segment (np.ndarray): The previous segment of the signal for overlap.
    """

    def __init__(self, cfg_dict: Dict = {}):
        """
        Initializes the Streaming_fft class with configuration parameters.

        Args:
            cfg_dict (Dict): Configuration dictionary to initialize parameters.
        """
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('fft_processing.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # STEP_2: Initialize parameters
        self.window = self._init_param(cfg_dict, 'window', DEFAULT_WINDOW)
        self.previous_segment = np.array([])

    def _init_param(self, cfg_dict: Dict, key: str, default):
        """
        Helper method to initialize a parameter with a default value if the key is missing.

        Args:
            cfg_dict (Dict): Configuration dictionary.
            key (str): The key to look for in the dictionary.
            default: The default value to use if the key is not found.

        Returns:
            The value from the dictionary or the default value.
        """
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %s", key, default)
        return value

    def get_cfg(self) -> Dict:
        """
        Returns the current configuration dictionary.

        Returns:
            Dict: The configuration dictionary with current parameter values.
        """
        return {'window': self.window}

    def nperseg(self) -> int:
        """
        Returns the number of samples per segment for FFT processing.

        Returns:
            int: The number of samples per segment.
        """
        return NPERSEG

    def fft(self, audio_freq_hz: int, current_segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs FFT on the current segment of the audio signal.

        Args:
            audio_freq_hz (int): Sampling frequency in Hz.
            current_segment (np.ndarray): The current segment of the signal.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Frequency bins, time bins, and FFT magnitude.
        """
        # STEP_3: Validate input
        if audio_freq_hz <= 0:
            self.logger.error("Invalid audio frequency: %d. Must be greater than 0.", audio_freq_hz)
            raise ValueError("Audio frequency must be greater than 0.")

        # STEP_4: Append previous segment
        segment = np.concatenate((self.previous_segment, current_segment))

        # STEP_5: Apply window function if specified
        if self.window != 'none':
            if self.window == 'hamming':
                segment *= np.hamming(len(segment))
            elif self.window == 'blackman':
                segment *= np.blackman(len(segment))
            else:
                self.logger.error("Invalid window function: %s", self.window)
                raise ValueError("Invalid window function specified.")

        # STEP_6: Perform FFT
        fft_result = fft(segment)

        # STEP_7: Compute magnitude
        fft_magnitude = np.abs(fft_result)

        # STEP_8: Update previous segment
        self.previous_segment = current_segment

        # STEP_9: Calculate frequency and time bins
        frequency_bins = np.fft.fftfreq(len(segment), d=1/audio_freq_hz)
        time_bins = np.arange(len(segment)) / audio_freq_hz

        # STEP_10: Return results
        return frequency_bins, time_bins, fft_magnitude
