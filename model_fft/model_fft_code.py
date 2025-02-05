#  TOKENS: 3085 (of:8000) = 1474 + 1611(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_fft/model_fft_code.py 
# dest: model_fft/model_fft_code.py 
"""
Module: SignalModelFft
Date: 2025-02-04
Description: This module is part of a Model/View/Controller architecture for processing Morse code audio data.
It provides functionalities for FFT analysis, bandpass filtering, and Goertzel algorithm-based frequency detection.
The code shall be saved in a file named: model_fft/model_fft_code.py

Example usage:
from model_fft.model_fft_code import SignalModelFft
"""

import logging
from typing import Dict, TypedDict
import numpy as np
from scipy.signal import butter, lfilter
from model_fft.fft.fft_code import Streaming_fft

# Constants
DEFAULT_ROLLING_BUFFER_SECONDS = 3.0
DEFAULT_FFT_RATE_HZ = 200
DEFAULT_FREQ_RANGE_MIN = 300
DEFAULT_FREQ_RANGE_MAX = 1000
DEFAULT_BANDPASS_FILTER_ORDER = 4
DEFAULT_AUDIO_RATE_HZ = 44100

class PlotDataDict(TypedDict):
    target: str
    color: str
    Hz: float
    x_origin: float
    y: np.ndarray

class SignalModelFft:
    def __init__(self, cfg_dict: Dict = {}):
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('signal_model_fft.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # STEP_2: Initialize configuration parameters
        self.rolling_buffer_seconds = self._init_param(cfg_dict, 'rolling_buffer_seconds', DEFAULT_ROLLING_BUFFER_SECONDS)
        self.fft_rate_hz = self._init_param(cfg_dict, 'fft_rate_hz', DEFAULT_FFT_RATE_HZ)
        self.freq_range_min = self._init_param(cfg_dict, 'freq_range_min', DEFAULT_FREQ_RANGE_MIN)
        self.freq_range_max = self._init_param(cfg_dict, 'freq_range_max', DEFAULT_FREQ_RANGE_MAX)
        self.bandpass_filter_order = self._init_param(cfg_dict, 'bandpass_filter_order', DEFAULT_BANDPASS_FILTER_ORDER)
        self.audio_rate_hz = DEFAULT_AUDIO_RATE_HZ

        # STEP_3: Initialize rolling buffers
        self.audio_data_rolling_buffer = np.zeros(int(self.rolling_buffer_seconds * self.audio_rate_hz))
        self.fft_magnitude_rolling_buffer = np.zeros(int(self.rolling_buffer_seconds * self.fft_rate_hz))

    def _init_param(self, cfg_dict: Dict, key: str, default):
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %s", key, default)
        return value

    def get_cfg(self) -> Dict:
        return {
            'rolling_buffer_seconds': self.rolling_buffer_seconds,
            'fft_rate_hz': self.fft_rate_hz,
            'freq_range_min': self.freq_range_min,
            'freq_range_max': self.freq_range_max,
            'bandpass_filter_order': self.bandpass_filter_order,
            'audio_rate_hz': self.audio_rate_hz
        }

    def get_fft_rate_hz(self) -> float:
        return self.fft_rate_hz

    def set_audio_rate_hz(self, audio_rate_hz: float):
        self.audio_rate_hz = audio_rate_hz
        self.audio_data_rolling_buffer = np.zeros(int(self.rolling_buffer_seconds * self.audio_rate_hz))
        self.fft_magnitude_rolling_buffer = np.zeros(int(self.rolling_buffer_seconds * self.fft_rate_hz))

    def goertzel(self, samples: np.ndarray, target_freq: float, fs: int = DEFAULT_AUDIO_RATE_HZ) -> float:
        try:
            # STEP_4: Goertzel algorithm implementation
            s_prev = 0.0
            s_prev2 = 0.0
            normalized_freq = target_freq / fs
            coeff = 2 * np.cos(2 * np.pi * normalized_freq)
            for sample in samples:
                s = sample + coeff * s_prev - s_prev2
                s_prev2 = s_prev
                s_prev = s
            power = s_prev2**2 + s_prev**2 - coeff * s_prev * s_prev2
            return power
        except Exception as e:
            self.logger.exception("Error in Goertzel algorithm: %s", e)
            raise

    def bandpass_filter(self, data: np.ndarray, fs: int = DEFAULT_AUDIO_RATE_HZ) -> np.ndarray:
        try:
            # STEP_5: Bandpass filter implementation
            nyquist = 0.5 * fs
            low = self.freq_range_min / nyquist
            high = self.freq_range_max / nyquist
            b, a = butter(self.bandpass_filter_order, [low, high], btype='band')
            y = lfilter(b, a, data)
            return y
        except Exception as e:
            self.logger.exception("Error in bandpass filter: %s", e)
            raise

    def cw_detection(self, audio_chunk: np.ndarray, cw_samples_hz: np.ndarray, fs: int = DEFAULT_AUDIO_RATE_HZ) -> np.ndarray:
        try:
            # STEP_6: CW detection implementation
            filtered_chunk = self.bandpass_filter(audio_chunk, fs)
            return np.array([self.goertzel(filtered_chunk, freq, fs) for freq in cw_samples_hz])
        except Exception as e:
            self.logger.exception("Error in CW detection: %s", e)
            raise

    def process_chunk(self, audio_chunk: np.ndarray) -> Dict[str, PlotDataDict]:
        try:
            # STEP_7: Process audio chunk
            self.audio_data_rolling_buffer = np.roll(self.audio_data_rolling_buffer, -len(audio_chunk))
            self.audio_data_rolling_buffer[-len(audio_chunk):] = audio_chunk

            cw_samples_hz = np.arange(self.freq_range_min, self.freq_range_max, 50)
            fft_magnitude_range = self.cw_detection(audio_chunk, cw_samples_hz, self.audio_rate_hz)
            fft_magnitude_range /= 1000000
            mag_max = np.max(fft_magnitude_range)


            self.fft_magnitude_rolling_buffer = np.roll(self.fft_magnitude_rolling_buffer, -1)
            self.fft_magnitude_rolling_buffer[-1] = mag_max

            frequency_bins = np.fft.fftfreq(len(audio_chunk), 1 / self.audio_rate_hz)
            fft_magnitude_hz = 1 / (frequency_bins[1] - frequency_bins[0])

            return {
                "Audio": {"target": "plot_1", "color": "Green", "Hz": self.audio_rate_hz, "x_origin": 0.0, "y": self.audio_data_rolling_buffer},
                "Freq": {"target": "plot_2", "color": "Green", "Hz": fft_magnitude_hz, "x_origin": self.freq_range_min, "y": fft_magnitude_range},
                "Magnitude": {"target": "plot_3", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.fft_magnitude_rolling_buffer}
            }
        except Exception as e:
            self.logger.exception("Error in processing chunk: %s", e)
            raise
