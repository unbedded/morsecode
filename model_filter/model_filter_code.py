#  TOKENS: 3957 (of:8000) = 1855 + 2102(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_filter/model_filter_code.py 
# dest: model_filter/model_filter_code.py 
"""
This module is part of a Model/View/Controller architecture for processing Morse code audio data.
It defines the SignalModelFilter class, which processes audio data and maintains rolling buffers
for signal normalization and filtering. The code is designed to be saved in a file named:
model_filter/model_filter_code.py

Date: 2025-02-06

Example usage:
# Initialize the model with default configuration
model = SignalModelFilter()

# Process a chunk of data
processed_data = model.process_chunk(plot_dict)

The class provides methods for setting FFT rate, processing data chunks, and maintaining
rolling buffers for signal processing. It uses Pythonic error handling and logging for
debugging and error reporting.
"""

import logging
from typing import Dict, Deque, TypedDict
from collections import deque
import numpy as np

class PlotDataDict(TypedDict):
    target: str
    color: str
    Hz: float
    x_origin: float
    y: np.ndarray

class SignalModelFilter:
    """
    SignalModelFilter processes Morse code audio data, maintaining rolling buffers for
    normalization and filtering. It supports configuration management and logging for
    debugging and error handling.
    """

    # Constants for default configuration values
    DEFAULT_CW_MAG_THRESH_SECONDS = 0.1
    DEFAULT_DISPLAY_SECONDS = 3.0
    DEFAULT_FFT_RATE_HZ = 100
    DEFAULT_CW_PEAK_RATIO_THRESHOLD = 4
    DEFAULT_CW_PEAK_RATIO_THRESHOLD_MIN = 3
    DEFAULT_CUTOFF_HZ = 15.0
    DEFAULT_N_MOVE_AVG_ELEMENTS = 6

    def __init__(self, cfg_dict: Dict = {}):
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='signal_model_filter.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.WARNING)
        self.logger.setLevel(logging.WARNING)

        # Initialize configuration parameters
        self.cw_mag_thresh_seconds = self._init_param(cfg_dict, 'cw_mag_thresh_seconds', self.DEFAULT_CW_MAG_THRESH_SECONDS)
        self.display_seconds = self._init_param(cfg_dict, 'display_seconds', self.DEFAULT_DISPLAY_SECONDS)
        self.fft_rate_hz = self._init_param(cfg_dict, 'fft_rate_hz', self.DEFAULT_FFT_RATE_HZ)
        self.cw_peak_ratio_threshold = self._init_param(cfg_dict, 'cw_peak_ratio_threshold', self.DEFAULT_CW_PEAK_RATIO_THRESHOLD)
        self.cw_peak_ratio_threshold_min = self._init_param(cfg_dict, 'cw_peak_ratio_threshold_min', self.DEFAULT_CW_PEAK_RATIO_THRESHOLD_MIN)
        self.cutoff_hz = self._init_param(cfg_dict, 'cutoff_hz', self.DEFAULT_CUTOFF_HZ)
        self.n_move_avg_elements = self._init_param(cfg_dict, 'n_move_avg_elements', self.DEFAULT_N_MOVE_AVG_ELEMENTS)

        # Initialize rolling buffers
        buffer_length = int(self.display_seconds * self.fft_rate_hz)
        self.cw_mag_rolling_buffer = np.zeros(buffer_length)
        self.cw_mag_thresh_rolling_buffer = np.zeros(buffer_length)
        self.norm_rolling_buffer = np.zeros(buffer_length)

        norm_gain_length = int(self.cw_mag_thresh_seconds * self.fft_rate_hz)
        self.cw_mag_for_norm_gain = np.zeros(norm_gain_length)

        # Initialize moving average queue
        self.move_avg_queue: Deque[np.ndarray] = deque()

    def _init_param(self, cfg_dict: Dict, key: str, default: float) -> float:
        """
        Helper method to initialize a configuration parameter with a default value if the key is missing.
        Logs a message if the parameter is not found in the dictionary.
        """
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default value: %f", key, default)
        return value

    def get_cfg(self) -> Dict:
        """
        Returns the current configuration parameters as a dictionary.
        """
        return {
            'cw_mag_thresh_seconds': self.cw_mag_thresh_seconds,
            'display_seconds': self.display_seconds,
            'fft_rate_hz': self.fft_rate_hz,
            'cw_peak_ratio_threshold': self.cw_peak_ratio_threshold,
            'cw_peak_ratio_threshold_min': self.cw_peak_ratio_threshold_min,
            'cutoff_hz': self.cutoff_hz,
            'n_move_avg_elements': self.n_move_avg_elements
        }

    def move_avg_elementwise(self, new_sample: np.ndarray) -> np.ndarray:
        """
        Maintains a queue of NumPy arrays and computes the element-wise average.
        """
        # STEP_2: Check if new_sample length differs from previous
        if self.move_avg_queue and len(new_sample) != len(self.move_avg_queue[0]):
            self.logger.debug("New sample length differs. Reinitializing queue.")
            self.move_avg_queue.clear()

        # STEP_3: Insert new_sample into the queue
        self.move_avg_queue.append(new_sample)

        # STEP_4: Discard oldest element if queue exceeds n_move_avg_elements
        if len(self.move_avg_queue) > self.n_move_avg_elements:
            self.move_avg_queue.popleft()

        # STEP_5: Compute element-wise average
        if not self.move_avg_queue:
            return np.zeros_like(new_sample)
        return np.mean(np.array(self.move_avg_queue), axis=0)

    def set_fft_rate_hz(self, fft_rate_hz: float):
        """
        Resizes and initializes rolling buffers based on the new FFT rate.
        """
        # STEP_6: Resize buffers
        self.fft_rate_hz = fft_rate_hz
        buffer_length = int(self.display_seconds * self.fft_rate_hz)
        self.cw_mag_rolling_buffer = np.zeros(buffer_length)
        self.cw_mag_thresh_rolling_buffer = np.zeros(buffer_length)
        self.norm_rolling_buffer = np.zeros(buffer_length)

        norm_gain_length = int(self.cw_mag_thresh_seconds * self.fft_rate_hz)
        self.cw_mag_for_norm_gain = np.zeros(norm_gain_length)

    def process_chunk(self, plot_dict: Dict[str, PlotDataDict]) -> Dict[str, PlotDataDict]:
        """
        Processes a chunk of data and updates the plot dictionary with new values.
        """
        try:
            # STEP_7: SOURCE DATA
            cw_frequencies = plot_dict['Freq']['y']
            magnitude = plot_dict['Magnitude']['y'][-1]

            # STEP_8: MOVE AVG FFT FILTER
            cw_frequencies_avg = self.move_avg_elementwise(cw_frequencies)

            # STEP_9: CALC PEAK RATIO
            cw_mag_max = np.max(cw_frequencies_avg)
            cw_mag_mean = np.mean(np.delete(cw_frequencies_avg, np.argmax(cw_frequencies_avg)))
            cw_mag_min = np.min(cw_frequencies_avg)
            cw_peak_ratio = cw_mag_max / cw_mag_mean

            # STEP_10: FILTER MAGNITUDE for NORMALIZATION GAIN/OFFSET
            tmp = 0
            if cw_peak_ratio > self.cw_peak_ratio_threshold:
                self.cw_mag_for_norm_gain = np.roll(self.cw_mag_for_norm_gain, -1)
                self.cw_mag_for_norm_gain[-1] = cw_mag_max
                tmp = cw_mag_max

            self.cw_mag_thresh_rolling_buffer = np.roll(self.cw_mag_thresh_rolling_buffer, -1)
            self.cw_mag_thresh_rolling_buffer[-1] = tmp

            self.cw_mag_rolling_buffer = np.roll(self.cw_mag_rolling_buffer, -1)
            self.cw_mag_rolling_buffer[-1] = cw_mag_max

            # STEP_11: NORMALIZE
            norm_min = np.min(self.cw_mag_for_norm_gain)
            norm_max = np.max(self.cw_mag_for_norm_gain)
            norm_mean = np.mean(self.cw_mag_for_norm_gain)

            if cw_peak_ratio > self.cw_peak_ratio_threshold_min:
                norm_range = norm_max - norm_min
                if norm_range < 0.1:
                    norm_range = 0.1
                norm = 2 * (magnitude - norm_mean / 2) / norm_range
                norm = max(min(norm, 1.0), -1.0)
            else:
                norm = -1.0

            # STEP_12: ROLLING BUFFER
            self.norm_rolling_buffer = np.roll(self.norm_rolling_buffer, -1)
            self.norm_rolling_buffer[-1] = norm

            # STEP_13: RETURN UPDATED PLOT_DICT
            plot_dict.update({
                "Magnitude": {"target": "plot_3", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_rolling_buffer},
                "Mag_Thresh": {"target": "plot_3", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_thresh_rolling_buffer},
                "norm_rolling_buffer": {"target": "plot_4", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.norm_rolling_buffer},
                "norm_gain": {"target": "plot_7", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_for_norm_gain}
            })

            return plot_dict

        except Exception as e:
            self.logger.exception("Error processing chunk: %s", e)
            raise
