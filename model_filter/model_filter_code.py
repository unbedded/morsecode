#  TOKENS: 3677 (of:8000) = 1841 + 1836(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_filter/model_filter_code.py 
# dest: model_filter/model_filter_code.py 
"""
Module: SignalModelFilter
Date: 2025-02-04
Description: This module is part of a Model/View/Controller architecture for processing Morse code audio data.
It processes audio data to filter and normalize Morse code signals using FFT and rolling buffer techniques.
The code is saved in a file named: model_filter/model_filter_code.py

Example usage:
# Instantiate the SignalModelFilter class
filter = SignalModelFilter(cfg_dict={'fft_rate_hz': 120})

# Process a chunk of data
plot_dict = {
    'Freq': {'y': np.array([...])},
    'Magnitude': {'y': np.array([...])}
}
result = filter.process_chunk(plot_dict)
"""

import numpy as np
import logging
from typing import Dict, TypedDict
from collections import deque

class PlotDataDict(TypedDict):
    target: str
    color: str
    Hz: float
    x_origin: float
    y: np.ndarray

class SignalModelFilter:
    def __init__(self, cfg_dict: Dict = {}):
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(filename='signal_model_filter.log',
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            level=logging.WARNING)
        self.logger.setLevel(logging.WARNING)

        # Configuration parameters
        self.cw_mag_thresh_seconds = self._init_param(cfg_dict, 'cw_mag_thresh_seconds', 0.1)
        self.display_seconds = self._init_param(cfg_dict, 'display_seconds', 3.0)
        self.fft_rate_hz = self._init_param(cfg_dict, 'fft_rate_hz', 100)
        self.cw_peak_ratio_threshold = self._init_param(cfg_dict, 'cw_peak_ratio_threshold', 4)
        self.cw_peak_ratio_threshold_min = self._init_param(cfg_dict, 'cw_peak_ratio_threshold_min', 3)
        self.cutoff_hz = self._init_param(cfg_dict, 'cutoff_hz', 15.0)
        self.n_move_avg_elements = self._init_param(cfg_dict, 'n_move_avg_elements', 6)

        # Initialize rolling buffers
        buffer_length = int(self.display_seconds * self.fft_rate_hz)
        self.cw_mag_rolling_buffer = np.zeros(buffer_length)
        self.cw_mag_thresh_rolling_buffer = np.zeros(buffer_length)
        self.norm_rolling_buffer = np.zeros(buffer_length)

        norm_gain_length = int(self.cw_mag_thresh_seconds * self.fft_rate_hz)
        self.cw_mag_for_norm_gain = np.zeros(norm_gain_length)

        # Queue for moving average
        self.move_avg_queue = deque(maxlen=self.n_move_avg_elements)

    def _init_param(self, cfg_dict: Dict, key: str, default):
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %s", key, default)
        return value

    def get_cfg(self) -> Dict:
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
        # STEP_2: Maintain a queue of nparrays
        if len(self.move_avg_queue) > 0 and len(new_sample) != len(self.move_avg_queue[0]):
            self.move_avg_queue.clear()
            self.logger.debug("Queue cleared due to size mismatch.")

        self.move_avg_queue.append(new_sample)
        self.logger.debug("New sample added to queue. Queue size: %d", len(self.move_avg_queue))

        if len(self.move_avg_queue) > self.n_move_avg_elements:
            self.move_avg_queue.popleft()
            self.logger.debug("Oldest sample removed from queue.")

        avg_array = np.mean(np.array(self.move_avg_queue), axis=0)
        self.logger.debug("Computed element-wise average.")
        return avg_array

    def set_fft_rate_hz(self, fft_rate_hz: float):
        # STEP_3: Resize and initialize buffers
        self.fft_rate_hz = fft_rate_hz
        buffer_length = int(self.display_seconds * self.fft_rate_hz)
        self.cw_mag_rolling_buffer = np.zeros(buffer_length)
        self.cw_mag_thresh_rolling_buffer = np.zeros(buffer_length)
        self.norm_rolling_buffer = np.zeros(buffer_length)

        norm_gain_length = int(self.cw_mag_thresh_seconds * self.fft_rate_hz)
        self.cw_mag_for_norm_gain = np.zeros(norm_gain_length)

        self.logger.info("FFT rate set to %f Hz. Buffers resized.", fft_rate_hz)

    def process_chunk(self, plot_dict: Dict[str, PlotDataDict]) -> Dict[str, PlotDataDict]:
        try:
            # STEP_4: Source data
            cw_frequencies = plot_dict['Freq']['y']
            magnitude = plot_dict['Magnitude']['y'][-1]

            # STEP_5: Move avg FFT filter
            cw_frequencies_avg = self.move_avg_elementwise(cw_frequencies)

            # STEP_6: Calc peak ratio
            cw_mag_max = np.max(cw_frequencies_avg)
            cw_mag_mean = np.mean(np.delete(cw_frequencies_avg, np.argmax(cw_frequencies_avg)))
            cw_peak_ratio = cw_mag_max / cw_mag_mean

            # STEP_7: Filter magnitude for normalization gain/offset
            tmp = 0
            if cw_peak_ratio > self.cw_peak_ratio_threshold:
                self.cw_mag_for_norm_gain = np.roll(self.cw_mag_for_norm_gain, -1)
                self.cw_mag_for_norm_gain[-1] = cw_mag_max
                tmp = cw_mag_max

            self.cw_mag_thresh_rolling_buffer = np.roll(self.cw_mag_thresh_rolling_buffer, -1)
            self.cw_mag_thresh_rolling_buffer[-1] = tmp

            self.cw_mag_rolling_buffer = np.roll(self.cw_mag_rolling_buffer, -1)
            self.cw_mag_rolling_buffer[-1] = cw_mag_max

            # STEP_8: Normalize
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

            # STEP_9: Rolling buffer
            self.norm_rolling_buffer = np.roll(self.norm_rolling_buffer, -1)
            self.norm_rolling_buffer[-1] = norm

            # STEP_10: Return updated plot_dict
            plot_dict.update({
                "Magnitude": {"target": "plot_3", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_rolling_buffer},
                "Mag_Thresh": {"target": "plot_3", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_thresh_rolling_buffer},
                "norm_rolling_buffer": {"target": "plot_4", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.norm_rolling_buffer},
                "norm_gain": {"target": "plot_7", "color": "Green", "Hz": self.fft_rate_hz, "x_origin": 0.0, "y": self.cw_mag_for_norm_gain}
            })
            self.logger.info("Chunk processed successfully.")
            return plot_dict

        except Exception as e:
            self.logger.exception("Error processing chunk: %s", str(e))
            raise
