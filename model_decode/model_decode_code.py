#  TOKENS: 6649 (of:8000) = 4337 + 2312(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_decode/model_decode_code.py 
# dest: model_decode/model_decode_code.py 
"""
Module: SignalModelDecode
Date: 2025-02-04
Description: This module is part of a Model/View/Controller architecture for processing Morse code audio data.
It implements the model component, which processes audio data to detect Morse code signals.
The code is designed to be saved in a file named: model_decode/model_decode_code.py

Example usage:
- Instantiate the SignalModelDecode class with optional configuration parameters.
- Use the process_chunk method to analyze Morse code signals from audio data.
"""

import numpy as np
import logging
from typing import Dict, TypedDict
from scipy.ndimage import gaussian_filter1d
from model_decode.synpat.synpat_code import Synpat

class PlotDataDict(TypedDict):
    target: str
    color: str
    Hz: float
    x_origin: float
    y: np.ndarray

class SignalModelDecode:
    def __init__(self, cfg_dict: Dict = {}):
        """
        Initialize the SignalModelDecode class with configuration parameters.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('signal_model_decode.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.wpm_start = self._init_param(cfg_dict, 'wpm_start', 24)
        self.wpm_end = self._init_param(cfg_dict, 'wpm_end', 12)
        self.p_thresh = self._init_param(cfg_dict, 'p_thresh', 0.5)
        self.default_dit_msec = self._init_param(cfg_dict, 'default_dit_msec', 66)

        self._synpat = Synpat()
        self._prev_dit_msec = self.default_dit_msec
        self._chunk_sec = None
        self._fft_rate_hz = None
        self._syn_wpm_dict = None

    def _init_param(self, cfg_dict: Dict, key: str, default: float) -> float:
        """
        Initialize a configuration parameter with a default value if the key is missing.
        Logs a message if the parameter is not found in the dictionary.
        """
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %.1f", key, default)
        return value

    def get_cfg(self) -> Dict:
        """
        Returns the configuration dictionary updated with current class configuration parameter values.
        """
        return {
            'wpm_start': self.wpm_start,
            'wpm_end': self.wpm_end,
            'p_thresh': self.p_thresh,
            'default_dit_msec': self.default_dit_msec
        }

    def process_sec(self, chunk_sec: float):
        """
        Set the chunk duration in seconds for processing.
        """
        self._chunk_sec = chunk_sec

    def set_fft_rate_hz(self, fft_rate_hz: float):
        """
        Set the FFT sampling rate and generate synthetic patterns.
        """
        self._fft_rate_hz = fft_rate_hz
        self._syn_wpm_dict = self._synpat.generate_synthetic_dict(self.wpm_start, self.wpm_end, fft_rate_hz)

    def apply_gaussian_filter(self, signal: np.ndarray, cutoff_hz: float, f_signal_Hz: float) -> np.ndarray:
        """
        Applies Gaussian smoothing to a 1D signal.
        Computes the standard deviation (sigma) for a Gaussian filter given a cutoff frequency and sampling rate.
        """
        sigma = (1 / (2 * np.pi * cutoff_hz)) * f_signal_Hz
        return gaussian_filter1d(signal, sigma)

    def process_chunk(self, plot_dict: Dict[str, PlotDataDict]) -> Dict[str, PlotDataDict]:
        """
        Process a chunk of audio data to detect Morse code signals.
        """
        # STEP_0: GET DATA
        norm_rolling_buffer = plot_dict['norm_rolling_buffer']['y']

        # STEP_1a: WPM - DETECT ZERO CROSSINGS
        norm_filtered = self.apply_gaussian_filter(norm_rolling_buffer, cutoff_hz=30, f_signal_Hz=self._fft_rate_hz)
        zero_crossings = np.where(np.diff(np.sign(norm_filtered)))[0]
        intervals = np.diff(zero_crossings)

        # STEP_1b: WPM - WRAP DAH intervals into DIT interval
        wpm_max = max(self.wpm_start, self.wpm_end)
        n_dits_in_dah = 3
        dit_min_idx = int(self._synpat.dit_sec(wpm_max) * self._fft_rate_hz)
        dit_max_idx = dit_min_idx * n_dits_in_dah
        intervals = np.where((intervals >= dit_min_idx) & (intervals < dit_max_idx), intervals, 0)
        intervals = np.where(intervals >= dit_max_idx, intervals // n_dits_in_dah, intervals)

        # STEP_2: WPM - HISTOGRAM
        histogram_array = np.zeros(dit_max_idx)
        for interval in intervals:
            if interval < len(histogram_array):
                histogram_array[interval] += 1
        histogram_array[0] = 0
        histogram_array = self.apply_gaussian_filter(histogram_array, cutoff_hz=30, f_signal_Hz=self._fft_rate_hz)

        # STEP_3: WPM - PEAK PROMINENCE
        first_largest_peak_index = np.argmax(histogram_array)
        first_largest_peak_value = histogram_array[first_largest_peak_index]

        # STEP_6: WPM - THRESHOLD
        if first_largest_peak_value > 1:
            dit_msec = (first_largest_peak_index / self._fft_rate_hz) * 1000
            if dit_msec < 30:
                dit_msec = self.default_dit_msec
            self._prev_dit_msec = dit_msec
        else:
            dit_msec = self._prev_dit_msec
        dit_sec_array = np.array([dit_msec * 1000])

        # STEP_7: CONVOLVE with synthetic patterns
        syn_dit = self._synpat.generate_synthetic_patterns(self._fft_rate_hz, dit_msec)
        dit_impulse = np.zeros_like(norm_filtered, dtype=bool)
        nit_impulse = np.zeros_like(norm_filtered, dtype=bool)
        dah_impulse = np.zeros_like(norm_filtered, dtype=bool)
        letter_impulse = np.zeros_like(norm_filtered, dtype=bool)
        word_impulse = np.zeros_like(norm_filtered, dtype=bool)

        conv = np.convolve(syn_dit['dit'], norm_filtered, mode='same')
        dit_probability = np.maximum(conv, 0)
        nit_probability = np.maximum(-conv, 0)
        deriv_it = np.diff(conv)
        it_impulse = np.diff(np.sign(deriv_it)) != 0

        conv = np.convolve(syn_dit['dah'], norm_filtered, mode='same')
        dah_probability = np.maximum(conv, 0)
        letter_probability = np.maximum(-conv, 0)
        deriv_ah = np.diff(conv)
        ah_impulse = np.diff(np.sign(deriv_ah)) != 0

        conv = np.convolve(syn_dit['word'], norm_filtered, mode='same')
        word_probability = np.maximum(conv, 0)
        deriv_word = np.diff(conv)
        w_impulse = np.diff(np.sign(deriv_word)) != 0

        for index in range(len(it_impulse) - 1):
            max_prob = max(dit_probability[index], nit_probability[index], dah_probability[index], letter_probability[index], word_probability[index])
            if max_prob > self.p_thresh:
                if it_impulse[index]:
                    dit_impulse[index] = (dit_probability[index] == max_prob)
                    nit_impulse[index] = (nit_probability[index] == max_prob)
                if ah_impulse[index]:
                    dah_impulse[index] = (dah_probability[index] == max_prob)
                    letter_impulse[index] = (letter_probability[index] == max_prob)
                if w_impulse[index]:
                    word_impulse[index] = (word_probability[index] == max_prob)

        # STEP_8: CLEANUP
        plot_dict.update({
            "norm_filtered": {"target": "plot_4", "color": "Green", "Hz": self._fft_rate_hz, "x_origin": 0.0, "y": norm_filtered},
            "DIT_impulse": {"target": "plot_6", "color": "Green", "Hz": 1, "x_origin": 0, "y": dit_impulse},
            "DIT": {"target": "plot_6", "color": "Green", "Hz": 1, "x_origin": 0, "y": dit_probability},
            "NIT_impulse": {"target": "plot_6", "color": "Red", "Hz": 1, "x_origin": 0, "y": nit_impulse},
            "NIT": {"target": "plot_6", "color": "Red", "Hz": 1, "x_origin": 0, "y": nit_probability},
            "DAH_impulse": {"target": "plot_6", "color": "Blue", "Hz": 1, "x_origin": 0, "y": dah_impulse},
            "DAH": {"target": "plot_6", "color": "Blue", "Hz": 1, "x_origin": 0, "y": dah_probability},
            "LETTER_impulse": {"target": "plot_6", "color": "Magenta", "Hz": 1, "x_origin": 0, "y": letter_impulse},
            "LETTER": {"target": "plot_6", "color": "Magenta", "Hz": 1, "x_origin": 0, "y": letter_probability},
            "WORD_impulse": {"target": "plot_6", "color": "Cyan", "Hz": 1, "x_origin": 0, "y": word_impulse},
            "WORD": {"target": "plot_6", "color": "Cyan", "Hz": 1, "x_origin": 0, "y": word_probability},
            "FreqHz": {"target": "plot_5", "color": "Green", "Hz": self._fft_rate_hz, "x_origin": 0, "y": histogram_array}
        })

        return plot_dict
