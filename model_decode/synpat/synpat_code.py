#  TOKENS: 4419 (of:8000) = 2400 + 2019(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_decode/synpat/synpat_code.py 
# dest: model_decode/synpat/synpat_code.py 
"""
Synpat Module
-------------
This module provides functionality for generating and analyzing synthetic waveforms
based on predefined patterns. It includes methods for creating synthetic arrays,
generating patterns, and analyzing waveforms to identify peaks and their characteristics.

Code shall be saved in a file named: model_decode/synpat/synpat_code.py

Date: 2025-02-04

Example usage:
--------------
# Instantiate the Synpat class
synpat = Synpat()

# Generate synthetic patterns
syn_dict = synpat.generate_synthetic_patterns(fft_rate_hz=44100, dit_msec=50)

# Analyze a waveform
analysis_info = synpat.analyze_waveform(waveform=np.array([0.0, 1.0, 0.5]), pulse_width_sec=0.06)
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, NamedTuple
from scipy.signal import find_peaks, peak_widths

# Constants
SYNTHETIC_X_DEFAULT = 26
DIT_GAIN = 0.333 * 1.3
NIT_GAIN = 0.333 * 1.3
DAH_GAIN = 0.2 * 1.1
LETTER_GAIN = 0.2 * 1.1
WORD_GAIN = 0.125 * 1.0
RELATIVE_HEIGHT = 0.5
THRESHOLD_FACTOR = 0.7
WPM_FACTOR = 1.2

class Synpat:
    def __init__(self, cfg_dict: Dict = {}):
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('synpat.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # STEP_2: Initialize configuration parameters
        self.synpat_x = self._init_param(cfg_dict, 'synpat_x', SYNTHETIC_X_DEFAULT)
        self.dit_gain = self._init_param(cfg_dict, 'DIT_GAIN', DIT_GAIN)
        self.nit_gain = self._init_param(cfg_dict, 'NIT_GAIN', NIT_GAIN)
        self.dah_gain = self._init_param(cfg_dict, 'DAH_GAIN', DAH_GAIN)
        self.letter_gain = self._init_param(cfg_dict, 'LETTER_GAIN', LETTER_GAIN)
        self.word_gain = self._init_param(cfg_dict, 'WORD_GAIN', WORD_GAIN)
        self.relative_height = self._init_param(cfg_dict, 'relative_height', RELATIVE_HEIGHT)
        self.threshold_factor = self._init_param(cfg_dict, 'threshold_factor', THRESHOLD_FACTOR)
        self.wpm_factor = self._init_param(cfg_dict, 'WPM_FACTOR', WPM_FACTOR)

    def _init_param(self, cfg_dict: Dict, key: str, default: float) -> float:
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %.1f", key, default)
        return value

    def get_cfg(self) -> Dict:
        # STEP_3: Return current configuration
        return {
            'synpat_x': self.synpat_x,
            'DIT_GAIN': self.dit_gain,
            'NIT_GAIN': self.nit_gain,
            'DAH_GAIN': self.dah_gain,
            'LETTER_GAIN': self.letter_gain,
            'WORD_GAIN': self.word_gain,
            'relative_height': self.relative_height,
            'threshold_factor': self.threshold_factor,
            'WPM_FACTOR': self.wpm_factor
        }

    def create_synthetic(self, fft_rate_hz: float, dit_msec: float, pattern: List[Tuple[float, float]], amplitude_multiplier: float) -> np.ndarray:
        # STEP_4: Convert dit_msec to samples
        samples_per_dit = (fft_rate_hz * dit_msec) / 1000
        synthetic_array = []

        # STEP_5: Construct the synthetic array
        for value, duration_in_dits in pattern:
            num_samples = int(samples_per_dit * duration_in_dits) if duration_in_dits != 0 else 1
            synthetic_array.extend([value] * num_samples)

        # STEP_6: Normalize the array
        synthetic_array = np.array(synthetic_array) * amplitude_multiplier
        return synthetic_array

    def generate_synthetic_patterns(self, fft_rate_hz: float, dit_msec: float) -> Dict[str, np.ndarray]:
        # STEP_7: Generate synthetic patterns
        patterns = {
            'dit': [(-1, 1), (+1, 1), (-1, 1)],
            'ndit': [(+1, 1), (-1, 1), (+1, 1)],
            'dah': [(-1, 1), (+1, 3), (-1, 1)],
            'letter': [(+1, 1), (-1, 3), (+1, 1)],
            'word': [(+1, 1), (-1, 7)]
        }
        gains = {
            'dit': self.dit_gain,
            'ndit': self.nit_gain,
            'dah': self.dah_gain,
            'letter': self.letter_gain,
            'word': self.word_gain
        }
        syn_dict = {}
        for key, pattern in patterns.items():
            synthetic_array = self.create_synthetic(fft_rate_hz, dit_msec, pattern, gains[key])
            synthetic_array /= (fft_rate_hz * dit_msec) / 1000
            syn_dict[key] = synthetic_array
        return syn_dict

    def wpm_from_dit(self, dit_sec: float) -> float:
        # STEP_8: Calculate WPM from dit duration
        return self.wpm_factor / dit_sec

    def dit_sec(self, wpm: float) -> float:
        # STEP_9: Calculate dit duration from WPM
        return self.wpm_factor / wpm

    def generate_synthetic_dict(self, wpm_start: int, wpm_end: int, fft_rate_hz: float) -> Dict[int, Dict[str, np.ndarray]]:
        # STEP_10: Generate synthetic dictionary for WPM range
        dit_msec_start = 1000 * self.dit_sec(wpm_start)
        dit_msec_end = 1000 * self.dit_sec(wpm_end)
        dit_msec_step = (dit_msec_end - dit_msec_start) / 8
        syn_dict_wpm = {}
        for dit_msec in np.arange(dit_msec_start, dit_msec_end, dit_msec_step):
            syn_dict = self.generate_synthetic_patterns(fft_rate_hz, dit_msec)
            key = int(self.wpm_from_dit(dit_msec / 1000))
            syn_dict_wpm[key] = syn_dict
        return syn_dict_wpm

    def analyze_waveform(self, waveform: np.ndarray, pulse_width_sec: float = 0.06, prominence_factor: float = 0.7, fwhm_min_factor: float = 0.7, fwhm_max_factor: float = 1.2) -> NamedTuple:
        # STEP_11: Analyze waveform to identify peaks
        nominal_width = int(pulse_width_sec * len(waveform))
        prominence_threshold = waveform.max() * prominence_factor
        peak_height_min = waveform.max() * self.threshold_factor

        # STEP_12: Identify peaks
        peaks, _ = find_peaks(waveform, height=peak_height_min, prominence=prominence_threshold)

        # STEP_13: Compute FWHM for each peak
        results_half = peak_widths(waveform, peaks, rel_height=self.relative_height)
        widths = results_half[0].astype(int)

        # STEP_14: Mark Peaks attribute is_unfiltered
        fwhm_min = nominal_width * fwhm_min_factor
        fwhm_max = nominal_width * fwhm_max_factor

        PeakInfo = NamedTuple('PeakInfo', [('index', int), ('prominence_percent', int), ('fwhm_width', int), ('is_unfiltered', bool)])
        AnalysisInfo = NamedTuple('AnalysisInfo', [('prominence_threshold', float), ('peak_height_min', float), ('fwhm_min', int), ('nominal_width', int), ('fwhm_max', int), ('peaks', List[PeakInfo])])

        peak_info_list = []
        for i, peak in enumerate(peaks):
            prominence_percent = int(100 * (waveform[peak] - prominence_threshold) / prominence_threshold)
            is_unfiltered = fwhm_min <= widths[i] <= fwhm_max
            peak_info = PeakInfo(index=peak, prominence_percent=prominence_percent, fwhm_width=widths[i], is_unfiltered=is_unfiltered)
            peak_info_list.append(peak_info)

        return AnalysisInfo(prominence_threshold=prominence_threshold, peak_height_min=peak_height_min, fwhm_min=fwhm_min, nominal_width=nominal_width, fwhm_max=fwhm_max, peaks=peak_info_list)
