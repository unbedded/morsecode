#  TOKENS: 5886 (of:8000) = 4960 + 926(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_pytest.yaml 
# code: model_decode/synpat/synpat_code.py 
# dest: model_decode/synpat/synpat_test.py 
# synpat_test.py
# ----------------
# Unit tests for the Synpat module
#
# This file contains unit tests for the Synpat class, which provides functionality
# for generating and analyzing synthetic waveforms based on predefined patterns.
# The tests cover various scenarios, including edge cases and expected outputs.
#
# Date: 2025-02-04

import pytest
import numpy as np
from numpy.testing import assert_array_equal
from model_decode.synpat.synpat_code import Synpat

@pytest.fixture
def synpat():
    """Fixture to create a Synpat instance."""
    return Synpat()

def test_generate_synthetic_patterns(synpat):
    """Test the generate_synthetic_patterns method for expected output."""
    fft_rate_hz = 100
    dit_msec = 10
    expected_output = {
        "dit": np.array([-9, 9, -9]),
        "ndit": np.array([9, -9, 9]),
        "dah": np.array([-5, 5, 5, 5, -5]),
        "letter": np.array([5, -5, -5, -5, 5]),
        "word": np.array([3, -3, -3, -3, -3, -3, -3, -3])
    }
    result = synpat.generate_synthetic_patterns(fft_rate_hz, dit_msec)
    for key in expected_output:
        assert_array_equal(result[key], expected_output[key])

@pytest.mark.parametrize("fft_rate_hz, dit_msec, pattern, amplitude_multiplier, expected_output", [
    (100, 50, [(1, 2), (-1, 1), (0, 1)], 2, np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, -2, -2, -2, -2, -2, 0, 0, 0, 0, 0])),
    (100, 20, [(1, 0.5), (0, 2)], 2, np.array([2, 0, 0, 0, 0]))
])
def test_create_synthetic(synpat, fft_rate_hz, dit_msec, pattern, amplitude_multiplier, expected_output):
    """Test the create_synthetic method for various patterns and multipliers."""
    result = synpat.create_synthetic(fft_rate_hz, dit_msec, pattern, amplitude_multiplier)
    assert_array_equal(result, expected_output)

@pytest.mark.parametrize("waveform_name, dit_msec, prominence_factor, fft_rate_hz, expected_index_range, expected_prominence_percent_range, expected_fwhm_width, expected_is_unfiltered", [
    ("WF1", 100, 0.5, 100, (33, 37), (98, 102), 10, True),
    ("WF2", 100, 0.5, 100, (33, 37), (98, 102), 10, True)
])
def test_analyze_waveform(synpat, waveform_name, dit_msec, prominence_factor, fft_rate_hz, expected_index_range, expected_prominence_percent_range, expected_fwhm_width, expected_is_unfiltered):
    """Test the analyze_waveform method with various parameters."""
    if waveform_name == "WF1":
        waveform = np.zeros(100)
        waveform[30:40] = 50
    elif waveform_name == "WF2":
        waveform = np.zeros(100)
        waveform[30:40] = 51

    pulse_width_sec = dit_msec / 1000
    analysis_info = synpat.analyze_waveform(waveform, pulse_width_sec, prominence_factor)

    assert len(analysis_info.peaks) == 1
    peak_info = analysis_info.peaks[0]
    assert expected_index_range[0] <= peak_info.index <= expected_index_range[1]
    assert expected_prominence_percent_range[0] <= peak_info.prominence_percent <= expected_prominence_percent_range[1]
    assert peak_info.fwhm_width == expected_fwhm_width
    assert peak_info.is_unfiltered == expected_is_unfiltered
