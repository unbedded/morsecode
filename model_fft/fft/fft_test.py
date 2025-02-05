#  TOKENS: 2913 (of:8000) = 2046 + 867(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_pytest.yaml 
# code: model_fft/fft/fft_code.py 
# dest: model_fft/fft/fft_test.py 
#  TOKENS: 2132 (of:8000) = 1087 + 1045(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: model_fft/fft/fft_test.py 
# dest: model_fft/fft/fft_test.py 
"""
Unit tests for the Streaming FFT processing class, designed to handle real-time
signal processing using Fast Fourier Transform (FFT). The tests cover various
scenarios including edge cases and ensure the class behaves as expected.

Code shall be saved in a file named: model_fft/fft/fft_test.py

<DATE>: 2025-02-04
"""

import pytest
import numpy as np
from model_fft.fft.fft_code import Streaming_fft
from typing import Tuple

@pytest.fixture
def fft_processor():
    """Fixture to initialize the Streaming_fft class with default parameters."""
    return Streaming_fft(cfg_dict={'window': 'hamming'})

@pytest.fixture
def sin_wave_test_signal() -> Tuple[np.ndarray, int]:
    """Fixture to generate a sine wave test signal."""
    sample_freq_hz = 10000
    sine_freq_hz = 500
    num_samples = 20000
    t = np.arange(num_samples) / sample_freq_hz
    signal = 0.5 * np.sin(2 * np.pi * sine_freq_hz * t)
    return signal, sample_freq_hz

def test_fft_with_sine_wave(fft_processor, sin_wave_test_signal):
    """
    Test the FFT method with a sine wave input signal.
    Ensures the peak frequency is detected correctly within the frequency bin resolution.
    """
    signal, sample_freq_hz = sin_wave_test_signal
    frequency_bins, time_bins, fft_magnitude = fft_processor.fft(audio_freq_hz=sample_freq_hz, current_segment=signal)

    # Find the peak frequency in the FFT magnitude
    peak_index = np.argmax(fft_magnitude)
    peak_frequency = abs(frequency_bins[peak_index])

    # Assert the peak frequency is within the expected range
    expected_frequency = 500  # Hz
    frequency_resolution = sample_freq_hz / len(frequency_bins)
    assert expected_frequency - frequency_resolution <= peak_frequency <= expected_frequency + frequency_resolution

def test_fft_invalid_audio_freq(fft_processor):
    """
    Test the FFT method with an invalid audio frequency.
    Ensures that a ValueError is raised for non-positive frequencies.
    """
    with pytest.raises(ValueError, match="Audio frequency must be greater than 0."):
        fft_processor.fft(audio_freq_hz=0, current_segment=np.array([1, 2, 3]))

def test_fft_invalid_window_function():
    """
    Test the initialization of Streaming_fft with an invalid window function.
    Ensures that a ValueError is raised for unsupported window functions.
    """
    with pytest.raises(ValueError, match="Invalid window function specified."):
        Streaming_fft(cfg_dict={'window': 'invalid'}).fft(audio_freq_hz=1000, current_segment=np.array([1, 2, 3]))

def test_fft_with_empty_segment(fft_processor):
    """
    Test the FFT method with an empty current segment.
    Ensures that the method handles empty input gracefully.
    """
    frequency_bins, time_bins, fft_magnitude = fft_processor.fft(audio_freq_hz=1000, current_segment=np.array([]))
    assert len(frequency_bins) == 0
    assert len(time_bins) == 0
    assert len(fft_magnitude) == 0

def test_plot_test():
    """
    Test the plot_test function with a sample WAV file.
    This test is a placeholder and should be implemented with actual plotting logic.
    """
    # Placeholder for actual plot test implementation
    wav_file_path = "/home/preact/sw/morsecode/wav/131210_20WPM.wav"
    assert isinstance(wav_file_path, str)
