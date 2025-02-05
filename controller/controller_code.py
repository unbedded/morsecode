#  TOKENS: 2550 (of:8000) = 1244 + 1306(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: controller/controller_code.py 
# dest: controller/controller_code.py 
"""
Controller module for managing the interaction between SignalModel and SignalView.
This module is part of a Model/View/Controller architecture and is responsible for
coordinating updates, processing, and visualization of signal data.

Code shall be saved in a file named: controller/controller_code.py

Example usage:
"""

import logging
from typing import Dict
from PyQt5.QtCore import QTimer

class SignalController:
    """
    SignalController class manages the interaction between the hardware abstraction layer,
    signal models, and the signal view. It periodically updates the signal processing
    and visualization based on the provided configuration.

    Attributes:
        hal: Hardware Abstraction Layer class instance.
        model_fft: Instance of the SignalModelFft class.
        model_filter: Instance of the SignalModelFilter class.
        model_decode: Instance of the SignalModelDecode class.
        view: Instance of the SignalView class.
    """

    # Constants for default configuration values
    DEFAULT_UPDATE_INTERVAL_MS = 5
    DEFAULT_RESCALE_INTERVAL_SEC = 3
    DEFAULT_PLOT_INTERVAL_SEC = 0.125
    DEFAULT_DECODE_INTERVAL_SEC = 0.5
    DEFAULT_FFT_RATE_HZ = 200

    def __init__(self, hal, model_fft, model_filter, model_decode, view, cfg_dict: Dict = {}):
        """
        Initializes the SignalController with the given hardware abstraction layer,
        signal models, and view. Configures the controller based on the provided
        configuration dictionary.

        :param hal: Hardware Abstraction Layer class instance.
        :param model_fft: Instance of the SignalModelFft class.
        :param model_filter: Instance of the SignalModelFilter class.
        :param model_decode: Instance of the SignalModelDecode class.
        :param view: Instance of the SignalView class.
        :param cfg_dict: Configuration dictionary for initializing parameters.
        """
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('controller.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Initialize class attributes
        self.hal = hal
        self.model_fft = model_fft
        self.model_filter = model_filter
        self.model_decode = model_decode
        self.view = view

        # Initialize configuration parameters
        self.update_interval_ms = self._init_param(cfg_dict, 'update_interval_ms', self.DEFAULT_UPDATE_INTERVAL_MS)
        self.rescale_interval_sec = self._init_param(cfg_dict, 'rescale_interval_sec', self.DEFAULT_RESCALE_INTERVAL_SEC)
        self.plot_interval_sec = self._init_param(cfg_dict, 'plot_interval_sec', self.DEFAULT_PLOT_INTERVAL_SEC)
        self.decode_interval_sec = self._init_param(cfg_dict, 'decode_interval_sec', self.DEFAULT_DECODE_INTERVAL_SEC)
        self.fft_rate_hz = self._init_param(cfg_dict, 'fft_rate_hz', self.DEFAULT_FFT_RATE_HZ)

        # Initialize counters
        self._plot_counter = 0
        self._rescale_counter = 0
        self._decode_counter = 0

        # STEP_2: Call model_decode.process_sec(decode_interval_sec)
        self.model_decode.process_sec(self.decode_interval_sec)

        # Setup QTimer for periodic updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)

    def _init_param(self, cfg_dict: Dict, key: str, default: int) -> int:
        """
        Helper method to initialize a configuration parameter with a default value
        if the key is missing in the configuration dictionary.

        :param cfg_dict: Configuration dictionary.
        :param key: Key for the configuration parameter.
        :param default: Default value if the key is not found.
        :return: The initialized parameter value.
        """
        value = cfg_dict.get(key, default)
        if key not in cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %d", key, default)
        return value

    def get_cfg(self) -> Dict:
        """
        Returns the current configuration parameters as a dictionary.

        :return: Dictionary of configuration parameters.
        """
        return {
            'update_interval_ms': self.update_interval_ms,
            'rescale_interval_sec': self.rescale_interval_sec,
            'plot_interval_sec': self.plot_interval_sec,
            'decode_interval_sec': self.decode_interval_sec,
            'fft_rate_hz': self.fft_rate_hz
        }

    def update(self):
        """
        Periodically updates the signal processing and visualization.
        """
        try:
            # STEP_1: INCREMENT COUNTERS
            self._plot_counter += 1
            self._rescale_counter += 1
            self._decode_counter += 1

            # STEP_2: PROCESS FFT CHUNK
            audio_data_chunk = self.hal.get_next_chunk(self.update_interval_ms)
            signal_dict = self.model_fft.process_chunk(audio_data_chunk)
            signal_dict = self.model_filter.process_chunk(signal_dict)

            # STEP_3: DECODE CHUNK
            if self._decode_counter >= self.fft_rate_hz * self.decode_interval_sec:
                signal_dict = self.model_decode.process_chunk(signal_dict)
                self.view.process_chunk(signal_dict)
                self._decode_counter = 0

            # STEP_4: PLOT CHUNK
            if self._plot_counter >= self.fft_rate_hz * self.plot_interval_sec:
                self.view.process_chunk(signal_dict)
                self._plot_counter = 0

            # STEP_5: RESCALE PLOT
            if self._rescale_counter >= self.fft_rate_hz * self.rescale_interval_sec:
                self.view.rescale_plot()
                self._rescale_counter = 0

        except Exception as e:
            self.logger.exception("An error occurred during update: %s", str(e))
            raise

    def start(self):
        """
        Starts the periodic updates and the display application loop.
        """
        self.timer.start(self.update_interval_ms)
        self.view.run()
