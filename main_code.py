#  TOKENS: 9289 (of:8000) = 8477 + 812(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: main_code.py 
# dest: main_code.py 
"""
This module implements a Model-View-Controller (MVC) architecture for processing and visualizing Morse code signals.
The code is designed to be saved in a file named: main_code.py
Date: 2025-02-04

The architecture consists of the following components:
- HardwareAbstractionLayer: Handles audio data input.
- SignalModelFft: Performs FFT analysis on audio signals.
- SignalModelFilter: Filters frequency domain signals.
- SignalModelDecode: Decodes Morse code from processed signals.
- SignalView: Visualizes the processed signals.
- SignalController: Manages the interaction between models and the view.

Example usage:
- Load configuration from a YAML file.
- Instantiate and set up MVC components.
- Save updated configuration.
- Start the controller to process and visualize signals.
"""

import logging
import yaml
from typing import Dict
from hal.hal_code import HardwareAbstractionLayer
from model_fft.model_fft_code import SignalModelFft
from model_filter.model_filter_code import SignalModelFilter
from model_decode.model_decode_code import SignalModelDecode
from view.view_code import SignalView
from controller.controller_code import SignalController

CONFIG_FILE_PATH = './morse_cfg.yaml'

def load_configuration(file_path: str) -> Dict:
    """Load configuration from a YAML file."""
    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file) or {}
    except FileNotFoundError:
        logging.warning("Configuration file not found. Creating a new one.")
        config = {}
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)
    except Exception as e:
        logging.exception("Error loading configuration: %s", e)
        raise
    return config

def save_configuration(file_path: str, config: Dict):
    """Save configuration to a YAML file."""
    try:
        with open(file_path, 'w') as file:
            yaml.safe_dump(config, file)
    except Exception as e:
        logging.exception("Error saving configuration: %s", e)
        raise

def main():
    # STEP_1: Load configuration
    config = load_configuration(CONFIG_FILE_PATH)

    # STEP_2a: Instantiate classes
    hal = HardwareAbstractionLayer(cfg_dict=config.get('HardwareAbstractionLayer', {}))
    model_fft = SignalModelFft(cfg_dict=config.get('SignalModelFft', {}))
    model_filter = SignalModelFilter(cfg_dict=config.get('SignalModelFilter', {}))
    model_decode = SignalModelDecode(cfg_dict=config.get('SignalModelDecode', {}))
    view = SignalView(cfg_dict=config.get('SignalView', {}))
    controller = SignalController(
        hal=hal,
        model_fft=model_fft,
        model_filter=model_filter,
        model_decode=model_decode,
        view=view,
        cfg_dict=config.get('SignalController', {})
    )

    # STEP_2b: Setup classes
    model_fft.set_audio_rate_hz(hal.get_audio_rate_hz())
    fft_rate_hz = model_fft.get_fft_rate_hz()
    model_filter.set_fft_rate_hz(fft_rate_hz)
    model_decode.set_fft_rate_hz(fft_rate_hz)

    # STEP_3: Save configuration
    config.update({
        'HardwareAbstractionLayer': hal.get_cfg(),
        'SignalModelFft': model_fft.get_cfg(),
        'SignalModelFilter': model_filter.get_cfg(),
        'SignalModelDecode': model_decode.get_cfg(),
        'SignalView': view.get_cfg(),
        'SignalController': controller.get_cfg()
    })
    save_configuration(CONFIG_FILE_PATH, config)

    # STEP_4: Start the controller
    controller.start()

if __name__ == "__main__":
    main()
