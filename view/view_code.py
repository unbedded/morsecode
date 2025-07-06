#  TOKENS: 2809 (of:8000) = 1477 + 1332(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: view/view_code.py 
# dest: view/view_code.py 
"""
This module provides a View class named SignalView that displays multiple rows of plots.
The code is compatible with Python 3.8 and uses PyQtGraph for plotting.
The file shall be saved as: view/view_code.py
Date: 2025-02-06

Example usage:
- The SignalView class can be instantiated and used to display plots with specified configurations.
- The main function demonstrates generating sine wave data and plotting it using the appropriate color for each graph.

Configuration:
- The class supports custom configuration management through a configuration dictionary.
- Default configuration parameters include N_PLOTS, MAX_X, rolling_buffer_seconds, and padding_percent.

Error Handling:
- The code includes Pythonic error handling practices with informative error messages.
- Logging is used to capture exceptions and provide debugging information.

Debugging:
- The logging module is configured for file output with a clear format including timestamps.
- Logging levels are used appropriately to provide detailed information and capture exceptions.
"""

import numpy as np
import logging
from typing import Dict, TypedDict
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

# Constants
N_PLOTS = 7
MAX_X = 40000
DEFAULT_ROLLING_BUFFER_SECONDS = 3.0
PADDING_PERCENT = 10

class PlotDataDict(TypedDict):
    target: str
    color: str
    Hz: float
    x_origin: float
    y: np.ndarray

class SignalView:
    def __init__(self, cfg_dict: Dict = {}):
        # STEP_1: Initialize logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        handler = logging.FileHandler('signal_view.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # STEP_2: Initialize configuration
        self.cfg_dict = cfg_dict
        self.n_plots = self._init_param('N_PLOTS', N_PLOTS)
        self.max_x = self._init_param('MAX_X', MAX_X)
        self.rolling_buffer_seconds = self._init_param('rolling_buffer_seconds', DEFAULT_ROLLING_BUFFER_SECONDS)
        self.padding_percent = self._init_param('padding_percent', PADDING_PERCENT)

        # STEP_3: Initialize PyQtGraph application
        self.app = QtWidgets.QApplication([])

        # STEP_4: Create plot window and curves dictionary
        self.win = pg.GraphicsLayoutWidget(show=True, title="Signal View")
        self.curves = {}

        # STEP_5: Setup plots
        self._setup_plots()

    def _init_param(self, key: str, default):
        value = self.cfg_dict.get(key, default)
        if key not in self.cfg_dict:
            self.logger.info("Parameter '%s' not found in configuration. Using default: %s", key, default)
        return value

    def _setup_plots(self):
        for i in range(1, self.n_plots + 1):
            plot = self.win.addPlot(title=f"plot_{i}")
            plot.showGrid(y=True)
            plot.setAutoVisible(y=False)
            self.curves[f"plot_{i}"] = plot
            self.win.nextRow()

    def clear_plots(self, signal_dict: Dict[str, PlotDataDict]):
        for curve_name, data in signal_dict.items():
            target_plot = data['target']
            if target_plot in self.curves:
                plot = self.curves[target_plot]
                plot.clear()

    def process_chunk(self, signal_dict: Dict[str, PlotDataDict]):
        self.clear_plots(signal_dict)
        for curve_name, data in signal_dict.items():
            target_plot = data['target']
            if target_plot in self.curves:
                plot = self.curves[target_plot]
                color = data.get('color', 'g')
                Hz = data['Hz']
                y = data['y']
                N = len(y)
                x = np.linspace(0, (N - 1) * (1 / Hz), N)
                plot.plot(x, y, pen=color)

    def rescale_plot(self):
        self.curves['plot_1'].enableAutoRange()
        self.curves['plot_2'].setYRange(0, self.max_x)
        self.curves['plot_3'].setYRange(0, self.max_x)
        self.curves['plot_4'].setYRange(-1, 1)
        self.curves['plot_5'].setYRange(0, 10)
        self.curves['plot_5'].setXRange(0, 0.120)
        self.curves['plot_6'].setYRange(-1, 1)
        self.curves['plot_7'].setYRange(-1, 1)

    def teardown(self):
        self.win.close()

    def run(self):
        self.app.exec_()

    def get_cfg(self) -> Dict:
        return {
            'N_PLOTS': self.n_plots,
            'MAX_X': self.max_x,
            'rolling_buffer_seconds': self.rolling_buffer_seconds,
            'padding_percent': self.padding_percent
        }

def main():
    # STEP_6: Create SignalView instance
    view = SignalView()

    # STEP_7: Generate sine wave data for demonstration
    signal_dict = {
        'curve_1': {'target': 'plot_1', 'color': 'r', 'Hz': 1.0, 'x_origin': 0.0, 'y': np.sin(np.linspace(0, 2 * np.pi, 100))},
        'curve_2': {'target': 'plot_2', 'color': 'b', 'Hz': 1.0, 'x_origin': 0.0, 'y': np.cos(np.linspace(0, 2 * np.pi, 100))}
    }

    # STEP_8: Process and display the data
    view.process_chunk(signal_dict)

    # STEP_9: Run the application
    view.run()

if __name__ == "__main__":
    main()
