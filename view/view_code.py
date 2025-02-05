#  TOKENS: 2730 (of:8000) = 1488 + 1242(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_python3.8.yaml 
# code: view/view_code.py 
# dest: view/view_code.py 
"""
This module provides a View class named SignalView that displays multiple rows of plots.
The code is compatible with Python 3.8 and uses the PyQtGraph library for plotting.
The code shall be saved in a file named: view/view_code.py
Date: 2025-02-04

Example usage:
- The SignalView class can be used to create a plot display window with multiple rows of graphs.
- The class supports configuration management and provides methods to process and clear plot data.
- The main function demonstrates how to generate sine wave data and plot it using the SignalView class.
"""

import numpy as np
import logging
from typing import Dict, TypedDict
from pyqtgraph.Qt import QtWidgets
import pyqtgraph as pg

# Constants
N_PLOTS = 7
MAX_X = 800000
PADDING_PERCENT = 10
DEFAULT_COLOR = 'g'  # Green
DEFAULT_ROLLING_BUFFER_SECONDS = 3.0

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
        logging.basicConfig(filename='view.log', level=logging.WARNING,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger.setLevel(logging.WARNING)

        # STEP_2: Initialize configuration
        self.cfg_dict = cfg_dict
        self.rolling_buffer_seconds = self._init_param('rolling_buffer_seconds', DEFAULT_ROLLING_BUFFER_SECONDS)

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
        for i in range(1, N_PLOTS + 1):
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
                Hz = data['Hz']
                y = data['y']
                N = len(y)
                x = np.linspace(0, (N - 1) * (1 / Hz), N)
                color = data.get('color', DEFAULT_COLOR)
                plot.plot(x, y, pen=color)

    def rescale_plot(self):
        self.curves['plot_2'].setYRange(0, MAX_X)
        self.curves['plot_3'].setYRange(0, MAX_X)
        self.curves['plot_5'].setYRange(0, 10)
        self.curves['plot_5'].setXRange(0, 0.120)
        self.curves['plot_4'].setYRange(-1, 1)
        self.curves['plot_6'].setYRange(-1, 1)
        self.curves['plot_7'].setYRange(-1, 1)

        # Rescale plot_1 with padding
        min_y, max_y = self.curves['plot_1'].viewRange()[1]
        padding = (max_y - min_y) * PADDING_PERCENT / 100
        self.curves['plot_1'].setYRange(min_y - padding, max_y + padding)

    def teardown(self):
        self.win.close()

    def run(self):
        self.app.exec_()

    def get_cfg(self) -> Dict:
        return {**self.cfg_dict, 'rolling_buffer_seconds': self.rolling_buffer_seconds}

def main():
    import sys
    import math

    # STEP_6: Create SignalView instance
    view = SignalView()

    # STEP_7: Generate sine wave data
    signal_dict = {}
    for i in range(1, N_PLOTS + 1):
        Hz = 1.0 + i
        x_origin = 0.0
        y = np.array([math.sin(2 * math.pi * Hz * t) for t in np.linspace(0, 1, 100)])
        signal_dict[f"curve_{i}"] = PlotDataDict(target=f"plot_{i}", color='b', Hz=Hz, x_origin=x_origin, y=y)

    # STEP_8: Process and display data
    try:
        view.process_chunk(signal_dict)
        view.rescale_plot()
        view.run()
    except Exception as e:
        view.logger.exception("An error occurred: %s", e)
    finally:
        view.teardown()

if __name__ == "__main__":
    main()
