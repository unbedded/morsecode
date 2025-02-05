#  TOKENS: 3289 (of:8000) = 2543 + 746(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_pytest.yaml 
# code: view/view_code.py 
# dest: view/view_test.py 
"""
Unit tests for the SignalView class in the view/view_code.py module.
This test suite verifies the functionality of the SignalView class, ensuring it correctly processes and displays plot data.
The tests cover initialization, data processing, plot clearing, and configuration retrieval.
Date: 2025-02-04
"""

import pytest
import numpy as np
from typing import Dict
from pyqtgraph.Qt import QtWidgets
from view.view_code import SignalView, PlotDataDict

@pytest.fixture(scope="function")
def signal_view():
    """Fixture to create and teardown a SignalView instance."""
    view = SignalView()
    yield view
    view.teardown()
    QtWidgets.QApplication.instance().quit()

def test_initialization(signal_view):
    """Test that the SignalView initializes with the correct number of plots."""
    assert len(signal_view.curves) == 7
    for i in range(1, 8):
        assert f"plot_{i}" in signal_view.curves

def test_clear_plots(signal_view):
    """Test that clear_plots method clears the specified plots."""
    signal_dict = {
        "curve_1": PlotDataDict(target="plot_1", color='b', Hz=1.0, x_origin=0.0, y=np.array([0, 1, 0])),
        "curve_2": PlotDataDict(target="plot_2", color='r', Hz=2.0, x_origin=0.0, y=np.array([1, 0, 1]))
    }
    signal_view.process_chunk(signal_dict)
    signal_view.clear_plots(signal_dict)
    for plot_name in signal_view.curves:
        assert len(signal_view.curves[plot_name].listDataItems()) == 0

@pytest.mark.parametrize("plot_name, expected_range", [
    ("plot_2", (0, 800000)),
    ("plot_3", (0, 800000)),
    ("plot_5", (0, 10)),
    ("plot_4", (-1, 1)),
    ("plot_6", (-1, 1)),
    ("plot_7", (-1, 1))
])
def test_rescale_plot(signal_view, plot_name, expected_range):
    """Test that rescale_plot method sets the correct Y-axis range."""
    signal_view.rescale_plot()
    y_range = signal_view.curves[plot_name].getViewBox().viewRange()[1]
    assert y_range == list(expected_range)

def test_process_chunk(signal_view):
    """Test that process_chunk method correctly processes and plots data."""
    signal_dict = {
        "curve_1": PlotDataDict(target="plot_1", color='b', Hz=1.0, x_origin=0.0, y=np.array([0, 1, 0])),
        "curve_2": PlotDataDict(target="plot_2", color='r', Hz=2.0, x_origin=0.0, y=np.array([1, 0, 1]))
    }
    signal_view.process_chunk(signal_dict)
    for curve_name, data in signal_dict.items():
        plot = signal_view.curves[data['target']]
        assert len(plot.listDataItems()) == 1

def test_get_cfg(signal_view):
    """Test that get_cfg method returns the correct configuration."""
    cfg = signal_view.get_cfg()
    assert cfg['rolling_buffer_seconds'] == 3.0
