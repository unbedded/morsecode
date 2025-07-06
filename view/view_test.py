#  TOKENS: 3225 (of:8000) = 2615 + 610(prompt+return) -- MODEL: gpt-4o 
# policy: ./ai_sw_workflow/policy/policy_pytest.yaml 
# code: view/view_code.py 
# dest: view/view_test.py 
"""
Unit tests for the SignalView class in the view/view_code.py module.
This test suite ensures that the SignalView class behaves as expected when processing valid data.
The tests cover initialization, plot processing, and configuration retrieval.
Date: 2025-02-06
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

def test_process_chunk(signal_view):
    """Test processing a chunk of data and plotting it."""
    signal_dict = {
        'curve_1': {'target': 'plot_1', 'color': 'r', 'Hz': 1.0, 'x_origin': 0.0, 'y': np.sin(np.linspace(0, 2 * np.pi, 100))},
        'curve_2': {'target': 'plot_2', 'color': 'b', 'Hz': 1.0, 'x_origin': 0.0, 'y': np.cos(np.linspace(0, 2 * np.pi, 100))}
    }
    signal_view.process_chunk(signal_dict)
    # Check if the plots have been updated with the correct number of curves
    assert len(signal_view.curves['plot_1'].listDataItems()) == 1
    assert len(signal_view.curves['plot_2'].listDataItems()) == 1

def test_rescale_plot(signal_view):
    """Test that the rescale_plot method sets the correct ranges."""
    signal_view.rescale_plot()
    assert signal_view.curves['plot_1'].getViewBox().autoRangeEnabled()
    assert signal_view.curves['plot_2'].getViewBox().state['viewRange'][1] == [0, 40000]
    assert signal_view.curves['plot_5'].getViewBox().state['viewRange'][0] == [0, 0.120]

def test_get_cfg(signal_view):
    """Test that the get_cfg method returns the correct configuration."""
    cfg = signal_view.get_cfg()
    assert cfg['N_PLOTS'] == 7
    assert cfg['MAX_X'] == 40000
    assert cfg['rolling_buffer_seconds'] == 3.0
    assert cfg['padding_percent'] == 10
