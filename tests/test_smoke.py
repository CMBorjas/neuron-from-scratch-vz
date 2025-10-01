import numpy as np
from neuron_vz import VZNetwork

def test_feedforward_runs_between_0_and_1():
    net = VZNetwork(seed=0)
    out = net.feedforward(np.array([0.0, 0.0]))
    assert 0.0 <= out <= 1.0
