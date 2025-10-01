import numpy as np
from neuron_vz import Neuron, VZNetwork

def test_single_neuron_example():
    # weights=[0,1], bias=4, x=[2,3] -> sigmoid(7) ≈ 0.99909
    n = Neuron([0, 1], 4)
    out = n.feedforward(np.array([2.0, 3.0]))
    assert 0.998 < out < 0.9995

def test_two_two_one_feedforward():
    # 2–2–1 net with weights=[0,1], bias=0 on all neurons
    net = VZNetwork()
    out = net.feedforward(np.array([2.0, 3.0]))
    assert 0.70 < out < 0.75