import numpy as np
from neuron_vz import Neuron, VZNetwork, sigmoid

def test_single_neuron_example():
    # weights=[0,1], bias=4, inputs=[2,3] -> sigmoid(7) ≈ 0.99908895
    n = Neuron(weights=[0, 1], bias=4)
    out = n.feedforward_float(np.array([2.0, 3.0]))
    assert 0.998 < out < 0.9995

def test_two_two_one_feedforward():
    # With all weights=[0,1], bias=0:
    # h1 = sigmoid(3), h2 = sigmoid(3), o1 = sigmoid(h2) ≈ 0.72163256
    net = VZNetwork()
    out = net.feedforward_float(np.array([2.0, 3.0]))
    assert 0.70 < out < 0.75
