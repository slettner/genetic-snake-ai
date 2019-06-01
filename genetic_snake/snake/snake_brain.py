# implements a policy the snake can use.

import abc
import h5py
import numpy as np
from ..nn import softmax


class AbstractSnakePolicy(abc.ABC):

    """ Interface for snake policies """

    @abc.abstractmethod
    def decide(self, reason):
        """
        Make a decision based on reason.
        Args:
            reason(dict): Maps names to numerical values. Usually returned from what the snake senses.

        Returns:
            decision(int): The decision represented by the index of the snakes actions.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, name):
        raise NotImplementedError

    @abc.abstractmethod
    def restore(self, name):
        raise NotImplementedError


class BinaryNeuralNetwork(AbstractSnakePolicy):

    """
    A binary neural networks (-1, 1) represents the policy function.
    It takes the state as input a probability distribution
    """

    def __init__(self, hidden, state_size, action_size, activations, seed=0, restore=None):
        """
        Construct a binary neural network to use as a snake policy

        Args:
            hidden(tuple): Size of each hidden layer
            state_size(int): Size of the input
            action_size(int): Size of the output
            activations(func): The activation function to apply for each hidden layer
            seed(int): The seed for weights initialization
            restore(str): If restore is given the instance will restore weights from the file
        """
        self.hidden = hidden
        self.state_size = state_size
        self.action_size = action_size
        self.activation = activations

        self.weights = {}
        self.bias = {}
        self.random = np.random.RandomState(seed=seed)

        if restore is not None:
            self.restore(restore)
        else:
            self._build_network()
            self._initialize_kernel()

    def _build_network(self):
        """ Create the weights """
        prev_size = self.state_size
        for i, layer in enumerate(self.hidden):
            self.weights["layer{}".format(i+1)+"-kernel"] = np.zeros((layer, prev_size))
            self.bias["layer{}".format(i+1)+"-bias"] = np.zeros(layer)
            prev_size = layer
        # add output layer
        self.weights["layer{}".format(len(self.hidden)+1)+"-kernel"] = np.zeros((self.action_size, prev_size))
        self.bias["layer{}".format(len(self.hidden)+1)+"-bias"] = np.zeros(self.action_size)

    def _initialize_kernel(self):
        """ Initialize weight randomly to -1 or 1"""
        for _, kernel in self.weights.items():
            kernel_shape_x, kernel_shape_y = kernel.shape
            for i in range(kernel_shape_x):
                for j in range(kernel_shape_y):
                    kernel[i, j] = self.random.choice([1, -1])

    def save(self, name):
        """ Save weights to hdf5 """
        with h5py.File(name + ".h5", "w") as h5:
            for key, kernel in self.weights.items():
                h5.create_dataset(name=key, shape=kernel.shape, data=kernel)
            for key, bias in self.bias.items():
                h5.create_dataset(name=key, shape=bias.shape, data=bias)

    def restore(self, name):
        """ restore weights from file. This replaces build_network and initialize_kernel """
        with h5py.File(name + ".h5", "r") as h5:
            for key, dataset in h5.items():
                if "kernel" in key:
                    self.weights[key] = dataset.value
                if "bias" in key:
                    self.bias[key] = dataset.value

    def _inference(self, x):
        """
        Make inference.

        Args:
            x(np.array): The inputs.
        """
        out = None
        out_activated = x
        for i in range(len(self.hidden)+1):
            weight = self.weights["layer{}".format(i+1)+"-kernel"]
            bias = self.bias["layer{}".format(i+1)+"-bias"]
            w_x = np.matmul(weight, out_activated)
            out = w_x + bias
            out_activated = self.activation(out)
        return softmax(out)

    def decide(self, reason):
        """
        Make a forward pass through the network

        Args:
            reason(dict): This class expects a single input of shape (self.state_size, )

        Returns:
            decision(int)
        """
        assert len(reason) == 1, "More than one input given"
        inputs = list(reason.values())[0]
        assert inputs.shape == (self.state_size, ), "Input has shape {} but expected ({}, )".format(
            inputs.shape,
            self.state_size
        )
        action_distribution = self._inference(x=inputs)
        return int(np.argmax(action_distribution))  # action with max prob
