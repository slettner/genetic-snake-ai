# implements a policy the snake can use.

import abc
import h5py
import gin
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


@gin.configurable
class NeuralNetwork(AbstractSnakePolicy):

    """ Defines a neural network representing a snake policy """

    def __init__(self, hidden, state_size, action_size, activations=np.tanh, seed=0, restore=None, initialize=True):
        """
        Construct a neural network to use as a snake policy

        Args:
            hidden(tuple): Size of each hidden layer
            state_size(int): Size of the input
            action_size(int): Size of the output
            activations(func): The activation function to apply for each hidden layer
            seed(int): The seed for weights initialization
            restore(str): If restore is given the instance will restore weights from the file
            initialize(bool): Initialize the weights if true
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
            if initialize:
                self._initialize_kernel()

    def _build_network(self):
        """ Create the weights """
        prev_size = self.state_size
        for i, layer in enumerate(self.hidden):
            self.weights["layer{}".format(i + 1) + "-kernel"] = np.zeros((layer, prev_size))
            self.bias["layer{}".format(i + 1) + "-bias"] = np.zeros(layer)
            prev_size = layer
        # add output layer
        self.weights["layer{}".format(len(self.hidden) + 1) + "-kernel"] = np.zeros((self.action_size, prev_size))
        self.bias["layer{}".format(len(self.hidden) + 1) + "-bias"] = np.zeros(self.action_size)

    def _initialize_kernel(self):
        """ Initialize weight randomly to -1 or 1"""
        for _, kernel in self.weights.items():
            kernel_shape_x, kernel_shape_y = kernel.shape
            for i in range(kernel_shape_x):
                for j in range(kernel_shape_y):
                    kernel[i, j] = self.random.normal(0, 0.05)

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
        for i in range(len(self.hidden) + 1):
            weight = self.weights["layer{}".format(i + 1) + "-kernel"]
            bias = self.bias["layer{}".format(i + 1) + "-bias"]
            w_x = np.matmul(weight, out_activated)
            out = w_x + bias
            out_activated = self.activation(out)
        return softmax(out)

    @property
    def num_weights(self):
        """ Gives the total number of weights back. Kernel and bias."""
        total_number_of_weights = 0
        for _, kernel in self.weights.items():
            total_number_of_weights += kernel.shape[0] * kernel.shape[1]
        for _, bias in self.bias.items():
            total_number_of_weights += bias.shape[0]
        return total_number_of_weights

    def set_from_list(self, all_weights):
        """
        Set all weights from a flat list. Assumes ordering of layers from left to right.
        Weights then bias.

        Args:
            all_weights(list): List of -1 or 1. Size matches the total number of weights.

        Returns:

        """
        assert len(all_weights) == self.num_weights
        num_layers = len(self.hidden) + 1  # plus one for the output layer.
        ptr = 0
        for i in range(num_layers):
            # modify kernel
            old_kernel = self.weights["layer{}".format(i + 1) + "-kernel"]
            num_kernel_weights = old_kernel.shape[0] * old_kernel.shape[1]
            new_kernel = np.array(all_weights[ptr:ptr + num_kernel_weights]).reshape(old_kernel.shape)
            self.weights["layer{}".format(i + 1) + "-kernel"] = new_kernel

            ptr += num_kernel_weights

            # modify bias
            old_bias = self.bias["layer{}".format(i + 1) + "-bias"]
            num_bias_weights = old_bias.shape[0]
            new_bias = np.array(all_weights[ptr:ptr + num_bias_weights]).reshape(old_bias.shape)
            self.bias["layer{}".format(i + 1) + "-bias"] = new_bias

            ptr += num_bias_weights

    def get_weights_as_list(self):
        """ get all weights as a 1-d list """
        all_weights = []
        for i in range(len(self.hidden) + 1):
            kernel = self.weights["layer{}".format(i + 1) + "-kernel"]
            bias = self.bias["layer{}".format(i + 1) + "-bias"]
            all_weights += kernel.flatten().tolist()
            all_weights += bias.flatten().tolist()
        return all_weights

    def get_weight_as_arrays(self):
        """ Return a list of weight arrays w1, b1, w2, b2, ... """
        all_weights = []
        for i in range(len(self.hidden) + 1):
            kernel = self.weights["layer{}".format(i + 1) + "-kernel"]
            bias = self.bias["layer{}".format(i + 1) + "-bias"]
            all_weights.append(kernel)
            all_weights.append(bias)
        return all_weights

    def set_from_arrays(self, weights):
        """
        Set the weights from a list of arrays

        Args:
            weights(list): List of arrays: w1, b1, w2, b2, ...

        Returns:

        """
        for i, (w, b) in enumerate(zip(weights[0::2], weights[1::2])):
            self.weights["layer{}".format(i + 1) + "-kernel"] = w
            self.bias["layer{}".format(i + 1) + "-bias"] = b

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
        assert inputs.shape == (self.state_size,), "Input has shape {} but expected ({}, )".format(
            inputs.shape,
            self.state_size
        )
        action_distribution = self._inference(x=inputs)
        return int(np.argmax(action_distribution))  # action with max prob


@gin.configurable
class BinaryNeuralNetwork(NeuralNetwork):

    """
    A binary neural networks (-1, 1) represents the policy function.
    It takes the state as input a probability distribution
    """

    def __init__(self, hidden, state_size, action_size, activations=np.tanh, seed=0, restore=None, initialize=True):
        """
        Construct a binary neural network to use as a snake policy

        Args:
            hidden(tuple): Size of each hidden layer
            state_size(int): Size of the input
            action_size(int): Size of the output
            activations(func): The activation function to apply for each hidden layer
            seed(int): The seed for weights initialization
            restore(str): If restore is given the instance will restore weights from the file
            initialize(bool): Initialize the weights if true
        """
        super(BinaryNeuralNetwork, self).__init__(
            hidden=hidden,
            state_size=state_size,
            action_size=action_size,
            activations=activations,
            seed=seed,
            restore=restore,
            initialize=initialize
        )

    def _initialize_kernel(self):
        """ Initialize weight randomly to -1 or 1"""
        for _, kernel in self.weights.items():
            kernel_shape_x, kernel_shape_y = kernel.shape
            for i in range(kernel_shape_x):
                for j in range(kernel_shape_y):
                    kernel[i, j] = self.random.choice([1, -1])
