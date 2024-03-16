import numpy as np
from classifier import Classifier
from layers import fc_forward, fc_backward, relu_forward, relu_backward


class TwoLayerNet(Classifier):
    """
    A neural network with two layers, using a ReLU nonlinearity on its one
    hidden layer. That is, the architecture should be:

    input -> FC layer -> ReLU layer -> FC layer -> scores
    """
    def __init__(self, input_dim=3072, num_classes=10, hidden_dim=512,
                 weight_scale=1e-3):
        """
        Initialize a new two layer network.

        Inputs:
        - input_dim: The number of dimensions in the input.
        - num_classes: The number of classes over which to classify
        - hidden_dim: The size of the hidden layer
        - weight_scale: The weight matrices of the model will be initialized
          from a Gaussian distribution with standard deviation equal to
          weight_scale. The bias vectors of the model will always be
          initialized to zero.
        """
        self.w1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros(hidden_dim)

        self.w2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.b2 = np.zeros(num_classes)


    def parameters(self):
        params = {}
        # TODO: Build a dict of all learnable parameters of this model.
        params["w1"] = self.w1
        params["b1"] = self.b1
        params["w2"] = self.w2
        params["b2"] = self.b2

        return params

    def forward(self, X):
        scores, cache = None, None
        # TODO: Implement the forward pass to compute classification scores
        # for the input data X. Store into cache any data that will be needed #
        # during the backward pass.                                           #
        out1, cache1 = fc_forward(X, self.w1, self.b1)
        out2, cache2 = relu_forward(out1)
        scores, cache3 = fc_forward(out2, self.w2, self.b2)
        cache = (cache1, cache2, cache3)


        return scores, cache

    def backward(self, grad_scores, cache):
        grads = {}
        #######################################################################
        # TODO: Implement the backward pass to compute gradients for all      #
        # learnable parameters of the model, storing them in the grads dict   #
        # above. The grads dict should give gradients for all parameters in   #
        # the dict returned by model.parameters().                            #
        #######################################################################
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
        cache1, cache2, cache3 = cache

        _, grads["w2"], grads["b2"] = fc_backward(grad_scores, cache3)
        grads2 = relu_backward(_, cache2)
        g, grads["w1"], grads["b1"] = fc_backward(grads2, cache1)

        return grads
