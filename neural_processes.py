import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.distributions import MultivariateNormal


class NeuralProcess(object):
    '''
    A class that represents a Neural Process model. This model can be
    used to learn a distribution over a family of functions.
    '''

    def __init__(self, parameters):
        '''
         Arguments:
        parameters: a dictionary storing all the parameters necessary to 
                    construct the encoder and decoder networks of the model.

                    x_dim: the input dimension
                    y_dim: the output dimension
                    r_dim: the representation dimension
                    z_dim: the latent variable dimension
                    h_hidden: the number of hidden neurons in each layer of
                              the encoder
                    g_hidden: the number of hidden neurons in each layer of
                              the decoder
                    h_activation: the nn.Module activation function used in
                                  each hidden layer of the encoder
                    g_activation: the nn.Module activation function used in
                                  each hidden layer of the decoder
        '''

        # Extracts the parameters from the dictionary.
        self.x_dim = parameters['x_dim']
        self.y_dim = parameters['y_dim']
        self.r_dim = parameters['r_dim']
        self.z_dim = parameters['z_dim']
        self.h_hidden = parameters['h_hidden']
        self.g_hidden = parameters['g_hidden']
        self.h_activation = parameters['h_activation']
        self.g_activation = parameters['g_activation']

        # Instantiate the encoder and decoder networks as None.
        self.encoder = None
        self.decoder = None

        # Instantiate Encoder and Decoder
        self.instantiate_networks()

    def instantiate_networks(self):
        '''
        Instantiates the encoder and decoder networks using
        the parameters extracted in the constructor.
        '''

        self.encoder = Encoder(input_size=self.x_dim + self.y_dim,
                               r_dim=self.r_dim,
                               z_dim=self.z_dim,
                               hidden_size=self.h_hidden,
                               activation=self.h_activation)

        self.decoder = Decoder(z_dim=self.z_dim,
                               x_dim=self.x_dim,
                               y_dim=self.y_dim,
                               hidden_size=self.g_hidden,
                               activation=self.g_activation)

    def fit(self, x_context, y_context, x_target, y_target, number_of_samples):
        '''
        Given context and target points, computes through the pipeline of
        the Neural Process. This includes calculating the latent distribution
        parameters through the encoder, sampling from these parameters, and 
        using the sample with x targets to produce a prediction for the y targets.

        Arguments:
            x_context: the x contextual points
            y_context: the y contextual points
            x_target: the x target points
            y_target: the y target points
            number_of_samples: the number of samples to take from the latent distribution.

        Returns:
            outputs: a dictionary containing:
                     y_pred_mu: the mean predicted value of y target
                     y_pred_std: the std of the predicted value of y target
                     z_mu_context: the latent variable mean for the context
                     z_std_context: the latent variable std for the context
                     z_mu_all: the latent variable mean for all the points
                     z_std_all: the latent variable std for all the points

        '''
        x_all = torch.cat((x_context, x_target), dim=1)
        y_all = torch.cat((y_context, y_target), dim=1)

        z_mu_context, z_std_context = self.encoder(x_context, y_context)
        z_mu_all, z_std_all = self.encoder(x_all, y_all)

        eps_samples = torch.randn((number_of_samples, self.z_dim))

        z_samples = z_std_all.mul(eps_samples) + z_mu_all
        z_samples = z_samples.reshape(
            z_samples.shape[1], z_samples.shape[0], -1)
        y_pred_mu, y_pred_std = self.decoder(x_target, z_samples)

        outputs = {
            'y_pred_mu': y_pred_mu,
            'y_pred_std': y_pred_std,
            'z_mu_context': z_mu_context,
            'z_std_context': z_std_context,
            'z_mu_all': z_mu_all,
            'z_std_all': z_std_all

        }

        return outputs

    def sample(self, x_target, number_of_samples, x_context=None, y_context=None, seed=None):
        '''
        Samples from the latent distribution. If no context is given, samples from a standard normal.

        Arguments:
            x_target: the x target points
            number_of_samples: the number of samples to take
            x_context: the x context points
            y_context: the y_context points
            seed: to provide deterministic results.
        '''
        batch_size = x_target.shape[0]
        if x_context is None:

            z_samples = MultivariateNormal(torch.zeros(self.z_dim), torch.eye(
                self.z_dim)).sample((batch_size * number_of_samples,))
            z_mu = None
            z_std = None

        else:
            if seed is not None:
                torch.manual_seed(seed)

            z_mu, z_std = self.encoder(x_context, y_context)

            eps_samples = torch.randn((number_of_samples, self.z_dim))

            z_samples = z_std.mul(eps_samples) + z_mu
            z_samples = z_samples.reshape(number_of_samples, -1, self.z_dim)

        y_star, y_std = self.decoder(x_target, z_samples)

        return y_star, y_std, z_mu, z_std


class Encoder(nn.Module):
    '''
    A class that represents the encoder part of the Neural Process model. This encodes pairs of (x, y) data into parameters (mu, sigma) of a normal
    distribution where a latent z can be sampled from.
    '''

    def __init__(self, input_size, r_dim, z_dim, hidden_size=10, activation=nn.Sigmoid()):
        '''
        Arguments:
            input_size: the input data dimension
            r_dim: the representation dimension
            z_dim: the latent variable dimension
            hidden_size: the number of hidden neurons in each layer of the network
            activation: the activation function to use for each layer.
        '''
        super().__init__()

        self.input_size = input_size
        self.r_dim = r_dim
        self.z_dim = z_dim

        # Instantiate the part of the network that encodes the
        # input and output pairs into a representation, r.
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, r_dim),
            activation
        )

        # Instantiate the part of the network that maps r to
        # the latent variable mean
        self.fc_context_mean = nn.Sequential(
            nn.Linear(r_dim, z_dim)
        )

        # Instantiate the part of the network that maps r to
        # the latent variable std
        self.fc_context_std = nn.Sequential(
            nn.Linear(r_dim, z_dim)
        )

        # Apply weight initialisation to the network.
        self.apply(self.init_weights)

    def init_weights(self, m):
        '''
        Instantiates the weights of the neural network

        Arguments:
            m: the layer in the network to instantiate.
        '''
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    def forward(self, x, y):
        '''
        Performs a forward pass of the neural network, to calculate
        the latent mean and std.

        Arguments:
            x: the 'context' input data 
            y: the 'context' output data

        Returns:
            context_mean: the latent mean
            context_std: the latent std.
        '''
        context = torch.cat((x, y), dim=2)
        r = self.fc(context)
        representation = self.aggregate(r)
        representation = representation.view(representation.shape[0], 1, -1)

        context_mean = self.fc_context_mean(representation)

        context_log_var = self.fc_context_std(representation)
        context_std = torch.exp(0.5 * context_log_var)

        return context_mean, context_std

    def aggregate(self, r):
        '''
        Aggregates a matrix of encoded representations by
        taking the mean over the representations. Provides
        context point order invariance.

        Input:
            r: a matrix of representations

        Output:
            r_bar: a mean over the representations
        '''
        r_bar = torch.mean(r, 1)
        return r_bar


class Decoder(nn.Module):
    '''
    A class that represents the decoder in the Neural Process model and
    takes care of decoding a latent sample + a target x into a target y.
    '''

    def __init__(self, z_dim, x_dim, y_dim, hidden_size=10, activation=nn.Sigmoid()):
        '''
        Arguments:
                z_dim: the latent dimension
                x_dim: the input data dimension
                y_dim: the output data dimension
                hidden_size: the number of hidden neurons in each layer
                activation: the activation function to be used in each layer.
        '''
        super().__init__()

        self.z_dim = z_dim
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.fc_decode = nn.Sequential(
            nn.Linear(z_dim + x_dim, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation
        )

        self.fc_mean = nn.Sequential(
            nn.Linear(hidden_size, y_dim)
        )

        self.fc_std = nn.Sequential(
            nn.Linear(hidden_size, y_dim)
        )

        self.apply(self.init_weights)

    def init_weights(self, m):
        '''
        Instantiates the weights of the neural network

        Arguments:
            m: the layer in the network to instantiate.
        '''
        if type(m) == nn.Linear:
            torch.nn.init.xavier_normal_(m.weight)

    def forward(self, target_x, z):
        '''
        Performs a forward pass through the network, 
        decoding the latent and target x into a prediction y.

        Arguments:
                mu: the target prediction mean
                std: the target prediction std.
        '''
        batch_size = target_x.shape[0]
        number_of_samples = z.shape[0]

        z = z.reshape(batch_size * number_of_samples, 1, -1)
        z = z.repeat(1, target_x.shape[1], 1)

        target_x_expand = target_x.repeat(number_of_samples, 1, 1)

        xr_concat = torch.cat((target_x_expand, z), dim=2)

        decode = self.fc_decode(xr_concat)
        mu, logvar = self.fc_mean(decode), self.fc_std(decode)

        std = torch.exp(0.5 * logvar)

        return mu, std
