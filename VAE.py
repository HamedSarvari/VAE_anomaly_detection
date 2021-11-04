import torch
from torch import nn
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import softplus
from hamed_utils import create_layer_dims
import pandas as pd
import os

def tabular_encoder_An(input_size: int, latent_size: int):
    """
    Simple encoder for tabular data.
    If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
    :param input_size: number of input variables
    :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
    :return: The untrained encoder model
    """
    return nn.Sequential(
        nn.Linear(input_size, 500),
        nn.ReLU(),
        nn.Linear(500, 200),
        nn.ReLU(),
        nn.Linear(200, latent_size * 2)  # times 2 because this is the concatenated vector of latent mean and variance
    )

def tabular_encoder_BAE(input_size: int, latent_size: int ):
    """
    Simple encoder for tabular data.
    If you want to feed image to a VAE make another encoder function with Conv2d instead of Linear layers.
    :param input_size: number of input variables
    :param latent_size: number of output variables i.e. the size of the latent space since it's the encoder of a VAE
    :return: The untrained encoder model
    """

    layers = create_layer_dims(input_size)
    print('layers:' , layers)

    return nn.Sequential(
        nn.Linear(input_size, layers[1]),
        nn.ReLU(),
        nn.Linear(layers[1], layers[2]),
        nn.ReLU(),
        nn.Linear(layers[2], latent_size * 2)
        # times 2 because this is the concatenated vector of latent mean and variance
    )



def tabular_decoder_An(latent_size: int, output_size: int):
    """
    Simple decoder for tabular data.
    :param latent_size: size of input latent space
    :param output_size: number of output parameters. Must have the same value of input_size
    :return: the untrained decoder
    """
    return nn.Sequential(
        nn.Linear(latent_size, 200),
        nn.ReLU(),
        nn.Linear(200, 500),
        nn.ReLU(),
        nn.Linear(500, output_size * 2)
        # times 2 because this is the concatenated vector of reconstructed mean and variance
    )

def tabular_decoder_BAE(latent_size: int, output_size: int):
    """
    Simple decoder for tabular data.
    :param latent_size: size of input latent space
    :param output_size: number of output parameters. Must have the same value of input_size
    :return: the untrained decoder
    """
    input_size = output_size
    layers = create_layer_dims(input_size)

    return nn.Sequential(
        nn.Linear(latent_size, layers[4]),
        nn.ReLU(),
        nn.Linear(layers[4], layers[5]),
        nn.ReLU(),
        nn.Linear(layers[5], output_size * 2)
        # times 2 because this is the concatenated vector of reconstructed mean and variance
    )

class VAEAnomaly(nn.Module):

    def __init__(self, input_size: int, latent_size: int, structure: str, L=10):
        """
        :param input_size: Number of input features
        :param latent_size: Size of the latent space
        :param L: Number of samples in the latent space (See paper for more details)
        """
        super().__init__()
        self.L = L
        self.input_size = input_size
        self.latent_size = latent_size
        if structure == 'BAE':
            self.encoder = tabular_encoder_BAE(input_size, latent_size)
            self.decoder = tabular_decoder_BAE(latent_size, input_size)
        elif structure == 'An_paper':
            self.encoder = tabular_encoder_An(input_size, latent_size)
            self.decoder = tabular_decoder_An(latent_size, input_size)
        self.prior = Normal(0, 1)

    def forward(self, x):
        pred_result = self.predict(x)
        x = x.unsqueeze(0)  # unsqueeze to broadcast input across sample dimension (L)
        log_lik = Normal(pred_result['recon_mu'], pred_result['recon_sigma']).log_prob(x).mean(
            dim=0)  # average over sample dimension
        log_lik = log_lik.mean(dim=0).sum()
        kl = kl_divergence(pred_result['latent_dist'], self.prior).mean(dim=0).sum()
        loss = kl - log_lik
        return dict(loss=loss, kl=kl, recon_loss=log_lik, **pred_result)

    def predict(self, x) -> dict:
        """
        :param x: tensor of shape [batch_size, num_features]
        :return: A dictionary containing prediction i.e.
        - latent_dist = torch.distributions.Normal instance of latent space
        - latent_mu = torch.Tensor mu (mean) parameter of latent Normal distribution
        - latent_sigma = torch.Tensor sigma (std) parameter of latent Normal distribution
        - recon_mu = torch.Tensor mu (mean) parameter of reconstructed Normal distribution
        - recon_sigma = torch.Tensor sigma (std) parameter of reconstructed Normal distribution
        - z = torch.Tensor sampled latent space from latent distribution
        """
        batch_size = len(x)
        latent_mu, latent_sigma = self.encoder(x).chunk(2, dim=1)  # both with size [batch_size, latent_size]
        print('mu',latent_mu.view())
        latent_sigma = softplus(latent_sigma)
        try:
            dist = Normal(latent_mu, latent_sigma)

        except ValueError:
            print("Oops!  Here is the error")
            df1 = pd.DataFrame(latent_sigma)
            df2 = pd.DataFrame(latent_mu)
            df1.to_csv(os.getcwd() + '/latent_sigma.csv')
            df2.to_csv(os.getcwd() + '/latent_mu.csv')

        z = dist.rsample([self.L])  # shape: [L, batch_size, latent_size]
        #print('shape:',z.shape)

        z = z.view(self.L * batch_size, self.latent_size)

        #print('new shape:', z.shape)
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)


        recon_sigma = softplus(recon_sigma)
        recon_mu = recon_mu.view(self.L, *x.shape)


        recon_sigma = recon_sigma.view(self.L, *x.shape)
        return dict(latent_dist=dist, latent_mu=latent_mu,
                    latent_sigma=latent_sigma, recon_mu=recon_mu,
                    recon_sigma=recon_sigma, z=z)

    def is_anomaly(self, x, alpha=0.05):
        """

        :param x:
        :param alpha: Anomaly threshold (see paper for more details)
        :return: Return a vector of boolean with shape [x.shape[0]]
                 which is true when an element is considered an anomaly
        """
        p = self.reconstructed_probability(x)
        return p < alpha

    def reconstructed_probability(self, x):
        with torch.no_grad():
            pred = self.predict(x)

        try:
            recon_dist = Normal(pred['recon_mu'], pred['recon_sigma'])

        except ValueError:

            print("Oops!  Here is the error")
            pred_df = pd.DataFrame(pred)
            pred_df.to_csv(os.getcwd() + 'error.csv')

        x = x.unsqueeze(0)
        p = recon_dist.log_prob(x).exp().mean(dim=0).mean(dim=-1)  # vector of shape [batch_size]
        return p

    def generate(self, batch_size: int = 1) -> torch.Tensor:
        """
        Sample from prior distribution, feed into decoder and get in output reconstructed samples
        :param batch_size:
        :return: Generated samples
        """
        z = self.prior.sample((batch_size, self.latent_size))
        recon_mu, recon_sigma = self.decoder(z).chunk(2, dim=1)
        recon_sigma = softplus(recon_sigma)
        return recon_mu + recon_sigma * torch.rand_like(recon_sigma)

