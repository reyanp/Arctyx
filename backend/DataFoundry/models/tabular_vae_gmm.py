import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Helper Modules (to be shared) ---
# We define these here, but they could be moved to a 'model_utils.py'
# to be shared with the original TabularCVAE.

class _Encoder(nn.Module):
    """Helper Encoder network."""
    
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(_Encoder, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(in_dim, output_dim)
        self.fc_logvar = nn.Linear(in_dim, output_dim)
        
    def forward(self, x):
        h = self.network(x)
        return self.fc_mu(h), self.fc_logvar(h)


class _Decoder(nn.Module):
    """Helper Decoder network."""
    
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(_Decoder, self).__init__()
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.fc_out = nn.Linear(in_dim, output_dim)
        
    def forward(self, x):
        h = self.network(x)
        return self.fc_out(h)


# --- Main Model Class ---

class TabularVAE_GMM(nn.Module):
    """
    Conditional VAE with a Gaussian Mixture Model (GMM) in the latent space.
    This helps model clustered data.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing:
                - input_dim: Dimension of input features
                - latent_dim: Dimension of latent space
                - condition_dim: Dimension of condition vector
                - n_clusters: Number of clusters (Gaussians) in the mixture
                - encoder_hidden_layers: List of hidden layer sizes
                - decoder_hidden_layers: List of hidden layer sizes
        """
        super(TabularVAE_GMM, self).__init__()
        
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.condition_dim = config['condition_dim']
        self.n_clusters = config['n_clusters']
        
        # --- Model Components ---
        self.encoder = _Encoder(
            self.input_dim + self.condition_dim,
            config.get('encoder_hidden_layers', [128, 64]),
            self.latent_dim
        )
        
        self.decoder = _Decoder(
            self.latent_dim + self.condition_dim,
            config.get('decoder_hidden_layers', [64, 128]),
            self.input_dim
        )
        
        # --- GMM Parameters ---
        # These are the parameters for the K clusters.
        # We will learn these.
        
        # 1. Cluster probabilities (logits)
        #    pi_k = probability of a point being in cluster k
        self.gmm_logits = nn.Parameter(torch.zeros(self.n_clusters))
        
        # 2. Cluster means (mu_k)
        self.gmm_means = nn.Parameter(torch.randn(self.n_clusters, self.latent_dim))
        
        # 3. Cluster log-variances (logvar_k)
        self.gmm_logvars = nn.Parameter(torch.zeros(self.n_clusters, self.latent_dim))
    
    def reparameterize(self, mu, logvar):
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x, c):
        """Standard encode."""
        inputs = torch.cat([x, c], dim=1)
        z_mu, z_logvar = self.encoder(inputs)
        return z_mu, z_logvar
    
    def decode(self, z, c):
        """Standard decode."""
        inputs = torch.cat([z, c], dim=1)
        return self.decoder(inputs)
    
    def forward(self, x, c):
        """
        Forward pass is standard, but the loss calculation (in trainer.py)
        will be what makes this model special.
        """
        z_mu, z_logvar = self.encode(x, c)
        z = self.reparameterize(z_mu, z_logvar)
        recon_x = self.decode(z, c)
        
        # We also need to return the GMM parameters and z_mu, z_logvar
        # so the special loss function can use them.
        return recon_x, z_mu, z_logvar
    
    def get_gmm_params(self):
        """Helper to return GMM parameters."""
        return F.softmax(self.gmm_logits, dim=0), self.gmm_means, self.gmm_logvars

