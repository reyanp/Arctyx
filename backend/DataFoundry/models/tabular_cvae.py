import torch
import torch.nn as nn


class TabularCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for tabular data.
    Takes features x and condition c, encodes to latent space, and decodes back.
    """
    
    def __init__(self, config):
        """
        Initializes the TabularCVAE model based on config parameters.
        
        Args:
            config: Dictionary containing:
                - input_dim: Dimension of input features
                - latent_dim: Dimension of latent space
                - condition_dim: Dimension of condition vector
                - encoder_hidden_layers: List of hidden layer sizes for encoder
                - decoder_hidden_layers: List of hidden layer sizes for decoder
        """
        super(TabularCVAE, self).__init__()
        
        self.input_dim = config['input_dim']
        self.latent_dim = config['latent_dim']
        self.condition_dim = config['condition_dim']
        
        # Encoder: input_dim + condition_dim -> hidden layers -> latent_dim
        encoder_layers = []
        encoder_input_dim = self.input_dim + self.condition_dim
        encoder_hidden = config.get('encoder_hidden_layers', [128, 64])
        
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(encoder_input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_input_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space projections
        self.fc_mu = nn.Linear(encoder_input_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_input_dim, self.latent_dim)
        
        # Decoder: latent_dim + condition_dim -> hidden layers -> input_dim
        decoder_layers = []
        decoder_input_dim = self.latent_dim + self.condition_dim
        decoder_hidden = config.get('decoder_hidden_layers', [64, 128])
        
        for hidden_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_input_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output layer
        self.fc_out = nn.Linear(decoder_input_dim, self.input_dim)
    
    def encode(self, x, c):
        """
        Encodes input features x and condition c into latent space parameters.
        
        Args:
            x: Input features tensor [batch_size, input_dim]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        inputs = torch.cat([x, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for VAE sampling.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """
        Decodes latent vector z and condition c back to input space.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            Reconstructed input [batch_size, input_dim]
        """
        inputs = torch.cat([z, c], dim=1)
        h = self.decoder(inputs)
        return self.fc_out(h)
    
    def forward(self, x, c):
        """
        Forward pass: encode -> reparameterize -> decode.
        
        Args:
            x: Input features tensor [batch_size, input_dim]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            recon_x: Reconstructed input [batch_size, input_dim]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

