import torch
import torch.nn as nn


class CTGAN_Generator(nn.Module):
    """
    Generator for a Conditional Tabular GAN (CTGAN).
    Takes a random noise vector 'z' and a condition 'c' to produce fake data.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing:
                - latent_dim: Dimension of input noise vector
                - condition_dim: Dimension of condition vector
                - output_dim: Dimension of the data features to be generated
                - gen_hidden_layers: List of hidden layer sizes
        """
        super(CTGAN_Generator, self).__init__()
        
        self.latent_dim = config['latent_dim']
        self.condition_dim = config['condition_dim']
        self.output_dim = config['output_dim']  # This is the same as 'input_dim'
        
        gen_layers = []
        gen_input_dim = self.latent_dim + self.condition_dim
        gen_hidden = config.get('gen_hidden_layers', [256, 256])
        
        for hidden_dim in gen_hidden:
            gen_layers.append(nn.Linear(gen_input_dim, hidden_dim))
            gen_layers.append(nn.ReLU())
            gen_input_dim = hidden_dim
            
        gen_layers.append(nn.Linear(gen_input_dim, self.output_dim))
        # Use Tanh to output data in a normalized range (e.g., -1 to 1)
        # This requires your preprocessor to scale data to this range.
        gen_layers.append(nn.Tanh())
        
        self.generator = nn.Sequential(*gen_layers)
    
    def forward(self, z, c):
        """
        Args:
            z: Noise vector [batch_size, latent_dim]
            c: Condition tensor [batch_size, condition_dim]
        """
        inputs = torch.cat([z, c], dim=1)
        return self.generator(inputs)


class CTGAN_Discriminator(nn.Module):
    """
    Discriminator for a Conditional Tabular GAN (CTGAN).
    Tries to distinguish real data from fake data.
    """
    
    def __init__(self, config):
        """
        Args:
            config: Dictionary containing:
                - input_dim: Dimension of the data features
                - condition_dim: Dimension of condition vector
                - disc_hidden_layers: List of hidden layer sizes
        """
        super(CTGAN_Discriminator, self).__init__()
        self.input_dim = config['input_dim']
        self.condition_dim = config['condition_dim']
        
        disc_layers = []
        disc_input_dim = self.input_dim + self.condition_dim
        disc_hidden = config.get('disc_hidden_layers', [256, 256])
        for hidden_dim in disc_hidden:
            disc_layers.append(nn.Linear(disc_input_dim, hidden_dim))
            disc_layers.append(nn.LeakyReLU(0.2))  # LeakyReLU is common in GANs
            disc_input_dim = hidden_dim
            
        disc_layers.append(nn.Linear(disc_input_dim, 1))
        # No Sigmoid here, as it's handled by the BCEWithLogitsLoss
        
        self.discriminator = nn.Sequential(*disc_layers)
    
    def forward(self, x, c):
        """
        Args:
            x: Data features [batch_size, input_dim]
            c: Condition tensor [batch_size, condition_dim]
        """
        inputs = torch.cat([x, c], dim=1)
        return self.discriminator(inputs)

