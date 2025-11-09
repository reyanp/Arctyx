import torch
import torch.nn as nn


class MixedDataCVAE(nn.Module):
    """
    Conditional VAE for mixed data types (numerical and categorical).
    It processes numerical and categorical features in separate 'towers'
    before combining them into a latent space.
    
    The decoder reconstructs numerical data (MSE) and categorical data (CrossEntropy).
    """
    
    def __init__(self, config):
        """
        Initializes the MixedDataCVAE.
        
        Args:
            config: Dictionary containing:
                - numerical_dim: Int, number of numerical features.
                - categorical_embed_dims: List of tuples (num_categories, embed_dim)
                                          for each categorical feature.
                                          e.g., [(10, 8), (5, 4)]
                - latent_dim: Dimension of latent space.
                - condition_dim: Dimension of condition vector.
                - encoder_hidden_layers: List of hidden layer sizes.
                - decoder_hidden_layers: List of hidden layer sizes.
        """
        super(MixedDataCVAE, self).__init__()
        
        self.config = config
        self.numerical_dim = config['numerical_dim']
        self.categorical_embed_dims = config['categorical_embed_dims']
        self.latent_dim = config['latent_dim']
        self.condition_dim = config['condition_dim']
        
        # --- Categorical Embedding ---
        # Create a list of embedding layers, one for each categorical feature
        self.embedding_layers = nn.ModuleList([
            nn.Embedding(num_categories, embed_dim)
            for num_categories, embed_dim in self.categorical_embed_dims
        ])
        
        # Calculate total dimension of concatenated embeddings
        total_categorical_embed_dim = sum([
            embed_dim for _, embed_dim in self.categorical_embed_dims
        ])
        
        # --- Encoder ---
        # We combine numerical, categorical, and condition features
        encoder_input_dim = (
            self.numerical_dim + 
            total_categorical_embed_dim + 
            self.condition_dim
        )
        
        encoder_layers = []
        encoder_hidden = config.get('encoder_hidden_layers', [128, 64])
        
        for hidden_dim in encoder_hidden:
            encoder_layers.append(nn.Linear(encoder_input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_input_dim = hidden_dim
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        self.fc_mu = nn.Linear(encoder_input_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_input_dim, self.latent_dim)
        
        # --- Decoder ---
        decoder_input_dim = self.latent_dim + self.condition_dim
        decoder_hidden = config.get('decoder_hidden_layers', [64, 128])
        
        decoder_layers_list = []
        for hidden_dim in decoder_hidden:
            decoder_layers_list.append(nn.Linear(decoder_input_dim, hidden_dim))
            decoder_layers_list.append(nn.ReLU())
            decoder_input_dim = hidden_dim
        
        self.decoder_base = nn.Sequential(*decoder_layers_list)
        
        # --- Multi-Headed Output ---
        # One head to reconstruct numerical features
        self.decoder_out_numerical = nn.Linear(decoder_input_dim, self.numerical_dim)
        
        # A list of heads, one for each *original* categorical feature
        self.decoder_out_categorical = nn.ModuleList([
            nn.Linear(decoder_input_dim, num_categories)
            for num_categories, _ in self.categorical_embed_dims
        ])
    
    def encode(self, x_num, x_cat, c):
        """
        Args:
            x_num: Tensor of numerical features [batch_size, numerical_dim]
            x_cat: Tensor of categorical features (as indices) [batch_size, num_categorical_features]
            c: Condition tensor [batch_size, condition_dim]
        """
        # Process categorical features through their respective embedding layers
        cat_embeds = [
            self.embedding_layers[i](x_cat[:, i])
            for i in range(len(self.embedding_layers))
        ]
        
        x_cat_embedded = torch.cat(cat_embeds, dim=1)
        
        # Combine all inputs
        inputs = torch.cat([x_num, x_cat_embedded, c], dim=1)
        h = self.encoder(inputs)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """
        Decodes z and c into *reconstructions* of numerical features
        and *logits* for categorical features.
        """
        inputs = torch.cat([z, c], dim=1)
        h_base = self.decoder_base(inputs)
        
        # Reconstruct numericals
        recon_num = self.decoder_out_numerical(h_base)
        
        # Reconstruct categorical logits
        recon_cat_logits = [
            out_head(h_base) for out_head in self.decoder_out_categorical
        ]
        
        return recon_num, recon_cat_logits
    
    def forward(self, x_num, x_cat, c):
        mu, logvar = self.encode(x_num, x_cat, c)
        z = self.reparameterize(mu, logvar)
        recon_num, recon_cat_logits = self.decode(z, c)
        return recon_num, recon_cat_logits, mu, logvar

