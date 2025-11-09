import torch
import torch.nn as nn
from transformers import AutoModel


class TextCVAE(nn.Module):
    """
    Conditional Variational Autoencoder for text data using Transformer architecture.
    This is a stretch goal implementation using pre-trained transformers as encoder/decoder base.
    """
    
    def __init__(self, config):
        """
        Initializes the TextCVAE model based on config parameters.
        
        Args:
            config: Dictionary containing:
                - text_model: Name of pre-trained transformer model (e.g., 'bert-base-uncased')
                - latent_dim: Dimension of latent space
                - condition_dim: Dimension of condition vector
                - max_length: Maximum sequence length for tokenization
        """
        super(TextCVAE, self).__init__()
        
        self.latent_dim = config['latent_dim']
        self.condition_dim = config['condition_dim']
        self.max_length = config.get('max_length', 128)
        text_model_name = config.get('text_model', 'bert-base-uncased')
        
        # Use pre-trained transformer as encoder base
        self.encoder_base = AutoModel.from_pretrained(text_model_name)
        encoder_hidden_dim = self.encoder_base.config.hidden_size
        
        # Project encoder output to latent space
        self.fc_mu = nn.Linear(encoder_hidden_dim + self.condition_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(encoder_hidden_dim + self.condition_dim, self.latent_dim)
        
        # Decoder: project from latent + condition back to hidden dimension
        self.fc_decoder_input = nn.Linear(self.latent_dim + self.condition_dim, encoder_hidden_dim)
        
        # Use transformer decoder or a simple MLP decoder
        # For simplicity, using a simple MLP decoder here
        # In a full implementation, you might use a transformer decoder
        decoder_hidden = config.get('decoder_hidden_layers', [512, 256])
        decoder_layers = []
        decoder_input_dim = encoder_hidden_dim
        
        for hidden_dim in decoder_hidden:
            decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_input_dim = hidden_dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output projection to vocabulary size (for text generation)
        # Note: This is a simplified version. Full implementation would use tokenizer vocab size
        vocab_size = self.encoder_base.config.vocab_size
        self.fc_out = nn.Linear(decoder_input_dim, vocab_size)
    
    def encode(self, input_ids, attention_mask, c):
        """
        Encodes input text and condition into latent space parameters.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Get encoder output (using [CLS] token representation)
        encoder_outputs = self.encoder_base(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (first token) as sentence representation
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Concatenate with condition
        inputs = torch.cat([pooled_output, c], dim=1)
        
        return self.fc_mu(inputs), self.fc_logvar(inputs)
    
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
    
    def decode(self, z, c, seq_len=None):
        """
        Decodes latent vector z and condition c back to vocabulary logits.
        
        Args:
            z: Latent vector [batch_size, latent_dim]
            c: Condition tensor [batch_size, condition_dim]
            seq_len: Sequence length to generate. If None, uses self.max_length
        
        Returns:
            Logits over vocabulary [batch_size, seq_len, vocab_size]
        """
        if seq_len is None:
            seq_len = self.max_length
        
        batch_size = z.size(0)
        inputs = torch.cat([z, c], dim=1)  # [batch_size, latent_dim + condition_dim]
        hidden = self.fc_decoder_input(inputs)  # [batch_size, encoder_hidden_dim]
        
        # Expand to sequence length: [batch_size, seq_len, encoder_hidden_dim]
        hidden_expanded = hidden.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # Reshape for MLP: [batch_size * seq_len, encoder_hidden_dim]
        hidden_flat = hidden_expanded.reshape(-1, hidden.size(-1))
        
        # Pass through decoder layers
        h = self.decoder(hidden_flat)  # [batch_size * seq_len, decoder_output_dim]
        
        # Output projection
        logits = self.fc_out(h)  # [batch_size * seq_len, vocab_size]
        
        # Reshape back to sequence: [batch_size, seq_len, vocab_size]
        return logits.view(batch_size, seq_len, -1)
    
    def forward(self, input_ids, attention_mask, c):
        """
        Forward pass: encode -> reparameterize -> decode.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            c: Condition tensor [batch_size, condition_dim]
        
        Returns:
            recon_logits: Reconstructed logits [batch_size, seq_len, vocab_size]
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        mu, logvar = self.encode(input_ids, attention_mask, c)
        z = self.reparameterize(mu, logvar)
        seq_len = input_ids.size(1)  # Get sequence length from input
        recon_logits = self.decode(z, c, seq_len=seq_len)
        return recon_logits, mu, logvar

