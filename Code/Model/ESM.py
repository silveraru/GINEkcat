import torch
import torch.nn as nn
from transformers import EsmModel, EsmTokenizer


class ESMEncoder(nn.Module):
    def __init__(self, model_name='facebook/esm2_t33_650M_UR50D', hidden_dim=1280, output_dim=64):
        """
        ESM (Evolutionary Scale Modeling) protein encoder
        
        Args:
            model_name: ESM model variant to use
                - facebook/esm2_t33_650M_UR50D (default, 33 layers, 650M params)
                - facebook/esm2_t30_150M_UR50D (30 layers, 150M params)
                - facebook/esm2_t12_35M_UR50D (12 layers, 35M params)
                - facebook/esm2_t6_8M_UR50D (6 layers, 8M params)
            hidden_dim: Hidden dimension of ESM model
            output_dim: Output dimension after projection
        """
        super(ESMEncoder, self).__init__()
        self.tokenizer = EsmTokenizer.from_pretrained(model_name)
        self.model = EsmModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, sequences):
        """
        Args:
            sequences: List of protein sequences or single sequence string
        Returns:
            Tensor of shape (batch_size, output_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        # Tokenize sequences
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get ESM embeddings
        outputs = self.model(**inputs)
        
        # Use [CLS] token representation (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project to desired output dimension
        projected = self.projection(cls_embeddings)
        
        return projected
    
    def get_per_residue_embeddings(self, sequences):
        """
        Get per-residue embeddings instead of sequence-level
        
        Returns:
            Tensor of shape (batch_size, seq_len, hidden_dim)
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # Return all token embeddings (excluding special tokens)
        return outputs.last_hidden_state
    
    def freeze_esm(self):
        """Freeze ESM weights for transfer learning"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_esm(self):
        """Unfreeze ESM weights for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
