import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ChemBERTaEncoder(nn.Module):
    def __init__(self, model_name='seyonec/ChemBERTa-zinc-base-v1', hidden_dim=768, output_dim=64):
        super(ChemBERTaEncoder, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, smiles_list):
        # Tokenize SMILES strings
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
            
        inputs = self.tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}
        
        # Get ChemBERTa embeddings
        outputs = self.model(**inputs)
        
        # Use [CLS] token representation
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project to desired output dimension
        projected = self.projection(cls_embeddings)
        
        return projected
    
    def freeze_chemberta(self):
        """Freeze ChemBERTa weights for transfer learning"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_chemberta(self):
        """Unfreeze ChemBERTa weights for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
