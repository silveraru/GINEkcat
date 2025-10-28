import torch
from torch import nn
from Code.Model.GINE import GINE
from Code.Model.ChemBERTa import ChemBERTaEncoder
from Code.Model.ESM import ESMEncoder
import torch.nn.functional as F


class GINEkcat(nn.Module):
    def __init__(self, dim=64, layer_output=3, dropout=0.3, 
                 gine_layers=3, edge_dim=10):
        super(GINEkcat, self).__init__()
        
        # Substrate encoder: ChemBERTa
        self.chemberta = ChemBERTaEncoder(output_dim=dim)
        
        # Substrate graph encoder: GINE
        self.substrate_gine = GINE(
            in_features=dim, 
            hidden_features=dim, 
            out_features=dim,
            edge_dim=edge_dim,
            num_layers=gine_layers,
            dropout=dropout
        )
        
        # Protein sequence encoder: ESM
        self.esm = ESMEncoder(output_dim=dim)
        
        # Protein structure encoder: GINE
        self.structure_gine = GINE(
            in_features=dim,
            hidden_features=dim,
            out_features=dim,
            edge_dim=edge_dim,
            num_layers=gine_layers,
            dropout=dropout
        )
        
        # Output layers
        self.W_out = nn.ModuleList([nn.Linear(4 * dim, 4 * dim) for _ in range(layer_output)])
        self.W_interaction = nn.Linear(4 * dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, layer_output, dropout):
        smiles, substrate_graph, protein_seq, structure_graph = inputs[:4]
        
        # 1. ChemBERTa encoding for substrate
        chemberta_vectors = self.chemberta(smiles)
        if chemberta_vectors.dim() > 1:
            chemberta_vectors = torch.mean(chemberta_vectors, 0, keepdim=True)
        else:
            chemberta_vectors = chemberta_vectors.unsqueeze(0)
        
        # 2. GINE encoding for substrate graph
        substrate_vectors = self.substrate_gine(
            substrate_graph['x'],
            substrate_graph['edge_index'],
            substrate_graph['edge_attr'],
            substrate_graph.get('batch', None)
        )
        if substrate_vectors.dim() > 1:
            substrate_vectors = torch.mean(substrate_vectors, 0, keepdim=True)
        else:
            substrate_vectors = substrate_vectors.unsqueeze(0)
        
        # 3. ESM encoding for protein sequence
        esm_vectors = self.esm(protein_seq)
        if esm_vectors.dim() > 1:
            esm_vectors = torch.mean(esm_vectors, 0, keepdim=True)
        else:
            esm_vectors = esm_vectors.unsqueeze(0)
        
        # 4. GINE encoding for protein structure (PDB)
        structure_vectors = self.structure_gine(
            structure_graph['x'],
            structure_graph['edge_index'],
            structure_graph['edge_attr'],
            structure_graph.get('batch', None)
        )
        if structure_vectors.dim() > 1:
            structure_vectors = torch.mean(structure_vectors, 0, keepdim=True)
        else:
            structure_vectors = structure_vectors.unsqueeze(0)
        
        # 5. Concatenate all representations
        cat_vector = torch.cat((chemberta_vectors, substrate_vectors, esm_vectors, structure_vectors), 1)
        
        # 6. Output layers
        for j in range(layer_output):
            cat_vector = F.relu(cat_vector)
            cat_vector = F.dropout(cat_vector, dropout, training=self.training)
            cat_vector = self.W_out[j](cat_vector)
        
        cat_vector = F.relu(cat_vector)
        cat_vector = F.dropout(cat_vector, dropout, training=self.training)
        interaction = self.W_interaction(cat_vector)
        interaction = torch.squeeze(interaction, 0)
        
        return interaction
    
    def freeze_pretrained(self):
        self.chemberta.freeze_chemberta()
        self.esm.freeze_esm()
    
    def unfreeze_pretrained(self):
        self.chemberta.unfreeze_chemberta()
        self.esm.unfreeze_esm()