import torch
import torch.nn as nn
from transformers import ChemBERTaTokenizer, ChemBERTaModel
from torch_geometric.nn import GINEConv
from esm import ProteinBertModel

class GINEkcat(nn.Module):
    def __init__(self):
        super(GINEkcat, self).__init__()

        # ChemBERTa Encoder for SMILES
        self.chemberta_tokenizer = ChemBERTaTokenizer.from_pretrained('seyonec/ChemBERTa-ZINC-250k')
        self.chemberta_model = ChemBERTaModel.from_pretrained('seyonec/ChemBERTa-ZINC-250k')

        # GINE Encoder for substrate graph
        self.gine_encoder = GINEConv(nn.Linear(128, 128))

        # ESM Encoder for protein sequences
        self.esm_model = ProteinBertModel.from_pretrained("facebookresearch/esm")
        
        # GINE Encoder for PDB structure
        self.gine_pdb_encoder = GINEConv(nn.Linear(128, 128))

        # Fusion Layers
        self.fusion_layer = nn.Linear(128 * 3, 128)  # Adjusted for combined inputs
        
        # Output Layer for kcat prediction
        self.output_layer = nn.Linear(128, 1)

    def forward(self, smiles, substrate_graph, protein_seq, pdb_structure):
        # ChemBERTa encoding
        chemberta_input = self.chemberta_tokenizer(smiles, return_tensors='pt')
        chemberta_output = self.chemberta_model(**chemberta_input).last_hidden_state

        # GINE encoding for substrate graph
        substrate_encoding = self.gine_encoder(substrate_graph)

        # ESM encoding for protein sequence
        esm_output = self.esm_model(protein_seq)

        # GINE encoding for PDB structure
        pdb_encoding = self.gine_pdb_encoder(pdb_structure)

        # Combine all encodings
        combined = torch.cat((chemberta_output, substrate_encoding, esm_output, pdb_encoding), dim=-1)
        fused_output = self.fusion_layer(combined)

        # Output for kcat prediction
        kcat_prediction = self.output_layer(fused_output)

        return kcat_prediction
