import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import numpy as np
from grover.model.layers import Readout

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')

def make_ffn(input_dim: int,
             hidden_dim: int,
             output_dim: int,
             num_layers: int,
             dropout: float,
             activation: str) -> nn.Sequential:
    """
    Build an MLP: Dropout -> Linear -> Act repeated, then final Dropout -> Linear -> (no act).
    """
    act = get_activation_function(activation)
    layers = [nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim), act]
    for _ in range(num_layers - 2):
        layers += [nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), act]
    layers += [nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*layers)

class FiLM(nn.Module):
    """
    Feature-wise linear modulation:
    gamma, beta <- projections(condition)
    output = gamma * x + beta
    """
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta  = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        g = self.gamma(cond)
        b = self.beta(cond)
        return g * x + b

class GroverFinetuneTask(nn.Module):
    """
    The finetune model combining GROVER with Protein Bi-LSTM
    """
    def __init__(self, pretrained_emb, finetune_args):
        super(GroverFinetuneTask, self).__init__()
        self.iscuda = finetune_args.cuda
        
        self.grover = pretrained_emb
        self.readout = Readout(rtype='mean', hidden_size=0)
        
        # Protein sequence
        self.prot_embedding = nn.Embedding(
            num_embeddings = 21,
            embedding_dim = finetune_args.prot_emb_dim
        )
        self.prot_rnn = nn.LSTM(
            input_size    = finetune_args.prot_emb_dim,
            hidden_size   = finetune_args.prot_hidden_size,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = True
        )
        self.cond_dim = 2 * finetune_args.prot_hidden_size # for bidirectionality

        # FiLM layers combines GROVER views with protein embedding
        self.graph_feat_dim = finetune_args.grover_hidden_size
        self.film_node = FiLM(feature_dim=self.graph_feat_dim, cond_dim=self.cond_dim)
        self.film_edge = FiLM(feature_dim=self.graph_feat_dim, cond_dim=self.cond_dim)

        # Attention merges two final view embeddings
        self.attn = nn.MultiheadAttention(
            embed_dim=self.graph_feat_dim, 
            num_heads=finetune_args.attn_nheads,
            dropout=finetune_args.dropout,
            batch_first=False
        )

        # Final layer performs regression
        self.mol_ffn = make_ffn(
            input_dim=self.graph_feat_dim,
            hidden_dim=finetune_args.ffn_hidden_size,
            output_dim=1,
            num_layers=finetune_args.ffn_num_layers,
            dropout=finetune_args.dropout,
            activation=finetune_args.activation
        )
    
    def forward(self,
                smiles_batch,
                graph_components,
                features_batch,
                mask,
                targets,
                prot_idxs,
                prot_lens):
        # (f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, b2a_scope)
        _, _, _, _, _, a_scope, _, _ = graph_components
        out    = self.grover(graph_components)
        g_node = self.readout(out['atom_from_bond'], a_scope)
        g_edge = self.readout(out['atom_from_atom'], a_scope)
        
        # Protein sequence
        self.prot_rnn.flatten_parameters()
        prot_padded = pad_sequence(prot_idxs, batch_first=True, padding_value=0)
        prot_emb = self.prot_embedding(prot_padded)
        prot_packed = pack_padded_sequence(prot_emb, prot_lens, batch_first=True, enforce_sorted=False)
        
        _, (h_n, _) = self.prot_rnn(prot_packed)
        
        cond = torch.cat([h_n[-2], h_n[-1]], dim=1) # [B, cond_dim] for biLSTM
        
        g_node_film = self.film_node(g_node, cond)
        g_edge_film = self.film_edge(g_edge, cond)
        
        attn_out, attn_prob = self.attn(
            g_node_film.unsqueeze(0), # q
            g_edge_film.unsqueeze(0), # k
            g_edge_film.unsqueeze(0)  # v
        )
        fused = attn_out.squeeze(0)

        if features_batch is not None and features_batch[0] is not None:
            F = torch.from_numpy(np.stack(features_batch)).float().to(fused)
            fused = torch.cat([fused, F], dim=1)
        
        y = self.mol_ffn(attn_out.squeeze(0))
        return y
