"""
Relational Graph Convolutional Network (R-GCN) Upgraded Knowledge Graph Embedding (KGE) Module for KG-CMI.

Replaces the baseline 1-layer homogeneous GAT with a 2-layer R-GCN featuring:
- Edge-type specific transformation matrices (e.g., organ-part, organ-finding, disease-symptom)
- Basis-decomposition for parameter regularization
- Multi-hop relational message passing
- Question-guided cross-attention fusion
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGraphConvLayer(nn.Module):
    """
    R-GCN Layer with Basis Decomposition for Relational Medical Knowledge Graphs.
    
    Formula:
        h_i^{(l+1)} = \sigma ( W_0 h_i^{(l)} + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}} W_r h_j^{(l)} )
    """
    def __init__(self, in_features, out_features, num_relations, num_bases=4, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases

        # Self-loop weight
        self.weight_self = nn.Parameter(torch.Tensor(in_features, out_features))

        # Basis decomposition to prevent overfitting on multiple relation types
        if num_bases > 0:
            self.bases = nn.Parameter(torch.Tensor(num_bases, in_features, out_features))
            self.rel_coeffs = nn.Parameter(torch.Tensor(num_relations, num_bases))
        else:
            self.weights = nn.Parameter(torch.Tensor(num_relations, in_features, out_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_self)
        if self.num_bases > 0:
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.rel_coeffs)
        else:
            nn.init.xavier_uniform_(self.weights)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: [N_v, in_features] Node feature matrix
            edge_index: [2, E] Graph edge index tensor (source, target)
            edge_type: [E] Tensor of edge relation types (0 to num_relations-1)
        Returns:
            [N_v, out_features] Updated node representations
        """
        N_v = x.size(0)
        target_dtype = x.dtype
        target_device = x.device

        # Self-loop message
        out = torch.matmul(x, self.weight_self.to(device=target_device, dtype=target_dtype))

        # Compute relation weights W_r
        if self.num_bases > 0:
            W_r = torch.einsum('rb, bio -> rio', self.rel_coeffs, self.bases).to(device=target_device, dtype=target_dtype)
        else:
            W_r = self.weights.to(device=target_device, dtype=target_dtype)

        src, dst = edge_index[0], edge_index[1]

        # Process each relation type
        for r in range(self.num_relations):
            rel_mask = (edge_type == r)
            if not rel_mask.any():
                continue

            r_src = src[rel_mask]
            r_dst = dst[rel_mask]

            # Source features for relation r
            msg = torch.matmul(x[r_src], W_r[r]) # [E_r, out_features]

            # Degree normalization c_{i,r}
            deg = torch.zeros(N_v, device=target_device, dtype=target_dtype).scatter_add_(
                0, r_dst, torch.ones_like(r_dst, dtype=target_dtype)
            ).clamp(min=1.0)
            norm = 1.0 / deg[r_dst].unsqueeze(-1)

            msg_norm = (msg * norm).to(dtype=out.dtype)

            # Scatter add messages to destination nodes
            out.scatter_add_(0, r_dst.unsqueeze(-1).expand(-1, self.out_features), msg_norm)

        if self.bias is not None:
            out = out + self.bias.to(device=target_device, dtype=target_dtype)

        return out


class QuestionGuidedRGCN(nn.Module):
    """
    Upgraded 2-Layer R-GCN Module with Question-guided Graph-Text Attention.
    """
    def __init__(self, d_model=768, num_relations=8, num_bases=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # 2-Layer R-GCN for multi-hop relational message passing
        self.rgcn1 = RelationalGraphConvLayer(d_model, d_model, num_relations=num_relations, num_bases=num_bases)
        self.rgcn2 = RelationalGraphConvLayer(d_model, d_model, num_relations=num_relations, num_bases=num_bases)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Cross-Attention to query graph features using question text embeddings
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, f_od, edge_index, edge_type, text_embeds):
        """
        Args:
            f_od: [N_v, d_model] Initial organ-disease node embeddings (from RoBERTa)
            edge_index: [2, E] Relation edge pairs (organ <-> disease, disease <-> finding, etc.)
            edge_type: [E] Integers representing edge relation type
            text_embeds: [B, L_t, d_model] Text embeddings of input question t
        Returns:
            k_prime: [B, N_v, d_model] Question-conditioned knowledge graph representation
        """
        target_dtype = text_embeds.dtype
        target_device = text_embeds.device

        f_od = f_od.to(device=target_device, dtype=target_dtype)
        edge_index = edge_index.to(device=target_device)
        edge_type = edge_type.to(device=target_device)

        # Layer 1 R-GCN + Activation + Dropout
        h = self.rgcn1(f_od, edge_index, edge_type)
        h = F.gelu(h)
        h = self.dropout(h)

        # Layer 2 R-GCN (Multi-hop aggregation)
        f_g = self.rgcn2(h, edge_index, edge_type)
        f_g = self.norm(f_od + f_g) # Residual connection

        # Question-guided Cross-Attention:
        # Query = f_g (Graph nodes), Key/Value = text_embeds (Question tokens)
        B = text_embeds.size(0)
        N_v = f_g.size(0)

        # Expand graph nodes across batch: [B, N_v, d_model]
        f_g_batch = f_g.unsqueeze(0).expand(B, -1, -1)

        Q = self.W_q(f_g_batch) # [B, N_v, d_model]
        K = self.W_k(text_embeds) # [B, L_t, d_model]
        V = self.W_v(text_embeds) # [B, L_t, d_model]

        scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_model) # [B, N_v, L_t]
        attn_weights = F.softmax(scores, dim=-1)
        
        k_prime = torch.bmm(attn_weights, V) # [B, N_v, d_model]
        k_prime = self.out_proj(k_prime)

        return k_prime


# Backwards compatibility alias
RGCNKGEModule = QuestionGuidedRGCN


if __name__ == "__main__":
    print("Testing RGCNKGEModule implementation under float32 and float16 (AMP)...")
    d_model = 768
    N_v = 30
    E = 60
    num_relations = 5

    for dtype in [torch.float32, torch.float16]:
        f_od = torch.randn(N_v, d_model, dtype=dtype)
        edge_index = torch.randint(0, N_v, (2, E))
        edge_type = torch.randint(0, num_relations, (E,))
        text_embeds = torch.randn(8, 32, d_model, dtype=dtype)

        model = RGCNKGEModule(d_model=d_model, num_relations=num_relations).to(dtype=dtype)
        output = model(f_od, edge_index, edge_type, text_embeds)
        
        assert output.shape == (8, N_v, d_model), "Shape mismatch!"
        assert output.dtype == dtype, "Dtype mismatch!"
        print(f"✅ R-GCN KGE Module test passed for dtype={dtype}!")
