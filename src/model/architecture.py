"""The neural diacritic restoration model: a small Transformer encoder over
per-syllable character-composed vectors, with a masked class-scoring head.
"""

import math

import torch
import torch.nn as nn

# Finite "effectively zero probability" fill for invalid candidate slots.
# A literal float('-inf') would turn a fully-invalid row (e.g. a padding
# position, where every candidate slot is invalid) into an all -inf row,
# and log_softmax over an all -inf row is NaN -- this stays finite instead.
NEGATIVE_INF = -1e9


class SyllableCharEncoder(nn.Module):
    """Composes a toneless syllable's character sequence into one fixed-size
    vector via multi-kernel 1D convolutions (a small char-CNN, in the spirit
    of Kim et al.'s character-aware word representations). This is what lets
    an unseen syllable -- most of the corpus's long tail of unique toneless
    forms -- still get a meaningful representation instead of collapsing to
    a generic <unk>, the same reasoning fastText uses for rare words.
    """

    def __init__(
        self,
        char_vocab_size: int,
        char_embed_dim: int,
        d_model: int,
        kernel_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 64,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(char_embed_dim, num_filters, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.proj = nn.Linear(num_filters * len(kernel_sizes), d_model)

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        """char_ids: (B, S, L) -> (B, S, d_model)"""
        b, s, l = char_ids.shape
        x = self.char_embedding(char_ids.view(b * s, l))  # (B*S, L, E)
        x = x.transpose(1, 2)  # (B*S, E, L)
        pooled = [torch.relu(conv(x)).amax(dim=-1) for conv in self.convs]  # each (B*S, F)
        composed = torch.cat(pooled, dim=-1)  # (B*S, F * len(kernel_sizes))
        out = self.proj(composed)  # (B*S, d_model)
        return out.view(b, s, -1)


class SinusoidalPositionalEncoding(nn.Module):
    """Computed on the fly for the input's actual sequence length (not a
    fixed buffer sized at init), so it generalizes cleanly to inference-time
    sentences longer than anything seen in training -- training caps
    sequence length for batching efficiency, but restore() at inference
    does not."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return x + pe.unsqueeze(0)


class NeuralDiacriticModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        num_classes: int,
        char_embed_dim: int = 32,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Bundled into checkpoints so the module can be reconstructed before
        # loading weights, per the plan's checkpoint format.
        self.config = dict(
            char_vocab_size=char_vocab_size,
            num_classes=num_classes,
            char_embed_dim=char_embed_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.char_encoder = SyllableCharEncoder(char_vocab_size, char_embed_dim, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.class_embeddings = nn.Embedding(num_classes, d_model)

    def forward(
        self,
        char_ids: torch.Tensor,  # (B, S, L)
        padding_mask: torch.Tensor,  # (B, S) bool, True = pad
        candidate_ids: torch.Tensor,  # (B, S, C) global class ids
        candidate_valid: torch.Tensor,  # (B, S, C) bool
    ) -> torch.Tensor:
        x = self.char_encoder(char_ids)  # (B, S, d_model)
        x = self.pos_encoding(x)
        hidden = self.transformer(x, src_key_padding_mask=padding_mask)  # (B, S, d_model)

        candidate_embeds = self.class_embeddings(candidate_ids)  # (B, S, C, d_model)
        scores = (hidden.unsqueeze(2) * candidate_embeds).sum(dim=-1)  # (B, S, C)
        scores = scores.masked_fill(~candidate_valid, NEGATIVE_INF)
        return scores
