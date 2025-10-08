import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import platform
import sys
import datetime
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
d_model = 128
nhead = 4
d_head = d_model // nhead
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

snapshots: Dict[int, Dict[str, Any]] = {}

# ═══════════════════════════════════════════════════════════════════
# Snapshot Semantics
# ═══════════════════════════════════════════════════════════════════
SNAP_SEMANTICS = {
    1: "Raw input token IDs (source sequence).",
    2: "Target token IDs (with EOS). Decoder input has BOS prepended.",
    3: "Embedding weight matrix slice (5×5). Full shape: (vocab_size, d_model).",
    4: "Source embeddings after lookup (batch, src_len, d_model), scaled by sqrt(d_model).",
    5: "Source embeddings + positional encoding (batch, src_len, d_model).",
    6: "Encoder block 1 input (batch, src_len, d_model).",
    7: "Encoder self-attention Query projection (batch, src_len, d_model).",
    8: "Encoder self-attention Key projection (batch, src_len, d_model).",
    9: "Encoder self-attention Value projection (batch, src_len, d_model).",
    10: "Encoder attention scores before softmax (batch, heads, src_len, src_len).",
    11: "Encoder attention weights after softmax (batch, heads, src_len, src_len).",
    12: "Encoder Q/K/V split into heads (3, batch, heads, src_len, d_head).",
    13: "Encoder multi-head output after concatenation (batch, src_len, d_model).",
    14: "Encoder residual after attention: src + attn_out (batch, src_len, d_model).",
    15: "Encoder layer normalization output after attention (batch, src_len, d_model).",
    16: "Encoder feed-forward input (batch, src_len, d_model).",
    17: "Encoder FF first linear layer output (batch, src_len, dim_feedforward).",
    18: "Encoder FF second linear layer output (batch, src_len, d_model).",
    19: "Encoder block 1 final output (batch, src_len, d_model).",
    20: "Decoder block 1 input (batch, tgt_len, d_model).",
    21: "Decoder masked self-attention Query (batch, tgt_len, d_model).",
    22: "Decoder masked self-attention Key (batch, tgt_len, d_model).",
    23: "Decoder masked self-attention Value (batch, tgt_len, d_model).",
    24: "Decoder masked attention scores BEFORE applying causal mask (batch, heads, tgt_len, tgt_len).",
    25: "Causal mask tensor (1, 1, tgt_len, tgt_len) with -inf above diagonal.",
    26: "Decoder masked attention weights after mask + softmax (batch, heads, tgt_len, tgt_len).",
    27: "Decoder masked Q/K/V split into heads (3, batch, heads, tgt_len, d_head).",
    28: "Decoder masked self-attention output after concatenation (batch, tgt_len, d_model).",
    29: "Decoder after residual + norm (masked self-attention) (batch, tgt_len, d_model).",
    30: "Cross-attention Query from decoder (batch, tgt_len, d_model).",
    31: "Cross-attention Key from encoder memory (batch, src_len, d_model).",
    32: "Cross-attention Value from encoder memory (batch, src_len, d_model).",
    33: "Cross-attention scores before softmax (batch, heads, tgt_len, src_len).",
    34: "Cross-attention weights after softmax (batch, heads, tgt_len, src_len).",
    35: "Cross-attention output after concatenation (batch, tgt_len, d_model).",
    36: "Decoder after residual + norm (cross-attention) (batch, tgt_len, d_model).",
    37: "Decoder feed-forward input (batch, tgt_len, d_model).",
    38: "Decoder FF first linear layer output (batch, tgt_len, dim_feedforward).",
    39: "Decoder FF second linear layer output (batch, tgt_len, d_model).",
    40: "Decoder block 1 final output (batch, tgt_len, d_model).",
    41: "Decoder final sequence output before projection (batch, tgt_len, d_model).",
    42: "Logits after final linear projection (batch, tgt_len, vocab_size).",
    43: "Logits slice for first token (first 10 vocabulary entries).",
}


# ═══════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════

def get_environment_info() -> Dict[str, Any]:
    """Collect essential environment information."""
    return {
        "timestamp": datetime.datetime.now().isoformat(),
        "python": {
            "version": sys.version.split()[0],
            "implementation": platform.python_implementation()
        },
        "os": {
            "system": platform.system(),
            "release": platform.release()
        },
        "wsl": {
            "is_wsl": ("microsoft" in platform.release().lower()) or
                      (os.environ.get("WSL_DISTRO_NAME") is not None),
            "distro": os.environ.get("WSL_DISTRO_NAME", "N/A")
        },
        "pytorch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "device": str(device),
    }


def save_snapshot(num: int, name: str, tensor: torch.Tensor, note: str = ""):
    """
    Save snapshot with intelligent slicing to minimize JSON size.

    Slicing Strategy (max 12 elements):
    - 1D tensors: [12]
    - 2D tensors: [4, 3] = 12 elements
    - 3D tensors: [2, 2, 3] = 12 elements
    - 4D+ tensors: [1, 1, ..., 12] = 12 elements (last dimension only)

    This ensures the JSON file stays under 500 lines.

    Breakpoint Strategy:
    Place breakpoint on the 'pass' line below. It will stop automatically
    at each of the 43 snapshots. Press F9 (Resume) to move to the next snapshot.

    Args:
        num: Snapshot number (1-43)
        name: Descriptive name
        tensor: PyTorch tensor to capture
        note: Optional additional description
    """
    arr = tensor.detach().cpu().numpy()

    # Maximum elements in slice (crucial for keeping JSON small)
    MAX_SLICE_ELEMENTS = 12

    # Calculate smart slice based on array dimensions
    if arr.ndim == 0:
        # Scalar value
        small = float(arr) if arr.dtype.kind == 'f' else arr.tolist()

    elif arr.size <= MAX_SLICE_ELEMENTS:
        # Array is small enough - take everything
        if arr.dtype.kind == 'f':
            small = np.round(arr, 4).tolist()
        else:
            small = arr.tolist()

    else:
        # Apply adaptive slicing based on dimensionality
        # Use if-elif chain instead of dictionary to avoid IndexError
        if arr.ndim == 1:
            idx = (slice(0, min(12, arr.shape[0])),)

        elif arr.ndim == 2:
            idx = (
                slice(0, min(4, arr.shape[0])),
                slice(0, min(3, arr.shape[1]))
            )

        elif arr.ndim == 3:
            idx = (
                slice(0, min(2, arr.shape[0])),
                slice(0, min(2, arr.shape[1])),
                slice(0, min(3, arr.shape[2]))
            )

        else:
            # 4D+: take first element from all dimensions except last
            idx = tuple([slice(0, 1)] * (arr.ndim - 1) + [slice(0, min(12, arr.shape[-1]))])

        small_arr = arr[idx]

        # Safety check: if still too large, flatten and limit
        if small_arr.size > MAX_SLICE_ELEMENTS:
            small_arr = small_arr.flatten()[:MAX_SLICE_ELEMENTS]

        # Format numbers based on dtype
        if small_arr.dtype.kind == 'f':
            small_arr = np.round(small_arr, 4)
        elif small_arr.dtype.kind not in ('i', 'b', 'u'):
            small_arr = small_arr.astype(str)

        small = small_arr.tolist()

    caption = note if note else SNAP_SEMANTICS.get(num, "")

    # ═══════════════════════════════════════════════════════════════
    # 🔴 BREAKPOINT HERE: Place breakpoint on the next line
    # ═══════════════════════════════════════════════════════════════
    pass  # Will stop at each of 43 snapshots. Press F9 to continue.
    # ═══════════════════════════════════════════════════════════════

    snapshots[num] = {
        "number": num,
        "name": name,
        "shape": list(arr.shape),
        "slice": small,
        "caption": caption
    }

    # Enhanced feedback with slice size
    num_elements = len(small) if isinstance(small, list) else 1
    print(f"✓ Snapshot {num:02d}/43: {name:<50} | shape={str(list(arr.shape)):20} | slice_elements={num_elements:2d}")


# ═══════════════════════════════════════════════════════════════════
# Data Preparation
# ═══════════════════════════════════════════════════════════════════

input_text = "Messi passes the ball to Neymar"
target_text = "Neymar receives the ball from Messi skillfully"


def simple_tokenize(s: str):
    """Simple whitespace tokenizer."""
    return s.replace("'", "'").strip().split()


tokens_src = simple_tokenize(input_text)
tokens_tgt = simple_tokenize(target_text)

# Build vocabulary
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
vocab_list = [PAD, BOS, EOS, UNK] + sorted(set(tokens_src + tokens_tgt))
vocab = {w: i for i, w in enumerate(vocab_list)}


def tokens_to_ids(tokens):
    """Convert tokens to vocabulary IDs."""
    return [vocab.get(t, vocab[UNK]) for t in tokens]


# Prepare sequences
src_ids = tokens_to_ids(tokens_src)
decoder_input_tokens = [BOS] + tokens_tgt
target_tokens = tokens_tgt + [EOS]
decoder_input_ids = tokens_to_ids(decoder_input_tokens)
tgt_ids = tokens_to_ids(target_tokens)

# Create tensors
src_seq = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
dec_in_seq = torch.tensor(decoder_input_ids, dtype=torch.long, device=device).unsqueeze(0)

# Save snapshots #1 and #2
save_snapshot(1, "Raw_input_tokens_IDS", torch.tensor(src_ids),
              note=f"Input text: '{input_text}' | Tokens: {tokens_src}")
save_snapshot(2, "Target_tokens_IDS", torch.tensor(tgt_ids),
              note=f"Target text: '{target_text}' | Target tokens (with EOS): {target_tokens} | Decoder input (with BOS): {decoder_input_tokens}")


# ═══════════════════════════════════════════════════════════════════
# Model Components
# ═══════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input embeddings."""
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return x


class MultiHeadAttentionCapture(nn.Module):
    """Multi-head attention with intermediate values capture."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass with intermediate values capture.

        Returns:
            out: Final output after projection
            extras: Dictionary of intermediate values for debugging
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        def split_heads(x):
            return x.view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)

        # Split into heads
        Qh = split_heads(Q)
        Kh = split_heads(K)
        Vh = split_heads(V)

        # Attention scores
        scores_before_mask = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.d_head ** 0.5)

        # Apply mask if provided
        if attn_mask is not None:
            scores_after_mask = scores_before_mask + attn_mask
        else:
            scores_after_mask = scores_before_mask

        # Softmax
        weights = F.softmax(scores_after_mask, dim=-1)

        # Apply attention to values
        attn_output_heads = torch.matmul(weights, Vh)

        # Concatenate heads
        attn_output_concat = (
            attn_output_heads.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        # Final projection
        out = self.out_proj(attn_output_concat)

        extras = {
            "Q": Q,
            "K": K,
            "V": V,
            "Qh": Qh,
            "Kh": Kh,
            "Vh": Vh,
            "scores_before_mask": scores_before_mask,
            "scores_before_softmax": scores_after_mask,
            "weights_after_softmax": weights,
            "attn_output_concat": attn_output_concat
        }
        return out, extras


class EncoderBlock(nn.Module):
    """Transformer encoder block."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.mha = MultiHeadAttentionCapture(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ff_activation = nn.ReLU()
        self.ff_linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, block_index=0):
        """
        Forward pass with snapshot capture for first block.

        Args:
            src: Input tensor
            block_index: Block number (0-indexed). Snapshots only captured for block 0.
        """
        capture = (block_index == 0)

        if capture:
            save_snapshot(6, "Encoder_block1_input", src,
                          note="Input to first encoder block")

        attn_out, extras = self.mha(src, src, src)

        if capture:
            save_snapshot(7, "Encoder_block1_Q", extras["Q"],
                          note="Query projection in encoder self-attention")
            save_snapshot(8, "Encoder_block1_K", extras["K"],
                          note="Key projection in encoder self-attention")
            save_snapshot(9, "Encoder_block1_V", extras["V"],
                          note="Value projection in encoder self-attention")
            save_snapshot(10, "Encoder_block1_attn_scores_before_softmax",
                          extras["scores_before_softmax"],
                          note="Attention scores before softmax (scaled dot-product)")
            save_snapshot(11, "Encoder_block1_attn_scores_after_softmax",
                          extras["weights_after_softmax"],
                          note="Attention weights after softmax (probabilities)")

            save_snapshot(12, "Encoder_block1_multihead_split_QKV",
                          torch.stack([extras["Qh"], extras["Kh"], extras["Vh"]], dim=0),
                          note="Q/K/V all split into heads: (3, batch, heads, seq, d_head)")

            save_snapshot(13, "Encoder_block1_attn_concat_output",
                          extras["attn_output_concat"],
                          note="Multi-head attention output after concatenation")

        res1 = src + attn_out
        if capture:
            save_snapshot(14, "Encoder_block1_residual_after_attn", res1,
                          note="After adding residual connection (src + attn_out)")

        norm1 = self.norm1(res1)
        if capture:
            save_snapshot(15, "Encoder_block1_layernorm1_output", norm1,
                          note="After layer normalization")
            save_snapshot(16, "Encoder_block1_ff_input", norm1,
                          note="Input to feed-forward network")

        ff1_out = self.ff_linear1(norm1)
        if capture:
            save_snapshot(17, "Encoder_block1_ff_linear1_output", ff1_out,
                          note=f"After first FF linear layer (expansion to {dim_feedforward})")

        ff_activated = self.ff_activation(ff1_out)
        ff_out = self.ff_linear2(ff_activated)
        if capture:
            save_snapshot(18, "Encoder_block1_ff_linear2_output", ff_out,
                          note=f"After second FF linear layer (projection back to {d_model})")

        res2 = norm1 + ff_out
        norm2 = self.norm2(res2)

        if capture:
            save_snapshot(19, "Encoder_block1_final_output", norm2,
                          note="Final output of first encoder block")

        return norm2


class DecoderBlock(nn.Module):
    """Transformer decoder block."""

    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.mha_masked = MultiHeadAttentionCapture(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)
        self.mha_cross = MultiHeadAttentionCapture(d_model, nhead)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff_linear1 = nn.Linear(d_model, dim_feedforward)
        self.ff_activation = nn.ReLU()
        self.ff_linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, block_index=0):
        """
        Forward pass with snapshot capture for first block.

        Args:
            tgt: Target sequence input
            memory: Encoder output (memory)
            tgt_mask: Causal mask for decoder self-attention
            block_index: Block number (0-indexed). Snapshots only captured for block 0.
        """
        capture = (block_index == 0)

        if capture:
            save_snapshot(20, "Decoder_block1_input", tgt,
                          note="Input to first decoder block")

        attn_out1, extras1 = self.mha_masked(tgt, tgt, tgt, attn_mask=tgt_mask)

        if capture:
            save_snapshot(21, "Decoder_block1_masked_Q", extras1["Q"],
                          note="Query in masked self-attention")
            save_snapshot(22, "Decoder_block1_masked_K", extras1["K"],
                          note="Key in masked self-attention")
            save_snapshot(23, "Decoder_block1_masked_V", extras1["V"],
                          note="Value in masked self-attention")
            save_snapshot(24, "Decoder_block1_masked_scores_before_mask",
                          extras1["scores_before_mask"],
                          note="Attention scores BEFORE applying causal mask")
            save_snapshot(25, "Decoder_block1_mask_tensor", tgt_mask,
                          note="Causal mask tensor (upper triangle = -inf)")
            save_snapshot(26, "Decoder_block1_masked_scores_after_softmax",
                          extras1["weights_after_softmax"],
                          note="Attention weights after mask + softmax (lower triangular)")

            save_snapshot(27, "Decoder_block1_masked_multihead_split_QKV",
                          torch.stack([extras1["Qh"], extras1["Kh"], extras1["Vh"]], dim=0),
                          note="Masked Q/K/V all split into heads: (3, batch, heads, seq, d_head)")

            save_snapshot(28, "Decoder_block1_masked_multihead_concat",
                          extras1["attn_output_concat"],
                          note="Masked self-attention output after concat")

        res1 = tgt + attn_out1
        norm1 = self.norm1(res1)

        if capture:
            save_snapshot(29, "Decoder_block1_residual_norm_after_masked", norm1,
                          note="After residual + norm following masked self-attention")

        attn_out2, extras2 = self.mha_cross(norm1, memory, memory)

        if capture:
            save_snapshot(30, "Decoder_block1_cross_Q", extras2["Q"],
                          note="Query from decoder in cross-attention")
            save_snapshot(31, "Decoder_block1_cross_K", extras2["K"],
                          note="Key from encoder in cross-attention")
            save_snapshot(32, "Decoder_block1_cross_V", extras2["V"],
                          note="Value from encoder in cross-attention")
            save_snapshot(33, "Decoder_block1_cross_scores_before_softmax",
                          extras2["scores_before_softmax"],
                          note="Cross-attention scores before softmax")
            save_snapshot(34, "Decoder_block1_cross_scores_after_softmax",
                          extras2["weights_after_softmax"],
                          note="Cross-attention weights: (batch, heads, tgt_len, src_len)")
            save_snapshot(35, "Decoder_block1_cross_attn_concat",
                          extras2["attn_output_concat"],
                          note="Cross-attention output after concat")

        res2 = norm1 + attn_out2
        norm2 = self.norm2(res2)

        if capture:
            save_snapshot(36, "Decoder_block1_residual_norm_after_cross", norm2,
                          note="After residual + norm following cross-attention")
            save_snapshot(37, "Decoder_block1_ff_input", norm2,
                          note="Input to decoder feed-forward")

        ff1_out = self.ff_linear1(norm2)
        if capture:
            save_snapshot(38, "Decoder_block1_ff_linear1_output", ff1_out,
                          note="After decoder FF first linear layer")

        ff_activated = self.ff_activation(ff1_out)
        ff_out = self.ff_linear2(ff_activated)
        if capture:
            save_snapshot(39, "Decoder_block1_ff_linear2_output", ff_out,
                          note="After decoder FF second linear layer")

        res3 = norm2 + ff_out
        norm3 = self.norm3(res3)

        if capture:
            save_snapshot(40, "Decoder_block1_final_output", norm3,
                          note="Final output of first decoder block")

        return norm3


class MiniTransformer(nn.Module):
    """Mini Transformer model for debugging."""

    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers,
                 num_decoder_layers, dim_feedforward):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.posenc = PositionalEncoding(d_model)

        self.enc_layers = nn.ModuleList([
            EncoderBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_encoder_layers)
        ])

        self.dec_layers = nn.ModuleList([
            DecoderBlock(d_model, nhead, dim_feedforward)
            for _ in range(num_decoder_layers)
        ])

        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_ids, tgt_input_ids):
        """
        Forward pass with snapshot capture.

        Args:
            src_ids: Source token IDs
            tgt_input_ids: Target token IDs (with BOS prepended)

        Returns:
            logits: Output logits over vocabulary
        """
        # Snapshot 3: Embedding matrix slice
        save_snapshot(3, "Embedding_weight_matrix_slice",
                      self.embedding.weight[:5, :5],
                      note=f"Embedding matrix 5×5 slice | Full shape: (vocab_size={self.embedding.weight.shape[0]}, d_model={self.d_model})")

        # Encoder path
        src_emb = self.embedding(src_ids) * (self.d_model ** 0.5)
        save_snapshot(4, "Input_embeddings_after_lookup", src_emb,
                      note="Source embeddings after lookup (scaled by sqrt(d_model))")

        src_emb_pos = self.posenc(src_emb)
        save_snapshot(5, "Embeddings_after_positional_encoding", src_emb_pos,
                      note="Source embeddings after adding positional encoding")

        memory = src_emb_pos
        for i, layer in enumerate(self.enc_layers):
            memory = layer(memory, block_index=i)

        # Decoder path
        tgt_emb = self.embedding(tgt_input_ids) * (self.d_model ** 0.5)
        tgt_emb_pos = self.posenc(tgt_emb)

        # Create causal mask: shape (1, 1, tgt_len, tgt_len)
        tgt_len = tgt_input_ids.size(1)
        attn_mask = torch.triu(
            torch.full((tgt_len, tgt_len), float('-inf'), device=device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        dec_output = tgt_emb_pos
        for i, layer in enumerate(self.dec_layers):
            dec_output = layer(dec_output, memory, tgt_mask=attn_mask, block_index=i)

        # Final outputs
        save_snapshot(41, "Decoder_final_sequence_output_before_projection",
                      dec_output,
                      note="Final decoder output before projection to vocabulary")

        logits = self.proj(dec_output)
        save_snapshot(42, "Logits_after_final_projection", logits,
                      note=f"Logits: (batch, tgt_len, vocab_size={self.embedding.weight.shape[0]})")

        save_snapshot(43, "Logits_slice_first_token", logits[0, 0, :10],
                      note="First 10 logits for first token (to predict next word)")

        return logits


# ═══════════════════════════════════════════════════════════════════
# Debug Entry Point
# ═══════════════════════════════════════════════════════════════════

def debug_forward_pass(model, src_seq, dec_in_seq):
    """
    Run forward pass without gradient computation.

    All snapshots are captured during this call via save_snapshot().
    """
    with torch.no_grad():
        logits = model(src_seq, dec_in_seq)
    return logits


# ═══════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 80)
    print(" " * 20 + "📸 Mini Transformer Debugging (Optimized)")
    print("=" * 80)
    print(f"Input:  '{input_text}'")
    print(f"Target: '{target_text}'")
    print(f"Device: {device}")
    print("=" * 80)

    # Validate input constraints
    assert 5 <= len(tokens_tgt) <= 12, f"Target must be 5-12 tokens, got {len(tokens_tgt)}"
    assert len(decoder_input_ids) == len(tgt_ids), "Length mismatch between decoder input and target"

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create model
    model = MiniTransformer(
        len(vocab), d_model, nhead,
        num_encoder_layers, num_decoder_layers,
        dim_feedforward
    ).to(device)

    print(f"\n🚀 Starting snapshot capture (expecting 43 snapshots)...\n")

    # Run forward pass and capture snapshots
    logits = debug_forward_pass(model, src_seq, dec_in_seq)

    # Summary
    print("\n" + "=" * 80)
    print(f"✅ Snapshot capture complete: {len(snapshots)}/43 snapshots")

    # Check for missing snapshots
    missing = [i for i in range(1, 44) if i not in snapshots]
    if missing:
        print(f"⚠️  WARNING: Missing snapshots: {missing}")
    else:
        print("✅ All snapshots captured successfully!")
    print("=" * 80)

    # Prepare summary data
    summary_data = {
        "metadata": {
            "project_name": "Mini Transformer Debugging (Optimized)",
            "created_at": datetime.datetime.now().isoformat(),
            "total_snapshots": len(snapshots),
            "expected_snapshots": 43,
            "status": "complete" if len(snapshots) == 43 else "incomplete"
        },
        "environment": get_environment_info(),
        "data": {
            "input_text": input_text,
            "target_text": target_text,
            "tokens_src": tokens_src,
            "tokens_tgt": tokens_tgt,
            "decoder_input_tokens": decoder_input_tokens,
            "target_tokens": target_tokens,
            "vocab_size": len(vocab)
        },
        "model_config": {
            "d_model": d_model,
            "nhead": nhead,
            "d_head": d_head,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "vocab_size": len(vocab),
            "device": str(device)
        },
        # Snapshots as array (sorted by number)
        "snapshots": [snapshots[i] for i in sorted(snapshots.keys())]
    }

    # Save summary with error handling
    summary_path = "snapshots_summary.json"

    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        file_size_kb = os.path.getsize(summary_path) / 1024
        print(f"\n💾 Summary saved successfully!")
        print(f"   📄 File: {summary_path}")
        print(f"   📊 Size: {file_size_kb:.1f} KB")
        print(f"   📝 Expected: ~15-25 KB (with smart slicing)")

    except Exception as e:
        print(f"\n❌ Error saving summary: {e}")
        backup_path = "summary_backup.json"
        try:
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2)
            print(f"💾 Saved to backup: {backup_path}")
        except Exception as e2:
            print(f"❌ Failed to save backup as well: {e2}")
