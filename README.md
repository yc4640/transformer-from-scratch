# Transformer Language Model from Scratch

[![中文文档](https://img.shields.io/badge/文档-中文版-blue.svg)](README_CN.md) [![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

> **[中文文档 / Chinese Documentation](README_CN.md)**

A **decoder-only Transformer** implementation from scratch for **autoregressive language modeling**. This project demonstrates deep understanding of the attention mechanism, modern normalization techniques, and end-to-end training pipeline for NLP tasks.

## Technical Highlights

- **Pure PyTorch Implementation**: All core components (Multi-Head Attention, RMSNorm, Positional Encoding) implemented from scratch without relying on `nn.TransformerDecoder`
- **Modern Architecture Choices**: Pre-LayerNorm, RMSNorm, GELU activation, AdamW optimizer with weight decay
- **Causal Masking**: Efficient upper-triangular mask generation for autoregressive decoding
- **BPE Tokenization**: Subword tokenization using Byte-Pair Encoding for optimal vocabulary coverage
- **Attention Visualization**: Built-in attention weight extraction for interpretability analysis

## Architecture

```
Input Token IDs
       │
       ▼
┌──────────────────┐
│  Token Embedding │  (vocab_size → d_model)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Sinusoidal PosEnc│  PE(pos,2i) = sin(pos/10000^(2i/d))
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│         Transformer Block × N        │
│  ┌─────────────────────────────────┐ │
│  │  RMSNorm → Multi-Head Attention │ │
│  │         (Causal Masked)         │ │
│  │              + Residual         │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │  RMSNorm → FFN (GELU)           │ │
│  │              + Residual         │ │
│  └─────────────────────────────────┘ │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│    RMSNorm       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   LM Head        │  (d_model → vocab_size)
└────────┬─────────┘
         │
         ▼
    Logits (B, T, V)
```

## Core Components

### 1. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

Where $M$ is the causal mask with $M_{ij} = -\infty$ for $j > i$.

### 2. Multi-Head Self-Attention

```python
# Parallel attention heads with learned projections
Q, K, V = W_q(x), W_k(x), W_v(x)  # Linear projections
Q, K, V = split_heads(Q, K, V)    # (B, H, T, d_k)
attn = softmax(QK^T / sqrt(d_k) + causal_mask) @ V
output = W_o(concat_heads(attn))
```

### 3. RMSNorm (Root Mean Square Normalization)

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}$$

Advantages over LayerNorm:
- No mean subtraction → faster computation
- Comparable performance with reduced complexity
- Used in LLaMA, Gemma, and other modern LLMs

### 4. Position-wise Feed-Forward Network

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

Expansion ratio: $d_{ff} / d_{model} = 2$

## Model Configuration

| Hyperparameter | Value | Description |
|----------------|-------|-------------|
| `d_model` | 64 | Hidden dimension |
| `n_heads` | 2 | Number of attention heads |
| `n_layers` | 2 | Transformer blocks |
| `d_ff` | 128 | FFN intermediate dimension |
| `dropout` | 0.2 | Dropout probability |
| `seq_len` | 50 | Context window size |
| `vocab_size` | 500 | BPE vocabulary size |

**Total Parameters**: 131K

## Training Pipeline

### Data Processing
- **Corpus**: Tiny Shakespeare (~1.1M characters, 447K BPE tokens)
- **Tokenization**: BPE with `min_frequency=2`, whitespace pre-tokenizer
- **Sequence Generation**: Overlapping sliding window (stride=1)
- **Split**: 80% train (358K sequences) / 20% validation (89K sequences)

### Optimization
- **Optimizer**: AdamW ($\beta_1=0.9,\ \beta_2=0.999,\ \text{weight decay}=0.01$)
- **Learning Rate**: $3 \times 10^{-4}$
- **Gradient Clipping**: Max norm = 1.0
- **Batch Size**: 64
- **Epochs**: 10

### Loss Function
Cross-entropy loss for next-token prediction:

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{\lt t})
$$


## Results

| Metric | Value |
|--------|-------|
| Final Training Loss | ~2.8 |
| Final Validation Loss | ~3.0 |
| Validation Perplexity | ~20 |

## Project Structure

```
transformer-from-scratch/
├── transformer.ipynb   # Complete implementation
├── input.txt           # Tiny Shakespeare corpus
├── README.md           # English documentation
└── README_CN.md        # Chinese documentation
```

## Quick Start

### Requirements

```bash
pip install torch numpy matplotlib tokenizers
```

### Run

```bash
jupyter notebook transformer.ipynb
```

Execute cells sequentially to:
1. Train BPE tokenizer on corpus
2. Build input/target sequence pairs
3. Initialize Transformer model
4. Train with validation monitoring
5. Visualize attention patterns

## Implementation Details

### Causal Mask Generation
```python
def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))
```

### Attention Weight Caching
Attention weights are cached during forward pass for visualization:
```python
self.last_attn_weights = attn.detach().cpu()
```

### Pre-LN Architecture
Normalization applied before attention/FFN (more stable training):
```python
h = self.norm1(x)           # Pre-norm
h = self.attn(h, mask)
x = x + self.dropout(h)     # Residual
```

## Attention Visualization

The implementation includes tools to visualize attention patterns, showing how the model learns token dependencies through training.

## References

1. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
2. Zhang & Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019)
3. Radford et al. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
4. Sennrich et al. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE, 2016)

## License

MIT License - Educational project for learning Transformer fundamentals.
