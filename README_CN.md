# 从零实现 Transformer 语言模型

[![English](https://img.shields.io/badge/Docs-English-blue.svg)](README.md) [![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)

基于 PyTorch 从零实现的 **Decoder-only Transformer**，用于**自回归语言建模**任务。本项目完整展示了对注意力机制、现代归一化技术以及 NLP 端到端训练流程的深入理解。

## 技术亮点

- **纯 PyTorch 实现**：所有核心组件（Multi-Head Attention、RMSNorm、Positional Encoding）均从零实现，未使用 `nn.TransformerDecoder`
- **现代架构设计**：Pre-LayerNorm、RMSNorm、GELU 激活函数、带权重衰减的 AdamW 优化器
- **因果掩码机制**：高效的上三角掩码生成，支持自回归解码
- **BPE 分词**：使用字节对编码实现子词分词，优化词表覆盖率
- **注意力可视化**：内置注意力权重提取，支持模型可解释性分析

## 模型架构

```
输入 Token IDs
       │
       ▼
┌──────────────────┐
│   Token 嵌入层   │  (vocab_size → d_model)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  正弦位置编码    │  PE(pos,2i) = sin(pos/10000^(2i/d))
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────┐
│       Transformer Block × N          │
│  ┌─────────────────────────────────┐ │
│  │ RMSNorm → 多头自注意力          │ │
│  │        (因果掩码)               │ │
│  │           + 残差连接            │ │
│  └─────────────────────────────────┘ │
│  ┌─────────────────────────────────┐ │
│  │ RMSNorm → FFN (GELU)            │ │
│  │           + 残差连接            │ │
│  └─────────────────────────────────┘ │
└────────┬─────────────────────────────┘
         │
         ▼
┌──────────────────┐
│     RMSNorm      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    LM Head       │  (d_model → vocab_size)
└────────┬─────────┘
         │
         ▼
   Logits (B, T, V)
```

## 核心组件

### 1. 缩放点积注意力

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

其中 $M$ 为因果掩码，$M_{ij} = -\infty$ 当 $j > i$。

### 2. 多头自注意力机制

```python
# 并行注意力头，带学习投影
Q, K, V = W_q(x), W_k(x), W_v(x)  # 线性投影
Q, K, V = split_heads(Q, K, V)    # (B, H, T, d_k)
attn = softmax(QK^T / sqrt(d_k) + causal_mask) @ V
output = W_o(concat_heads(attn))
```

### 3. RMSNorm（均方根归一化）

$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}$$

相比 LayerNorm 的优势：
- 无需均值减法 → 计算更快
- 性能相当但复杂度更低
- 被 LLaMA、Gemma 等现代大语言模型采用

### 4. 逐位置前馈网络

$$\text{FFN}(x) = W_2 \cdot \text{GELU}(W_1 \cdot x + b_1) + b_2$$

扩展比例：$d_{ff} / d_{model} = 2$

## 模型配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| `d_model` | 64 | 隐藏层维度 |
| `n_heads` | 2 | 注意力头数 |
| `n_layers` | 2 | Transformer 层数 |
| `d_ff` | 128 | FFN 中间层维度 |
| `dropout` | 0.2 | Dropout 概率 |
| `seq_len` | 50 | 上下文窗口大小 |
| `vocab_size` | 500 | BPE 词表大小 |

**总参数量**：131K

## 训练流程

### 数据处理
- **语料库**：Tiny Shakespeare（约 110 万字符，44.7 万 BPE tokens）
- **分词方式**：BPE，`min_frequency=2`，空格预分词
- **序列生成**：滑动窗口（步长=1）
- **数据划分**：80% 训练集（35.8 万序列）/ 20% 验证集（8.9 万序列）

### 优化策略
- **优化器**：AdamW（$\beta_1=0.9$，$\beta_2=0.999$，权重衰减=$0.01$）
- **学习率**：$3 \times 10^{-4}$
- **梯度裁剪**：最大范数 = 1.0
- **批次大小**：64
- **训练轮数**：10

### 损失函数
下一个 token 预测的交叉熵损失：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}\log P(x_t | x_{<t})$$

## 实验结果

| 指标 | 数值 |
|------|------|
| 最终训练损失 | ~2.8 |
| 最终验证损失 | ~3.0 |
| 验证集困惑度 | ~20 |

## 项目结构

```
transformer-from-scratch/
├── transformer.ipynb   # 完整实现代码
├── input.txt           # Tiny Shakespeare 语料库
├── README.md           # 英文文档
└── README_CN.md        # 中文文档
```

## 快速开始

### 环境依赖

```bash
pip install torch numpy matplotlib tokenizers
```

### 运行

```bash
jupyter notebook transformer.ipynb
```

按顺序执行 notebook 单元格：
1. 在语料库上训练 BPE 分词器
2. 构建输入/目标序列对
3. 初始化 Transformer 模型
4. 训练并监控验证集表现
5. 可视化注意力模式

## 实现细节

### 因果掩码生成
```python
def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float("-inf"))
```

### 注意力权重缓存
前向传播时缓存注意力权重，用于可视化：
```python
self.last_attn_weights = attn.detach().cpu()
```

### Pre-LN 架构
归一化在注意力/FFN 之前应用（训练更稳定）：
```python
h = self.norm1(x)           # 前置归一化
h = self.attn(h, mask)
x = x + self.dropout(h)     # 残差连接
```

## 注意力可视化

项目内置注意力模式可视化工具，展示模型如何在训练过程中学习 token 间的依赖关系。

## 参考文献

1. Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
2. Zhang & Sennrich. [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019)
3. Radford et al. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2, 2019)
4. Sennrich et al. [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) (BPE, 2016)

## 许可证

MIT License - 用于学习 Transformer 基础的教育项目。
