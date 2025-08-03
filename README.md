# GPT2025

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Stars](https://img.shields.io/github/stars/dustinwloring1988/gpt2025?style=social)](https://github.com/dustinwloring1988/gpt2025/stargazers)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**GPT2025** is an advanced, research-grade implementation of a GPT-style language model that integrates cutting-edge efficiency techniques including FlashAttention, RMSNorm, RoPE/NoPE hybrid positional embeddings, and SwiGLU activations. This repository also includes support scripts for large-scale pretraining and evaluation on datasets like FineWeb and HellaSwag.

---

## 🚀 Features

- 🔁 **Hybrid RoPE + NoPE Positional Embeddings**
- ⚡ **FlashAttention 3** integration (optional)
- 🧠 **RMSNorm** over traditional LayerNorm
- 🧬 **SwiGLU Activation** support
- 🧮 **KV Caching** for efficient generation
- 🧰 **From-pretrained GPT-2 compatibility**
- 🏋️‍♂️ **Efficient MLP and attention layers**
- 🔄 **Gradient checkpointing (optional)**
- 📉 **Cosine annealing learning rate scheduler**

---

## 📁 Repository Structure

```text
.
├── enhanced_gpt2_model.py   # Core GPT2025 model with all optimizations
├── fineweb.py               # Dataset preprocessor for FineWeb-Edu
├── hellaswag.py             # Script for downloading and evaluating HellaSwag
└── README.md                # This file
````

---

## 🧠 Model Variants

The `create_optimized_model` function lets you choose between different model sizes and positional encoding strategies:

```python
model = create_optimized_model(model_size='small', position_strategy='hybrid')
```

Available options:

* `model_size`: `'small'`, `'medium'`, `'large'`
* `position_strategy`: `'hybrid'`, `'rope'`, `'nope'`

---

## 📦 Pretraining: FineWeb-Edu

The `fineweb.py` script tokenizes and shards the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset:

```bash
python fineweb.py
```

This will download and preprocess the dataset into the `edu_fineweb10B` directory as NumPy `.npy` shards.

---

## 📊 Evaluation: HellaSwag

To evaluate GPT2-family models on HellaSwag:

```bash
python hellaswag.py --model_type gpt2-xl --device cuda
```

Supports both `acc` and `acc_norm` scoring strategies for multiple-choice evaluation.

---

## 🛠️ Requirements

* Python 3.8+
* PyTorch 2.x
* `transformers`, `datasets`, `tiktoken`, `tqdm`, `numpy`
* Optional: FlashAttention, Triton, A100 for optimal performance

Install with pip:

```bash
pip install torch transformers datasets tiktoken tqdm numpy
```

---

## 🧪 Example Forward Pass

```python
model = create_optimized_model('small', 'hybrid').to('cuda')
input_ids = torch.randint(0, model.config.vocab_size, (4, 512)).to('cuda')
logits, loss = model(input_ids)
```

---

## 📈 Training Utilities

The model provides:

* `configure_optimizers`: optimizer setup with parameter group separation
* `reset_cache`: clears KV-cache for generation
* `estimate_mfu`: estimate model FLOP utilization (A100 reference)

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built with insights from NanoGPT, FlashAttention, HellaSwag, and the open-source transformer community.
