# 🏥 English-Vietnamese Medical Machine Translation
### Building and Fine-Tuning a Transformer for Specialized Medical Translation

A two-stage neural machine translation system that trains a Transformer from scratch on general English-Vietnamese data, then fine-tunes it for the medical domain — achieving a **10–12 BLEU point improvement** over the pre-trained baseline on medical text.

---

<!-- Add a diagram or sample translation screenshot here -->
<!-- ![Architecture Overview](architecture.png) -->

---

## Results Summary

| Stage | Dataset | BLEU Score |
|---|---|---|
| Pre-training | General (960K pairs) | 38.05 |
| Before fine-tuning | Medical test set | 28.60 |
| After fine-tuning | Medical test set | 43.44 |

Fine-tuning for just **4 epochs** closes a 14.84-point domain gap and surpasses the general BLEU score — using only ~1.5–2 hours of GPU time.

---

## Architecture

The model implements the standard Transformer architecture built from scratch in PyTorch:

| Hyperparameter | Value |
|---|---|
| Encoder / Decoder layers | 4 each |
| Embedding dimension (d_model) | 256 |
| Attention heads | 8 |
| FFN dimension | 1,024 |
| Dropout | 0.15 |
| Max sequence length | 128 |
| Vocabulary size | 40,000 |
| Total parameters | **38.1M** |

The tokenizer uses **Byte-Level BPE** with byte fallback, guaranteeing 0% unknown tokens even on specialized medical vocabulary and Vietnamese diacritics.

---

## Project Structure

```
├── training/
│   ├── pretrain.py           # Pre-training on general data
│   └── finetune.py           # Fine-tuning on medical data
├── model/
│   ├── transformer.py        # Transformer architecture (encoder, decoder, attention)
│   ├── positional_encoding.py
│   └── tokenizer.py          # Byte-Level BPE tokenizer training
├── data/
│   ├── general/              # ~960K English-Vietnamese sentence pairs
│   └── medical/              # ~322K medical sentence pairs
├── evaluate.py               # BLEU scoring with SacreBLEU
└── translate.py              # Inference script
```

---

## How It Works

```
Stage 1 — Pre-training
  960K general EN-VI pairs  →  Transformer (38M params)  →  BLEU 38.05

Stage 2 — Fine-tuning
  322K medical EN-VI pairs  →  Fine-tuned model  →  Medical BLEU 43.44
```

**Stage 1 — Pre-training on General Data**

The Transformer is trained for 20 epochs on ~960K cleaned general-domain sentence pairs. A `ReduceLROnPlateau` scheduler automatically reduces the learning rate when validation loss stops improving.

**Stage 2 — Fine-tuning on Medical Data**

The pre-trained model is adapted to the medical domain using a lower learning rate and stronger regularization to prevent catastrophic forgetting:

| Parameter | Pre-training | Fine-tuning |
|---|---|---|
| Batch size | 64 | 32 |
| Learning rate | 3e-4 | **5e-5** |
| Optimizer | Adam | AdamW |
| Weight decay | 0.0 | 0.01 |
| Label smoothing | 0.0 | 0.1 |
| Warmup | 4,000 steps | 10% of steps |
| Epochs | 20 | 4 |

The learning rate of `5e-5` proved optimal — conservative enough to preserve general linguistic knowledge while still adapting to medical terminology.

---

## Data

### General Dataset (~981K pairs)

- Source: Publicly available English-Vietnamese parallel corpora
- Cleaning: removes pairs shorter than 3 tokens, length ratio filtering (0.5–2.0), language detection via `langid`, Unicode normalization, deduplication

### Medical Dataset (~322K pairs)

- Sources: medical journal abstracts, clinical treatment guidelines, case reports, and specialized terminology
- Additional filters: medical term consistency checks, numerical/statistical consistency validation, metadata removal (e.g. "Page X", "Abstract")

---

## Setup

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (paper uses NVIDIA Tesla T4 16GB)

### 1. Install dependencies

```bash
pip install torch torchtext sacrebleu tokenizers langid
```

### 2. Train the tokenizer

```bash
python model/tokenizer.py --data data/general/ --vocab-size 40000
```

### 3. Pre-train on general data

```bash
python training/pretrain.py \
  --data data/general/ \
  --epochs 20 \
  --batch-size 64 \
  --lr 3e-4 \
  --warmup-steps 4000
```

Expected training time: ~11.5 hours on a T4 GPU.

### 4. Fine-tune on medical data

```bash
python training/finetune.py \
  --checkpoint checkpoints/pretrained_best.pt \
  --data data/medical/ \
  --epochs 4 \
  --batch-size 32 \
  --lr 5e-5 \
  --weight-decay 0.01 \
  --label-smoothing 0.1
```

Expected training time: ~1.5–2 hours on a T4 GPU.

### 5. Evaluate

```bash
python evaluate.py \
  --checkpoint checkpoints/finetuned_best.pt \
  --test-data data/medical/test.txt
```

### 6. Translate

```bash
python translate.py \
  --checkpoint checkpoints/finetuned_best.pt \
  --text "The patient has hypertension and diabetes mellitus type 2."
```

Expected output:
```
Bệnh nhân có tăng huyết áp và đái tháo đường type 2.
```


## Translation Examples

**Medical terminology:**
```
EN:   The patient has hypertension and diabetes mellitus type 2.
Ref:  Bệnh nhân bị tăng huyết áp và đái tháo đường type 2.
Pre:  Bệnh nhân có huyết áp cao và bệnh đái tháo đường loại 2.   ← colloquial term
Fine: Bệnh nhân có tăng huyết áp và đái tháo đường type 2.       ← correct clinical term ✓
```

**Numerical statistics:**
```
EN:   Mean age was 5.1 years (SD ± 2.3).
Ref:  Tuổi trung bình là 5,1 năm (SD ± 2,3).
Pre:  Tuổi trung bình là năm 5.1 (SD ± 2,3).   ← word order error ✗
Fine: Tuổi trung bình 5,1 năm (SD ± 2,3).      ← correct structure ✓
```

---

## Environment

| Component | Version |
|---|---|
| Framework | PyTorch 2.6.0 |
| GPU | NVIDIA Tesla T4 (16GB) |
| CUDA | 12.4 |
| Evaluation metric | SacreBLEU |

Training was conducted on Google Colab and Kaggle.

---

## Limitations

- Model size (38M parameters) is small compared to SOTA systems like mBART and mT5
- Evaluated on a single medical sub-domain; performance on other specialties is untested
- No human evaluation has been conducted
- Out-of-domain robustness has not been assessed

## Future Work

- Scale model to 100–200M parameters
- Multi-domain adaptation (legal, financial, etc.)
- Few-shot domain adaptation
- Constrained decoding with medical terminology dictionaries
- Detailed human evaluation and error analysis

---

```

---

## Acknowledgements

Compute resources provided by Google Colab and Kaggle.
Built with [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) and [PyTorch](https://pytorch.org/).
