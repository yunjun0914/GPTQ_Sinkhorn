# LLM Quantization Ablation Results

**Calibration**: 128 samples, seqlen 2048 (c4)  
**Date**: 2026-06-12  

---

## 1. LLaMA-2-7B · 4bit · per-row scale (g=-1)

| # | Condition | Sinkhorn | Hadamard | WikiText2 ↓ | C4 ↓ |
|---|-----------|:--------:|:--------:|:-----------:|:----:|
| 0 | FP16 | ✗ | ✗ | 5.47 | 6.97 |
| 1 | Baseline (GPTQ only) | ✗ | ✗ | 6.33 | 8.07 |
| 2 | Sinkhorn only | ✓ | ✗ | 5.94 | 7.80 |
| 3 | Hadamard only | ✗ | ✓ | 5.85 | 7.73 |
| 4 | had → gh | ✓ | ✓ | 5.84 | 7.72 |
| 5 | **gh → had** | ✓ | ✓ | **5.78** | **7.63** |

---

## 2. LLaMA-2-7B · 4bit · group_size=128

| # | Condition | Sinkhorn | Hadamard | WikiText2 ↓ | C4 ↓ |
|---|-----------|:--------:|:--------:|:-----------:|:----:|
| 0 | FP16 | ✗ | ✗ | 5.47 | 6.97 |
| 1 | Baseline (GPTQ only) | ✗ | ✗ | 5.86 | 7.53 |
| 2 | Sinkhorn only | ✓ | ✗ | 5.63 | 7.42 |
| 3 | Hadamard only | ✗ | ✓ | 5.68 | 7.45 |
| 4 | had → gh | ✓ | ✓ | 5.66 | 7.47 |
| 5 | **gh → had** | ✓ | ✓ | **5.65** | **7.42** |

---

## 3. LLaMA-3-8B · 4bit · per-row scale (g=-1)

| # | Condition | Sinkhorn | Hadamard | WikiText2 ↓ | C4 ↓ |
|---|-----------|:--------:|:--------:|:-----------:|:----:|
| 0 | FP16 | ✗ | ✗ | - | - |
| 1 | Baseline (GPTQ only) | ✗ | ✗ | 1119.88 ❌ | 188.82 ❌ |
| 2 | Sinkhorn only | ✓ | ✗ | 7.62 | 11.69 |
| 3 | Hadamard only | ✗ | ✓ | 7.10 | 11.11 |
| 4 | had → gh | ✓ | ✓ | 7.12 | 11.10 |
| 5 | **gh → had** | ✓ | ✓ | **6.96** | **10.84** |

---

## 4. LLaMA-3-8B · 4bit · group_size=128

| # | Condition | Sinkhorn | Hadamard | WikiText2 ↓ | C4 ↓ |
|---|-----------|:--------:|:--------:|:-----------:|:----:|
| 0 | FP16 | ✗ | ✗ | - | - |
| 1 | Baseline (GPTQ only) | ✗ | ✗ | 61.44 ❌ | 29.13 ❌ |
| 2 | Sinkhorn only | ✓ | ✗ | 6.63 | 10.26 |
| 3 | Hadamard only | ✗ | ✓ | 6.63 | 10.26 |
| 4 | had → gh | ✓ | ✓ | 6.62 | 10.25 |
| 5 | **gh → had** | ✓ | ✓ | **6.57** | **10.15** |
