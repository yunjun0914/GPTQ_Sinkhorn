# GPTQ-Sinkhorn

GPTQ error propagation + symmetric RTN 기반 LLM 양자화 라이브러리.  
**Sinkhorn normalization**과 **randomized Hadamard rotation**을 결합하여 양자화 품질을 향상시킨다.

## 핵심 아이디어

| 구성요소 | 설명 |
|---------|------|
| **Sinkhorn normalization** | 반복적 row/col 정규화(`comp_gh`)로 양자화 전 weight 분포를 균등화 |
| **Hadamard rotation** | 랜덤 block-Hadamard transform으로 weight를 회전 → outlier 에너지 분산 |
| **GPTQ** | Hessian 기반 column-wise 양자화. 이전 column의 오차를 이후에 전파해 보정 (Cholesky 기반) |
| **Symmetric RTN** | 단순 정수 양자화: `Q = round(W / scale).clamp(-maxq, maxq)` |

### 파이프라인 (linear layer당)

```
W, H = 2·X^T·X / N
  → comp_gh(W)            [--sinkhorn]          Sinkhorn → g, h
  → W /= g·h,  H *= h²
  → Hadamard(W, H)        [--hadamard_rotation] incoherence 처리
  → GPTQ + sym RTN                              column-wise 양자화 + 오차 전파
```

## 설치

```bash
git clone https://github.com/yunjun0914/GPTQ_Sinkhorn.git
cd GPTQ_Sinkhorn
pip install -e .
```

**요구사항:** Python ≥ 3.10, PyTorch ≥ 2.0, transformers, datasets

## 사용법

### 1. 양자화

```bash
python quantize.py <model_name_or_path> --output_dir <output_dir> [options]
```

**권장 설정 (Llama-2-7b):**

```bash
python quantize.py meta-llama/Llama-2-7b-hf \
    --output_dir ./quantized_llama2-7b \
    --bits 4 \
    --sinkhorn \
    --hadamard_rotation
```

**전체 옵션:**

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--bits` | `4` | 양자화 비트 수 (2/3/4/8) |
| `--group_size` | `-1` | 그룹 크기. `-1` = per-row scale (출력 채널당 1개 scale). `128` = per-row-group scale |
| `--sinkhorn` / `--no_sinkhorn` | on | Sinkhorn row/col 정규화 적용 여부 |
| `--hadamard_rotation` | off | Hadamard rotation 적용 여부 |
| `--rotation_first` | off | Hadamard를 Sinkhorn보다 먼저 적용 (had → gh). 기본값: Sinkhorn 먼저 (gh → had, 권장) |
| `--percdamp` | `0.01` | Hessian damping 비율 |
| `--blocksize` | `128` | GPTQ 블록 크기 |
| `--n_samples` | `128` | calibration 샘플 수 |
| `--seqlen` | `2048` | calibration 시퀀스 길이 |
| `--calib_dataset` | `c4` | calibration 데이터셋 (`c4` 또는 `wikitext2`) |
| `--device` | `cuda` | 실행 디바이스 |

### 2. 복원 (dequantize → fp16)

```bash
python reconstruct.py \
    --quantized_dir ./quantized_llama2-7b \
    --output_dir ./reconstructed_llama2-7b
```

`from_pretrained`으로 로드 가능한 표준 HuggingFace 모델 디렉토리를 생성한다.

### 3. PPL 평가

```bash
# 복원된 모델 평가
python eval_ppl.py --model_dir ./reconstructed_llama2-7b --dataset wikitext2 c4

# 원본 fp16 모델 평가 (baseline)
python eval_ppl.py --model_name meta-llama/Llama-2-7b-hf --dataset wikitext2 c4
```

## 출력 포맷

```
quantized_model/
├── quant_config.json          # QuantizationConfig
├── global_params.pt           # embeddings, final norm, lm_head
├── tokenizer/
└── layer_N/
    ├── <layer_name>.pt        # linear layer당 QuantizedLayerData
    └── non_quantized.pt       # layer norm 등 비양자화 파라미터
```

**QuantizedLayerData 필드:**

| 필드 | 타입 | Shape | 설명 |
|------|------|-------|------|
| `Q` | int8 | (out, in) | `[-maxq, maxq]` 범위의 양자화 인덱스 |
| `scales` | fp16 | (out,) or (out, n_groups) | 출력 채널별 또는 그룹별 scale |
| `g` | fp16 | (out,) | Sinkhorn row scale |
| `h` | fp16 | (in,) | Sinkhorn col scale |
| `had_d` | int8 | (in,) | Hadamard ±1 부호 벡터 |
| `had_K_col` | int | scalar | Hadamard 블록 수 |

## 지원 모델

- `meta-llama/Llama-2-*`, `meta-llama/Llama-3-*`
- `mistralai/Mistral-*`
- `Qwen/Qwen*`
- `facebook/opt-*`

## 프로젝트 구조

```
GPTQ_Sinkhorn/
├── gptq_sinkhorn/
│   ├── config.py              # QuantizationConfig, QuantizedLayerData
│   ├── algorithms/
│   │   └── gptq.py            # comp_gh (Sinkhorn), gptq_quantize, dequantize_layer
│   ├── models/
│   │   ├── llama.py           # LlamaHandler (Llama, Mistral, Qwen)
│   │   └── opt.py             # OPTHandler
│   └── utils/
│       ├── hadamard.py        # Block-Hadamard transform 유틸리티
│       ├── hessian.py         # forward hook 기반 Hessian 계산
│       └── calibration.py     # C4 / wikitext2 calibration 데이터
├── quantize.py                # 양자화 실행
├── reconstruct.py             # fp16으로 복원
└── eval_ppl.py                # PPL 평가 (wikitext2, c4)
```

## License

MIT
