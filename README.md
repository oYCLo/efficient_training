# efficient_training

In recent years, deep learning has advanced at an extremely rapid pace, and the number of parameters and complexity of its models have grown accordingly. As ordinary developers, we often hope to validate our ideas with relatively low overhead. Therefore, this repository provides some current methods that can help reduce the cost of validation, as well as explanations of some fundamental concepts.

## PyTorch Data Types Detailed Summary

Below is a comprehensive table summarizing the most common numerical data types used in PyTorch—including floating point, integer, quantized, and emerging 8‑bit floating point formats—with information on their aliases, bit-width, approximate representable range, and key notes.

> **Note:** The ranges provided are approximate values for reference only. The internal formats follow the IEEE 754 standard (for floating point types) or two's complement (for integers).

---

### 1. Floating‑Point Types

| Data Type                | Alias(s)                | Bit‑Width | Approximate Range                                      | Description                                                                                                                                  |
|--------------------------|-------------------------|-----------|--------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `torch.float64`          | `torch.double`          | 64        | ±[2.22507×10⁻³⁰⁸, 1.79769×10³⁰⁸]                         | Double‑precision floating point; ~15–17 decimal digits of precision; used in scenarios requiring very high numerical accuracy.              |
| `torch.float32`          | `torch.float`           | 32        | ±[1.17549×10⁻³⁸, 3.40282×10³⁸]                          | Single‑precision floating point; the default data type; provides about 6–7 decimal digits of precision.                                        |
| `torch.float16`          | `torch.half`            | 16        | ±[6.10×10⁻⁵, 65504]                                    | Half‑precision floating point; reduces memory usage and speeds up computation (especially in mixed‑precision training) but has a narrow range. |
| `torch.bfloat16`         | —                       | 16        | Approximately the same as FP32: ±[1.17549×10⁻³⁸, 3.40282×10³⁸] | Bfloat16 retains the 8‑bit exponent from FP32 but has only 7 bits for the significand; similar range as FP32 but with lower precision.        |
| `torch.float8_e4m3fn`     | —                       | 8         | Approximately ±[~0.0156, 240]                           | An 8‑bit float format (e4m3): 1 sign bit, 4 exponent bits, and 3 fraction bits; very limited precision.                                        |
| `torch.float8_e5m2`       | —                       | 8         | Approximately ±[~6.10×10⁻⁵, 57344]                      | Another 8‑bit float format (e5m2): 1 sign bit, 5 exponent bits, and 2 fraction bits; offers a wider range than e4m3 but with even less precision. |

---

## 2. Integer Types

| Data Type          | Alias(s)          | Bit‑Width | Approximate Range                  | Description                                                            |
|--------------------|-------------------|-----------|------------------------------------|------------------------------------------------------------------------|
| `torch.int64`      | `torch.long`      | 64        | [−9.22×10¹⁸, 9.22×10¹⁸]             | 64‑bit signed integer; commonly used for indexing in deep learning.    |
| `torch.int32`      | `torch.int`       | 32        | [−2.147×10⁹, 2.147×10⁹]             | 32‑bit signed integer.                                                 |
| `torch.int16`      | `torch.short`     | 16        | [−32768, 32767]                    | 16‑bit signed integer.                                                 |
| `torch.int8`       | —                 | 8         | [−128, 127]                        | 8‑bit signed integer.                                                  |
| `torch.uint8`      | —                 | 8         | [0, 255]                           | 8‑bit unsigned integer.                                                |
| `torch.bool`       | —                 | 1 (logical) | {False, True}                    | Boolean type; represents only True or False.                           |

---

### 3. Quantized Types (Used for Model Quantization)

| Data Type            | Alias  | Bit‑Width | Approximate Range              | Description                                                                                                  |
|----------------------|--------|-----------|--------------------------------|--------------------------------------------------------------------------------------------------------------|
| `torch.qint8`        | —      | 8         | [−128, 127]                    | Quantized signed 8‑bit integer, often used for quantizing weights or activations.                           |
| `torch.quint8`       | —      | 8         | [0, 255]                       | Quantized unsigned 8‑bit integer.                                                                            |
| `torch.qint32`       | —      | 32        | [−2.147×10⁹, 2.147×10⁹]          | Quantized signed 32‑bit integer.                                                                             |
| `torch.quint4x2`     | —      | 8 (2×4‑bit per byte) | Each 4‑bit unit: [0, 15]    | Encodes two 4‑bit unsigned integers per byte; used in specialized quantized operations (e.g., EmbeddingBag).    |

---

### 4. Additional Notes

- **TF32 (TensorFloat‑32):**  
  TF32 is not a separate storage data type. It is an operational mode available on NVIDIA’s Ampere GPUs that uses the FP32 8‑bit exponent and a reduced 10‑bit significand to speed up matrix operations while maintaining the dynamic range of FP32.

- **Mixed Precision Training:**  
  To balance computational efficiency and training stability, mixed precision training typically uses a combination of lower precision (FP16 or BF16) for most computations and FP32 for accumulating weight updates. Techniques such as weight backup and dynamic loss scaling are employed to avoid numerical underflow or rounding errors.

- **Range vs. Precision:**  
  In floating‑point numbers, the exponent bits determine the representable range, while the significand bits determine the precision. For example, FP16 (with about 10‑bit significand) provides roughly 3 decimal digits of precision, whereas bfloat16, with only 7‑bit significand but an exponent matching FP32, has lower precision but retains the wide range of FP32.

# Structure of repo
----
```
efficient_torch/
│
├── core/
│   ├── __init__.py
│   ├── trainer.py          # 训练器核心实现
│   ├── inference.py        # 推理核心实现
│   └── optimization.py     # 优化器相关实现
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py         # 评估指标
│   ├── logger.py          # 日志工具
│   └── profiler.py        # 性能分析工具
│
├── accelerate/
│   ├── __init__.py
│   ├── mixed_precision.py  # 混合精度训练
│   ├── distributed.py      # 分布式训练
│   └── parallel.py        # 模型并行化
│
├── memory/
│   ├── __init__.py
│   ├── gradient_checkpointing.py  # 梯度检查点
│   └── memory_efficient.py        # 内存优化
│
├── examples/
│   ├── train_example.py
│   └── inference_example.py
│
├── tests/
│   └── test_*.py
│
├── setup.py
└── README.md
```