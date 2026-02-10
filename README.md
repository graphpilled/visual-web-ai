# visual-web-ai

A neural network compiler and runtime that runs entirely in the browser. ReScript generates optimized WGSL compute shaders. WebGPU executes them on your GPU. Autograd handles backpropagation. No server required.

**[Live Demo →](https://graphpilled.github.io/visual-web-ai/demo.html)**&ensp;·&ensp;**[Project Page →](https://graphpilled.github.io/visual-web-ai/)**

---

## How it works

```
Types.res → Shape.res → Codegen.res → Compiler.res → runtime.js
 (ops)     (inference)    (WGSL)       (graph)        (WebGPU)
```

1. **Types.res** — 208 operations defined as algebraic types (unary, binary, reductions, convolutions, attention, recurrent, normalization, and more)
2. **Shape.res** — Shape inference with broadcasting, reduction, and convolution rules
3. **Codegen.res** — 3500+ lines that emit optimized WGSL compute shaders for every operation
4. **Compiler.res** — Graph compilation, topological sorting, buffer allocation, dispatch scheduling
5. **runtime.js** — WebGPU execution engine with pipeline caching, buffer management, and multi-output support
6. **nn.js** — PyTorch-style high-level API built on top of everything

## Usage

```js
import { Tensor, nn, optim, init } from './nn.js';

await init();  // Initialize WebGPU

// Define model
const model = new nn.Sequential(
  new nn.Linear(2, 16),
  new nn.Tanh(),
  new nn.Linear(16, 1),
  new nn.Sigmoid()
);

const optimizer = new optim.Adam(model.parameters(), 0.01);

// Training loop
for (let i = 0; i < 100; i++) {
  const y = await model.forward(x);
  const loss = await nn.mseLoss(y, target);
  await loss.backward();        // Autograd on GPU
  await optimizer.step();       // Adam update via WGSL
}
```

## Operations

**Element-wise (30+)** — Neg, Abs, ReLU, GeLU, SiLU, Sigmoid, Tanh, Mish, Softplus, Exp, Log, Sqrt, Sin, Cos, Erf, Add, Mul, Div, Pow, Min, Max, Equal, Greater, And, Or, Where, Clamp, …

**Linear algebra** — MatMul (tiled), BatchedMatMul, INT4 quantized MatMul (GPTQ, configurable group size, ~103% memory bandwidth efficiency), Gemm

**Reductions** — Sum, Mean, Max, Min, Prod, L1, L2, LogSum, LogSumExp, SumSquare — arbitrary axes, keepDims

**Normalization** — LayerNorm, BatchNorm, RMSNorm, InstanceNorm, GroupNorm, LRN

**Convolution & Pooling** — Conv1D/2D/3D, transposed convolutions, depthwise separable, MaxPool, AvgPool, GlobalAvgPool, AdaptiveAvgPool

**Attention & Sequence** — ScaledDotProductAttention (causal masking), LSTM, GRU, RNN cells

**Shape** — Reshape, Flatten, Transpose, Permute, Squeeze, Unsqueeze, Concat, Split, Gather, Scatter, Pad, Slice, Tile, Expand

**Autograd** — Full automatic differentiation via GradTape. Backward kernel generation for every differentiable op. Gradient accumulation, optimizer updates (Adam, AdamW, SGD) — all executed as WGSL shaders on GPU.

## nn.js API

| Layer | Description |
|---|---|
| `nn.Linear(in, out)` | Fully connected |
| `nn.Conv2D(in_ch, out_ch, kernel)` | 2D convolution |
| `nn.MaxPool2D(k, s)` / `nn.AvgPool2D(k, s)` | Pooling |
| `nn.LSTM(input, hidden)` / `nn.GRU(input, hidden)` | Recurrent |
| `nn.Embedding(vocab, dim)` | Lookup table |
| `nn.LayerNorm(shape)` / `nn.BatchNorm1D(n)` / `nn.BatchNorm2D(n)` | Normalization |
| `nn.ReLU` / `nn.GeLU` / `nn.Sigmoid` / `nn.Tanh` / `nn.Softmax` | Activations |
| `nn.Dropout(rate)` | Regularization |
| `nn.Sequential(...)` | Container |
| `nn.Flatten()` | Reshape |

| Optimizer | Description |
|---|---|
| `optim.SGD(params, lr)` | Stochastic gradient descent |
| `optim.Adam(params, lr)` | Adam with weight decay support |

## INT4 Quantized Inference

The compiler supports GPTQ-quantized models with INT4 weights. The matmul kernel is fully unrolled with column-major weight layout for coalesced memory access, vec4 dot products, and per-group scale application:

```
Input activations (F32) × Packed weights (INT4, column-major) → Output (F32)
                           ↑ scales per group (128 weights)
```

Tested with Qwen 2.5 (0.5B and 7B parameter variants).

## Project structure

```
src/
├── Types.res              # 208 op types as algebraic variants (417 lines)
├── Shape.res              # Shape inference with broadcasting & convolution rules (500 lines)
├── Codegen.res            # WGSL compute shader generation (3544 lines)
├── Compiler.res           # Graph compiler, buffer allocation, scheduling (1305 lines)
├── Autograd.res           # Backward pass kernel generation (1254 lines)
├── AutogradEngine.res     # Forward/backward execution engine (737 lines)
├── GradTape.res           # Gradient recording tape (330 lines)
├── Main.res               # Entry point
├── runtime.js             # WebGPU runtime — pipeline caching, buffer mgmt (420 lines)
├── nn.js                  # PyTorch-style high-level API (2000 lines)
├── bridge.js              # ReScript ↔ JS interop
└── trainer.js             # Training loop utilities

dist/
└── bundle.js              # esbuild output

test/                      # WebGPU test pages for individual ops and models
index.html                 # Project landing page
demo.html                  # Interactive 3D block builder
```

## Requirements

- A browser with WebGPU support (Chrome 113+, Edge 113+, Firefox Nightly)
- For development: ReScript compiler, esbuild

## Building

```bash
npm install
npx rescript build
npx esbuild src/Main.res.mjs --bundle --outfile=dist/bundle.js --format=esm
```

## License

MIT
