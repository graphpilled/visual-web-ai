// ============================================
// Shape Inference Module
// ============================================

open Types

// ----------------------------------------
// Helper Functions
// ----------------------------------------

let len = Array.length
let at = (arr, i) => arr[i]->Option.getOr(0)
let numElements = shape => Array.reduce(shape, 1, (a, b) => a * b)
let normalizeAxis = (axis, rank) => axis < 0 ? rank + axis : axis

let broadcast = (s1: shape, s2: shape): option<shape> => {
  let r1 = len(s1)
  let r2 = len(s2)
  let maxRank = max(r1, r2)

  let getDim = (s, r, i) => {
    let offset = maxRank - r
    i < offset ? 1 : at(s, i - offset)
  }

  let result = Array.fromInitializer(~length=maxRank, i => {
    let d1 = getDim(s1, r1, i)
    let d2 = getDim(s2, r2, i)
    if d1 == d2 { Some(d1) }
    else if d1 == 1 { Some(d2) }
    else if d2 == 1 { Some(d1) }
    else { None }
  })

  Array.every(result, Option.isSome)
    ? Some(Array.map(result, Option.getOr(_, 0)))
    : None
}

// ----------------------------------------
// Inference Helpers
// ----------------------------------------

let inferReduce = (input: shape, axes: array<int>, keepDims: bool) => {
  let rank = len(input)
  let normAxes = Array.map(axes, a => normalizeAxis(a, rank))
  keepDims
    ? Array.mapWithIndex(input, (d, i) => Array.includes(normAxes, i) ? 1 : d)
    : Array.filterWithIndex(input, (_, i) => !Array.includes(normAxes, i))
}

let inferMatMul = (s1: shape, s2: shape): option<shape> => {
  let r1 = len(s1)
  let r2 = len(s2)

  switch (r1, r2) {
  | (0, _) | (_, 0) => None
  | (1, 1) => at(s1, 0) == at(s2, 0) ? Some([1]) : None
  | (1, _) => {
      let k = at(s1, 0)
      let k2 = at(s2, r2 - 2)
      let n = at(s2, r2 - 1)
      k == k2 ? Some(Array.concat(Array.slice(s2, ~start=0, ~end=r2-2), [n])) : None
    }
  | (_, 1) => {
      let k = at(s1, r1 - 1)
      k == at(s2, 0) ? Some(Array.slice(s1, ~start=0, ~end=r1-1)) : None
    }
  | _ => {
      let m = at(s1, r1 - 2)
      let k1 = at(s1, r1 - 1)
      let k2 = at(s2, r2 - 2)
      let n = at(s2, r2 - 1)
      if k1 != k2 { None }
      else {
        let b1 = Array.slice(s1, ~start=0, ~end=r1-2)
        let b2 = Array.slice(s2, ~start=0, ~end=r2-2)
        broadcast(b1, b2)->Option.map(batch => Array.concat(batch, [m, n]))
      }
    }
  }
}

let inferConv2D = (input, filters, (kH, kW), (sH, sW), padding, (dH, dW)) => {
  if len(input) != 4 { None }
  else {
    let batch = at(input, 0)
    let inH = at(input, 1)
    let inW = at(input, 2)
    let effKH = (kH - 1) * dH + 1
    let effKW = (kW - 1) * dW + 1
    let (outH, outW) = switch padding {
    | Same => ((inH + sH - 1) / sH, (inW + sW - 1) / sW)
    | Valid => ((inH - effKH) / sH + 1, (inW - effKW) / sW + 1)
    | Explicit({pads}) => {
        let p = i => pads[i]->Option.getOr(0)
        ((inH + p(0) + p(2) - effKH) / sH + 1, (inW + p(1) + p(3) - effKW) / sW + 1)
      }
    }
    Some([batch, outH, outW, filters])
  }
}

let inferPool2D = (input, (kH, kW), (sH, sW), padding) => {
  if len(input) != 4 { None }
  else {
    let batch = at(input, 0)
    let inH = at(input, 1)
    let inW = at(input, 2)
    let channels = at(input, 3)
    let (outH, outW) = switch padding {
    | Same => ((inH + sH - 1) / sH, (inW + sW - 1) / sW)
    | Valid => ((inH - kH) / sH + 1, (inW - kW) / sW + 1)
    | Explicit({pads}) => {
        let p = i => pads[i]->Option.getOr(0)
        ((inH + p(0) + p(2) - kH) / sH + 1, (inW + p(1) + p(3) - kW) / sW + 1)
      }
    }
    Some([batch, outH, outW, channels])
  }
}

let inferReshape = (input: shape, newShape: shape) => {
  let total = numElements(input)
  let neg1Count = Array.reduce(newShape, 0, (acc, d) => d == -1 ? acc + 1 : acc)
  switch neg1Count {
  | 0 => numElements(newShape) == total ? Some(newShape) : None
  | 1 => {
      let known = Array.reduce(newShape, 1, (acc, d) => d == -1 ? acc : acc * d)
      let inferred = total / known
      inferred * known == total
        ? Some(Array.map(newShape, d => d == -1 ? inferred : d))
        : None
    }
  | _ => None
  }
}

let inferConcat = (inputs: array<shape>, axis) => {
  if len(inputs) == 0 { None }
  else {
    let first = inputs[0]->Option.getOr([])
    let rank = len(first)
    let normAxis = normalizeAxis(axis, rank)
    let concatDim = Array.reduce(inputs, 0, (acc, s) => acc + at(s, normAxis))
    Some(Array.mapWithIndex(first, (d, i) => i == normAxis ? concatDim : d))
  }
}

// ----------------------------------------
// Main Inference Function
// ----------------------------------------

let infer = (op: op, inputs: array<shape>): option<shape> => {
  let get = i => inputs[i]->Option.getOr([])
  let input = get(0)
  let r = len(input)

  switch op {
  // Constants
  | Input({shape}) | Const({shape}) => Some(shape)
  | Identity => Some(input)

  // Unary ops (same shape)
  | Neg | Abs | Sign | Reciprocal | Floor | Ceil | Round
  | Sqrt | Exp | Log | Log2 | Log10
  | Sin | Cos | Tan | Asin | Acos | Atan
  | Sinh | Cosh | Tanh | Asinh | Acosh | Atanh
  | ReLU | LeakyReLU(_) | PReLU | ELU(_) | SELU
  | Sigmoid | HardSigmoid(_) | Softplus | Softsign
  | GeLU | SiLU | Mish | Erf | Not | IsNaN | IsInf(_)
  | Clip(_) | Cast(_) | Dropout(_) | AlphaDropout(_)
  | BatchNorm(_) | LayerNorm(_) | InstanceNorm(_)
  | GroupNorm(_) | LRN(_) | RMSNorm(_)
  | Softmax(_) | LogSoftmax(_) | Hardmax(_)
  | QuantizeLinear(_) | DequantizeLinear(_)
  | Sort(_) | Reverse(_) | Bernoulli(_) =>
    Some(input)

  // Binary ops (broadcast)
  | Add | Sub | Mul | Div | Pow | Mod | FloorDiv
  | Maximum | Minimum | Atan2
  | Equal | NotEqual | Greater | GreaterEqual | Less | LessEqual
  | And | Or | Xor =>
    broadcast(input, get(1))

  // Ternary
  | Where => broadcast(input, get(1))->Option.flatMap(s => broadcast(s, get(2)))

  // Reductions
  | Reduce({axes, keepDims}) => Some(inferReduce(input, axes, keepDims))
  | ArgMax({axis, keepDims}) | ArgMin({axis, keepDims}) =>
    Some(inferReduce(input, [axis], keepDims))
  | CumSum(_) => Some(input)
  | CumProd(_) => Some(input)

  // Linear algebra
  | MatMul | BatchedMatMul | Gemm(_) => inferMatMul(input, get(1))
  | Dot => Some([1])
  | Einsum(_) => None
  | MatMulInt4(_) => {
      // input: [M, K], packed weights: [N, K/8], scales: [N, numGroups]
      // output: [M, N]
      let s2 = get(1)
      let m = at(input, r - 2)
      let n = at(s2, 0)
      Some([m, n])
    }

  // Shape ops
  | Reshape({newShape}) => inferReshape(input, newShape)
  | Transpose({perm}) =>
    len(perm) == r ? Some(Array.map(perm, p => at(input, p))) : None
  | Flatten({axis}) => {
      let a = normalizeAxis(axis, r)
      Some([
        numElements(Array.slice(input, ~start=0, ~end=a)),
        numElements(Array.slice(input, ~start=a, ~end=r))
      ])
    }
  | Squeeze({axes}) =>
    Some(Array.filterWithIndex(input, (d, i) => !(Array.includes(axes, i) && d == 1)))
  | Unsqueeze({axes}) => {
      let newRank = r + len(axes)
      let sortedAxes = Array.toSorted(axes, Int.compare)
      let result = ref([])
      let inputIdx = ref(0)
      for i in 0 to newRank - 1 {
        if Array.includes(sortedAxes, i) {
          result := Array.concat(result.contents, [1])
        } else {
          result := Array.concat(result.contents, [at(input, inputIdx.contents)])
          inputIdx := inputIdx.contents + 1
        }
      }
      Some(result.contents)
    }
  | Broadcast({targetShape}) => Some(targetShape)
  | ExpandDims({axis}) => {
      let normAxis = normalizeAxis(axis, r + 1)
      let before = Array.slice(input, ~start=0, ~end=normAxis)
      let after = Array.slice(input, ~start=normAxis, ~end=r)
      Some(Array.concat(Array.concat(before, [1]), after))
    }
  | Concat({axis}) => inferConcat(inputs, axis)
  | Stack({axis}) => {
      let numInputs = len(inputs)
      let normAxis = normalizeAxis(axis, r + 1)
      let before = Array.slice(input, ~start=0, ~end=normAxis)
      let after = Array.slice(input, ~start=normAxis, ~end=r)
      Some(Array.concat(Array.concat(before, [numInputs]), after))
    }
  | Split({axis, splitSizes}) => {
      let normAxis = normalizeAxis(axis, r)
      let firstSize = splitSizes[0]->Option.getOr(at(input, normAxis))
      Some(Array.mapWithIndex(input, (d, i) => i == normAxis ? firstSize : d))
    }

  // Indexing
  | Slice({starts, ends, axes, steps}) => {
      let result = Array.copy(input)
      Array.forEachWithIndex(axes, (ax, i) => {
        let dimSize = at(input, ax)
        let start = starts[i]->Option.getOr(0)
        let end_ = ends[i]->Option.getOr(dimSize)
        let step = steps[i]->Option.getOr(1)
        let s = start < 0 ? dimSize + start : start
        let e = end_ < 0 ? dimSize + end_ : end_
        let size = (e - s + step - 1) / step
        ignore(result->Array.set(ax, size))
      })
      Some(result)
    }
  | Gather({axis}) | GatherElements({axis}) => {
      let indices = get(1)
      let normAxis = normalizeAxis(axis, r)
      let before = Array.slice(input, ~start=0, ~end=normAxis)
      let after = Array.slice(input, ~start=normAxis + 1, ~end=r)
      Some(Array.concat(Array.concat(before, indices), after))
    }
  | GatherND(_) => None // complex shape logic
  | Scatter(_) | ScatterElements(_) | ScatterND => Some(input)
  | Pad({pads}) => {
      Some(Array.fromInitializer(~length=r, i => {
        let before = pads[i]->Option.getOr(0)
        let after = pads[r + i]->Option.getOr(0)
        at(input, i) + before + after
      }))
    }
  | Tile({repeats}) =>
    Some(Array.mapWithIndex(input, (d, i) => d * (repeats[i]->Option.getOr(1))))

  // Conv layers
  | Conv1D({filters, kernel, stride, padding, dilation}) => {
      if r != 3 { None }
      else {
        let batch = at(input, 0)
        let inLen = at(input, 1)
        let effK = (kernel - 1) * dilation + 1
        let outLen = switch padding {
        | Same => (inLen + stride - 1) / stride
        | Valid => (inLen - effK) / stride + 1
        | Explicit({pads}) => (inLen + at(pads, 0) + at(pads, 1) - effK) / stride + 1
        }
        Some([batch, outLen, filters])
      }
    }
  | Conv2D({filters, kernel, stride, padding, dilation}) =>
    inferConv2D(input, filters, kernel, stride, padding, dilation)
  | Conv3D({filters, kernel: (kD, kH, kW), stride: (sD, sH, sW), padding, dilation: (dD, dH, dW)}) => {
      if r != 5 { None }
      else {
        let batch = at(input, 0)
        let calc = (inSize, k, s, d) => {
          let effK = (k - 1) * d + 1
          switch padding {
          | Same => (inSize + s - 1) / s
          | Valid => (inSize - effK) / s + 1
          | Explicit(_) => (inSize - effK) / s + 1
          }
        }
        Some([batch, calc(at(input, 1), kD, sD, dD), calc(at(input, 2), kH, sH, dH), calc(at(input, 3), kW, sW, dW), filters])
      }
    }
  | DepthwiseConv2D({kernel: (kH, kW), stride: (sH, sW), padding, dilation: (dH, dW), depthMultiplier}) => {
      if r != 4 { None }
      else {
        let batch = at(input, 0)
        let inH = at(input, 1)
        let inW = at(input, 2)
        let channels = at(input, 3)
        let effKH = (kH - 1) * dH + 1
        let effKW = (kW - 1) * dW + 1
        let (outH, outW) = switch padding {
        | Same => ((inH + sH - 1) / sH, (inW + sW - 1) / sW)
        | Valid => ((inH - effKH) / sH + 1, (inW - effKW) / sW + 1)
        | Explicit(_) => ((inH - effKH) / sH + 1, (inW - effKW) / sW + 1)
        }
        Some([batch, outH, outW, channels * depthMultiplier])
      }
    }
  | ConvTranspose1D({filters, kernel, stride, padding, outputPadding}) => {
      if r != 3 { None }
      else {
        let batch = at(input, 0)
        let inLen = at(input, 1)
        let outLen = switch padding {
        | Same => inLen * stride
        | Valid | Explicit(_) => inLen * stride + max(kernel - stride, 0)
        } + outputPadding
        Some([batch, outLen, filters])
      }
    }
  | ConvTranspose2D({filters, kernel: (kH, kW), stride: (sH, sW), padding, outputPadding: (opH, opW)}) => {
      if r != 4 { None }
      else {
        let batch = at(input, 0)
        let inH = at(input, 1)
        let inW = at(input, 2)
        let (outH, outW) = switch padding {
        | Same => (inH * sH + opH, inW * sW + opW)
        | Valid | Explicit(_) => (inH * sH + max(kH - sH, 0) + opH, inW * sW + max(kW - sW, 0) + opW)
        }
        Some([batch, outH, outW, filters])
      }
    }
  | ConvTranspose3D({filters, kernel: (kD, kH, kW), stride: (sD, sH, sW), padding, outputPadding: (opD, opH, opW)}) => {
      if r != 5 { None }
      else {
        let batch = at(input, 0)
        let calc = (inSize, k, s, op) => switch padding {
        | Same => inSize * s + op
        | Valid | Explicit(_) => inSize * s + max(k - s, 0) + op
        }
        Some([batch, calc(at(input, 1), kD, sD, opD), calc(at(input, 2), kH, sH, opH), calc(at(input, 3), kW, sW, opW), filters])
      }
    }

  // Pooling
  | MaxPool1D({kernel, stride, padding}) => {
      if r != 3 { None }
      else {
        let batch = at(input, 0)
        let inLen = at(input, 1)
        let channels = at(input, 2)
        let outLen = switch padding {
        | Same => (inLen + stride - 1) / stride
        | Valid | Explicit(_) => (inLen - kernel) / stride + 1
        }
        Some([batch, outLen, channels])
      }
    }
  | AvgPool1D({kernel, stride, padding}) => {
      if r != 3 { None }
      else {
        let batch = at(input, 0)
        let inLen = at(input, 1)
        let channels = at(input, 2)
        let outLen = switch padding {
        | Same => (inLen + stride - 1) / stride
        | Valid | Explicit(_) => (inLen - kernel) / stride + 1
        }
        Some([batch, outLen, channels])
      }
    }
  | MaxPool2D({kernel, stride, padding}) | AvgPool2D({kernel, stride, padding}) =>
    inferPool2D(input, kernel, stride, padding)
  | MaxPool3D({kernel: (kD, kH, kW), stride: (sD, sH, sW), padding})
  | AvgPool3D({kernel: (kD, kH, kW), stride: (sD, sH, sW), padding}) => {
      if r != 5 { None }
      else {
        let batch = at(input, 0)
        let channels = at(input, 4)
        let calc = (inSize, k, s) => switch padding {
        | Same => (inSize + s - 1) / s
        | Valid | Explicit(_) => (inSize - k) / s + 1
        }
        Some([batch, calc(at(input, 1), kD, sD), calc(at(input, 2), kH, sH), calc(at(input, 3), kW, sW), channels])
      }
    }
  | GlobalMaxPool | GlobalAvgPool =>
    r == 4 ? Some([at(input, 0), 1, 1, at(input, 3)]) : None
  | LpPool({kernel: (kH, kW), stride: (sH, sW)}) =>
    inferPool2D(input, (kH, kW), (sH, sW), Valid)
  | AdaptiveMaxPool1D({outputSize}) | AdaptiveAvgPool1D({outputSize}) =>
    r == 3 ? Some([at(input, 0), outputSize, at(input, 2)]) : None
  | AdaptiveMaxPool2D({outputSize: (h, w)}) | AdaptiveAvgPool2D({outputSize: (h, w)}) =>
    r == 4 ? Some([at(input, 0), h, w, at(input, 3)]) : None

  // Dense
  | Dense({units}) => {
      r > 0 ? Some(Array.concat(Array.slice(input, ~start=0, ~end=r-1), [units])) : None
    }

  // RNN - output is [batch, seq, hidden] for sequence or [batch, hidden] for final
  | RNN({hiddenSize}) | LSTM({hiddenSize}) | GRU({hiddenSize}) =>
    r >= 2 ? Some([at(input, 0), at(input, 1), hiddenSize]) : None

  // Attention
  | Attention({dim}) | MultiHeadAttention({dim}) =>
    r >= 2 ? Some([at(input, 0), at(input, 1), dim]) : None
  | ScaledDotProductAttention(_) => Some(input)

  // Embedding/OneHot
  | Embedding({embeddingDim}) => Some(Array.concat(input, [embeddingDim]))
  | OneHot({depth, axis}) => {
      let normAxis = normalizeAxis(axis, r + 1)
      let before = Array.slice(input, ~start=0, ~end=normAxis)
      let after = Array.slice(input, ~start=normAxis, ~end=r)
      Some(Array.concat(Array.concat(before, [depth]), after))
    }

  // TopK
  | TopK({k, axis}) => {
      let normAxis = normalizeAxis(axis, r)
      Some(Array.mapWithIndex(input, (d, i) => i == normAxis ? k : d))
    }

  // Shape queries
  | Shape => Some([r])
  | Size => Some([1])
  | Rank => Some([1])

  // Image ops
  | Resize({sizes}) => {
      switch sizes {
      | Some(s) => r == 4 ? Some([at(input, 0), at(s, 0), at(s, 1), at(input, 3)]) : None
      | None => Some(input) // scales case, would need runtime info
      }
    }
  | SpaceToDepth({blockSize}) => {
      let b = blockSize
      r == 4 ? Some([at(input, 0), at(input, 1) / b, at(input, 2) / b, at(input, 3) * b * b]) : None
    }
  | DepthToSpace({blockSize}) => {
      let b = blockSize
      r == 4 ? Some([at(input, 0), at(input, 1) * b, at(input, 2) * b, at(input, 3) / (b * b)]) : None
    }
  | GridSample(_) => {
      let grid = get(1)
      r == 4 && len(grid) == 4 ? Some([at(input, 0), at(grid, 1), at(grid, 2), at(input, 3)]) : None
    }
  | RoiAlign({outputHeight, outputWidth}) => {
      let rois = get(1)
      r == 4 && len(rois) == 2 ? Some([at(rois, 0), outputHeight, outputWidth, at(input, 1)]) : None
    }
  | NonMaxSuppression(_) => None

  // Loss functions -> scalar
  | MSELoss(_) | CrossEntropyLoss(_) | BCELoss(_) | BCEWithLogitsLoss(_)
  | NLLLoss(_) | CTCLoss(_) | HuberLoss(_) | SmoothL1Loss(_)
  | TripletMarginLoss(_) | CosineEmbeddingLoss(_) => Some([1])

  // Random ops - shape from params
  | RandomNormal({shape}) | RandomUniform({shape}) => Some(shape)

  // Dynamic output size
  | Unique(_) | NonZero => None
  }
}
