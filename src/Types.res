// ============================================
// Visual Web AI Compiler - Core Type Definitions
// ============================================

// Tensor shape is an array of dimensions
type shape = array<int>

// Supported data types
type dtype =
  | F32
  | F16
  | I32
  | U32
  | I8
  | U8
  | Bool

// Unique identifier for tensors in the graph
type tensorId = TensorId(int)

// Reduction operations
type reduceOp =
  | Sum
  | Mean
  | Max
  | Min
  | Prod
  | L1
  | L2
  | LogSum
  | LogSumExp
  | SumSquare

// Padding modes
type padding =
  | Same
  | Valid
  | Explicit({pads: array<int>})

// Pad modes for Pad operation
type padMode =
  | Constant
  | Reflect
  | Edge

// RNN direction
type rnnDirection =
  | Forward
  | Reverse
  | Bidirectional

// Interpolation mode for resize
type interpolationMode =
  | Nearest
  | Linear
  | Cubic

// ============================================
// All Supported Operations
// ============================================
type rec op =
  // ----------------------------------------
  // Input/Output/Constants
  // ----------------------------------------
  | Input({shape: shape, dtype: dtype})
  | Const({shape: shape, dtype: dtype})
  | Identity

  // ----------------------------------------
  // Element-wise Unary - Arithmetic
  // ----------------------------------------
  | Neg
  | Abs
  | Sign
  | Reciprocal
  | Floor
  | Ceil
  | Round
  | Sqrt
  | Exp
  | Log
  | Log2
  | Log10

  // ----------------------------------------
  // Element-wise Unary - Trigonometric
  // ----------------------------------------
  | Sin
  | Cos
  | Tan
  | Asin
  | Acos
  | Atan
  | Sinh
  | Cosh
  | Tanh
  | Asinh
  | Acosh
  | Atanh

  // ----------------------------------------
  // Element-wise Unary - Activations
  // ----------------------------------------
  | ReLU
  | LeakyReLU({alpha: float})
  | PReLU
  | ELU({alpha: float})
  | SELU
  | Sigmoid
  | HardSigmoid({alpha: float, beta: float})
  | Softplus
  | Softsign
  | GeLU
  | SiLU
  | Mish
  | Erf

  // ----------------------------------------
  // Element-wise Binary - Arithmetic
  // ----------------------------------------
  | Add
  | Sub
  | Mul
  | Div
  | Pow
  | Mod
  | FloorDiv
  | Maximum
  | Minimum
  | Atan2

  // ----------------------------------------
  // Element-wise Binary - Comparison
  // ----------------------------------------
  | Equal
  | NotEqual
  | Greater
  | GreaterEqual
  | Less
  | LessEqual

  // ----------------------------------------
  // Element-wise Binary/Unary - Logical
  // ----------------------------------------
  | And
  | Or
  | Xor
  | Not

  // ----------------------------------------
  // Element-wise Ternary
  // ----------------------------------------
  | Where
  | Clip({min: option<float>, max: option<float>})

  // ----------------------------------------
  // Reduction Operations
  // ----------------------------------------
  | Reduce({op: reduceOp, axes: array<int>, keepDims: bool})
  | ArgMax({axis: int, keepDims: bool, selectLastIndex: bool})
  | ArgMin({axis: int, keepDims: bool, selectLastIndex: bool})
  | CumSum({axis: int, exclusive: bool, reverse: bool})
  | CumProd({axis: int, exclusive: bool, reverse: bool})

  // ----------------------------------------
  // Linear Algebra
  // ----------------------------------------
  | MatMul
  | BatchedMatMul
  | MatMulInt4({groupSize: int})
  | Gemm({alpha: float, beta: float, transA: bool, transB: bool})
  | Dot
  | Einsum({equation: string})

  // ----------------------------------------
  // Shape Operations
  // ----------------------------------------
  | Reshape({newShape: shape})
  | Squeeze({axes: array<int>})
  | Unsqueeze({axes: array<int>})
  | Flatten({axis: int})
  | Transpose({perm: array<int>})
  | Broadcast({targetShape: shape})
  | ExpandDims({axis: int})

  // ----------------------------------------
  // Slice/Index Operations
  // ----------------------------------------
  | Slice({starts: array<int>, ends: array<int>, axes: array<int>, steps: array<int>})
  | Gather({axis: int})
  | GatherElements({axis: int})
  | GatherND({batchDims: int})
  | Scatter({axis: int})
  | ScatterElements({axis: int})
  | ScatterND
  | Concat({axis: int})
  | Split({axis: int, splitSizes: array<int>})
  | Stack({axis: int})
  | Tile({repeats: array<int>})
  | Pad({pads: array<int>, mode: padMode, constantValue: float})
  | Reverse({axes: array<int>})

  // ----------------------------------------
  // Shape Query
  // ----------------------------------------
  | Shape
  | Size
  | Rank

  // ----------------------------------------
  // Data Type Operations
  // ----------------------------------------
  | Cast({dtype: dtype})
  | IsNaN
  | IsInf({detectPositive: bool, detectNegative: bool})

  // ----------------------------------------
  // Convolution Operations
  // ----------------------------------------
  | Conv1D({filters: int, kernel: int, stride: int, padding: padding, dilation: int, groups: int})
  | Conv2D({filters: int, kernel: (int, int), stride: (int, int), padding: padding, dilation: (int, int), groups: int})
  | Conv3D({filters: int, kernel: (int, int, int), stride: (int, int, int), padding: padding, dilation: (int, int, int), groups: int})
  | ConvTranspose1D({filters: int, kernel: int, stride: int, padding: padding, outputPadding: int, dilation: int, groups: int})
  | ConvTranspose2D({filters: int, kernel: (int, int), stride: (int, int), padding: padding, outputPadding: (int, int), dilation: (int, int), groups: int})
  | ConvTranspose3D({filters: int, kernel: (int, int, int), stride: (int, int, int), padding: padding, outputPadding: (int, int, int), dilation: (int, int, int), groups: int})
  | DepthwiseConv2D({kernel: (int, int), stride: (int, int), padding: padding, dilation: (int, int), depthMultiplier: int})

  // ----------------------------------------
  // Pooling Operations
  // ----------------------------------------
  | MaxPool1D({kernel: int, stride: int, padding: padding})
  | MaxPool2D({kernel: (int, int), stride: (int, int), padding: padding})
  | MaxPool3D({kernel: (int, int, int), stride: (int, int, int), padding: padding})
  | AvgPool1D({kernel: int, stride: int, padding: padding, countIncludePad: bool})
  | AvgPool2D({kernel: (int, int), stride: (int, int), padding: padding, countIncludePad: bool})
  | AvgPool3D({kernel: (int, int, int), stride: (int, int, int), padding: padding, countIncludePad: bool})
  | GlobalMaxPool
  | GlobalAvgPool
  | LpPool({p: int, kernel: (int, int), stride: (int, int)})
  | AdaptiveAvgPool1D({outputSize: int})
  | AdaptiveAvgPool2D({outputSize: (int, int)})
  | AdaptiveMaxPool1D({outputSize: int})
  | AdaptiveMaxPool2D({outputSize: (int, int)})

  // ----------------------------------------
  // Normalization Operations
  // ----------------------------------------
  | BatchNorm({epsilon: float, momentum: float})
  | LayerNorm({axes: array<int>, epsilon: float})
  | InstanceNorm({epsilon: float})
  | GroupNorm({numGroups: int, epsilon: float})
  | LRN({size: int, alpha: float, beta: float, bias: float})
  | RMSNorm({epsilon: float})

  // ----------------------------------------
  // Softmax Family
  // ----------------------------------------
  | Softmax({axis: int})
  | LogSoftmax({axis: int})
  | Hardmax({axis: int})

  // ----------------------------------------
  // Dropout/Regularization
  // ----------------------------------------
  | Dropout({rate: float})
  | AlphaDropout({rate: float})

  // ----------------------------------------
  // Dense/Linear
  // ----------------------------------------
  | Dense({units: int, useBias: bool})

  // ----------------------------------------
  // Attention Mechanisms
  // ----------------------------------------
  | Attention({heads: int, dim: int, dropout: float, causal: bool})
  | MultiHeadAttention({heads: int, dim: int, dropout: float})
  | ScaledDotProductAttention({dropout: float, causal: bool})

  // ----------------------------------------
  // RNN Operations
  // ----------------------------------------
  | RNN({hiddenSize: int, direction: rnnDirection, activation: string})
  | LSTM({hiddenSize: int, direction: rnnDirection})
  | GRU({hiddenSize: int, direction: rnnDirection})

  // ----------------------------------------
  // Image Operations
  // ----------------------------------------
  | Resize({mode: interpolationMode, scales: option<array<float>>, sizes: option<array<int>>})
  | SpaceToDepth({blockSize: int})
  | DepthToSpace({blockSize: int})
  | GridSample({mode: interpolationMode, paddingMode: padMode, alignCorners: bool})
  | RoiAlign({outputHeight: int, outputWidth: int, samplingRatio: int, spatialScale: float})
  | NonMaxSuppression({maxOutputBoxesPerClass: int, iouThreshold: float, scoreThreshold: float})

  // ----------------------------------------
  // One-Hot / Embedding
  // ----------------------------------------
  | OneHot({depth: int, axis: int})
  | Embedding({numEmbeddings: int, embeddingDim: int})

  // ----------------------------------------
  // Sorting/Selection
  // ----------------------------------------
  | TopK({k: int, axis: int, largest: bool, sorted: bool})
  | Sort({axis: int, descending: bool})
  | Unique({axis: option<int>, sorted: bool})
  | NonZero

  // ----------------------------------------
  // Random Operations
  // ----------------------------------------
  | RandomNormal({shape: shape, mean: float, stddev: float})
  | RandomUniform({shape: shape, low: float, high: float})
  | Bernoulli({p: float})

  // ----------------------------------------
  // Loss Functions
  // ----------------------------------------
  | MSELoss({reduction: string})
  | CrossEntropyLoss({reduction: string, ignoreIndex: option<int>})
  | BCELoss({reduction: string})
  | BCEWithLogitsLoss({reduction: string})
  | NLLLoss({reduction: string, ignoreIndex: option<int>})
  | CTCLoss({blank: int, reduction: string})
  | HuberLoss({delta: float, reduction: string})
  | SmoothL1Loss({beta: float, reduction: string})
  | TripletMarginLoss({margin: float, p: float, reduction: string})
  | CosineEmbeddingLoss({margin: float, reduction: string})

  // ----------------------------------------
  // Quantization
  // ----------------------------------------
  | QuantizeLinear({scale: float, zeroPoint: int})
  | DequantizeLinear({scale: float, zeroPoint: int})

// ============================================
// Graph Structures
// ============================================

type node = {
  id: int,
  op: op,
  inputs: array<int>,
  shape: option<shape>,
  dtype: option<dtype>,
  name: option<string>,
}

type graph = {
  nodes: array<node>,
  inputIds: array<int>,
  outputIds: array<int>,
  nextId: int,
}

// ============================================
// Compilation Results
// ============================================

type bufferUsage =
  | Storage
  | Uniform
  | ReadOnly
  | ReadWrite

type bufferBinding = {
  binding: int,
  size: int,
  usage: bufferUsage,
  name: string,
}

type dispatch = {
  workgroupSize: (int, int, int),
  workgroupCount: (int, int, int),
  kernelName: string,
  pipelineIndex: int,
}

type kernel = {
  name: string,
  wgsl: string,
  bindings: array<bufferBinding>,
}

type program = {
  kernels: array<kernel>,
  dispatches: array<dispatch>,
  totalBufferSize: int,
}

// ============================================
// Error Types
// ============================================

type shapeError =
  | DimensionMismatch({expected: int, got: int, nodeId: int})
  | IncompatibleShapes({shape1: shape, shape2: shape, nodeId: int})
  | InvalidAxis({axis: int, rank: int, nodeId: int})
  | UnknownShape({nodeId: int})

type compileError =
  | ShapeError(shapeError)
  | InvalidOperation({message: string, nodeId: int})
  | MissingInput({nodeId: int, inputIndex: int})
  | UnsupportedOp({opName: string, nodeId: int})
  | CyclicGraph({nodeIds: array<int>})
  | InvalidGraph({message: string})
  | CodegenError({message: string, nodeId: int})

// ============================================
// Result Type Alias
// ============================================
type compileResult<'a> = result<'a, compileError>
type shapeResult<'a> = result<'a, shapeError>
