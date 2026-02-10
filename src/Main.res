// src/Main.res
open Types

let log = Console.log

let printShape = (name, shape) => {
  let strs = Array.map(shape, d => Int.toString(d))
  let joined = Array.join(strs, ", ")
  log(name ++ ": [" ++ joined ++ "]")
}

let testInfer = (name, op, inputs) => {
  switch Shape.infer(op, inputs) {
  | Some(s) => printShape(name, s)
  | None => log(name ++ ": None")
  }
}

let testCodegen = (name, op, inputs) => {
  switch Codegen.generate(op, inputs) {
  | Some((kernel, dispatch)) => {
      let (wx, wy, _) = dispatch.workgroupCount
      log(name ++ ": " ++ kernel.name ++ " (" ++ Int.toString(wx) ++ "x" ++ Int.toString(wy) ++ ")")
    }
  | None => log(name ++ ": FAILED")
  }
}

log("=== Shape Inference ===")
testInfer("Input", Input({shape: [32, 224, 224, 3], dtype: F32}), [])
testInfer("Add broadcast", Add, [[32, 1, 64], [1, 10, 64]])
testInfer("MatMul", MatMul, [[32, 128, 64], [32, 64, 256]])

log("\n=== Codegen ===")

// Element-wise
testCodegen("ReLU", ReLU, [[4, 256]])
testCodegen("Add", Add, [[4, 256], [4, 256]])

// MatMul
testCodegen("MatMul", MatMul, [[64, 128], [128, 64]])

// Reductions
testCodegen("Reduce Sum", Reduce({op: Sum, axes: [-1], keepDims: false}), [[32, 128]])
testCodegen("Reduce Mean", Reduce({op: Mean, axes: [1, 2], keepDims: false}), [[8, 32, 32, 3]])

// Softmax
testCodegen("Softmax", Softmax({axis: -1}), [[4, 1000]])

// Dense
testCodegen("Dense", Dense({units: 256, useBias: true}), [[8, 512]])

// Pooling
testCodegen("MaxPool2D", MaxPool2D({kernel: (2, 2), stride: (2, 2), padding: Valid}), [[1, 224, 224, 64]])
testCodegen("AvgPool2D", AvgPool2D({kernel: (2, 2), stride: (2, 2), padding: Valid, countIncludePad: false}), [[1, 112, 112, 128]])

// BatchNorm
testCodegen("BatchNorm", BatchNorm({epsilon: 1e-5, momentum: 0.1}), [[1, 56, 56, 256]])

// Conv1D
testCodegen("Conv1D", Conv1D({filters: 64, kernel: 3, stride: 1, padding: Same, dilation: 1, groups: 1}), [[4, 128, 32]])

// Global pooling
testCodegen("GlobalMaxPool", GlobalMaxPool, [[1, 7, 7, 512]])
testCodegen("GlobalAvgPool", GlobalAvgPool, [[1, 7, 7, 512]])

// LayerNorm - needs axes field
testCodegen("LayerNorm", LayerNorm({axes: [-1], epsilon: 1e-5}), [[4, 128, 256]])

// Attention
testCodegen("Attention", ScaledDotProductAttention({dropout: 0.0, causal: false}), [[2, 64, 128]])

// Embedding - fields are numEmbeddings and embeddingDim
testCodegen("Embedding", Embedding({numEmbeddings: 50000, embeddingDim: 512}), [[4, 128]])

// Clip - min/max are option<float>
testCodegen("Clip", Clip({min: Some(0.0), max: Some(6.0)}), [[4, 256]])

log("\n=== Done ===")
