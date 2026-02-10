// src/GradTape.res
// Automatic differentiation via gradient tape
//
// Records forward operations and constructs backward pass automatically.
// Usage:
//   let tape = GradTape.create()
//   let y = tape->GradTape.record(graph, () => {
//     // forward pass operations
//   })
//   let grads = tape->GradTape.backward(y, runtime)

open Types

// A recorded operation in the tape
type recordedOp = {
  nodeId: int,
  op: op,
  inputIds: array<int>,        // Node IDs of inputs
  inputShapes: array<shape>,   // Shapes of inputs
  outputShape: shape,          // Shape of output
  outputId: int,               // Node ID of output (same as nodeId for most ops)
}

// Gradient tape that records operations
type t = {
  mutable ops: array<recordedOp>,
  mutable parameterIds: Belt.Set.Int.t,  // Node IDs that are trainable parameters
  mutable gradients: Belt.Map.Int.t<int>, // nodeId -> gradient buffer ID
}

// Create a new gradient tape
let create = (): t => {
  ops: [],
  parameterIds: Belt.Set.Int.empty,
  gradients: Belt.Map.Int.empty,
}

// Mark a node as a trainable parameter
let markParameter = (tape: t, nodeId: int): unit => {
  tape.parameterIds = Belt.Set.Int.add(tape.parameterIds, nodeId)
}

// Check if a node is a parameter
let isParameter = (tape: t, nodeId: int): bool => {
  Belt.Set.Int.has(tape.parameterIds, nodeId)
}

// Record an operation to the tape
let recordOp = (tape: t, nodeId: int, op: op, inputIds: array<int>, inputShapes: array<shape>, outputShape: shape): unit => {
  let record = {
    nodeId,
    op,
    inputIds,
    inputShapes,
    outputShape,
    outputId: nodeId,
  }
  tape.ops = Array.concat(tape.ops, [record])
}

// Clear the tape (but keep parameter markings)
let reset = (tape: t): unit => {
  tape.ops = []
  tape.gradients = Belt.Map.Int.empty
}

// Get the backward kernel and dispatch info for an operation
// Returns: (kernels, buffer bindings) for the backward pass
let getBackwardKernels = (
  record: recordedOp,
  _gradOutputShape: shape,
): array<(kernel, array<string>)> => {
  let outputSize = Shape.numElements(record.outputShape)
  
  switch record.op {
  // ==========================================
  // Unary operations
  // ==========================================
  | Neg => [(Autograd.genNegBackwardKernel(outputSize), ["grad_out", "grad_x"])]
  
  | Abs => [(Autograd.genAbsBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  | Sqrt => [(Autograd.genSqrtBackwardKernel(outputSize), ["grad_out", "out", "grad_x"])]
  
  | Exp => [(Autograd.genExpBackwardKernel(outputSize), ["grad_out", "out", "grad_x"])]
  
  | Log => [(Autograd.genLogBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  | Sin => [(Autograd.genSinBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  | Cos => [(Autograd.genCosBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  | Tanh => [(Autograd.genTanhBackwardKernel(outputSize), ["grad_out", "out", "grad_x"])]
  
  | Sigmoid => [(Autograd.genSigmoidBackwardKernel(outputSize), ["grad_out", "out", "grad_x"])]
  
  | ReLU => [(Autograd.genReLUBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  | LeakyReLU({alpha}) => [(Autograd.genLeakyReLUBackwardKernel(outputSize, alpha), ["grad_out", "x", "grad_x"])]
  
  | GeLU => [(Autograd.genGeLUBackwardKernel(outputSize), ["grad_out", "x", "grad_x"])]
  
  // ==========================================
  // Binary operations
  // ==========================================
  | Add => {
      let inputShape0 = record.inputShapes[0]->Option.getOr([])
      let inputShape1 = record.inputShapes[1]->Option.getOr([])
      let kernels = [(Autograd.genAddBackwardKernel(outputSize), ["grad_out", "grad_a", "grad_b"])]
      
      // Add reduction kernels if broadcasting occurred
      let input0Size = Shape.numElements(inputShape0)
      let input1Size = Shape.numElements(inputShape1)
      
      if input0Size < outputSize {
        let reduceKernel = Autograd.genGradReduceKernel(record.outputShape, inputShape0)
        Array.concat(kernels, [(reduceKernel, ["grad_a_full", "grad_a_reduced"])])
      } else if input1Size < outputSize {
        let reduceKernel = Autograd.genGradReduceKernel(record.outputShape, inputShape1)
        Array.concat(kernels, [(reduceKernel, ["grad_b_full", "grad_b_reduced"])])
      } else {
        kernels
      }
    }
  
  | Sub => [(Autograd.genSubBackwardKernel(outputSize), ["grad_out", "grad_a", "grad_b"])]
  
  | Mul => [(Autograd.genMulBackwardKernel(outputSize), ["grad_out", "a", "b", "grad_a", "grad_b"])]
  
  | Div => [(Autograd.genDivBackwardKernel(outputSize), ["grad_out", "a", "b", "grad_a", "grad_b"])]
  
  | Pow => [(Autograd.genPowBackwardKernel(outputSize), ["grad_out", "a", "b", "out", "grad_a", "grad_b"])]
  
  | Maximum => [(Autograd.genMaximumBackwardKernel(outputSize), ["grad_out", "a", "b", "grad_a", "grad_b"])]
  
  | Minimum => [(Autograd.genMinimumBackwardKernel(outputSize), ["grad_out", "a", "b", "grad_a", "grad_b"])]
  
  // ==========================================
  // MatMul
  // ==========================================
  | MatMul => {
      let shape0 = record.inputShapes[0]->Option.getOr([])
      let shape1 = record.inputShapes[1]->Option.getOr([])
      let r0 = Array.length(shape0)
      let r1 = Array.length(shape1)
      
      if r0 >= 2 && r1 >= 2 {
        let m = shape0[r0 - 2]->Option.getOr(1)
        let k = shape0[r0 - 1]->Option.getOr(1)
        let n = shape1[r1 - 1]->Option.getOr(1)
        
        // Check for batched matmul
        let batchDims = Array.slice(record.outputShape, ~start=0, ~end=Array.length(record.outputShape) - 2)
        let batchSize = Array.reduce(batchDims, 1, (a, b) => a * b)
        
        if batchSize > 1 {
          [
            (Autograd.genBatchedMatMulBackwardAKernel(batchSize, m, k, n), ["grad_out", "b", "grad_a"]),
            (Autograd.genBatchedMatMulBackwardBKernel(batchSize, m, k, n), ["grad_out", "a", "grad_b"]),
          ]
        } else {
          [
            (Autograd.genMatMulBackwardAKernel(m, k, n), ["grad_out", "b", "grad_a"]),
            (Autograd.genMatMulBackwardBKernel(m, k, n), ["grad_out", "a", "grad_b"]),
          ]
        }
      } else {
        []
      }
    }
  
  // ==========================================
  // Reductions
  // ==========================================
  | Reduce({op: Sum, axes}) => {
      let inputShape = record.inputShapes[0]->Option.getOr([])
      [(Autograd.genSumBackwardKernel(inputShape, record.outputShape, axes), ["grad_out", "grad_x"])]
    }
  
  | Reduce({op: Mean, axes}) => {
      let inputShape = record.inputShapes[0]->Option.getOr([])
      [(Autograd.genMeanBackwardKernel(inputShape, record.outputShape, axes), ["grad_out", "grad_x"])]
    }
  
  // ==========================================
  // Softmax
  // ==========================================
  | Softmax({axis}) => {
      let inputShape = record.inputShapes[0]->Option.getOr([])
      let rank = Array.length(inputShape)
      let normAxis = axis < 0 ? rank + axis : axis
      let axisSize = inputShape[normAxis]->Option.getOr(1)
      let outerSize = Shape.numElements(inputShape) / axisSize
      [(Autograd.genSoftmaxBackwardKernel(outerSize, axisSize), ["grad_out", "softmax_out", "grad_x"])]
    }
  
  // ==========================================
  // LayerNorm
  // ==========================================
  | LayerNorm({axes, epsilon}) => {
      let inputShape = record.inputShapes[0]->Option.getOr([])
      let rank = Array.length(inputShape)
      let normAxes = Array.map(axes, a => a < 0 ? rank + a : a)
      let normSize = Array.reduce(normAxes, 1, (acc, axis) => 
        acc * inputShape[axis]->Option.getOr(1)
      )
      let outerSize = Shape.numElements(inputShape) / normSize
      [(Autograd.genLayerNormBackwardKernel(outerSize, normSize, epsilon), 
        ["grad_out", "x", "gamma", "grad_x", "grad_gamma", "grad_beta"])]
    }
  
  // ==========================================
  // Shape operations (pass-through gradients)
  // ==========================================
  | Reshape(_) | Flatten(_) | Squeeze(_) | Unsqueeze(_) | ExpandDims(_) => {
      [(Autograd.genCopyBackwardKernel(outputSize), ["grad_out", "grad_x"])]
    }
  
  | Transpose(_) => {
      // TODO: Need transpose backward kernel
      [(Autograd.genCopyBackwardKernel(outputSize), ["grad_out", "grad_x"])]
    }
  
  // Identity just passes gradient through
  | Identity => [(Autograd.genCopyBackwardKernel(outputSize), ["grad_out", "grad_x"])]
  
  // Operations without gradients or not yet implemented
  | _ => []
  }
}

// Check if an operation supports gradients
let supportsGradient = (op: op): bool => {
  switch op {
  | Neg | Abs | Sqrt | Exp | Log | Sin | Cos | Tanh | Sigmoid | ReLU 
  | LeakyReLU(_) | GeLU | Add | Sub | Mul | Div | Pow | Maximum | Minimum
  | MatMul | Reduce({op: Sum}) | Reduce({op: Mean}) | Softmax(_) 
  | LayerNorm(_) | Reshape(_) | Flatten(_) | Squeeze(_) | Unsqueeze(_) 
  | ExpandDims(_) | Transpose(_) | Identity => true
  | _ => false
  }
}

// Topological sort of operations for backward pass (reverse order)
let topoSortBackward = (ops: array<recordedOp>): array<recordedOp> => {
  // Already in forward order, just reverse for backward
  Array.toReversed(ops)
}

// Structure for compiled backward pass
type backwardPass = {
  // Kernels to execute in order
  kernels: array<kernel>,
  dispatches: array<dispatch>,
  // Buffer allocations
  buffers: array<{id: int, size: int, name: string}>,
  // Mapping from node ID to gradient buffer ID
  gradientBufferIds: Belt.Map.Int.t<int>,
  // Which buffers are parameter gradients (to be used by optimizer)
  parameterGradientIds: array<(int, int)>, // (paramNodeId, gradBufferId)
}

// Compile backward pass from tape
let compileBackward = (tape: t, lossNodeId: int, nodeShapes: Belt.Map.Int.t<shape>): option<backwardPass> => {
  let ops = topoSortBackward(tape.ops)
  
  // Filter to only ops that contribute to the loss
  // For now, include all ops (could optimize with reachability analysis)
  
  let kernels = ref([])
  let dispatches = ref([])
  let buffers = ref([])
  let nextBufferId = ref(0)
  let gradientBufferIds = ref(Belt.Map.Int.empty)
  let paramGradIds = ref([])
  
  // Allocate gradient buffer for loss (initialized to 1.0)
  let lossShape = nodeShapes->Belt.Map.Int.get(lossNodeId)->Option.getOr([1])
  let lossGradSize = Shape.numElements(lossShape)
  let lossGradBufferId = nextBufferId.contents
  nextBufferId := nextBufferId.contents + 1
  buffers := Array.concat(buffers.contents, [{id: lossGradBufferId, size: lossGradSize * 4, name: "grad_loss"}])
  gradientBufferIds := Belt.Map.Int.set(gradientBufferIds.contents, lossNodeId, lossGradBufferId)
  
  // Process each operation in reverse order
  Array.forEach(ops, record => {
    if supportsGradient(record.op) {
      let gradOutShape = nodeShapes->Belt.Map.Int.get(record.outputId)->Option.getOr([])
      let backwardKernels = getBackwardKernels(record, gradOutShape)
      
      Array.forEach(backwardKernels, ((kernel, _bindings)) => {
        // Allocate gradient buffers for inputs
        Array.forEachWithIndex(record.inputIds, (inputId, _idx) => {
          if !(Belt.Map.Int.has(gradientBufferIds.contents, inputId)) {
            let inputShape = record.inputShapes[_idx]->Option.getOr([])
            let gradSize = Shape.numElements(inputShape)
            let gradBufferId = nextBufferId.contents
            nextBufferId := nextBufferId.contents + 1
            buffers := Array.concat(buffers.contents, [{id: gradBufferId, size: gradSize * 4, name: "grad_" ++ Int.toString(inputId)}])
            gradientBufferIds := Belt.Map.Int.set(gradientBufferIds.contents, inputId, gradBufferId)
            
            // Track if this is a parameter gradient
            if Belt.Set.Int.has(tape.parameterIds, inputId) {
              paramGradIds := Array.concat(paramGradIds.contents, [(inputId, gradBufferId)])
            }
          }
        })
        
        // Add kernel and dispatch
        kernels := Array.concat(kernels.contents, [kernel])
        let totalElements = Shape.numElements(gradOutShape)
        let workgroupCount = (totalElements + 255) / 256
        dispatches := Array.concat(dispatches.contents, [{
          workgroupSize: (256, 1, 1),
          workgroupCount: (workgroupCount, 1, 1),
          kernelName: kernel.name,
          pipelineIndex: Array.length(kernels.contents) - 1,
        }])
      })
    }
  })
  
  Some({
    kernels: kernels.contents,
    dispatches: dispatches.contents,
    buffers: buffers.contents,
    gradientBufferIds: gradientBufferIds.contents,
    parameterGradientIds: paramGradIds.contents,
  })
}
