// src/AutogradEngine.res
// Complete automatic differentiation engine
//
// This connects forward execution with automatic backward pass generation.
// Usage:
//   let engine = AutogradEngine.create(runtime)
//   engine->markRequiresGrad(weightNodeId)
//   let loss = engine->forward(graph, inputs)
//   engine->backward()  // Computes all gradients
//   engine->step(~lr=0.01)  // Updates parameters

open Types

// Saved tensor info for backward pass
type savedTensor = {
  nodeId: int,
  bufferId: int,
  shape: shape,
  data: option<array<float>>, // CPU copy if needed
}

// Recorded operation with all info needed for backward
type recordedOp = {
  nodeId: int,
  op: op,
  inputs: array<savedTensor>,
  output: savedTensor,
}

// Parameter info
type parameter = {
  nodeId: int,
  bufferId: int,
  shape: shape,
  name: string,
}

// Optimizer state for Adam
type adamState = {
  m: int, // buffer ID for first moment
  v: int, // buffer ID for second moment
}

// The autograd engine
type t = {
  // Recorded operations (forward order)
  mutable tape: array<recordedOp>,
  // Parameters that require gradients
  mutable parameters: array<parameter>,
  // Node IDs that require gradients
  mutable requiresGrad: Belt.Set.Int.t,
  // Gradient buffers: nodeId -> bufferId
  mutable gradients: Belt.Map.Int.t<int>,
  // Adam optimizer state: nodeId -> adamState
  mutable adamStates: Belt.Map.Int.t<adamState>,
  // Optimizer timestep
  mutable timestep: int,
  // Buffer cache for intermediate values: nodeId -> bufferId
  mutable bufferCache: Belt.Map.Int.t<int>,
  // Shape cache: nodeId -> shape
  mutable shapeCache: Belt.Map.Int.t<shape>,
}

let create = (): t => {
  tape: [],
  parameters: [],
  requiresGrad: Belt.Set.Int.empty,
  gradients: Belt.Map.Int.empty,
  adamStates: Belt.Map.Int.empty,
  timestep: 0,
  bufferCache: Belt.Map.Int.empty,
  shapeCache: Belt.Map.Int.empty,
}

// Mark a node as requiring gradients (trainable parameter)
let markRequiresGrad = (engine: t, nodeId: int, shape: shape, name: string): unit => {
  engine.requiresGrad = Belt.Set.Int.add(engine.requiresGrad, nodeId)
  engine.parameters = Array.concat(engine.parameters, [{
    nodeId,
    bufferId: -1, // Will be set during forward
    shape,
    name,
  }])
  engine.shapeCache = Belt.Map.Int.set(engine.shapeCache, nodeId, shape)
}

// Check if a node requires gradients
let needsGrad = (engine: t, nodeId: int): bool => {
  Belt.Set.Int.has(engine.requiresGrad, nodeId)
}

// Record an operation to the tape
let record = (
  engine: t,
  nodeId: int,
  op: op,
  inputNodeIds: array<int>,
  inputBufferIds: array<int>,
  inputShapes: array<shape>,
  outputBufferId: int,
  outputShape: shape,
): unit => {
  // Only record if any input requires grad or is a parameter
  let anyRequiresGrad = Array.some(inputNodeIds, id => needsGrad(engine, id))
  
  if anyRequiresGrad {
    let inputs = Array.mapWithIndex(inputNodeIds, (id, idx) => {
      {
        nodeId: id,
        bufferId: inputBufferIds[idx]->Option.getOr(-1),
        shape: inputShapes[idx]->Option.getOr([]),
        data: None,
      }
    })
    
    let output = {
      nodeId,
      bufferId: outputBufferId,
      shape: outputShape,
      data: None,
    }
    
    engine.tape = Array.concat(engine.tape, [{nodeId, op, inputs, output}])
    engine.bufferCache = Belt.Map.Int.set(engine.bufferCache, nodeId, outputBufferId)
    engine.shapeCache = Belt.Map.Int.set(engine.shapeCache, nodeId, outputShape)
    
    // Output also requires grad if inputs do
    engine.requiresGrad = Belt.Set.Int.add(engine.requiresGrad, nodeId)
  }
}

// Clear the tape (keep parameters)
let clearTape = (engine: t): unit => {
  engine.tape = []
}

// Get gradient buffer for a node (allocate if needed)
let getOrAllocateGradient = (engine: t, nodeId: int, _size: int): int => {
  switch Belt.Map.Int.get(engine.gradients, nodeId) {
  | Some(bufferId) => bufferId
  | None => {
      // Return a placeholder - actual allocation happens in JS
      let bufferId = nodeId * 1000 + 500 // Unique ID scheme for grad buffers
      engine.gradients = Belt.Map.Int.set(engine.gradients, nodeId, bufferId)
      bufferId
    }
  }
}

// Generate backward pass operations
// Returns array of (kernel, inputBufferIds, outputBufferIds, dispatch)
type backwardOp = {
  kernel: kernel,
  // Mapping of kernel binding names to buffer IDs or special values
  bindings: array<(string, int)>,
  outputSize: int,
}

let generateBackwardOps = (engine: t, lossNodeId: int): array<backwardOp> => {
  let ops = ref([])
  
  // Process tape in reverse order
  let reversedTape = Array.toReversed(engine.tape)
  
  // Initialize loss gradient to 1.0 (handled separately in runtime)
  let lossShape = engine.shapeCache->Belt.Map.Int.get(lossNodeId)->Option.getOr([1])
  let lossSize = Shape.numElements(lossShape)
  let _lossGradBufferId = getOrAllocateGradient(engine, lossNodeId, lossSize)
  
  Array.forEach(reversedTape, record => {
    let outputSize = Shape.numElements(record.output.shape)
    let gradOutBufferId = getOrAllocateGradient(engine, record.nodeId, outputSize)
    
    switch record.op {
    // ==========================================
    // Unary operations
    // ==========================================
    | Neg => {
        let inputId = record.inputs[0]->Option.map(i => i.nodeId)->Option.getOr(-1)
        let inputSize = record.inputs[0]->Option.map(i => Shape.numElements(i.shape))->Option.getOr(0)
        let gradInBufferId = getOrAllocateGradient(engine, inputId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genNegBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Abs => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genAbsBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Sqrt => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSqrtBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("out", record.output.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Exp => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genExpBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("out", record.output.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Log => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genLogBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Sin => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSinBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Cos => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genCosBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Tanh => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genTanhBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("out", record.output.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | Sigmoid => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSigmoidBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("out", record.output.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | ReLU => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genReLUBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | LeakyReLU({alpha}) => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genLeakyReLUBackwardKernel(outputSize, alpha),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    | GeLU => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradInBufferId = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genGeLUBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("x", input.bufferId),
            ("grad_x", gradInBufferId),
          ],
          outputSize,
        }])
      }
    
    // ==========================================
    // Binary operations
    // ==========================================
    | Add => {
        let input0 = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let input1 = record.inputs[1]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let size0 = Shape.numElements(input0.shape)
        let size1 = Shape.numElements(input1.shape)
        let gradA = getOrAllocateGradient(engine, input0.nodeId, size0)
        let gradB = getOrAllocateGradient(engine, input1.nodeId, size1)
        
        // If shapes match, simple backward
        if size0 == outputSize && size1 == outputSize {
          ops := Array.concat(ops.contents, [{
            kernel: Autograd.genAddBackwardKernel(outputSize),
            bindings: [
              ("grad_out", gradOutBufferId),
              ("grad_a", gradA),
              ("grad_b", gradB),
            ],
            outputSize,
          }])
        } else {
          // Need to handle broadcasting - add reduction
          // For now, use simple backward (TODO: add proper reduction)
          ops := Array.concat(ops.contents, [{
            kernel: Autograd.genAddBackwardKernel(outputSize),
            bindings: [
              ("grad_out", gradOutBufferId),
              ("grad_a", gradA),
              ("grad_b", gradB),
            ],
            outputSize,
          }])
        }
      }
    
    | Sub => {
        let input0 = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let input1 = record.inputs[1]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let size0 = Shape.numElements(input0.shape)
        let size1 = Shape.numElements(input1.shape)
        let gradA = getOrAllocateGradient(engine, input0.nodeId, size0)
        let gradB = getOrAllocateGradient(engine, input1.nodeId, size1)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSubBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("grad_a", gradA),
            ("grad_b", gradB),
          ],
          outputSize,
        }])
      }
    
    | Mul => {
        let input0 = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let input1 = record.inputs[1]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let size0 = Shape.numElements(input0.shape)
        let size1 = Shape.numElements(input1.shape)
        let gradA = getOrAllocateGradient(engine, input0.nodeId, size0)
        let gradB = getOrAllocateGradient(engine, input1.nodeId, size1)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genMulBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("a", input0.bufferId),
            ("b", input1.bufferId),
            ("grad_a", gradA),
            ("grad_b", gradB),
          ],
          outputSize,
        }])
      }
    
    | Div => {
        let input0 = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let input1 = record.inputs[1]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let size0 = Shape.numElements(input0.shape)
        let size1 = Shape.numElements(input1.shape)
        let gradA = getOrAllocateGradient(engine, input0.nodeId, size0)
        let gradB = getOrAllocateGradient(engine, input1.nodeId, size1)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genDivBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("a", input0.bufferId),
            ("b", input1.bufferId),
            ("grad_a", gradA),
            ("grad_b", gradB),
          ],
          outputSize,
        }])
      }
    
    // ==========================================
    // MatMul
    // ==========================================
    | MatMul => {
        let input0 = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let input1 = record.inputs[1]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let shape0 = input0.shape
        let shape1 = input1.shape
        let r0 = Array.length(shape0)
        let r1 = Array.length(shape1)
        
        if r0 >= 2 && r1 >= 2 {
          let m = shape0[r0 - 2]->Option.getOr(1)
          let k = shape0[r0 - 1]->Option.getOr(1)
          let n = shape1[r1 - 1]->Option.getOr(1)
          
          let size0 = Shape.numElements(shape0)
          let size1 = Shape.numElements(shape1)
          let gradA = getOrAllocateGradient(engine, input0.nodeId, size0)
          let gradB = getOrAllocateGradient(engine, input1.nodeId, size1)
          
          // Check for batched
          let batchDims = Array.slice(record.output.shape, ~start=0, ~end=Array.length(record.output.shape) - 2)
          let batchSize = Array.reduce(batchDims, 1, (a, b) => a * b)
          
          if batchSize > 1 {
            // Batched matmul backward
            ops := Array.concat(ops.contents, [
              {
                kernel: Autograd.genBatchedMatMulBackwardAKernel(batchSize, m, k, n),
                bindings: [
                  ("grad_out", gradOutBufferId),
                  ("b", input1.bufferId),
                  ("grad_a", gradA),
                ],
                outputSize: size0,
              },
              {
                kernel: Autograd.genBatchedMatMulBackwardBKernel(batchSize, m, k, n),
                bindings: [
                  ("grad_out", gradOutBufferId),
                  ("a", input0.bufferId),
                  ("grad_b", gradB),
                ],
                outputSize: size1,
              },
            ])
          } else {
            // Regular 2D matmul backward
            ops := Array.concat(ops.contents, [
              {
                kernel: Autograd.genMatMulBackwardAKernel(m, k, n),
                bindings: [
                  ("grad_out", gradOutBufferId),
                  ("b", input1.bufferId),
                  ("grad_a", gradA),
                ],
                outputSize: size0,
              },
              {
                kernel: Autograd.genMatMulBackwardBKernel(m, k, n),
                bindings: [
                  ("grad_out", gradOutBufferId),
                  ("a", input0.bufferId),
                  ("grad_b", gradB),
                ],
                outputSize: size1,
              },
            ])
          }
        }
      }
    
    // ==========================================
    // Reductions
    // ==========================================
    | Reduce({op: Sum, axes}) => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradIn = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSumBackwardKernel(input.shape, record.output.shape, axes),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("grad_x", gradIn),
          ],
          outputSize: inputSize,
        }])
      }
    
    | Reduce({op: Mean, axes}) => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradIn = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genMeanBackwardKernel(input.shape, record.output.shape, axes),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("grad_x", gradIn),
          ],
          outputSize: inputSize,
        }])
      }
    
    // ==========================================
    // Softmax
    // ==========================================
    | Softmax({axis}) => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputShape = input.shape
        let rank = Array.length(inputShape)
        let normAxis = axis < 0 ? rank + axis : axis
        let axisSize = inputShape[normAxis]->Option.getOr(1)
        let outerSize = outputSize / axisSize
        let inputSize = Shape.numElements(input.shape)
        let gradIn = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genSoftmaxBackwardKernel(outerSize, axisSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("softmax_out", record.output.bufferId),
            ("grad_x", gradIn),
          ],
          outputSize: inputSize,
        }])
      }
    
    // ==========================================
    // Shape operations (pass-through)
    // ==========================================
    | Reshape(_) | Flatten(_) | Squeeze(_) | Unsqueeze(_) | ExpandDims(_) | Identity => {
        let input = record.inputs[0]->Option.getOr({nodeId: -1, bufferId: -1, shape: [], data: None})
        let inputSize = Shape.numElements(input.shape)
        let gradIn = getOrAllocateGradient(engine, input.nodeId, inputSize)
        
        ops := Array.concat(ops.contents, [{
          kernel: Autograd.genCopyBackwardKernel(outputSize),
          bindings: [
            ("grad_out", gradOutBufferId),
            ("grad_x", gradIn),
          ],
          outputSize: inputSize,
        }])
      }
    
    | _ => () // Skip unsupported ops
    }
  })
  
  ops.contents
}

// Get parameter gradients (for optimizer)
let getParameterGradients = (engine: t): array<(int, int, shape)> => {
  // Returns: (paramNodeId, gradBufferId, shape)
  Array.filterMap(engine.parameters, param => {
    switch Belt.Map.Int.get(engine.gradients, param.nodeId) {
    | Some(gradBufferId) => Some((param.nodeId, gradBufferId, param.shape))
    | None => None
    }
  })
}

// Generate optimizer step operations
let generateOptimizerOps = (
  engine: t,
  optimizer: string,
  lr: float,
  beta1: float,
  beta2: float,
  epsilon: float,
  weightDecay: float,
): array<backwardOp> => {
  engine.timestep = engine.timestep + 1
  let t = engine.timestep
  
  Array.filterMap(engine.parameters, param => {
    let gradBufferId = Belt.Map.Int.get(engine.gradients, param.nodeId)
    
    switch gradBufferId {
    | None => None
    | Some(gradId) => {
        let size = Shape.numElements(param.shape)
        
        switch optimizer {
        | "sgd" => Some({
            kernel: Autograd.genSGDKernel(size, lr),
            bindings: [
              ("param", param.bufferId),
              ("grad", gradId),
            ],
            outputSize: size,
          })
        
        | "adam" => {
            // Get or create Adam state
            let adamState = switch Belt.Map.Int.get(engine.adamStates, param.nodeId) {
            | Some(state) => state
            | None => {
                let mId = param.nodeId * 1000 + 600
                let vId = param.nodeId * 1000 + 700
                let state = {m: mId, v: vId}
                engine.adamStates = Belt.Map.Int.set(engine.adamStates, param.nodeId, state)
                state
              }
            }
            
            Some({
              kernel: Autograd.genAdamKernel(size, lr, beta1, beta2, epsilon, t),
              bindings: [
                ("param", param.bufferId),
                ("grad", gradId),
                ("m", adamState.m),
                ("v", adamState.v),
              ],
              outputSize: size,
            })
          }
        
        | "adamw" => {
            let adamState = switch Belt.Map.Int.get(engine.adamStates, param.nodeId) {
            | Some(state) => state
            | None => {
                let mId = param.nodeId * 1000 + 600
                let vId = param.nodeId * 1000 + 700
                let state = {m: mId, v: vId}
                engine.adamStates = Belt.Map.Int.set(engine.adamStates, param.nodeId, state)
                state
              }
            }
            
            Some({
              kernel: Autograd.genAdamWKernel(size, lr, beta1, beta2, epsilon, weightDecay, t),
              bindings: [
                ("param", param.bufferId),
                ("grad", gradId),
                ("m", adamState.m),
                ("v", adamState.v),
              ],
              outputSize: size,
            })
          }
        
        | _ => None
        }
      }
    }
  })
}

// Generate zero grad operations
let generateZeroGradOps = (engine: t): array<backwardOp> => {
  Array.filterMap(engine.parameters, param => {
    let gradBufferId = Belt.Map.Int.get(engine.gradients, param.nodeId)
    
    switch gradBufferId {
    | None => None
    | Some(gradId) => {
        let size = Shape.numElements(param.shape)
        Some({
          kernel: Autograd.genGradZeroKernel(size),
          bindings: [("grad", gradId)],
          outputSize: size,
        })
      }
    }
  })
}
