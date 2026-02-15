// src/Compiler.res
// Graph compiler: takes a computation graph and produces an executable program

open Types

// ============================================
// Graph Building API
// ============================================

// Tensor types
type tensorKind =
  | Input        // User-provided input data
  | Weight       // Trainable parameters (Dense weights, Conv filters)
  | Constant     // Non-trainable constants
  | Intermediate // Computed during forward pass
  | Output       // Final outputs

// Reference to a specific output of a node
type nodeRef = {
  nodeId: int,
  outputIndex: int,  // Which output of the node (0 for single-output ops)
}

// Helper to create a nodeRef (default outputIndex = 0)
let mkRef = (nodeId: int): nodeRef => {nodeId, outputIndex: 0}
let refOutput = (nodeId: int, outputIndex: int): nodeRef => {nodeId, outputIndex}

// LSTM cell result type
type lstmCellResult = {h: int, c: int}

// Extended node with tensor metadata
type graphNode = {
  id: int,
  op: op,
  inputs: array<nodeRef>,  // Changed from array<int>
  outputShapes: array<shape>,  // Multiple output shapes
  dtype: option<dtype>,
  name: option<string>,
  kind: tensorKind,
  data: option<array<float>>,  // For constants/initial weights
}

// Mutable graph builder
type graphBuilder = {
  mutable nodes: array<graphNode>,
  mutable nextId: int,
  mutable outputIds: array<nodeRef>,  // Changed to nodeRef
}

let createGraph = (): graphBuilder => {
  nodes: [],
  nextId: 0,
  outputIds: [],
}

// Get number of outputs for an op
let getOpOutputCount = (op: op): int => {
  switch op {
  | TopK(_) => 2  // values, indices
  | Split({splitSizes}) => Array.length(splitSizes)
  | _ => 1
  }
}

// Internal: add node helper
let addNodeInternal = (
  graph: graphBuilder,
  op: op,
  inputs: array<nodeRef>,
  name: option<string>,
  kind: tensorKind,
  data: option<array<float>>
): int => {
  let id = graph.nextId
  
  // Get input shapes (from the specific output of each input node)
  let inputShapes = Array.map(inputs, inputRef => {
    let inputNode = graph.nodes->Array.find(n => n.id == inputRef.nodeId)
    switch inputNode {
    | Some(node) => node.outputShapes->Array.get(inputRef.outputIndex)->Option.getOr([])
    | None => []
    }
  })

  // Infer output shape(s)
  let outputShapes = switch op {
  | TopK({k, axis}) => {
      // TopK returns two outputs with same shape
      let inputShape = inputShapes->Array.get(0)->Option.getOr([])
      let outShape = Array.mapWithIndex(inputShape, (dim, i) => {
        if i == axis || (axis < 0 && i == Array.length(inputShape) + axis) {
          k
        } else {
          dim
        }
      })
      [outShape, outShape]  // values and indices have same shape
    }
  | Split({axis, splitSizes}) => {
      let inputShape = inputShapes->Array.get(0)->Option.getOr([])
      Array.map(splitSizes, splitSize => {
        Array.mapWithIndex(inputShape, (dim, i) => {
          if i == axis { splitSize } else { dim }
        })
      })
    }
  | _ => {
      let shape = Shape.infer(op, inputShapes)
      switch shape {
      | Some(s) => [s]
      | None => [[]]
      }
    }
  }

  let node: graphNode = {
    id,
    op,
    inputs,
    outputShapes,
    dtype: Some(F32),
    name,
    kind,
    data,
  }

  graph.nodes = Array.concat(graph.nodes, [node])
  graph.nextId = graph.nextId + 1
  id
}

// Add a node to the graph and return its ID
let addNode = (graph: graphBuilder, op: op, inputs: array<int>, name: option<string>): int => {
  let inputRefs = Array.map(inputs, id => mkRef(id))
  addNodeInternal(graph, op, inputRefs, name, Intermediate, None)
}

// Add node with explicit nodeRefs (for multi-output ops)
let addNodeWithRefs = (graph: graphBuilder, op: op, inputs: array<nodeRef>, name: option<string>): int => {
  addNodeInternal(graph, op, inputs, name, Intermediate, None)
}

// ============================================
// Input/Weight/Constant API
// ============================================

// User input tensor
let input = (graph: graphBuilder, shape: shape, name: string): int => {
  let id = graph.nextId
  let node: graphNode = {
    id,
    op: Input({shape, dtype: F32}),
    inputs: [],
    outputShapes: [shape],
    dtype: Some(F32),
    name: Some(name),
    kind: Input,
    data: None,
  }
  graph.nodes = Array.concat(graph.nodes, [node])
  graph.nextId = graph.nextId + 1
  id
}

// Trainable weight tensor
let weight = (graph: graphBuilder, shape: shape, name: string): int => {
  let size = Shape.numElements(shape)
  let zeros = Array.make(~length=size, 0.0)
  let id = graph.nextId
  let node: graphNode = {
    id,
    op: Input({shape, dtype: F32}),
    inputs: [],
    outputShapes: [shape],
    dtype: Some(F32),
    name: Some(name),
    kind: Weight,
    data: Some(zeros),
  }
  graph.nodes = Array.concat(graph.nodes, [node])
  graph.nextId = graph.nextId + 1
  id
}

// Weight with initial data
let weightWithData = (graph: graphBuilder, shape: shape, name: string, data: array<float>): int => {
  let id = graph.nextId
  let node: graphNode = {
    id,
    op: Input({shape, dtype: F32}),
    inputs: [],
    outputShapes: [shape],
    dtype: Some(F32),
    name: Some(name),
    kind: Weight,
    data: Some(data),
  }
  graph.nodes = Array.concat(graph.nodes, [node])
  graph.nextId = graph.nextId + 1
  id
}

// Constant tensor (non-trainable)
let constant = (graph: graphBuilder, shape: shape, name: string, data: array<float>): int => {
  let id = graph.nextId
  let node: graphNode = {
    id,
    op: Input({shape, dtype: F32}),
    inputs: [],
    outputShapes: [shape],
    dtype: Some(F32),
    name: Some(name),
    kind: Constant,
    data: Some(data),
  }
  graph.nodes = Array.concat(graph.nodes, [node])
  graph.nextId = graph.nextId + 1
  id
}

// Mark node as output (specific output index)
let markOutputRef = (graph: graphBuilder, nodeRef: nodeRef): unit => {
  graph.outputIds = Array.concat(graph.outputIds, [nodeRef])
}

// Mark node as output (default output 0)
let markOutput = (graph: graphBuilder, nodeId: int): unit => {
  markOutputRef(graph, mkRef(nodeId))
}

// ============================================
// Unary Operations
// ============================================

let relu = (graph: graphBuilder, x: int): int => addNode(graph, ReLU, [x], None)
let sigmoid = (graph: graphBuilder, x: int): int => addNode(graph, Sigmoid, [x], None)
let tanh_ = (graph: graphBuilder, x: int): int => addNode(graph, Tanh, [x], None)
let gelu = (graph: graphBuilder, x: int): int => addNode(graph, GeLU, [x], None)
let silu = (graph: graphBuilder, x: int): int => addNode(graph, SiLU, [x], None)
let neg = (graph: graphBuilder, x: int): int => addNode(graph, Neg, [x], None)
let exp_ = (graph: graphBuilder, x: int): int => addNode(graph, Exp, [x], None)
let log_ = (graph: graphBuilder, x: int): int => addNode(graph, Log, [x], None)
let sqrt_ = (graph: graphBuilder, x: int): int => addNode(graph, Sqrt, [x], None)
let abs_ = (graph: graphBuilder, x: int): int => addNode(graph, Abs, [x], None)
let sin_ = (graph: graphBuilder, x: int): int => addNode(graph, Sin, [x], None)
let cos_ = (graph: graphBuilder, x: int): int => addNode(graph, Cos, [x], None)

// ============================================
// Binary Operations
// ============================================

let add = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Add, [a, b], None)
let sub = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Sub, [a, b], None)
let mul = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Mul, [a, b], None)
let div = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Div, [a, b], None)
let pow_ = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Pow, [a, b], None)
let maximum = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Maximum, [a, b], None)
let minimum = (graph: graphBuilder, a: int, b: int): int => addNode(graph, Minimum, [a, b], None)

// ============================================
// Matrix Operations
// ============================================

let matmul = (graph: graphBuilder, a: int, b: int): int => addNode(graph, MatMul, [a, b], None)

// Dense layer with explicit weight and bias
let denseWithWeights = (graph: graphBuilder, x: int, weights: int, bias: option<int>): int => {
  let out = matmul(graph, x, weights)
  switch bias {
  | Some(b) => add(graph, out, b)
  | None => out
  }
}

// Dense layer (creates internal weights)
let dense = (graph: graphBuilder, x: int, units: int, name: string): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let inputDim = inputShape->Array.get(Array.length(inputShape) - 1)->Option.getOr(1)

  let w = weight(graph, [inputDim, units], name ++ "_weight")
  let b = weight(graph, [units], name ++ "_bias")

  denseWithWeights(graph, x, w, Some(b))
}

// ============================================
// Reduction Operations
// ============================================

let softmax = (graph: graphBuilder, x: int, axis: int): int =>
  addNode(graph, Softmax({axis: axis}), [x], None)

let reduce = (graph: graphBuilder, x: int, op: reduceOp, axes: array<int>, keepDims: bool): int =>
  addNode(graph, Reduce({op, axes, keepDims}), [x], None)

let reduceSum = (graph: graphBuilder, x: int, axes: array<int>, keepDims: bool): int =>
  reduce(graph, x, Sum, axes, keepDims)

let reduceMean = (graph: graphBuilder, x: int, axes: array<int>, keepDims: bool): int =>
  reduce(graph, x, Mean, axes, keepDims)

let reduceMax = (graph: graphBuilder, x: int, axes: array<int>, keepDims: bool): int =>
  reduce(graph, x, Max, axes, keepDims)

// ============================================
// Convolution Operations
// ============================================

let conv2dWithWeights = (
  graph: graphBuilder,
  x: int,
  weights: int,
  bias: option<int>,
  stride: (int, int),
  padding: padding
): int => {
  let weightsNode = graph.nodes->Array.find(n => n.id == weights)
  let weightsShape = switch weightsNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let filters = weightsShape->Array.get(3)->Option.getOr(1)
  let kH = weightsShape->Array.get(0)->Option.getOr(1)
  let kW = weightsShape->Array.get(1)->Option.getOr(1)

  let conv = addNode(graph, Conv2D({
    filters,
    kernel: (kH, kW),
    stride,
    padding,
    dilation: (1, 1),
    groups: 1
  }), [x, weights], None)

  switch bias {
  | Some(b) => add(graph, conv, b)
  | None => conv
  }
}

let conv2d = (
  graph: graphBuilder,
  x: int,
  filters: int,
  kernelSize: int,
  stride: int,
  padding: padding,
  name: string
): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let inChannels = inputShape->Array.get(3)->Option.getOr(1)

  let w = weight(graph, [kernelSize, kernelSize, inChannels, filters], name ++ "_weight")
  let b = weight(graph, [filters], name ++ "_bias")

  conv2dWithWeights(graph, x, w, Some(b), (stride, stride), padding)
}

// ============================================
// Pooling Operations
// ============================================

let maxPool2d = (graph: graphBuilder, x: int, size: int, stride: int): int =>
  addNode(graph, MaxPool2D({
    kernel: (size, size),
    stride: (stride, stride),
    padding: Valid
  }), [x], None)

let avgPool2d = (graph: graphBuilder, x: int, size: int, stride: int): int =>
  addNode(graph, AvgPool2D({
    kernel: (size, size),
    stride: (stride, stride),
    padding: Valid,
    countIncludePad: true
  }), [x], None)

let globalAvgPool = (graph: graphBuilder, x: int): int =>
  addNode(graph, GlobalAvgPool, [x], None)

let globalMaxPool = (graph: graphBuilder, x: int): int =>
  addNode(graph, GlobalMaxPool, [x], None)

// ============================================
// Normalization Operations
// ============================================

let batchNormWithParams = (
  graph: graphBuilder,
  x: int,
  gamma: int,
  beta: int,
  mean: int,
  variance: int,
  epsilon: float
): int => {
  addNode(graph, BatchNorm({epsilon, momentum: 0.1}), [x, gamma, beta, mean, variance], None)
}

let layerNorm = (graph: graphBuilder, x: int, axes: array<int>, epsilon: float): int =>
  addNode(graph, LayerNorm({axes, epsilon}), [x], None)

// ============================================
// Shape Operations
// ============================================

let reshape = (graph: graphBuilder, x: int, targetShape: shape): int =>
  addNode(graph, Reshape({newShape: targetShape}), [x], None)

let flatten = (graph: graphBuilder, x: int): int =>
  addNode(graph, Flatten({axis: 1}), [x], None)

let transpose = (graph: graphBuilder, x: int, permutation: array<int>): int =>
  addNode(graph, Transpose({perm: permutation}), [x], None)

// ============================================
// Multi-Output Operations
// ============================================

// TopK - returns (nodeId, valuesOutputIndex, indicesOutputIndex)
type topkResult = {
  nodeId: int,
  values: nodeRef,
  indices: nodeRef,
}

let topk = (graph: graphBuilder, x: int, k: int, axis: int): topkResult => {
  let nodeId = addNode(graph, TopK({k: k, axis: axis, largest: true, sorted: true}), [x], None)
  {
    nodeId,
    values: {nodeId, outputIndex: 0},
    indices: {nodeId, outputIndex: 1},
  }
}

// Convenience: just get values from topk
let topkValues = (graph: graphBuilder, x: int, k: int, axis: int): int => {
  let result = topk(graph, x, k, axis)
  result.nodeId  // Node ID, use result.values for explicit reference
}

// Split - returns array of nodeRefs
type splitResult = {
  nodeId: int,
  outputs: array<nodeRef>,
}

let split = (graph: graphBuilder, x: int, axis: int, splitSizes: array<int>): splitResult => {
  let nodeId = addNode(graph, Split({axis: axis, splitSizes: splitSizes}), [x], None)
  let outputs = Array.mapWithIndex(splitSizes, (_, i) => {nodeId, outputIndex: i})
  {nodeId, outputs}
}

// Convenience: split into equal chunks
let chunk = (graph: graphBuilder, x: int, axis: int, numChunks: int): splitResult => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let axisSize = inputShape->Array.get(axis)->Option.getOr(numChunks)
  let chunkSize = axisSize / numChunks
  let splitSizes = Array.make(~length=numChunks, chunkSize)
  split(graph, x, axis, splitSizes)
}

// ============================================
// Conditional Operations
// ============================================

let where = (graph: graphBuilder, condition: int, xTrue: int, xFalse: int): int =>
  addNode(graph, Where, [condition, xTrue, xFalse], None)

let gather = (graph: graphBuilder, data: int, indices: int, axis: int): int =>
  addNode(graph, Gather({axis: axis}), [data, indices], None)

let clip = (graph: graphBuilder, x: int, minVal: option<float>, maxVal: option<float>): int =>
  addNode(graph, Clip({min: minVal, max: maxVal}), [x], None)

let clamp = (graph: graphBuilder, x: int, minVal: float, maxVal: float): int =>
  clip(graph, x, Some(minVal), Some(maxVal))

// ============================================
// Topological Sort
// ============================================

let topologicalSort = (nodes: array<graphNode>): array<graphNode> => {
  let visitedIds = Dict.make()
  let resultArr = []
  
  let rec visit = (nodeId: int): unit => {
    let key = Int.toString(nodeId)
    if Dict.get(visitedIds, key)->Option.isNone {
      Dict.set(visitedIds, key, true)
      
      let nodeOpt = nodes->Array.find(n => n.id == nodeId)
      switch nodeOpt {
      | Some(node) => {
          Array.forEach(node.inputs, inputRef => visit(inputRef.nodeId))
          Array.push(resultArr, node)
        }
      | None => ()
      }
    }
  }
  
  Array.forEach(nodes, node => visit(node.id))
  resultArr
}

// ============================================
// Buffer Allocation
// ============================================

// Buffer info now tracks which output of a node it represents
type bufferInfo = {
  id: int,
  size: int,
  nodeId: int,
  outputIndex: int,  // Which output of the node
  kind: tensorKind,
  data: option<array<float>>,
}

let allocateBuffers = (
  sortedNodes: array<graphNode>,
  outputRefs: array<nodeRef>
): array<bufferInfo> => {
  let buffers: array<bufferInfo> = []
  let bufferId = {contents: 0}

  Array.forEach(sortedNodes, node => {
    // Create a buffer for each output of the node
    Array.forEachWithIndex(node.outputShapes, (shape, outputIndex) => {
      let size = Shape.numElements(shape) * 4

      let isGraphOutput = Array.some(outputRefs, outRef => 
        outRef.nodeId == node.id && outRef.outputIndex == outputIndex
      )

      let kind = if isGraphOutput { Output } else { node.kind }

      let buffer: bufferInfo = {
        id: bufferId.contents,
        size,
        nodeId: node.id,
        outputIndex,
        kind,
        data: if outputIndex == 0 { node.data } else { None },
      }

      Array.push(buffers, buffer)
      bufferId.contents = bufferId.contents + 1
    })
  })

  buffers
}

// ============================================
// Compiled Op with Multiple Outputs
// ============================================

type compiledOp = {
  nodeId: int,
  kernel: kernel,
  dispatch: dispatch,
  inputBufferIds: array<int>,
  outputBufferIds: array<int>,  // Changed: now an array for multi-output ops
}

// Find buffer ID for a nodeRef
let findBufferId = (buffers: array<bufferInfo>, nodeRef: nodeRef): int => {
  let buf = buffers->Array.find(b => 
    b.nodeId == nodeRef.nodeId && b.outputIndex == nodeRef.outputIndex
  )
  buf->Option.map(b => b.id)->Option.getOr(-1)
}

let compileNode = (
  node: graphNode,
  allNodes: array<graphNode>,
  buffers: array<bufferInfo>
): option<compiledOp> => {
  // Get input shapes
  let inputShapes = Array.map(node.inputs, inputRef => {
    let inputNode = allNodes->Array.find(n => n.id == inputRef.nodeId)
    switch inputNode {
    | Some(n) => n.outputShapes->Array.get(inputRef.outputIndex)->Option.getOr([])
    | None => []
    }
  })

  // Skip input/weight/constant nodes
  switch node.kind {
  | Input | Weight | Constant => None
  | Intermediate | Output => {
      let result = Codegen.generate(node.op, inputShapes)

      result->Option.map(((kernel, dispatch)) => {
        let inputBufferIds = Array.map(node.inputs, inputRef =>
          findBufferId(buffers, inputRef)
        )
        
        // Get output buffer IDs for all outputs of this node
        let outputBufferIds = Array.filterMap(
          Array.mapWithIndex(node.outputShapes, (_, i) => i),
          outputIndex => {
            let buf = buffers->Array.find(b => 
              b.nodeId == node.id && b.outputIndex == outputIndex
            )
            buf->Option.map(b => b.id)
          }
        )

        {
          nodeId: node.id,
          kernel,
          dispatch,
          inputBufferIds,
          outputBufferIds,
        }
      })
    }
  }
}

// ============================================
// Compiled Graph Types
// ============================================

type compiledGraph = {
  buffers: array<bufferInfo>,
  ops: array<compiledOp>,
  inputBufferIds: array<int>,
  weightBufferIds: array<int>,
  constantBufferIds: array<int>,
  outputBufferIds: array<int>,
  totalBufferSize: int,
  weightNames: array<string>,
  weightShapes: array<shape>,
}

// ============================================
// Main Compilation Entry Point
// ============================================

let compile = (graph: graphBuilder): option<compiledGraph> => {
  // Use marked outputs, or find leaf nodes
  let outputRefs = if Array.length(graph.outputIds) > 0 {
    graph.outputIds
  } else {
    // Find nodes that aren't inputs to any other node
    let usedAsInput = Belt.MutableSet.Int.make()
    Array.forEach(graph.nodes, node => {
      Array.forEach(node.inputs, inputRef => {
        Belt.MutableSet.Int.add(usedAsInput, inputRef.nodeId)
      })
    })
    Array.filterMap(graph.nodes, node => {
      if !Belt.MutableSet.Int.has(usedAsInput, node.id) && 
         node.kind != Input && node.kind != Weight && node.kind != Constant {
        Some(mkRef(node.id))  // Default to output 0
      } else {
        None
      }
    })
  }

  // 1. Topological sort
  let sorted = topologicalSort(graph.nodes)

  // 2. Allocate buffers (one per output of each node)
  let buffers = allocateBuffers(sorted, outputRefs)

  // 3. Compile each compute node
  let compiledOps = Array.filterMap(sorted, node =>
    compileNode(node, graph.nodes, buffers)
  )

  // 4. Categorize buffer IDs
  let inputBufferIds = Array.filterMap(buffers, buf =>
    buf.kind == Input && buf.outputIndex == 0 ? Some(buf.id) : None
  )
  let weightBufferIds = Array.filterMap(buffers, buf =>
    buf.kind == Weight && buf.outputIndex == 0 ? Some(buf.id) : None
  )
  let constantBufferIds = Array.filterMap(buffers, buf =>
    buf.kind == Constant && buf.outputIndex == 0 ? Some(buf.id) : None
  )
  let outputBufferIds = Array.filterMap(buffers, buf =>
    buf.kind == Output ? Some(buf.id) : None
  )

  // 5. Extract weight metadata
  let weightNodes = Array.filter(sorted, n => n.kind == Weight)
  let weightNames = Array.map(weightNodes, n => n.name->Option.getOr("unnamed"))
  let weightShapes = Array.map(weightNodes, n => 
    n.outputShapes->Array.get(0)->Option.getOr([])
  )

  // 6. Calculate total buffer size
  let totalBufferSize = Array.reduce(buffers, 0, (acc, buf) => acc + buf.size)

  Some({
    buffers,
    ops: compiledOps,
    inputBufferIds,
    weightBufferIds,
    constantBufferIds,
    outputBufferIds,
    totalBufferSize,
    weightNames,
    weightShapes,
  })
}

// ============================================
// Utility Functions
// ============================================

let compileWithOutputs = (graph: graphBuilder, outputIds: array<int>): option<compiledGraph> => {
  graph.outputIds = Array.map(outputIds, id => mkRef(id))
  compile(graph)
}

// ============================================
// Scalar Operations (for Transformer support)
// ============================================

let scale = (graph: graphBuilder, x: int, scalar: float): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let size = Shape.numElements(inputShape)
  let scalarData = Array.make(~length=size, scalar)
  let scalarTensor = constant(graph, inputShape, "scale_const", scalarData)
  mul(graph, x, scalarTensor)
}

let divByScalar = (graph: graphBuilder, x: int, scalar: float): int => {
  scale(graph, x, 1.0 /. scalar)
}

let addScalar = (graph: graphBuilder, x: int, scalar: float): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let size = Shape.numElements(inputShape)
  let scalarData = Array.make(~length=size, scalar)
  let scalarTensor = constant(graph, inputShape, "add_const", scalarData)
  add(graph, x, scalarTensor)
}

// ============================================
// Transformer Building Blocks
// ============================================
let embedding = (graph: graphBuilder, indices: int, weights: int): int => {
  let weightsNode = graph.nodes->Array.find(n => n.id == weights)
  let weightsShape = switch weightsNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let numEmbeddings = weightsShape->Array.get(0)->Option.getOr(0)
  let embeddingDim = weightsShape->Array.get(1)->Option.getOr(0)
  addNode(graph, Embedding({numEmbeddings, embeddingDim}), [indices, weights], None)
}

let layerNormWithParams = (
  graph: graphBuilder,
  x: int,
  gamma: int,
  beta: int,
  epsilon: float
): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let lastAxis = Array.length(inputShape) - 1
  addNode(graph, LayerNorm({axes: [lastAxis], epsilon}), [x, gamma, beta], None)
}

let concat = (graph: graphBuilder, inputs: array<int>, axis: int): int => {
  addNode(graph, Concat({axis: axis}), inputs, None)
}

let scaledDotProductAttention = (
  graph: graphBuilder,
  query: int,
  key: int,
  value: int,
  scaleFactor: float
): int => {
  let keyT = transpose(graph, key, [0, 2, 1])
  let scores = matmul(graph, query, keyT)
  let scaledScores = scale(graph, scores, scaleFactor)
  let attnWeights = softmax(graph, scaledScores, -1)
  matmul(graph, attnWeights, value)
}

let multiHeadAttention = (
  graph: graphBuilder,
  query: int,
  key: int,
  value: int,
  _numHeads: int,
  headDim: int
): int => {
  let scaleFactor = 1.0 /. Math.sqrt(Int.toFloat(headDim))
  scaledDotProductAttention(graph, query, key, value, scaleFactor)
}

let linear = (graph: graphBuilder, x: int, outFeatures: int, name: string): int => {
  dense(graph, x, outFeatures, name)
}

let feedForward = (
  graph: graphBuilder,
  x: int,
  hiddenDim: int,
  outDim: int,
  name: string
): int => {
  let h = linear(graph, x, hiddenDim, name ++ "_fc1")
  let a = gelu(graph, h)
  linear(graph, a, outDim, name ++ "_fc2")
}

let transformerBlock = (
  graph: graphBuilder,
  x: int,
  numHeads: int,
  headDim: int,
  ffnDim: int,
  name: string
): int => {
  let inputNode = graph.nodes->Array.find(n => n.id == x)
  let inputShape = switch inputNode {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let modelDim = inputShape->Array.get(Array.length(inputShape) - 1)->Option.getOr(64)

  let ln1Gamma = weight(graph, [modelDim], name ++ "_ln1_gamma")
  let ln1Beta = weight(graph, [modelDim], name ++ "_ln1_beta")
  let normed1 = layerNormWithParams(graph, x, ln1Gamma, ln1Beta, 1e-5)

  let q = linear(graph, normed1, numHeads * headDim, name ++ "_q_proj")
  let k = linear(graph, normed1, numHeads * headDim, name ++ "_k_proj")
  let v = linear(graph, normed1, numHeads * headDim, name ++ "_v_proj")

  let attnOut = multiHeadAttention(graph, q, k, v, numHeads, headDim)
  let attnProj = linear(graph, attnOut, modelDim, name ++ "_out_proj")

  let res1 = add(graph, x, attnProj)

  let ln2Gamma = weight(graph, [modelDim], name ++ "_ln2_gamma")
  let ln2Beta = weight(graph, [modelDim], name ++ "_ln2_beta")
  let normed2 = layerNormWithParams(graph, res1, ln2Gamma, ln2Beta, 1e-5)

  let ffnOut = feedForward(graph, normed2, ffnDim, modelDim, name ++ "_ffn")

  add(graph, res1, ffnOut)
}

// Causal mask
let makeCausalMaskData = (seqLen: int): array<float> => {
  let size = seqLen * seqLen
  let mask = Array.make(~length=size, 0.0)
  for i in 0 to seqLen - 1 {
    for j in 0 to seqLen - 1 {
      let idx = i * seqLen + j
      if j > i {
        Array.set(mask, idx, -1000000000.0)->ignore
      }
    }
  }
  mask
}

let causalMask = (graph: graphBuilder, seqLen: int): int => {
  let maskData = makeCausalMaskData(seqLen)
  constant(graph, [seqLen, seqLen], "causal_mask", maskData)
}

let maskedAttention = (
  graph: graphBuilder,
  query: int,
  key: int,
  value: int,
  mask: int,
  scaleFactor: float
): int => {
  let keyT = transpose(graph, key, [0, 2, 1])
  let scores = matmul(graph, query, keyT)
  let scaledScores = scale(graph, scores, scaleFactor)
  let maskedScores = add(graph, scaledScores, mask)
  let attnWeights = softmax(graph, maskedScores, -1)
  matmul(graph, attnWeights, value)
}

// Positional encoding
let sinusoidalPositionalEncoding = (graph: graphBuilder, seqLen: int, dim: int): int => {
  let posData = Array.fromInitializer(~length=seqLen * dim, i => {
    let pos = Float.fromInt(i / dim)
    let idx = mod(i, dim)
    let divTerm = Math.pow(10000.0, ~exp=Float.fromInt(idx / 2 * 2) /. Float.fromInt(dim))
    if mod(idx, 2) == 0 {
      Math.sin(pos /. divTerm)
    } else {
      Math.cos(pos /. divTerm)
    }
  })
  constant(graph, [seqLen, dim], "pos_encoding", posData)
}

// Convenience for argmax
let argmax = (graph: graphBuilder, x: int, axis: int): topkResult => {
  topk(graph, x, 1, axis)
}

// ArgMin - returns index of minimum value
let argmin = (graph: graphBuilder, x: int, axis: int): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let rank = Array.length(xShape)
  let normAxis = if axis < 0 { rank + axis } else { axis }
  addNode(graph, ArgMin({axis: normAxis, keepDims: false, selectLastIndex: false}), [x], None)
}

// Pad - pad tensor with constant value  
let pad = (graph: graphBuilder, x: int, pads: array<int>, constantValue: float): int => {
  addNode(graph, Pad({pads: pads, mode: Constant, constantValue: constantValue}), [x], None)
}

// Tile - repeat tensor along dimensions
let tile = (graph: graphBuilder, x: int, repeats: array<int>): int => {
  addNode(graph, Tile({repeats: repeats}), [x], None)
}

// Slice - extract a slice from tensor
let slice_ = (graph: graphBuilder, x: int, starts: array<int>, ends: array<int>, axes: array<int>, steps: array<int>): int => {
  addNode(graph, Slice({starts: starts, ends: ends, axes: axes, steps: steps}), [x], None)
}

// OneHot - convert indices to one-hot encoding
let oneHot = (graph: graphBuilder, x: int, depth: int): int => {
  addNode(graph, OneHot({depth, axis: -1}), [x], None)
}

// Scatter - scatter updates into tensor at indices
let scatter = (graph: graphBuilder, data: int, indices: int, updates: int, axis: int): int => {
  addNode(graph, Scatter({axis: axis}), [data, indices, updates], None)
}

// Cast - type conversion (f32 only for now)
let cast = (graph: graphBuilder, x: int): int => {
  addNode(graph, Cast({dtype: F32}), [x], None)
}

// CumSum - cumulative sum along axis
let cumsum = (graph: graphBuilder, x: int, axis: int): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let normAxis = if axis < 0 { Array.length(xShape) + axis } else { axis }
  addNode(graph, CumSum({axis: normAxis, exclusive: false, reverse: false}), [x], None)
}

// Squeeze - remove dimensions of size 1
let squeeze = (graph: graphBuilder, x: int, axes: array<int>): int => {
  addNode(graph, Squeeze({axes: axes}), [x], None)
}

// Unsqueeze - add dimensions of size 1
let unsqueeze = (graph: graphBuilder, x: int, axes: array<int>): int => {
  addNode(graph, Unsqueeze({axes: axes}), [x], None)
}

// ExpandDims - add a single dimension
let expandDims = (graph: graphBuilder, x: int, axis: int): int => {
  addNode(graph, ExpandDims({axis: axis}), [x], None)
}

// Broadcast - expand to target shape
let broadcast = (graph: graphBuilder, x: int, targetShape: array<int>): int => {
  addNode(graph, Broadcast({targetShape: targetShape}), [x], None)
}

// Stack - stack tensors along new axis
let stack = (graph: graphBuilder, inputs: array<int>, axis: int): int => {
  addNode(graph, Stack({axis: axis}), inputs, None)
}

// CumProd - cumulative product along axis
let cumprod = (graph: graphBuilder, x: int, axis: int): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let normAxis = if axis < 0 { Array.length(xShape) + axis } else { axis }
  addNode(graph, CumProd({axis: normAxis, exclusive: false, reverse: false}), [x], None)
}

// Reverse - reverse tensor along axes
let reverse = (graph: graphBuilder, x: int, axes: array<int>): int => {
  addNode(graph, Reverse({axes: axes}), [x], None)
}

// LogSoftmax - log of softmax (numerically stable)
let logSoftmax = (graph: graphBuilder, x: int, axis: int): int => {
  addNode(graph, LogSoftmax({axis: axis}), [x], None)
}

// Sort - sort tensor along axis
let sort = (graph: graphBuilder, x: int, axis: int, ~descending: bool=false): int => {
  addNode(graph, Sort({axis: axis, descending: descending}), [x], None)
}

// Arange - generate a sequence [start, start+step, start+2*step, ...]
let arange = (graph: graphBuilder, size: int, ~start: float=0.0, ~step: float=1.0): int => {
  let data = Array.fromInitializer(~length=size, i => start +. Float.fromInt(i) *. step)
  constant(graph, [size], "arange", data)
}

// ============================================
// RNN Operations
// ============================================

// LSTMCell - single timestep LSTM
// Inputs: x [batch, input_size], h_prev [batch, hidden], c_prev [batch, hidden]
// Outputs: h_new [batch, hidden], c_new [batch, hidden]
let lstmCell = (graph: graphBuilder, x: int, hPrev: int, cPrev: int, hiddenSize: int, name: string): lstmCellResult => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let inputSize = xShape[Array.length(xShape) - 1]->Option.getOr(1)
  
  let gateSize = 4 * hiddenSize
  let wIh = weight(graph, [gateSize, inputSize], name ++ "_weight_ih")
  let wHh = weight(graph, [gateSize, hiddenSize], name ++ "_weight_hh")
  let bIh = weight(graph, [gateSize], name ++ "_bias_ih")
  let bHh = weight(graph, [gateSize], name ++ "_bias_hh")
  
  // gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
  let wIhT = transpose(graph, wIh, [1, 0])
  let wHhT = transpose(graph, wHh, [1, 0])
  
  let xW = matmul(graph, x, wIhT)
  let hW = matmul(graph, hPrev, wHhT)
  let gates1 = add(graph, xW, hW)
  let gates2 = add(graph, gates1, bIh)
  let gates = add(graph, gates2, bHh)
  
  // Split into 4 gates
  let iGatePre = slice_(graph, gates, [0], [hiddenSize], [1], [1])
  let fGatePre = slice_(graph, gates, [hiddenSize], [2*hiddenSize], [1], [1])
  let gGatePre = slice_(graph, gates, [2*hiddenSize], [3*hiddenSize], [1], [1])
  let oGatePre = slice_(graph, gates, [3*hiddenSize], [4*hiddenSize], [1], [1])
  
  // Apply activations
  let iGate = sigmoid(graph, iGatePre)
  let fGate = sigmoid(graph, fGatePre)
  let gGate = tanh_(graph, gGatePre)
  let oGate = sigmoid(graph, oGatePre)
  
  // Cell state: c = f * c_prev + i * g
  let fc = mul(graph, fGate, cPrev)
  let ig = mul(graph, iGate, gGate)
  let cNew = add(graph, fc, ig)
  
  // Hidden state: h = o * tanh(c)
  let cTanh = tanh_(graph, cNew)
  let hNew = mul(graph, oGate, cTanh)
  
  {h: hNew, c: cNew}
}

// GRUCell - single timestep GRU
// Inputs: x [batch, input_size], h_prev [batch, hidden]
// Output: h_new [batch, hidden]
let gruCell = (graph: graphBuilder, x: int, hPrev: int, hiddenSize: int, name: string): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let inputSize = xShape[Array.length(xShape) - 1]->Option.getOr(1)
  
  let gateSize = 3 * hiddenSize
  let wIh = weight(graph, [gateSize, inputSize], name ++ "_weight_ih")
  let wHh = weight(graph, [gateSize, hiddenSize], name ++ "_weight_hh")
  let bIh = weight(graph, [gateSize], name ++ "_bias_ih")
  let bHh = weight(graph, [gateSize], name ++ "_bias_hh")
  
  let wIhT = transpose(graph, wIh, [1, 0])
  let wHhT = transpose(graph, wHh, [1, 0])
  
  // Compute all gates at once
  let xW = matmul(graph, x, wIhT)
  let hW = matmul(graph, hPrev, wHhT)
  
  // Split x*W into r, z, n parts
  let xR = slice_(graph, xW, [0], [hiddenSize], [1], [1])
  let xZ = slice_(graph, xW, [hiddenSize], [2*hiddenSize], [1], [1])
  let xN = slice_(graph, xW, [2*hiddenSize], [3*hiddenSize], [1], [1])
  
  // Split h*W into r, z, n parts
  let hR = slice_(graph, hW, [0], [hiddenSize], [1], [1])
  let hZ = slice_(graph, hW, [hiddenSize], [2*hiddenSize], [1], [1])
  let hN = slice_(graph, hW, [2*hiddenSize], [3*hiddenSize], [1], [1])
  
  // Split biases
  let bIhR = slice_(graph, bIh, [0], [hiddenSize], [0], [1])
  let bIhZ = slice_(graph, bIh, [hiddenSize], [2*hiddenSize], [0], [1])
  let bIhN = slice_(graph, bIh, [2*hiddenSize], [3*hiddenSize], [0], [1])
  let bHhR = slice_(graph, bHh, [0], [hiddenSize], [0], [1])
  let bHhZ = slice_(graph, bHh, [hiddenSize], [2*hiddenSize], [0], [1])
  let bHhN = slice_(graph, bHh, [2*hiddenSize], [3*hiddenSize], [0], [1])
  
  // r = sigmoid(x*W_r + h*W_r + b_r)
  let rPre1 = add(graph, xR, hR)
  let rPre2 = add(graph, rPre1, bIhR)
  let rPre3 = add(graph, rPre2, bHhR)
  let rGate = sigmoid(graph, rPre3)
  
  // z = sigmoid(x*W_z + h*W_z + b_z)
  let zPre1 = add(graph, xZ, hZ)
  let zPre2 = add(graph, zPre1, bIhZ)
  let zPre3 = add(graph, zPre2, bHhZ)
  let zGate = sigmoid(graph, zPre3)
  
  // n = tanh(x*W_n + b_ih_n + r*(h*W_n + b_hh_n))
  let hNBias = add(graph, hN, bHhN)
  let rH = mul(graph, rGate, hNBias)
  let nPre1 = add(graph, xN, bIhN)
  let nPre2 = add(graph, nPre1, rH)
  let nGate = tanh_(graph, nPre2)
  
  // h = (1 - z) * n + z * h_prev
  let ones = constant(graph, [hiddenSize], "ones", Array.fromInitializer(~length=hiddenSize, _ => 1.0))
  let oneMinusZ = sub(graph, ones, zGate)
  let term1 = mul(graph, oneMinusZ, nGate)
  let term2 = mul(graph, zGate, hPrev)
  add(graph, term1, term2)
}

// LSTM - process full sequence by unrolling
// Input: x [batch, seq_len, input_size]
// Returns: outputs [batch, seq_len, hidden_size]
let lstm = (graph: graphBuilder, x: int, hiddenSize: int, name: string): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let batchSize = xShape[0]->Option.getOr(1)
  let seqLen = xShape[1]->Option.getOr(1)
  let inputSize = xShape[2]->Option.getOr(1)
  
  // Initialize hidden and cell states to zero
  let h0 = constant(graph, [batchSize, hiddenSize], name ++ "_h0", 
    Array.fromInitializer(~length=batchSize * hiddenSize, _ => 0.0))
  let c0 = constant(graph, [batchSize, hiddenSize], name ++ "_c0",
    Array.fromInitializer(~length=batchSize * hiddenSize, _ => 0.0))
  
  // Process each timestep
  let outputs = ref([]: array<int>)
  let h = ref(h0)
  let c = ref(c0)
  
  for t in 0 to seqLen - 1 {
    // Extract timestep t: x[:, t, :]
    let xt = slice_(graph, x, [0, t, 0], [batchSize, t + 1, inputSize], [0, 1, 2], [1, 1, 1])
    let xtFlat = reshape(graph, xt, [batchSize, inputSize])
    
    // LSTM cell
    let result = lstmCell(graph, xtFlat, h.contents, c.contents, hiddenSize, name ++ "_t" ++ Int.toString(t))
    h := result.h
    c := result.c
    
    // Reshape h to [batch, 1, hidden] for concatenation
    let hExp = reshape(graph, result.h, [batchSize, 1, hiddenSize])
    outputs := Array.concat(outputs.contents, [hExp])
  }
  
  // Concatenate all outputs along seq dimension
  if Array.length(outputs.contents) == 1 {
    outputs.contents[0]->Option.getOr(h0)
  } else {
    let result = ref(outputs.contents[0]->Option.getOr(h0))
    for i in 1 to Array.length(outputs.contents) - 1 {
      result := concat(graph, [result.contents, outputs.contents[i]->Option.getOr(h0)], 1)
    }
    result.contents
  }
}

// GRU - process full sequence by unrolling
// Input: x [batch, seq_len, input_size]
// Returns: outputs [batch, seq_len, hidden_size]
let gru = (graph: graphBuilder, x: int, hiddenSize: int, name: string): int => {
  let xShape = switch graph.nodes->Array.find(n => n.id == x) {
  | Some(node) => node.outputShapes->Array.get(0)->Option.getOr([])
  | None => []
  }
  let batchSize = xShape[0]->Option.getOr(1)
  let seqLen = xShape[1]->Option.getOr(1)
  let inputSize = xShape[2]->Option.getOr(1)
  
  // Initialize hidden state to zero
  let h0 = constant(graph, [batchSize, hiddenSize], name ++ "_h0",
    Array.fromInitializer(~length=batchSize * hiddenSize, _ => 0.0))
  
  // Process each timestep
  let outputs = ref([]: array<int>)
  let h = ref(h0)
  
  for t in 0 to seqLen - 1 {
    // Extract timestep t
    let xt = slice_(graph, x, [0, t, 0], [batchSize, t + 1, inputSize], [0, 1, 2], [1, 1, 1])
    let xtFlat = reshape(graph, xt, [batchSize, inputSize])
    
    // GRU cell
    let hNew = gruCell(graph, xtFlat, h.contents, hiddenSize, name ++ "_t" ++ Int.toString(t))
    h := hNew
    
    // Reshape h to [batch, 1, hidden] for concatenation
    let hExp = reshape(graph, hNew, [batchSize, 1, hiddenSize])
    outputs := Array.concat(outputs.contents, [hExp])
  }
  
  // Concatenate all outputs
  if Array.length(outputs.contents) == 1 {
    outputs.contents[0]->Option.getOr(h0)
  } else {
    let result = ref(outputs.contents[0]->Option.getOr(h0))
    for i in 1 to Array.length(outputs.contents) - 1 {
      result := concat(graph, [result.contents, outputs.contents[i]->Option.getOr(h0)], 1)
    }
    result.contents
  }
}

// ============================================
// Debug / Print utilities
// ============================================

let tensorKindToString = (kind: tensorKind): string => {
  switch kind {
  | Input => "Input"
  | Weight => "Weight"
  | Constant => "Constant"
  | Intermediate => "Intermediate"
  | Output => "Output"
  }
}

let printGraph = (graph: graphBuilder): unit => {
  Console.log("=== Computation Graph ===")
  Console.log("Nodes: " ++ Int.toString(Array.length(graph.nodes)))

  Array.forEach(graph.nodes, node => {
    let name = node.name->Option.getOr("unnamed")
    let kind = tensorKindToString(node.kind)
    let shapesStr = Array.map(node.outputShapes, s => 
      "[" ++ Array.map(s, d => Int.toString(d))->Array.join(", ") ++ "]"
    )->Array.join(", ")
    let inputsStr = "[" ++ Array.map(node.inputs, r => 
      Int.toString(r.nodeId) ++ ":" ++ Int.toString(r.outputIndex)
    )->Array.join(", ") ++ "]"
    Console.log("  Node " ++ Int.toString(node.id) ++ " (" ++ kind ++ "): " ++ name ++ 
      " outputs=" ++ shapesStr ++ " inputs=" ++ inputsStr)
  })
}

let printCompiled = (compiled: compiledGraph): unit => {
  Console.log("\n=== Compiled Graph ===")
  Console.log("Buffers: " ++ Int.toString(Array.length(compiled.buffers)))
  Console.log("Ops: " ++ Int.toString(Array.length(compiled.ops)))
  Console.log("Total buffer size: " ++ Int.toString(compiled.totalBufferSize) ++ " bytes")

  Console.log("\nInputs: [" ++ Array.map(compiled.inputBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]")
  Console.log("Weights: [" ++ Array.map(compiled.weightBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]")
  Console.log("Constants: [" ++ Array.map(compiled.constantBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]")
  Console.log("Outputs: [" ++ Array.map(compiled.outputBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]")

  Console.log("\nWeight names: [" ++ Array.join(compiled.weightNames, ", ") ++ "]")

  Console.log("\nOps:")
  Array.forEach(compiled.ops, op => {
    let inputsStr = "[" ++ Array.map(op.inputBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]"
    let outputsStr = "[" ++ Array.map(op.outputBufferIds, i => Int.toString(i))->Array.join(", ") ++ "]"
    Console.log("  " ++ op.kernel.name ++ ": inputs=" ++ inputsStr ++ " outputs=" ++ outputsStr)
  })
}
