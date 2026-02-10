// src/Codegen.res
open Types

// ----------------------------------------
// WGSL Code Templates
// ----------------------------------------

let workgroupSize = 256

let binding = idx => `@group(0) @binding(${Int.toString(idx)})`

let storageBuffer = (idx, name, access) => {
  let accessMode = switch access {
  | ReadOnly => "read"
  | ReadWrite => "read_write"
  | Storage => "read_write"
  | Uniform => "read"
  }
  `${binding(idx)} var<storage, ${accessMode}> ${name}: array<f32>;`
}

let shaderHeader = (numBuffers) => {
  let buffers = Array.fromInitializer(~length=numBuffers, i => {
    let name = i == numBuffers - 1 ? "output" : "input" ++ Int.toString(i)
    let access = i == numBuffers - 1 ? ReadWrite : ReadOnly
    storageBuffer(i, name, access)
  })
  Array.join(buffers, "\n")
}

let uniformsStruct = (fields) => {
  `struct Uniforms {
${fields}
}
@group(0) @binding(0) var<uniform> uniforms: Uniforms;`
}

let mainSignature = `@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;`

let mainEnd = "}"

// ----------------------------------------
// Element-wise Unary Operations
// ----------------------------------------

let unaryExpr = (op: op, input: string): option<string> => {
  switch op {
  | Neg => Some(`-${input}`)
  | Abs => Some(`abs(${input})`)
  | Sign => Some(`sign(${input})`)
  | Reciprocal => Some(`1.0 / ${input}`)
  | Floor => Some(`floor(${input})`)
  | Ceil => Some(`ceil(${input})`)
  | Round => Some(`round(${input})`)
  | Sqrt => Some(`sqrt(${input})`)
  | Exp => Some(`exp(${input})`)
  | Log => Some(`log(${input})`)
  | Log2 => Some(`log2(${input})`)
  | Log10 => Some(`log(${input}) / 2.302585`)
  | Sin => Some(`sin(${input})`)
  | Cos => Some(`cos(${input})`)
  | Tan => Some(`tan(${input})`)
  | Asin => Some(`asin(${input})`)
  | Acos => Some(`acos(${input})`)
  | Atan => Some(`atan(${input})`)
  | Sinh => Some(`sinh(${input})`)
  | Cosh => Some(`cosh(${input})`)
  | Tanh => Some(`tanh(${input})`)
  | Asinh => Some(`asinh(${input})`)
  | Acosh => Some(`acosh(${input})`)
  | Atanh => Some(`atanh(${input})`)
  | ReLU => Some(`max(${input}, 0.0)`)
  | LeakyReLU({alpha}) => Some(`select(${Float.toString(alpha)} * ${input}, ${input}, ${input} > 0.0)`)
  | ELU({alpha}) => Some(`select(${Float.toString(alpha)} * (exp(${input}) - 1.0), ${input}, ${input} > 0.0)`)
  | Sigmoid => Some(`1.0 / (1.0 + exp(-${input}))`)
  | GeLU => Some(`0.5 * ${input} * (1.0 + tanh(0.7978845608 * (${input} + 0.044715 * ${input} * ${input} * ${input})))`)
  | SiLU => Some(`${input} / (1.0 + exp(-${input}))`)
  | Mish => Some(`${input} * tanh(log(1.0 + exp(${input})))`)
  | Softplus => Some(`log(1.0 + exp(${input}))`)
  | Softsign => Some(`${input} / (1.0 + abs(${input}))`)
  | Not => Some(`f32(${input} == 0.0)`)
  | Identity => Some(input)
  | _ => None
  }
}

// ----------------------------------------
// Element-wise Binary Operations
// ----------------------------------------

let binaryExpr = (op: op, a: string, b: string): option<string> => {
  switch op {
  | Add => Some(`${a} + ${b}`)
  | Sub => Some(`${a} - ${b}`)
  | Mul => Some(`${a} * ${b}`)
  | Div => Some(`${a} / ${b}`)
  | Pow => Some(`pow(${a}, ${b})`)
  | Mod => Some(`${a} % ${b}`)
  | FloorDiv => Some(`floor(${a} / ${b})`)
  | Maximum => Some(`max(${a}, ${b})`)
  | Minimum => Some(`min(${a}, ${b})`)
  | Atan2 => Some(`atan2(${a}, ${b})`)
  | Equal => Some(`f32(${a} == ${b})`)
  | NotEqual => Some(`f32(${a} != ${b})`)
  | Greater => Some(`f32(${a} > ${b})`)
  | GreaterEqual => Some(`f32(${a} >= ${b})`)
  | Less => Some(`f32(${a} < ${b})`)
  | LessEqual => Some(`f32(${a} <= ${b})`)
  | And => Some(`f32(${a} != 0.0 && ${b} != 0.0)`)
  | Or => Some(`f32(${a} != 0.0 || ${b} != 0.0)`)
  | Xor => Some(`f32((${a} != 0.0) != (${b} != 0.0))`)
  | _ => None
  }
}

// ----------------------------------------
// Reduction Operations
// ----------------------------------------

let reduceIdentity = (op: reduceOp): string => {
  switch op {
  | Sum | Mean | L1 | L2 | LogSum | LogSumExp | SumSquare => "0.0"
  | Prod => "1.0"
  | Max => "-3.402823e+38"
  | Min => "3.402823e+38"
  }
}

let reduceOp = (op: reduceOp, acc: string, val: string): string => {
  switch op {
  | Sum | Mean => `${acc} + ${val}`
  | Prod => `${acc} * ${val}`
  | Max => `max(${acc}, ${val})`
  | Min => `min(${acc}, ${val})`
  | L1 => `${acc} + abs(${val})`
  | L2 | SumSquare => `${acc} + ${val} * ${val}`
  | LogSum => `${acc} + ${val}`
  | LogSumExp => `${acc} + exp(${val})`
  }
}

let reduceFinalize = (op: reduceOp, acc: string, count: string): string => {
  switch op {
  | Mean => `${acc} / f32(${count})`
  | L2 => `sqrt(${acc})`
  | LogSum | LogSumExp => `log(${acc})`
  | _ => acc
  }
}

// Generate reduction kernel for arbitrary axes
let genReduceKernel = (reduceType: reduceOp, inputShape: shape, axes: array<int>, _keepDims: bool): option<kernel> => {
  let rank = Array.length(inputShape)
  if rank < 1 {
    None
  } else {
    // Normalize axes
    let normAxes = Array.map(axes, a => a < 0 ? rank + a : a)
    
    // Compute output shape and sizes
    let isReducedAxis = i => Array.includes(normAxes, i)
    
    // Calculate strides for input
    let inputStrides = Array.fromInitializer(~length=rank, i => {
      let stride = ref(1)
      for j in i + 1 to rank - 1 {
        stride := stride.contents * (inputShape[j]->Option.getOr(1))
      }
      stride.contents
    })
    
    let inputSize = Shape.numElements(inputShape)
    let reduceSize = Array.reduceWithIndex(inputShape, 1, (acc, dim, i) => 
      isReducedAxis(i) ? acc * dim : acc
    )
    let outputSize = inputSize / reduceSize
    
    let shapeStr = Array.map(inputShape, d => Int.toString(d))->Array.join(", ")
    let stridesStr = Array.map(inputStrides, d => Int.toString(d))->Array.join(", ")
    let axesStr = Array.map(normAxes, d => Int.toString(d))->Array.join(", ")
    let numAxes = Array.length(normAxes)
    
    let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const RANK = ${Int.toString(rank)}u;
const INPUT_SIZE = ${Int.toString(inputSize)}u;
const OUTPUT_SIZE = ${Int.toString(outputSize)}u;
const REDUCE_SIZE = ${Int.toString(reduceSize)}u;
const NUM_AXES = ${Int.toString(numAxes)}u;
const SHAPE = array<u32, ${Int.toString(rank)}>(${shapeStr});
const STRIDES = array<u32, ${Int.toString(rank)}>(${stridesStr});
const AXES = array<u32, ${Int.toString(numAxes)}>(${axesStr});

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
  if (outIdx >= OUTPUT_SIZE) { return; }
  
  // Convert output index to output coordinates (skipping reduced axes)
  var outCoords: array<u32, ${Int.toString(rank)}>;
  var remaining = outIdx;
  var outDimIdx = 0u;
  for (var i = 0u; i < RANK; i = i + 1u) {
    if (!isReduceAxis(i)) {
      var outDimSize = 1u;
      for (var j = i + 1u; j < RANK; j = j + 1u) {
        if (!isReduceAxis(j)) {
          outDimSize = outDimSize * SHAPE[j];
        }
      }
      if (outDimSize > 0u) {
        outCoords[i] = remaining / outDimSize;
        remaining = remaining % outDimSize;
      }
    } else {
      outCoords[i] = 0u;
    }
  }
  
  // Iterate over all reduction indices
  var acc = ${reduceIdentity(reduceType)};
  var count = 0u;
  
  for (var reduceIdx = 0u; reduceIdx < REDUCE_SIZE; reduceIdx = reduceIdx + 1u) {
    // Convert reduceIdx to coordinates for reduced axes
    var coords = outCoords;
    var rem = reduceIdx;
    for (var i = 0u; i < RANK; i = i + 1u) {
      let ri = RANK - 1u - i;
      if (isReduceAxis(ri)) {
        coords[ri] = rem % SHAPE[ri];
        rem = rem / SHAPE[ri];
      }
    }
    
    // Compute input index
    var inIdx = 0u;
    for (var i = 0u; i < RANK; i = i + 1u) {
      inIdx = inIdx + coords[i] * STRIDES[i];
    }
    
    let val = input[inIdx];
    acc = ${reduceOp(reduceType, "acc", "val")};
    count = count + 1u;
  }
  
  output[outIdx] = ${reduceFinalize(reduceType, "acc", "count")};
}`

    Some({
      name: "reduce_" ++ Int.toString(outputSize),
      wgsl,
      bindings: [
        {binding: 0, size: inputSize * 4, usage: ReadOnly, name: "input"},
        {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
      ]
    })
  }
}

// ----------------------------------------
// Optimized MatMul for small M (M <= 16)
// No tiling overhead - each thread computes one output element
// ----------------------------------------
let genSmallMMatMulKernel = (m: int, k: int, n: int): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let total = M * N;
  if (idx >= total) { return; }
  
  let row = idx / N;
  let col = idx % N;
  
  var sum = 0.0;
  for (var i = 0u; i < K; i = i + 1u) {
    sum = sum + a[row * K + i] * b[i * N + col];
  }
  output[idx] = sum;
}`
  {
    name: "matmul_small_" ++ Int.toString(m) ++ "x" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 1, size: k * n * 4, usage: ReadOnly, name: "b"},
      {binding: 2, size: m * n * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// INT4 Quantized MatMul - Optimized Version
// Achieves ~103% memory bandwidth efficiency
// 
// Key optimizations:
// 1. Column-major weight layout for coalesced memory access
// 2. Fully unrolled inner loop (128 weights per group)
// 3. Tuned workgroup size (64, adjustable)
// 4. Scale applied once per group, not per weight
// 5. vec4<f32> loads + dot() for 4 MACs per instruction
// ----------------------------------------

let genInt4MatMulKernel = (m: int, k: int, n: int, groupSize: int): kernel => {
  // Validate groupSize is 128 (required for this unrolled version)
  let actualGroupSize = groupSize == 128 ? 128 : 128
  let numGroups = (k + actualGroupSize - 1) / actualGroupSize
  let packedPerGroup = actualGroupSize / 8  // 16 packed u32 per group
  
  // Tuned workgroup size - 64 is a good default, but 48 may be better on some GPUs
  let wgSize = 64
  
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const N = ${Int.toString(n)}u;
const NUM_GROUPS = ${Int.toString(numGroups)}u;
const PACKED_PER_GROUP = ${Int.toString(packedPerGroup)}u;

@compute @workgroup_size(${Int.toString(wgSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= N) { return; }
  
  var sum = 0.0;
  
  for (var g = 0u; g < NUM_GROUPS; g++) {
    // Load scale once per group (not per weight)
    let s = scales[g * N + col];
    let base = g * PACKED_PER_GROUP;
    var acc = 0.0;
    
    // ========================================
    // First half of group: 64 weights (8 packed u32)
    // ========================================
    // Coalesced loads: adjacent threads read adjacent memory
    let p0 = b_packed[base * N + col];
    let p1 = b_packed[(base + 1u) * N + col];
    let p2 = b_packed[(base + 2u) * N + col];
    let p3 = b_packed[(base + 3u) * N + col];
    let p4 = b_packed[(base + 4u) * N + col];
    let p5 = b_packed[(base + 5u) * N + col];
    let p6 = b_packed[(base + 6u) * N + col];
    let p7 = b_packed[(base + 7u) * N + col];
    
    // Load activations as vec4 for efficient dot products
    let k0 = base * 8u;
    let a0 = a[k0/4u]; let a1 = a[k0/4u+1u]; let a2 = a[k0/4u+2u]; let a3 = a[k0/4u+3u];
    let a4 = a[k0/4u+4u]; let a5 = a[k0/4u+5u]; let a6 = a[k0/4u+6u]; let a7 = a[k0/4u+7u];
    let a8 = a[k0/4u+8u]; let a9 = a[k0/4u+9u]; let a10 = a[k0/4u+10u]; let a11 = a[k0/4u+11u];
    let a12 = a[k0/4u+12u]; let a13 = a[k0/4u+13u]; let a14 = a[k0/4u+14u]; let a15 = a[k0/4u+15u];
    
    // Unrolled dot products - dequantize inline (subtract 8 for signed int4)
    acc += dot(a0, vec4<f32>(f32((p0>>0u)&0xFu)-8.0, f32((p0>>4u)&0xFu)-8.0, f32((p0>>8u)&0xFu)-8.0, f32((p0>>12u)&0xFu)-8.0));
    acc += dot(a1, vec4<f32>(f32((p0>>16u)&0xFu)-8.0, f32((p0>>20u)&0xFu)-8.0, f32((p0>>24u)&0xFu)-8.0, f32((p0>>28u)&0xFu)-8.0));
    acc += dot(a2, vec4<f32>(f32((p1>>0u)&0xFu)-8.0, f32((p1>>4u)&0xFu)-8.0, f32((p1>>8u)&0xFu)-8.0, f32((p1>>12u)&0xFu)-8.0));
    acc += dot(a3, vec4<f32>(f32((p1>>16u)&0xFu)-8.0, f32((p1>>20u)&0xFu)-8.0, f32((p1>>24u)&0xFu)-8.0, f32((p1>>28u)&0xFu)-8.0));
    acc += dot(a4, vec4<f32>(f32((p2>>0u)&0xFu)-8.0, f32((p2>>4u)&0xFu)-8.0, f32((p2>>8u)&0xFu)-8.0, f32((p2>>12u)&0xFu)-8.0));
    acc += dot(a5, vec4<f32>(f32((p2>>16u)&0xFu)-8.0, f32((p2>>20u)&0xFu)-8.0, f32((p2>>24u)&0xFu)-8.0, f32((p2>>28u)&0xFu)-8.0));
    acc += dot(a6, vec4<f32>(f32((p3>>0u)&0xFu)-8.0, f32((p3>>4u)&0xFu)-8.0, f32((p3>>8u)&0xFu)-8.0, f32((p3>>12u)&0xFu)-8.0));
    acc += dot(a7, vec4<f32>(f32((p3>>16u)&0xFu)-8.0, f32((p3>>20u)&0xFu)-8.0, f32((p3>>24u)&0xFu)-8.0, f32((p3>>28u)&0xFu)-8.0));
    acc += dot(a8, vec4<f32>(f32((p4>>0u)&0xFu)-8.0, f32((p4>>4u)&0xFu)-8.0, f32((p4>>8u)&0xFu)-8.0, f32((p4>>12u)&0xFu)-8.0));
    acc += dot(a9, vec4<f32>(f32((p4>>16u)&0xFu)-8.0, f32((p4>>20u)&0xFu)-8.0, f32((p4>>24u)&0xFu)-8.0, f32((p4>>28u)&0xFu)-8.0));
    acc += dot(a10, vec4<f32>(f32((p5>>0u)&0xFu)-8.0, f32((p5>>4u)&0xFu)-8.0, f32((p5>>8u)&0xFu)-8.0, f32((p5>>12u)&0xFu)-8.0));
    acc += dot(a11, vec4<f32>(f32((p5>>16u)&0xFu)-8.0, f32((p5>>20u)&0xFu)-8.0, f32((p5>>24u)&0xFu)-8.0, f32((p5>>28u)&0xFu)-8.0));
    acc += dot(a12, vec4<f32>(f32((p6>>0u)&0xFu)-8.0, f32((p6>>4u)&0xFu)-8.0, f32((p6>>8u)&0xFu)-8.0, f32((p6>>12u)&0xFu)-8.0));
    acc += dot(a13, vec4<f32>(f32((p6>>16u)&0xFu)-8.0, f32((p6>>20u)&0xFu)-8.0, f32((p6>>24u)&0xFu)-8.0, f32((p6>>28u)&0xFu)-8.0));
    acc += dot(a14, vec4<f32>(f32((p7>>0u)&0xFu)-8.0, f32((p7>>4u)&0xFu)-8.0, f32((p7>>8u)&0xFu)-8.0, f32((p7>>12u)&0xFu)-8.0));
    acc += dot(a15, vec4<f32>(f32((p7>>16u)&0xFu)-8.0, f32((p7>>20u)&0xFu)-8.0, f32((p7>>24u)&0xFu)-8.0, f32((p7>>28u)&0xFu)-8.0));
    
    // ========================================
    // Second half of group: 64 weights (8 packed u32)
    // ========================================
    let q0 = b_packed[(base + 8u) * N + col];
    let q1 = b_packed[(base + 9u) * N + col];
    let q2 = b_packed[(base + 10u) * N + col];
    let q3 = b_packed[(base + 11u) * N + col];
    let q4 = b_packed[(base + 12u) * N + col];
    let q5 = b_packed[(base + 13u) * N + col];
    let q6 = b_packed[(base + 14u) * N + col];
    let q7 = b_packed[(base + 15u) * N + col];
    
    let k1 = (base + 8u) * 8u;
    let b0 = a[k1/4u]; let b1 = a[k1/4u+1u]; let b2 = a[k1/4u+2u]; let b3 = a[k1/4u+3u];
    let b4 = a[k1/4u+4u]; let b5 = a[k1/4u+5u]; let b6 = a[k1/4u+6u]; let b7 = a[k1/4u+7u];
    let b8 = a[k1/4u+8u]; let b9 = a[k1/4u+9u]; let b10 = a[k1/4u+10u]; let b11 = a[k1/4u+11u];
    let b12 = a[k1/4u+12u]; let b13 = a[k1/4u+13u]; let b14 = a[k1/4u+14u]; let b15 = a[k1/4u+15u];
    
    acc += dot(b0, vec4<f32>(f32((q0>>0u)&0xFu)-8.0, f32((q0>>4u)&0xFu)-8.0, f32((q0>>8u)&0xFu)-8.0, f32((q0>>12u)&0xFu)-8.0));
    acc += dot(b1, vec4<f32>(f32((q0>>16u)&0xFu)-8.0, f32((q0>>20u)&0xFu)-8.0, f32((q0>>24u)&0xFu)-8.0, f32((q0>>28u)&0xFu)-8.0));
    acc += dot(b2, vec4<f32>(f32((q1>>0u)&0xFu)-8.0, f32((q1>>4u)&0xFu)-8.0, f32((q1>>8u)&0xFu)-8.0, f32((q1>>12u)&0xFu)-8.0));
    acc += dot(b3, vec4<f32>(f32((q1>>16u)&0xFu)-8.0, f32((q1>>20u)&0xFu)-8.0, f32((q1>>24u)&0xFu)-8.0, f32((q1>>28u)&0xFu)-8.0));
    acc += dot(b4, vec4<f32>(f32((q2>>0u)&0xFu)-8.0, f32((q2>>4u)&0xFu)-8.0, f32((q2>>8u)&0xFu)-8.0, f32((q2>>12u)&0xFu)-8.0));
    acc += dot(b5, vec4<f32>(f32((q2>>16u)&0xFu)-8.0, f32((q2>>20u)&0xFu)-8.0, f32((q2>>24u)&0xFu)-8.0, f32((q2>>28u)&0xFu)-8.0));
    acc += dot(b6, vec4<f32>(f32((q3>>0u)&0xFu)-8.0, f32((q3>>4u)&0xFu)-8.0, f32((q3>>8u)&0xFu)-8.0, f32((q3>>12u)&0xFu)-8.0));
    acc += dot(b7, vec4<f32>(f32((q3>>16u)&0xFu)-8.0, f32((q3>>20u)&0xFu)-8.0, f32((q3>>24u)&0xFu)-8.0, f32((q3>>28u)&0xFu)-8.0));
    acc += dot(b8, vec4<f32>(f32((q4>>0u)&0xFu)-8.0, f32((q4>>4u)&0xFu)-8.0, f32((q4>>8u)&0xFu)-8.0, f32((q4>>12u)&0xFu)-8.0));
    acc += dot(b9, vec4<f32>(f32((q4>>16u)&0xFu)-8.0, f32((q4>>20u)&0xFu)-8.0, f32((q4>>24u)&0xFu)-8.0, f32((q4>>28u)&0xFu)-8.0));
    acc += dot(b10, vec4<f32>(f32((q5>>0u)&0xFu)-8.0, f32((q5>>4u)&0xFu)-8.0, f32((q5>>8u)&0xFu)-8.0, f32((q5>>12u)&0xFu)-8.0));
    acc += dot(b11, vec4<f32>(f32((q5>>16u)&0xFu)-8.0, f32((q5>>20u)&0xFu)-8.0, f32((q5>>24u)&0xFu)-8.0, f32((q5>>28u)&0xFu)-8.0));
    acc += dot(b12, vec4<f32>(f32((q6>>0u)&0xFu)-8.0, f32((q6>>4u)&0xFu)-8.0, f32((q6>>8u)&0xFu)-8.0, f32((q6>>12u)&0xFu)-8.0));
    acc += dot(b13, vec4<f32>(f32((q6>>16u)&0xFu)-8.0, f32((q6>>20u)&0xFu)-8.0, f32((q6>>24u)&0xFu)-8.0, f32((q6>>28u)&0xFu)-8.0));
    acc += dot(b14, vec4<f32>(f32((q7>>0u)&0xFu)-8.0, f32((q7>>4u)&0xFu)-8.0, f32((q7>>8u)&0xFu)-8.0, f32((q7>>12u)&0xFu)-8.0));
    acc += dot(b15, vec4<f32>(f32((q7>>16u)&0xFu)-8.0, f32((q7>>20u)&0xFu)-8.0, f32((q7>>24u)&0xFu)-8.0, f32((q7>>28u)&0xFu)-8.0));
    
    // Apply scale once per group (not per weight)
    sum += acc * s;
  }
  
  output[col] = sum;
}`

  {
    name: "matmul_int4_opt_" ++ Int.toString(m) ++ "x" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      // Input activations: [K] as vec4<f32> (for M=1 inference)
      {binding: 0, size: k * 4, usage: ReadOnly, name: "a"},
      // Packed weights: [PACKED_PER_GROUP * NUM_GROUPS, N] column-major for coalescing
      {binding: 1, size: numGroups * packedPerGroup * n * 4, usage: ReadOnly, name: "b_packed"},
      // Scales: [NUM_GROUPS, N] column-major
      {binding: 2, size: numGroups * n * 4, usage: ReadOnly, name: "scales"},
      // Output: [N]
      {binding: 3, size: n * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// Dispatch helper for INT4 optimized kernel
let computeDispatchInt4Opt = (n: int, wgSize: int, kernelName: string, pipelineIndex: int): dispatch => {
  let workgroupCount = (n + wgSize - 1) / wgSize
  {
    workgroupSize: (wgSize, 1, 1),
    workgroupCount: (workgroupCount, 1, 1),
    kernelName,
    pipelineIndex,
  }
}

// ----------------------------------------
// INT4 Quantized MatMul - Tiled version for large M (M > 16)
// Uses 16x16 tiles with shared memory
// ----------------------------------------
let genInt4MatMulTiledKernel = (m: int, k: int, n: int, groupSize: int): kernel => {
  let tileSize = 16
  let numGroups = (k + groupSize - 1) / groupSize
  let packedK = (k + 7) / 8
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b_packed: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
const TILE = ${Int.toString(tileSize)}u;
const GROUP_SIZE = ${Int.toString(groupSize)}u;
const NUM_GROUPS = ${Int.toString(numGroups)}u;
const PACKED_K = ${Int.toString(packedK)}u;

var<workgroup> tileA: array<f32, ${Int.toString(tileSize * tileSize)}>;
var<workgroup> tileB: array<f32, ${Int.toString(tileSize * tileSize)}>;

fn dequantize(col: u32, k_idx: u32) -> f32 {
  let packed_idx = k_idx / 8u;
  let sub_idx = k_idx % 8u;
  let packed = b_packed[col * PACKED_K + packed_idx];
  let shift = sub_idx * 4u;
  let int4_val = (packed >> shift) & 0xFu;
  let group_idx = k_idx / GROUP_SIZE;
  let scale = scales[col * NUM_GROUPS + group_idx];
  return (f32(int4_val) - 8.0) * scale;
}

@compute @workgroup_size(${Int.toString(tileSize)}, ${Int.toString(tileSize)})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  
  var sum = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;
  
  for (var t = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A (activations - FP32)
    let aRow = wid.y * TILE + localRow;
    let aCol = t * TILE + localCol;
    if (aRow < M && aCol < K) {
      tileA[localRow * TILE + localCol] = a[aRow * K + aCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    
    // Load tile of B (dequantize INT4 on the fly)
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = dequantize(bCol, bRow);
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    
    workgroupBarrier();
  }
  
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}`
  {
    name: "matmul_int4_tiled_" ++ Int.toString(m) ++ "x" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 1, size: n * packedK * 4, usage: ReadOnly, name: "b_packed"},
      {binding: 2, size: n * numGroups * 4, usage: ReadOnly, name: "scales"},
      {binding: 3, size: m * n * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Matrix Multiplication
// ----------------------------------------

let genMatMulKernel = (m: int, k: int, n: int): kernel => {
  let tileSize = 16
  
  let wgsl = `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
const TILE = ${Int.toString(tileSize)}u;

var<workgroup> tileA: array<f32, ${Int.toString(tileSize * tileSize)}>;
var<workgroup> tileB: array<f32, ${Int.toString(tileSize * tileSize)}>;

@compute @workgroup_size(${Int.toString(tileSize)}, ${Int.toString(tileSize)})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  
  var sum = 0.0;
  
  let numTiles = (K + TILE - 1u) / TILE;
  
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let tiledRow = wid.y * TILE + localRow;
    let tiledCol = t * TILE + localCol;
    
    if (tiledRow < M && tiledCol < K) {
      tileA[localRow * TILE + localCol] = a[tiledRow * K + tiledCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = b[bRow * N + bCol];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    
    workgroupBarrier();
    
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    
    workgroupBarrier();
  }
  
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}`

  {
    name: "matmul_" ++ Int.toString(m) ++ "x" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 1, size: k * n * 4, usage: ReadOnly, name: "b"},
      {binding: 2, size: m * n * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Batched MatMul with broadcasting support
// Supports arbitrary batch dimensions: [...batchA, M, K] @ [...batchB, K, N] -> [...batchOut, M, N]
// ----------------------------------------
let genBatchedMatMulKernel = (
  batchShapeA: array<int>,  // Batch dims for A (excluding M,K)
  batchShapeB: array<int>,  // Batch dims for B (excluding K,N)
  batchShapeOut: array<int>, // Output batch dims (broadcast result)
  m: int, k: int, n: int
): kernel => {
  let batchSizeA = Array.reduce(batchShapeA, 1, (a, b) => a * b)
  let batchSizeB = Array.reduce(batchShapeB, 1, (a, b) => a * b)
  let batchSizeOut = Array.reduce(batchShapeOut, 1, (a, b) => a * b)
  let batchRank = Array.length(batchShapeOut)
  
  // Calculate strides for batch dimensions
  let outBatchStrides = Array.fromInitializer(~length=batchRank, i => {
    let stride = ref(1)
    for j in i + 1 to batchRank - 1 {
      stride := stride.contents * batchShapeOut[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Calculate strides for A's batch dims (with broadcasting)
  let rankA = Array.length(batchShapeA)
  let aBatchStrides = Array.fromInitializer(~length=batchRank, i => {
    let aIdx = i - (batchRank - rankA)
    if aIdx < 0 {
      0  // Broadcast: this dim doesn't exist in A
    } else {
      let aDim = batchShapeA[aIdx]->Option.getOr(1)
      let outDim = batchShapeOut[i]->Option.getOr(1)
      if aDim == 1 && outDim > 1 {
        0  // Broadcast: size 1 -> size N
      } else {
        let stride = ref(1)
        for j in aIdx + 1 to rankA - 1 {
          stride := stride.contents * batchShapeA[j]->Option.getOr(1)
        }
        stride.contents
      }
    }
  })
  
  // Calculate strides for B's batch dims (with broadcasting)
  let rankB = Array.length(batchShapeB)
  let bBatchStrides = Array.fromInitializer(~length=batchRank, i => {
    let bIdx = i - (batchRank - rankB)
    if bIdx < 0 {
      0
    } else {
      let bDim = batchShapeB[bIdx]->Option.getOr(1)
      let outDim = batchShapeOut[i]->Option.getOr(1)
      if bDim == 1 && outDim > 1 {
        0
      } else {
        let stride = ref(1)
        for j in bIdx + 1 to rankB - 1 {
          stride := stride.contents * batchShapeB[j]->Option.getOr(1)
        }
        stride.contents
      }
    }
  })
  
  let outBatchStridesStr = Array.map(outBatchStrides, d => Int.toString(d))->Array.join(", ")
  let aBatchStridesStr = Array.map(aBatchStrides, d => Int.toString(d))->Array.join(", ")
  let bBatchStridesStr = Array.map(bBatchStrides, d => Int.toString(d))->Array.join(", ")
  
  let totalOutput = batchSizeOut * m * n
  
  let wgsl = if batchSizeOut == 1 {
    // No batch dimension - use optimized tiled kernel
    let tileSize = 16
    `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
const TILE = ${Int.toString(tileSize)}u;
var<workgroup> tileA: array<f32, ${Int.toString(tileSize * tileSize)}>;
var<workgroup> tileB: array<f32, ${Int.toString(tileSize * tileSize)}>;
@compute @workgroup_size(${Int.toString(tileSize)}, ${Int.toString(tileSize)})
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>
) {
  let row = gid.y;
  let col = gid.x;
  let localRow = lid.y;
  let localCol = lid.x;
  var sum = 0.0;
  let numTiles = (K + TILE - 1u) / TILE;
  for (var t = 0u; t < numTiles; t = t + 1u) {
    let tiledRow = wid.y * TILE + localRow;
    let tiledCol = t * TILE + localCol;
    if (tiledRow < M && tiledCol < K) {
      tileA[localRow * TILE + localCol] = a[tiledRow * K + tiledCol];
    } else {
      tileA[localRow * TILE + localCol] = 0.0;
    }
    let bRow = t * TILE + localRow;
    let bCol = wid.x * TILE + localCol;
    if (bRow < K && bCol < N) {
      tileB[localRow * TILE + localCol] = b[bRow * N + bCol];
    } else {
      tileB[localRow * TILE + localCol] = 0.0;
    }
    workgroupBarrier();
    for (var i = 0u; i < TILE; i = i + 1u) {
      sum = sum + tileA[localRow * TILE + i] * tileB[i * TILE + localCol];
    }
    workgroupBarrier();
  }
  if (row < M && col < N) {
    output[row * N + col] = sum;
  }
}`
  } else {
    // Batched version - simpler kernel, iterate over batch
    `@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
const BATCH_OUT = ${Int.toString(batchSizeOut)}u;
const BATCH_RANK = ${Int.toString(batchRank)}u;
const OUT_BATCH_STRIDES = array<u32, ${Int.toString(max(batchRank, 1))}>(${if batchRank > 0 { outBatchStridesStr } else { "0" }});
const A_BATCH_STRIDES = array<u32, ${Int.toString(max(batchRank, 1))}>(${if batchRank > 0 { aBatchStridesStr } else { "0" }});
const B_BATCH_STRIDES = array<u32, ${Int.toString(max(batchRank, 1))}>(${if batchRank > 0 { bBatchStridesStr } else { "0" }});
@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${Int.toString(totalOutput)}u) { return; }
  
  // Decompose idx into batch_idx, row, col
  let matrix_size = M * N;
  let batch_idx = idx / matrix_size;
  let matrix_idx = idx % matrix_size;
  let row = matrix_idx / N;
  let col = matrix_idx % N;
  
  // Compute batch indices for A and B using broadcasting strides
  var a_batch_idx = 0u;
  var b_batch_idx = 0u;
  var remaining = batch_idx;
  
  for (var d = 0u; d < BATCH_RANK; d = d + 1u) {
    let coord = remaining / OUT_BATCH_STRIDES[d];
    remaining = remaining % OUT_BATCH_STRIDES[d];
    a_batch_idx = a_batch_idx + coord * A_BATCH_STRIDES[d];
    b_batch_idx = b_batch_idx + coord * B_BATCH_STRIDES[d];
  }
  
  // Compute matmul for this element
  var sum = 0.0;
  let a_base = a_batch_idx * M * K;
  let b_base = b_batch_idx * K * N;
  
  for (var i = 0u; i < K; i = i + 1u) {
    sum = sum + a[a_base + row * K + i] * b[b_base + i * N + col];
  }
  
  output[idx] = sum;
}`
  }
  
  {
    name: "batched_matmul_" ++ Int.toString(batchSizeOut) ++ "_" ++ Int.toString(m) ++ "x" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: batchSizeA * m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 1, size: batchSizeB * k * n * 4, usage: ReadOnly, name: "b"},
      {binding: 2, size: batchSizeOut * m * n * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Conv2D (direct convolution)
// ----------------------------------------

let genConv2DKernel = (
  batch: int, inH: int, inW: int, inC: int,
  outH: int, outW: int, outC: int,
  kH: int, kW: int,
  strideH: int, strideW: int,
  padH: int, padW: int
): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batch)}u;
const IN_H = ${Int.toString(inH)}u;
const IN_W = ${Int.toString(inW)}u;
const IN_C = ${Int.toString(inC)}u;
const OUT_H = ${Int.toString(outH)}u;
const OUT_W = ${Int.toString(outW)}u;
const OUT_C = ${Int.toString(outC)}u;
const K_H = ${Int.toString(kH)}u;
const K_W = ${Int.toString(kW)}u;
const STRIDE_H = ${Int.toString(strideH)}u;
const STRIDE_W = ${Int.toString(strideW)}u;
const PAD_H = ${Int.toString(padH)}u;
const PAD_W = ${Int.toString(padW)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_H * OUT_W * OUT_C;
  if (idx >= totalOut) { return; }
  
  let oc = idx % OUT_C;
  let tmp1 = idx / OUT_C;
  let ow = tmp1 % OUT_W;
  let tmp2 = tmp1 / OUT_W;
  let oh = tmp2 % OUT_H;
  let b = tmp2 / OUT_H;
  
  var sum = bias[oc];
  
  for (var ic = 0u; ic < IN_C; ic = ic + 1u) {
    for (var kh = 0u; kh < K_H; kh = kh + 1u) {
      for (var kw = 0u; kw < K_W; kw = kw + 1u) {
        let ih = oh * STRIDE_H + kh - PAD_H;
        let iw = ow * STRIDE_W + kw - PAD_W;
        
        if (ih < IN_H && iw < IN_W) {
          let inIdx = b * IN_H * IN_W * IN_C + ih * IN_W * IN_C + iw * IN_C + ic;
          let wIdx = oc * K_H * K_W * IN_C + kh * K_W * IN_C + kw * IN_C + ic;
          sum = sum + input[inIdx] * weight[wIdx];
        }
      }
    }
  }
  
  output[idx] = sum;
}`

  {
    name: "conv2d_" ++ Int.toString(outH) ++ "x" ++ Int.toString(outW) ++ "x" ++ Int.toString(outC),
    wgsl,
    bindings: [
      {binding: 0, size: batch * inH * inW * inC * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outC * kH * kW * inC * 4, usage: ReadOnly, name: "weight"},
      {binding: 2, size: outC * 4, usage: ReadOnly, name: "bias"},
      {binding: 3, size: batch * outH * outW * outC * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Dense / Fully Connected
// ----------------------------------------

let genDenseKernel = (batchSize: int, inFeatures: int, outFeatures: int): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batchSize)}u;
const IN_F = ${Int.toString(inFeatures)}u;
const OUT_F = ${Int.toString(outFeatures)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * OUT_F) { return; }
  
  let b = idx / OUT_F;
  let o = idx % OUT_F;
  
  var sum = bias[o];
  for (var i = 0u; i < IN_F; i = i + 1u) {
    sum = sum + input[b * IN_F + i] * weight[o * IN_F + i];
  }
  
  output[idx] = sum;
}`

  {
    name: "dense_" ++ Int.toString(batchSize) ++ "x" ++ Int.toString(inFeatures) ++ "x" ++ Int.toString(outFeatures),
    wgsl,
    bindings: [
      {binding: 0, size: batchSize * inFeatures * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outFeatures * inFeatures * 4, usage: ReadOnly, name: "weight"},
      {binding: 2, size: outFeatures * 4, usage: ReadOnly, name: "bias"},
      {binding: 3, size: batchSize * outFeatures * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Softmax (two-pass: max, then exp/sum)
// ----------------------------------------

let genSoftmaxKernel = (outerSize: int, axisSize: int): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const OUTER = ${Int.toString(outerSize)}u;
const AXIS = ${Int.toString(axisSize)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outerIdx = gid.x;
  if (outerIdx >= OUTER) { return; }
  
  let baseIdx = outerIdx * AXIS;
  
  // Find max for numerical stability
  var maxVal = input[baseIdx];
  for (var i = 1u; i < AXIS; i = i + 1u) {
    maxVal = max(maxVal, input[baseIdx + i]);
  }
  
  // Compute exp and sum
  var sum = 0.0;
  for (var i = 0u; i < AXIS; i = i + 1u) {
    let e = exp(input[baseIdx + i] - maxVal);
    output[baseIdx + i] = e;
    sum = sum + e;
  }
  
  // Normalize
  for (var i = 0u; i < AXIS; i = i + 1u) {
    output[baseIdx + i] = output[baseIdx + i] / sum;
  }
}`

  {
    name: "softmax_" ++ Int.toString(outerSize) ++ "x" ++ Int.toString(axisSize),
    wgsl,
    bindings: [
      {binding: 0, size: outerSize * axisSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outerSize * axisSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Transpose
// ----------------------------------------

let genTransposeKernel = (inputShape: shape, perm: array<int>): option<kernel> => {
  let rank = Array.length(inputShape)
  if rank != Array.length(perm) {
    None
  } else {
    let totalSize = Shape.numElements(inputShape)
    let outputShape = Array.map(perm, p => inputShape[p]->Option.getOr(1))
    
    // Compute strides for input and output
    let inputStrides = Array.fromInitializer(~length=rank, i => {
      let stride = ref(1)
      for j in i + 1 to rank - 1 {
        stride := stride.contents * (inputShape[j]->Option.getOr(1))
      }
      stride.contents
    })
    
    let outputStrides = Array.fromInitializer(~length=rank, i => {
      let stride = ref(1)
      for j in i + 1 to rank - 1 {
        stride := stride.contents * (outputShape[j]->Option.getOr(1))
      }
      stride.contents
    })
    
    let shapeStr = Array.map(inputShape, d => Int.toString(d))->Array.join(", ")
    let permStr = Array.map(perm, d => Int.toString(d))->Array.join(", ")
    let inStrideStr = Array.map(inputStrides, d => Int.toString(d))->Array.join(", ")
    let outStrideStr = Array.map(outputStrides, d => Int.toString(d))->Array.join(", ")
    
    let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const RANK = ${Int.toString(rank)}u;
const TOTAL = ${Int.toString(totalSize)}u;
const SHAPE = array<u32, ${Int.toString(rank)}>(${shapeStr});
const PERM = array<u32, ${Int.toString(rank)}>(${permStr});
const IN_STRIDES = array<u32, ${Int.toString(rank)}>(${inStrideStr});
const OUT_STRIDES = array<u32, ${Int.toString(rank)}>(${outStrideStr});

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outIdx = gid.x;
  if (outIdx >= TOTAL) { return; }
  
  // Convert output index to coordinates
  var coords: array<u32, ${Int.toString(rank)}>;
  var remaining = outIdx;
  for (var i = 0u; i < RANK; i = i + 1u) {
    coords[i] = remaining / OUT_STRIDES[i];
    remaining = remaining % OUT_STRIDES[i];
  }
  
  // Apply inverse permutation to get input coords
  var inIdx = 0u;
  for (var i = 0u; i < RANK; i = i + 1u) {
    inIdx = inIdx + coords[i] * IN_STRIDES[PERM[i]];
  }
  
  output[outIdx] = input[inIdx];
}`

    Some({
      name: "transpose_" ++ Int.toString(totalSize),
      wgsl,
      bindings: [
        {binding: 0, size: totalSize * 4, usage: ReadOnly, name: "input"},
        {binding: 1, size: totalSize * 4, usage: ReadWrite, name: "output"}
      ]
    })
  }
}

// ----------------------------------------
// Reshape (just a copy, memory layout same)
// ----------------------------------------

let genReshapeKernel = (size: int): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${Int.toString(size)}u) { return; }
  output[idx] = input[idx];
}`

  {
    name: "reshape_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Generate Element-wise Kernels
// ----------------------------------------

let genUnaryKernel = (op: op, size: int): option<kernel> => {
  unaryExpr(op, "input0[idx]")->Option.map(expr => {
    let wgsl = `${shaderHeader(2)}

${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  output[idx] = ${expr};
${mainEnd}`
    
    {
      name: "unary_" ++ Int.toString(size),
      wgsl,
      bindings: [
        {binding: 0, size: size * 4, usage: ReadOnly, name: "input0"},
        {binding: 1, size: size * 4, usage: ReadWrite, name: "output"}
      ]
    }
  })
}

// ----------------------------------------
// Binary Operations with Broadcasting
// ----------------------------------------
let genBinaryBroadcastKernel = (op: op, inputShape0: shape, inputShape1: shape, outputShape: shape): option<kernel> => {
  let outSize = Shape.numElements(outputShape)
  let inSize0 = Shape.numElements(inputShape0)
  let inSize1 = Shape.numElements(inputShape1)
  let outRank = Array.length(outputShape)
  let inRank0 = Array.length(inputShape0)
  let inRank1 = Array.length(inputShape1)
  
  // Calculate strides for output
  let outStrides = Array.fromInitializer(~length=outRank, i => {
    let stride = ref(1)
    for j in i + 1 to outRank - 1 {
      stride := stride.contents * outputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Calculate strides for input0 (with broadcasting)
  let inStrides0 = Array.fromInitializer(~length=outRank, i => {
    let inIdx = i - (outRank - inRank0)
    if inIdx < 0 {
      0  // This dimension is broadcast (doesn't exist in input)
    } else {
      let inDim = inputShape0[inIdx]->Option.getOr(1)
      let outDim = outputShape[i]->Option.getOr(1)
      if inDim == 1 && outDim > 1 {
        0  // This dimension is broadcast (size 1 -> size N)
      } else {
        // Calculate actual stride
        let stride = ref(1)
        for j in inIdx + 1 to inRank0 - 1 {
          stride := stride.contents * inputShape0[j]->Option.getOr(1)
        }
        stride.contents
      }
    }
  })
  
  // Calculate strides for input1 (with broadcasting)
  let inStrides1 = Array.fromInitializer(~length=outRank, i => {
    let inIdx = i - (outRank - inRank1)
    if inIdx < 0 {
      0
    } else {
      let inDim = inputShape1[inIdx]->Option.getOr(1)
      let outDim = outputShape[i]->Option.getOr(1)
      if inDim == 1 && outDim > 1 {
        0
      } else {
        let stride = ref(1)
        for j in inIdx + 1 to inRank1 - 1 {
          stride := stride.contents * inputShape1[j]->Option.getOr(1)
        }
        stride.contents
      }
    }
  })
  
  let outShapeStr = Array.map(outputShape, d => Int.toString(d))->Array.join(", ")
  let outStridesStr = Array.map(outStrides, d => Int.toString(d))->Array.join(", ")
  let inStrides0Str = Array.map(inStrides0, d => Int.toString(d))->Array.join(", ")
  let inStrides1Str = Array.map(inStrides1, d => Int.toString(d))->Array.join(", ")
  
  binaryExpr(op, "val0", "val1")->Option.map(expr => {
    let wgsl = if inSize0 == outSize && inSize1 == outSize {
      // Fast path: no broadcasting needed
      `${storageBuffer(0, "input0", ReadOnly)}
${storageBuffer(1, "input1", ReadOnly)}
${storageBuffer(2, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outSize)}u) { return; }
  let val0 = input0[idx];
  let val1 = input1[idx];
  output[idx] = ${expr};
${mainEnd}`
    } else {
      // General path: with broadcasting
      `${storageBuffer(0, "input0", ReadOnly)}
${storageBuffer(1, "input1", ReadOnly)}
${storageBuffer(2, "output", ReadWrite)}
const OUT_RANK = ${Int.toString(outRank)}u;
const OUT_SHAPE = array<u32, ${Int.toString(outRank)}>(${outShapeStr});
const OUT_STRIDES = array<u32, ${Int.toString(outRank)}>(${outStridesStr});
const IN_STRIDES0 = array<u32, ${Int.toString(outRank)}>(${inStrides0Str});
const IN_STRIDES1 = array<u32, ${Int.toString(outRank)}>(${inStrides1Str});
${mainSignature}
  if (idx >= ${Int.toString(outSize)}u) { return; }
  
  // Convert flat index to coordinates
  var remaining = idx;
  var in_idx0 = 0u;
  var in_idx1 = 0u;
  
  for (var d = 0u; d < OUT_RANK; d = d + 1u) {
    let coord = remaining / OUT_STRIDES[d];
    remaining = remaining % OUT_STRIDES[d];
    in_idx0 = in_idx0 + coord * IN_STRIDES0[d];
    in_idx1 = in_idx1 + coord * IN_STRIDES1[d];
  }
  
  let val0 = input0[in_idx0];
  let val1 = input1[in_idx1];
  output[idx] = ${expr};
${mainEnd}`
    }
    {
      name: "binary_broadcast_" ++ Int.toString(outSize),
      wgsl,
      bindings: [
        {binding: 0, size: inSize0 * 4, usage: ReadOnly, name: "input0"},
        {binding: 1, size: inSize1 * 4, usage: ReadOnly, name: "input1"},
        {binding: 2, size: outSize * 4, usage: ReadWrite, name: "output"}
      ]
    }
  })
}

// ----------------------------------------
// Compute Dispatch Size
// ----------------------------------------

let computeDispatch = (totalElements: int, kernelName: string, pipelineIndex: int): dispatch => {
  let workgroupCount = (totalElements + workgroupSize - 1) / workgroupSize
  {
    workgroupSize: (workgroupSize, 1, 1),
    workgroupCount: (workgroupCount, 1, 1),
    kernelName,
    pipelineIndex
  }
}

let computeDispatch2D = (x: int, y: int, tileSize: int, kernelName: string, pipelineIndex: int): dispatch => {
  let countX = (x + tileSize - 1) / tileSize
  let countY = (y + tileSize - 1) / tileSize
  {
    workgroupSize: (tileSize, tileSize, 1),
    workgroupCount: (countX, countY, 1),
    kernelName,
    pipelineIndex
  }
}

// ----------------------------------------
// MaxPool2D / AvgPool2D
// ----------------------------------------

let genPool2DKernel = (
  poolType: string,  // "max" or "avg"
  batch: int, inH: int, inW: int, channels: int,
  outH: int, outW: int,
  kH: int, kW: int,
  strideH: int, strideW: int,
  padH: int, padW: int
): kernel => {
  let identity = poolType == "max" ? "-3.402823e+38" : "0.0"
  let accumulate = poolType == "max" ? "acc = max(acc, val);" : "acc = acc + val;"
  let finalize = poolType == "max" ? "acc" : "acc / f32(K_H * K_W)"
  
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batch)}u;
const IN_H = ${Int.toString(inH)}u;
const IN_W = ${Int.toString(inW)}u;
const CHANNELS = ${Int.toString(channels)}u;
const OUT_H = ${Int.toString(outH)}u;
const OUT_W = ${Int.toString(outW)}u;
const K_H = ${Int.toString(kH)}u;
const K_W = ${Int.toString(kW)}u;
const STRIDE_H = ${Int.toString(strideH)}u;
const STRIDE_W = ${Int.toString(strideW)}u;
const PAD_H = ${Int.toString(padH)}u;
const PAD_W = ${Int.toString(padW)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_H * OUT_W * CHANNELS;
  if (idx >= totalOut) { return; }

  let c = idx % CHANNELS;
  let tmp1 = idx / CHANNELS;
  let ow = tmp1 % OUT_W;
  let tmp2 = tmp1 / OUT_W;
  let oh = tmp2 % OUT_H;
  let b = tmp2 / OUT_H;

  var acc = ${identity};

  for (var kh = 0u; kh < K_H; kh = kh + 1u) {
    for (var kw = 0u; kw < K_W; kw = kw + 1u) {
      let ih = oh * STRIDE_H + kh - PAD_H;
      let iw = ow * STRIDE_W + kw - PAD_W;

      if (ih < IN_H && iw < IN_W) {
        let inIdx = b * IN_H * IN_W * CHANNELS + ih * IN_W * CHANNELS + iw * CHANNELS + c;
        let val = input[inIdx];
        ${accumulate}
      }
    }
  }

  output[idx] = ${finalize};
}`

  let outSize = batch * outH * outW * channels
  {
    name: poolType ++ "pool2d_" ++ Int.toString(outH) ++ "x" ++ Int.toString(outW),
    wgsl,
    bindings: [
      {binding: 0, size: batch * inH * inW * channels * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// BatchNorm (inference mode)
// ----------------------------------------

let genBatchNormKernel = (batch: int, height: int, width: int, channels: int): kernel => {
  let size = batch * height * width * channels
  
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read> mean: array<f32>;
@group(0) @binding(4) var<storage, read> variance: array<f32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

const SIZE = ${Int.toString(size)}u;
const CHANNELS = ${Int.toString(channels)}u;
const EPSILON = 1e-5;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }

  let c = idx % CHANNELS;
  let x = input[idx];
  let normalized = (x - mean[c]) / sqrt(variance[c] + EPSILON);
  output[idx] = gamma[c] * normalized + beta[c];
}`

  {
    name: "batchnorm_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: channels * 4, usage: ReadOnly, name: "gamma"},
      {binding: 2, size: channels * 4, usage: ReadOnly, name: "beta"},
      {binding: 3, size: channels * 4, usage: ReadOnly, name: "mean"},
      {binding: 4, size: channels * 4, usage: ReadOnly, name: "variance"},
      {binding: 5, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Conv1D
// ----------------------------------------

let genConv1DKernel = (
  batch: int, inLen: int, inC: int,
  outLen: int, outC: int,
  kernel: int, stride: int, pad: int
): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batch)}u;
const IN_LEN = ${Int.toString(inLen)}u;
const IN_C = ${Int.toString(inC)}u;
const OUT_LEN = ${Int.toString(outLen)}u;
const OUT_C = ${Int.toString(outC)}u;
const K = ${Int.toString(kernel)}u;
const STRIDE = ${Int.toString(stride)}u;
const PAD = ${Int.toString(pad)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let totalOut = BATCH * OUT_LEN * OUT_C;
  if (idx >= totalOut) { return; }

  let oc = idx % OUT_C;
  let tmp = idx / OUT_C;
  let ol = tmp % OUT_LEN;
  let b = tmp / OUT_LEN;

  var sum = bias[oc];

  for (var ic = 0u; ic < IN_C; ic = ic + 1u) {
    for (var k = 0u; k < K; k = k + 1u) {
      let il = ol * STRIDE + k - PAD;
      if (il < IN_LEN) {
        let inIdx = b * IN_LEN * IN_C + il * IN_C + ic;
        let wIdx = oc * K * IN_C + k * IN_C + ic;
        sum = sum + input[inIdx] * weight[wIdx];
      }
    }
  }

  output[idx] = sum;
}`

  {
    name: "conv1d_" ++ Int.toString(outLen) ++ "x" ++ Int.toString(outC),
    wgsl,
    bindings: [
      {binding: 0, size: batch * inLen * inC * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outC * kernel * inC * 4, usage: ReadOnly, name: "weight"},
      {binding: 2, size: outC * 4, usage: ReadOnly, name: "bias"},
      {binding: 3, size: batch * outLen * outC * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// GlobalMaxPool / GlobalAvgPool
// ----------------------------------------

let genGlobalPoolKernel = (poolType: string, batch: int, height: int, width: int, channels: int): kernel => {
  let identity = poolType == "max" ? "-3.402823e+38" : "0.0"
  let accumulate = poolType == "max" ? "acc = max(acc, val);" : "acc = acc + val;"
  let finalize = poolType == "max" ? "acc" : "acc / f32(H * W)"
  
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batch)}u;
const H = ${Int.toString(height)}u;
const W = ${Int.toString(width)}u;
const C = ${Int.toString(channels)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * C) { return; }

  let c = idx % C;
  let b = idx / C;

  var acc = ${identity};

  for (var h = 0u; h < H; h = h + 1u) {
    for (var w = 0u; w < W; w = w + 1u) {
      let inIdx = b * H * W * C + h * W * C + w * C + c;
      let val = input[inIdx];
      ${accumulate}
    }
  }

  // Output shape is [batch, 1, 1, channels]
  let outIdx = b * C + c;
  output[outIdx] = ${finalize};
}`

  {
    name: poolType ++ "globalpool_" ++ Int.toString(batch) ++ "x" ++ Int.toString(channels),
    wgsl,
    bindings: [
      {binding: 0, size: batch * height * width * channels * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: batch * channels * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// LayerNorm
// ----------------------------------------

let genLayerNormKernel = (outerSize: int, normSize: int, epsilon: float): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const OUTER = ${Int.toString(outerSize)}u;
const NORM = ${Int.toString(normSize)}u;
const EPSILON = ${Float.toString(epsilon)};

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let outerIdx = gid.x;
  if (outerIdx >= OUTER) { return; }

  let baseIdx = outerIdx * NORM;

  // Compute mean
  var sum = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    sum = sum + input[baseIdx + i];
  }
  let mean = sum / f32(NORM);

  // Compute variance
  var varSum = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let diff = input[baseIdx + i] - mean;
    varSum = varSum + diff * diff;
  }
  let variance = varSum / f32(NORM);
  let invStd = 1.0 / sqrt(variance + EPSILON);

  // Normalize and scale
  for (var i = 0u; i < NORM; i = i + 1u) {
    let normalized = (input[baseIdx + i] - mean) * invStd;
    output[baseIdx + i] = gamma[i] * normalized + beta[i];
  }
}`

  {
    name: "layernorm_" ++ Int.toString(outerSize) ++ "x" ++ Int.toString(normSize),
    wgsl,
    bindings: [
      {binding: 0, size: outerSize * normSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: normSize * 4, usage: ReadOnly, name: "gamma"},
      {binding: 2, size: normSize * 4, usage: ReadOnly, name: "beta"},
      {binding: 3, size: outerSize * normSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// RMSNorm (Root Mean Square Layer Normalization)
// Used in Qwen2, LLaMA, Mistral, etc.
// 
// y = x * rsqrt(mean(x²) + eps) * weight
// ----------------------------------------

let genRMSNormKernel = (outerSize: int, normSize: int, epsilon: float): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const OUTER = ${Int.toString(outerSize)}u;
const NORM = ${Int.toString(normSize)}u;
const EPSILON = ${Float.toString(epsilon)};

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= OUTER) { return; }

  let baseIdx = row * NORM;

  // Compute mean of squares
  var sumSq = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let x = input[baseIdx + i];
    sumSq = sumSq + x * x;
  }
  let rms = sqrt(sumSq / f32(NORM) + EPSILON);
  let scale = 1.0 / rms;

  // Normalize and apply weight
  for (var i = 0u; i < NORM; i = i + 1u) {
    output[baseIdx + i] = input[baseIdx + i] * scale * weight[i];
  }
}`

  {
    name: "rmsnorm_" ++ Int.toString(outerSize) ++ "x" ++ Int.toString(normSize),
    wgsl,
    bindings: [
      {binding: 0, size: outerSize * normSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: normSize * 4, usage: ReadOnly, name: "weight"},
      {binding: 2, size: outerSize * normSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// RoPE (Rotary Position Embedding)
// Used in Qwen2, LLaMA, Mistral, etc.
//
// For each pair of dimensions (2i, 2i+1):
//   q'[2i]   = q[2i] * cos(θ) - q[2i+1] * sin(θ)
//   q'[2i+1] = q[2i] * sin(θ) + q[2i+1] * cos(θ)
//
// where θ = position * (base ^ (-2i / dim))
// ----------------------------------------

let genRoPEKernel = (numQHeads: int, numKVHeads: int, headDim: int, ropeTheta: float): kernel => {
  let totalQDim = numQHeads * headDim
  let totalKDim = numKVHeads * headDim
  let halfDim = headDim / 2
  let maxWork = Math.Int.max(numQHeads, numKVHeads) * halfDim
  let workgroups = (maxWork + workgroupSize - 1) / workgroupSize
  
  let wgsl = `@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> position: array<u32>;

const NUM_Q_HEADS = ${Int.toString(numQHeads)}u;
const NUM_KV_HEADS = ${Int.toString(numKVHeads)}u;
const HEAD_DIM = ${Int.toString(headDim)}u;
const HALF_DIM = ${Int.toString(halfDim)}u;
const ROPE_THETA = ${Float.toString(ropeTheta)};

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pos = f32(position[0]);
  
  // Process Q heads
  if (idx < NUM_Q_HEADS * HALF_DIM) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(2u * pair) / f32(HEAD_DIM));
    let angle = pos * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);
    
    let base_idx = head * HEAD_DIM + pair * 2u;
    let q0 = q[base_idx];
    let q1 = q[base_idx + 1u];
    
    q[base_idx] = q0 * cos_val - q1 * sin_val;
    q[base_idx + 1u] = q0 * sin_val + q1 * cos_val;
  }
  
  // Process K heads
  if (idx < NUM_KV_HEADS * HALF_DIM) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(2u * pair) / f32(HEAD_DIM));
    let angle = pos * freq;
    let cos_val = cos(angle);
    let sin_val = sin(angle);
    
    let base_idx = head * HEAD_DIM + pair * 2u;
    let k0 = k[base_idx];
    let k1 = k[base_idx + 1u];
    
    k[base_idx] = k0 * cos_val - k1 * sin_val;
    k[base_idx + 1u] = k0 * sin_val + k1 * cos_val;
  }
}`

  {
    name: "rope_" ++ Int.toString(numQHeads) ++ "x" ++ Int.toString(numKVHeads) ++ "x" ++ Int.toString(headDim),
    wgsl,
    bindings: [
      {binding: 0, size: totalQDim * 4, usage: ReadWrite, name: "q"},
      {binding: 1, size: totalKDim * 4, usage: ReadWrite, name: "k"},
      {binding: 2, size: 4, usage: ReadOnly, name: "position"}
    ]
  }
}

// ----------------------------------------
// GQA (Grouped Query Attention)
// Used in Qwen2.5, LLaMA 2/3, Mistral, etc.
//
// Qwen2.5-7B: 28 Q heads, 4 KV heads (7:1 ratio)
// ----------------------------------------

let genGQAAttentionKernel = (numQHeads: int, numKVHeads: int, headDim: int, maxSeqLen: int): kernel => {
  let headsPerGroup = numQHeads / numKVHeads
  let scale = 1.0 /. Math.sqrt(Int.toFloat(headDim))
  let wgSize = 256
  
  let wgsl = `@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> seq_len: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const NUM_Q_HEADS = ${Int.toString(numQHeads)}u;
const NUM_KV_HEADS = ${Int.toString(numKVHeads)}u;
const HEADS_PER_GROUP = ${Int.toString(headsPerGroup)}u;
const HEAD_DIM = ${Int.toString(headDim)}u;
const MAX_SEQ_LEN = ${Int.toString(maxSeqLen)}u;
const SCALE = ${Float.toString(scale)};
const WG_SIZE = ${Int.toString(wgSize)}u;

var<workgroup> wg_q: array<f32, ${Int.toString(headDim)}>;
var<workgroup> wg_scores: array<f32, ${Int.toString(maxSeqLen)}>;
var<workgroup> wg_reduce: array<f32, ${Int.toString(wgSize)}>;

@compute @workgroup_size(${Int.toString(wgSize)})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let q_head = wgid.x;
  let tid = lid.x;
  let kv_head = q_head / HEADS_PER_GROUP;
  let cur_seq_len = seq_len[0];
  
  // Load Q into shared memory
  let q_base = q_head * HEAD_DIM;
  for (var d = tid; d < HEAD_DIM; d = d + WG_SIZE) {
    wg_q[d] = q[q_base + d];
  }
  workgroupBarrier();
  
  // Phase 1: Compute scores Q @ K^T
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    var score = 0.0;
    let k_base = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
    
    for (var d = 0u; d < HEAD_DIM; d = d + 1u) {
      score = score + wg_q[d] * k_cache[k_base + d];
    }
    wg_scores[pos] = score * SCALE;
  }
  workgroupBarrier();
  
  // Phase 2: Softmax - find max
  var local_max = -1e30;
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    local_max = max(local_max, wg_scores[pos]);
  }
  wg_reduce[tid] = local_max;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_reduce[tid] = max(wg_reduce[tid], wg_reduce[tid + stride]);
    }
    workgroupBarrier();
  }
  let max_val = wg_reduce[0];
  
  // Phase 3: Softmax - exp and sum
  var local_sum = 0.0;
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    let e = exp(wg_scores[pos] - max_val);
    wg_scores[pos] = e;
    local_sum = local_sum + e;
  }
  wg_reduce[tid] = local_sum;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + stride];
    }
    workgroupBarrier();
  }
  let sum_val = wg_reduce[0];
  
  // Phase 4: Normalize probabilities
  for (var pos = tid; pos < cur_seq_len; pos = pos + WG_SIZE) {
    wg_scores[pos] = wg_scores[pos] / sum_val;
  }
  workgroupBarrier();
  
  // Phase 5: Output = probs @ V
  for (var d = tid; d < HEAD_DIM; d = d + WG_SIZE) {
    var out_val = 0.0;
    for (var pos = 0u; pos < cur_seq_len; pos = pos + 1u) {
      let v_idx = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d;
      out_val = out_val + wg_scores[pos] * v_cache[v_idx];
    }
    output[q_head * HEAD_DIM + d] = out_val;
  }
}`

  {
    name: "gqa_attention_" ++ Int.toString(numQHeads) ++ "x" ++ Int.toString(numKVHeads) ++ "x" ++ Int.toString(headDim),
    wgsl,
    bindings: [
      {binding: 0, size: numQHeads * headDim * 4, usage: ReadOnly, name: "q"},
      {binding: 1, size: maxSeqLen * numKVHeads * headDim * 4, usage: ReadOnly, name: "k_cache"},
      {binding: 2, size: maxSeqLen * numKVHeads * headDim * 4, usage: ReadOnly, name: "v_cache"},
      {binding: 3, size: 4, usage: ReadOnly, name: "seq_len"},
      {binding: 4, size: numQHeads * headDim * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Scaled Dot-Product Attention
// ----------------------------------------

let genAttentionKernel = (batch: int, seqLen: int, dim: int): kernel => {
  let scale = 1.0 /. Math.sqrt(Int.toFloat(dim))
  
  let wgsl = `@group(0) @binding(0) var<storage, read> query: array<f32>;
@group(0) @binding(1) var<storage, read> key: array<f32>;
@group(0) @binding(2) var<storage, read> value: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

const BATCH = ${Int.toString(batch)}u;
const SEQ = ${Int.toString(seqLen)}u;
const DIM = ${Int.toString(dim)}u;
const SCALE = ${Float.toString(scale)};

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH * SEQ * DIM) { return; }

  let d = idx % DIM;
  let tmp = idx / DIM;
  let qPos = tmp % SEQ;
  let b = tmp / SEQ;

  let qBase = b * SEQ * DIM + qPos * DIM;

  // Compute attention scores and find max for stability
  var maxScore = -3.402823e+38;
  for (var kPos = 0u; kPos < SEQ; kPos = kPos + 1u) {
    let kBase = b * SEQ * DIM + kPos * DIM;
    var score = 0.0;
    for (var i = 0u; i < DIM; i = i + 1u) {
      score = score + query[qBase + i] * key[kBase + i];
    }
    score = score * SCALE;
    maxScore = max(maxScore, score);
  }

  // Compute softmax and weighted sum
  var expSum = 0.0;
  var result = 0.0;

  for (var kPos = 0u; kPos < SEQ; kPos = kPos + 1u) {
    let kBase = b * SEQ * DIM + kPos * DIM;
    var score = 0.0;
    for (var i = 0u; i < DIM; i = i + 1u) {
      score = score + query[qBase + i] * key[kBase + i];
    }
    score = score * SCALE;
    let expScore = exp(score - maxScore);
    expSum = expSum + expScore;

    let vBase = b * SEQ * DIM + kPos * DIM;
    result = result + expScore * value[vBase + d];
  }

  output[idx] = result / expSum;
}`

  {
    name: "attention_" ++ Int.toString(batch) ++ "x" ++ Int.toString(seqLen) ++ "x" ++ Int.toString(dim),
    wgsl,
    bindings: [
      {binding: 0, size: batch * seqLen * dim * 4, usage: ReadOnly, name: "query"},
      {binding: 1, size: batch * seqLen * dim * 4, usage: ReadOnly, name: "key"},
      {binding: 2, size: batch * seqLen * dim * 4, usage: ReadOnly, name: "value"},
      {binding: 3, size: batch * seqLen * dim * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Embedding lookup
// ----------------------------------------

let genEmbeddingKernel = (batchSeq: int, vocabSize: int, embDim: int): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read> embeddings: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const BATCH_SEQ = ${Int.toString(batchSeq)}u;
const VOCAB = ${Int.toString(vocabSize)}u;
const DIM = ${Int.toString(embDim)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= BATCH_SEQ * DIM) { return; }

  let d = idx % DIM;
  let pos = idx / DIM;

  let tokenId = indices[pos];
  output[idx] = embeddings[tokenId * DIM + d];
}`

  {
    name: "embedding_" ++ Int.toString(batchSeq) ++ "x" ++ Int.toString(embDim),
    wgsl,
    bindings: [
      {binding: 0, size: batchSeq * 4, usage: ReadOnly, name: "indices"},
      {binding: 1, size: vocabSize * embDim * 4, usage: ReadOnly, name: "embeddings"},
      {binding: 2, size: batchSeq * embDim * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Concat along axis
// ----------------------------------------

let genConcatKernel = (inputShapes: array<shape>, axis: int): option<kernel> => {
  let numInputs = Array.length(inputShapes)
  if numInputs == 0 {
    None
  } else {
    let first = inputShapes[0]->Option.getOr([])
    let rank = Array.length(first)
    let normAxis = axis < 0 ? rank + axis : axis
    
    // Calculate output shape
    let concatDim = Array.reduce(inputShapes, 0, (acc, s) => 
      acc + (s[normAxis]->Option.getOr(0))
    )
    let outputShape = Array.mapWithIndex(first, (d, i) => 
      i == normAxis ? concatDim : d
    )
    let outSize = Shape.numElements(outputShape)
    
    // Calculate offsets for each input along concat axis
    let offsets = Array.fromInitializer(~length=numInputs, i => {
      let offset = ref(0)
      for j in 0 to i - 1 {
        let s = inputShapes[j]->Option.getOr([])
        offset := offset.contents + (s[normAxis]->Option.getOr(0))
      }
      offset.contents
    })
    
    let _offsetsStr = Array.map(offsets, o => Int.toString(o))->Array.join(", ")
    let _sizesStr = Array.map(inputShapes, s =>
      Int.toString(s[normAxis]->Option.getOr(0))
    )->Array.join(", ")
    
    // For simplicity, generate for 2 inputs (most common case)
    if numInputs == 2 {
      let size0 = Shape.numElements(inputShapes[0]->Option.getOr([]))
      let size1 = Shape.numElements(inputShapes[1]->Option.getOr([]))
      let axisDim0 = (inputShapes[0]->Option.getOr([]))[normAxis]->Option.getOr(0)
      
      let wgsl = `@group(0) @binding(0) var<storage, read> input0: array<f32>;
@group(0) @binding(1) var<storage, read> input1: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const OUT_SIZE = ${Int.toString(outSize)}u;
const AXIS = ${Int.toString(normAxis)}u;
const AXIS_DIM0 = ${Int.toString(axisDim0)}u;
const CONCAT_DIM = ${Int.toString(concatDim)}u;

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= OUT_SIZE) { return; }
  
  // Simple copy - determine which input based on position
  if (idx < ${Int.toString(size0)}u) {
    output[idx] = input0[idx];
  } else {
    output[idx] = input1[idx - ${Int.toString(size0)}u];
  }
}`

      Some({
        name: "concat_" ++ Int.toString(outSize),
        wgsl,
        bindings: [
          {binding: 0, size: size0 * 4, usage: ReadOnly, name: "input0"},
          {binding: 1, size: size1 * 4, usage: ReadOnly, name: "input1"},
          {binding: 2, size: outSize * 4, usage: ReadWrite, name: "output"}
        ]
      })
    } else {
      None
    }
  }
}

// ----------------------------------------
// Clip (clamp values)
// ----------------------------------------

let genClipKernel = (size: int, minVal: float, maxVal: float): kernel => {
  let wgsl = `@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE = ${Int.toString(size)}u;
const MIN_VAL = ${Float.toString(minVal)};
const MAX_VAL = ${Float.toString(maxVal)};

@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  output[idx] = clamp(input[idx], MIN_VAL, MAX_VAL);
}`

  {
    name: "clip_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}


// ----------------------------------------
// Where Kernel (conditional selection)
// ----------------------------------------
let genWhereKernel = (size: int): kernel => {
  let wgsl = `${storageBuffer(0, "condition", ReadOnly)}
${storageBuffer(1, "input_true", ReadOnly)}
${storageBuffer(2, "input_false", ReadOnly)}
${storageBuffer(3, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let cond = condition[idx];
  output[idx] = select(input_false[idx], input_true[idx], cond > 0.0);
${mainEnd}`
  {
    name: "where_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "condition"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "input_true"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "input_false"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Gather Kernel (index selection along axis)
// ----------------------------------------
let genGatherKernel = (dataShape: shape, indicesSize: int, axis: int): kernel => {
  let rank = Array.length(dataShape)
  let outerSize = ref(1)
  let axisSize = dataShape[axis]->Option.getOr(1)
  let innerSize = ref(1)
  
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * dataShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * dataShape[i]->Option.getOr(1)
  }
  
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = outer * indicesSize * inner
  
  let wgsl = `${storageBuffer(0, "data", ReadOnly)}
${storageBuffer(1, "indices", ReadOnly)}
${storageBuffer(2, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  
  let inner_size = ${Int.toString(inner)}u;
  let indices_size = ${Int.toString(indicesSize)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  
  let outer_idx = idx / (indices_size * inner_size);
  let remainder = idx % (indices_size * inner_size);
  let index_idx = remainder / inner_size;
  let inner_idx = remainder % inner_size;
  
  let gather_idx = u32(indices[index_idx]);
  let data_idx = outer_idx * (axis_size * inner_size) + gather_idx * inner_size + inner_idx;
  
  output[idx] = data[data_idx];
${mainEnd}`
  
  {
    name: "gather_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(dataShape) * 4, usage: ReadOnly, name: "data"},
      {binding: 1, size: indicesSize * 4, usage: ReadOnly, name: "indices"},
      {binding: 2, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Split Kernel (split tensor along axis)
// Note: Returns multiple kernels, one per output
// ----------------------------------------

let genSplitKernel = (inputShape: shape, axis: int, splitIndex: int, splitSize: int): kernel => {
  let rank = Array.length(inputShape)
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = outer * splitSize * inner
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  let inner_size = ${Int.toString(inner)}u;
  let split_size = ${Int.toString(splitSize)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  let offset = ${Int.toString(splitIndex)}u;
  let outer_idx = idx / (split_size * inner_size);
  let remainder = idx % (split_size * inner_size);
  let axis_idx = remainder / inner_size;
  let inner_idx = remainder % inner_size;
  let input_idx = outer_idx * (axis_size * inner_size) + (offset + axis_idx) * inner_size + inner_idx;
  output[idx] = input[input_idx];
${mainEnd}`
  {
    name: "split_" ++ Int.toString(splitIndex) ++ "_" ++ Int.toString(splitSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// TopK Kernel (find top k values)
// Simplified: uses parallel bubble sort for small k
// ----------------------------------------
let genTopKKernel = (inputShape: shape, k: int, axis: int): kernel => {
  let rank = Array.length(inputShape)
  let outerSize = ref(1)
  let axisSize = inputShape[axis]->Option.getOr(1)
  let innerSize = ref(1)
  
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  
  let outer = outerSize.contents
  let inner = innerSize.contents
  let numSlices = outer * inner
  let outputSize = numSlices * k
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output_values", ReadWrite)}
${storageBuffer(2, "output_indices", ReadWrite)}

${mainSignature}
  let slice_idx = gid.x;
  if (slice_idx >= ${Int.toString(numSlices)}u) { return; }
  
  let k = ${Int.toString(k)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  let inner_size = ${Int.toString(inner)}u;
  
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  
  // Simple selection sort for top-k (works well for small k)
  var top_vals: array<f32, ${Int.toString(k)}>;
  var top_idxs: array<i32, ${Int.toString(k)}>;
  
  // Initialize with very small values
  for (var i = 0u; i < k; i = i + 1u) {
    top_vals[i] = -3.402823e+38;
    top_idxs[i] = -1;
  }
  
  // Find top k
  for (var i = 0u; i < axis_size; i = i + 1u) {
    let input_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let val = input[input_idx];
    
    // Check if this value is larger than the smallest in top_vals
    if (val > top_vals[k - 1u]) {
      // Insert in sorted position
      var j = k - 1u;
      while (j > 0u && val > top_vals[j - 1u]) {
        top_vals[j] = top_vals[j - 1u];
        top_idxs[j] = top_idxs[j - 1u];
        j = j - 1u;
      }
      top_vals[j] = val;
      top_idxs[j] = i32(i);
    }
  }
  
  // Write output
  for (var i = 0u; i < k; i = i + 1u) {
    let out_idx = slice_idx * k + i;
    output_values[out_idx] = top_vals[i];
    output_indices[out_idx] = f32(top_idxs[i]);
  }
${mainEnd}`
  
  {
    name: "topk_" ++ Int.toString(k) ++ "_" ++ Int.toString(numSlices),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output_values"},
      {binding: 2, size: outputSize * 4, usage: ReadWrite, name: "output_indices"}
    ]
  }
}

// ----------------------------------------
// ArgMax/ArgMin Kernel
// ----------------------------------------
let genArgMaxKernel = (inputShape: shape, axis: int, selectLastIndex: bool): kernel => {
  let rank = Array.length(inputShape)
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = outer * inner
  let cmpOp = if selectLastIndex { ">=" } else { ">" }
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  let inner_size = ${Int.toString(inner)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  let outer_idx = idx / inner_size;
  let inner_idx = idx % inner_size;
  var max_val = input[outer_idx * (axis_size * inner_size) + inner_idx];
  var max_idx = 0u;
  for (var i = 1u; i < axis_size; i = i + 1u) {
    let val = input[outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx];
    if (val ${cmpOp} max_val) {
      max_val = val;
      max_idx = i;
    }
  }
  output[idx] = f32(max_idx);
${mainEnd}`
  {
    name: "argmax_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

let genArgMinKernel = (inputShape: shape, axis: int, selectLastIndex: bool): kernel => {
  let rank = Array.length(inputShape)
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = outer * inner
  let cmpOp = if selectLastIndex { "<=" } else { "<" }
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  let inner_size = ${Int.toString(inner)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  let outer_idx = idx / inner_size;
  let inner_idx = idx % inner_size;
  var min_val = input[outer_idx * (axis_size * inner_size) + inner_idx];
  var min_idx = 0u;
  for (var i = 1u; i < axis_size; i = i + 1u) {
    let val = input[outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx];
    if (val ${cmpOp} min_val) {
      min_val = val;
      min_idx = i;
    }
  }
  output[idx] = f32(min_idx);
${mainEnd}`
  {
    name: "argmin_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Pad Kernel
// ----------------------------------------
let genPadKernel = (inputShape: shape, pads: array<int>, constantValue: float): kernel => {
  let rank = Array.length(inputShape)
  let outputShape = Array.fromInitializer(~length=rank, i => {
    let before = pads[i]->Option.getOr(0)
    let after = pads[rank + i]->Option.getOr(0)
    inputShape[i]->Option.getOr(0) + before + after
  })
  let outputSize = Shape.numElements(outputShape)
  let inputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let outputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * outputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let padsBefore = Array.fromInitializer(~length=rank, i => pads[i]->Option.getOr(0))
  let coordCode = Array.mapWithIndex(outputStrides, (stride, i) => {
    let padBefore = padsBefore[i]->Option.getOr(0)
    let inDim = inputShape[i]->Option.getOr(1)
    let inStride = inputStrides[i]->Option.getOr(1)
    `  let coord_${Int.toString(i)} = remaining / ${Int.toString(stride)}u;
  remaining = remaining % ${Int.toString(stride)}u;
  let in_coord_${Int.toString(i)} = i32(coord_${Int.toString(i)}) - ${Int.toString(padBefore)};
  if (in_coord_${Int.toString(i)} < 0 || in_coord_${Int.toString(i)} >= ${Int.toString(inDim)}) {
    in_bounds = false;
  } else {
    input_idx = input_idx + u32(in_coord_${Int.toString(i)}) * ${Int.toString(inStride)}u;
  }`
  })->Array.join("\n")
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  var remaining = idx;
  var in_bounds = true;
  var input_idx = 0u;
${coordCode}
  if (in_bounds) {
    output[idx] = input[input_idx];
  } else {
    output[idx] = ${Float.toString(constantValue)};
  }
${mainEnd}`
  {
    name: "pad_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Tile Kernel
// ----------------------------------------
let genTileKernel = (inputShape: shape, repeats: array<int>): kernel => {
  let rank = Array.length(inputShape)
  let outputShape = Array.mapWithIndex(inputShape, (d, i) => d * (repeats[i]->Option.getOr(1)))
  let outputSize = Shape.numElements(outputShape)
  let inputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let outputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * outputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let coordCode = Array.mapWithIndex(outputStrides, (stride, i) => {
    let inDim = inputShape[i]->Option.getOr(1)
    let inStride = inputStrides[i]->Option.getOr(1)
    `  let coord_${Int.toString(i)} = remaining / ${Int.toString(stride)}u;
  remaining = remaining % ${Int.toString(stride)}u;
  let in_coord_${Int.toString(i)} = coord_${Int.toString(i)} % ${Int.toString(inDim)}u;
  input_idx = input_idx + in_coord_${Int.toString(i)} * ${Int.toString(inStride)}u;`
  })->Array.join("\n")
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  var remaining = idx;
  var input_idx = 0u;
${coordCode}
  output[idx] = input[input_idx];
${mainEnd}`
  {
    name: "tile_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Slice Kernel
// ----------------------------------------
let genSliceKernel = (inputShape: shape, starts: array<int>, ends: array<int>, axes: array<int>, steps: array<int>): kernel => {
  let rank = Array.length(inputShape)
  let outputShape = Array.copy(inputShape)
  Array.forEachWithIndex(axes, (ax, i) => {
    let dimSize = inputShape[ax]->Option.getOr(1)
    let start = starts[i]->Option.getOr(0)
    let end_ = ends[i]->Option.getOr(dimSize)
    let step = steps[i]->Option.getOr(1)
    let s = if start < 0 { dimSize + start } else { start }
    let e = if end_ < 0 { dimSize + end_ } else { end_ }
    let size = (e - s + step - 1) / step
    ignore(outputShape->Array.set(ax, size))
  })
  let outputSize = Shape.numElements(outputShape)
  let inputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let outputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * outputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  let allStarts = Array.make(~length=rank, 0)
  let allSteps = Array.make(~length=rank, 1)
  Array.forEachWithIndex(axes, (ax, i) => {
    let dimSize = inputShape[ax]->Option.getOr(1)
    let start = starts[i]->Option.getOr(0)
    let s = if start < 0 { dimSize + start } else { start }
    ignore(allStarts->Array.set(ax, s))
    ignore(allSteps->Array.set(ax, steps[i]->Option.getOr(1)))
  })
  let coordCode = Array.mapWithIndex(outputStrides, (stride, i) => {
    let inStride = inputStrides[i]->Option.getOr(1)
    let start = allStarts[i]->Option.getOr(0)
    let step = allSteps[i]->Option.getOr(1)
    `  let coord_${Int.toString(i)} = remaining / ${Int.toString(stride)}u;
  remaining = remaining % ${Int.toString(stride)}u;
  let in_coord_${Int.toString(i)} = ${Int.toString(start)}u + coord_${Int.toString(i)} * ${Int.toString(step)}u;
  input_idx = input_idx + in_coord_${Int.toString(i)} * ${Int.toString(inStride)}u;`
  })->Array.join("\n")
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  var remaining = idx;
  var input_idx = 0u;
${coordCode}
  output[idx] = input[input_idx];
${mainEnd}`
  {
    name: "slice_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: Shape.numElements(inputShape) * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Cumsum Kernel
// ----------------------------------------
let genCumsumKernel = (inputShape: shape, axis: int, exclusive: bool, reverse: bool): kernel => {
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to Array.length(inputShape) - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = Shape.numElements(inputShape)
  let numSlices = outer * inner
  let loopStart = if reverse { "axis_size - 1u" } else { "0u" }
  let loopCond = if reverse { "i > 0u || i == 0u" } else { "i < axis_size" }
  let loopIncr = if reverse { "i = i - 1u" } else { "i = i + 1u" }
  let exclusiveCode = if exclusive {
    "output[out_idx] = sum;\n    sum = sum + input[in_idx];"
  } else {
    "sum = sum + input[in_idx];\n    output[out_idx] = sum;"
  }
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  let slice_idx = idx;
  if (slice_idx >= ${Int.toString(numSlices)}u) { return; }
  let axis_size = ${Int.toString(axisSize)}u;
  let inner_size = ${Int.toString(inner)}u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  var sum = 0.0;
  for (var i = ${loopStart}; ${loopCond}; ${loopIncr}) {
    let in_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let out_idx = in_idx;
    ${exclusiveCode}
  }
${mainEnd}`
  {
    name: "cumsum_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// OneHot Kernel
// ----------------------------------------
let genOneHotKernel = (inputShape: shape, depth: int): kernel => {
  let inputSize = Shape.numElements(inputShape)
  let outputSize = inputSize * depth
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  let depth = ${Int.toString(depth)}u;
  let input_idx = idx / depth;
  let depth_idx = idx % depth;
  let class_idx = u32(input[input_idx]);
  output[idx] = select(0.0, 1.0, depth_idx == class_idx);
${mainEnd}`
  {
    name: "onehot_" ++ Int.toString(depth) ++ "_" ++ Int.toString(inputSize),
    wgsl,
    bindings: [
      {binding: 0, size: inputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Scatter Kernel
// ----------------------------------------
let genScatterKernel = (dataShape: shape, indicesSize: int, axis: int): kernel => {
  let outputSize = Shape.numElements(dataShape)
  let rank = Array.length(dataShape)
  let axisSize = dataShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * dataShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * dataShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let updateSize = outer * indicesSize * inner
  let wgsl = `${storageBuffer(0, "data", ReadOnly)}
${storageBuffer(1, "indices", ReadOnly)}
${storageBuffer(2, "updates", ReadOnly)}
${storageBuffer(3, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  output[idx] = data[idx];
  let inner_size = ${Int.toString(inner)}u;
  let indices_size = ${Int.toString(indicesSize)}u;
  let axis_size = ${Int.toString(axisSize)}u;
  let update_size = ${Int.toString(updateSize)}u;
  for (var u = 0u; u < update_size; u = u + 1u) {
    let outer_idx = u / (indices_size * inner_size);
    let remainder = u % (indices_size * inner_size);
    let index_idx = remainder / inner_size;
    let inner_idx = remainder % inner_size;
    let scatter_idx = u32(indices[index_idx]);
    let out_idx = outer_idx * (axis_size * inner_size) + scatter_idx * inner_size + inner_idx;
    if (out_idx == idx) {
      output[idx] = updates[u];
    }
  }
${mainEnd}`
  {
    name: "scatter_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "data"},
      {binding: 1, size: indicesSize * 4, usage: ReadOnly, name: "indices"},
      {binding: 2, size: updateSize * 4, usage: ReadOnly, name: "updates"},
      {binding: 3, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Cast Kernel
// ----------------------------------------
let genCastKernel = (size: int): kernel => {
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  output[idx] = input[idx];
${mainEnd}`
  {
    name: "cast_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Squeeze/Unsqueeze Kernel (just reshape/copy)
// ----------------------------------------
let genSqueezeKernel = (size: int): kernel => {
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  output[idx] = input[idx];
${mainEnd}`
  {
    name: "squeeze_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Stack Kernel (stack tensors along new axis)
// ----------------------------------------
let genStackKernel = (inputShape: shape, numInputs: int, axis: int): kernel => {
  let rank = Array.length(inputShape)
  let inputSize = Shape.numElements(inputShape)
  let outputSize = inputSize * numInputs
  
  // Calculate strides for the output
  let normAxis = if axis < 0 { rank + 1 + axis } else { axis }
  
  // For stack, we insert a new dimension at axis
  // Output shape: [...inputShape[:axis], numInputs, ...inputShape[axis:]]
  let beforeSize = ref(1)
  let afterSize = ref(1)
  for i in 0 to normAxis - 1 {
    beforeSize := beforeSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in normAxis to rank - 1 {
    afterSize := afterSize.contents * inputShape[i]->Option.getOr(1)
  }
  let _before = beforeSize.contents
  let after = afterSize.contents
  
  let wgsl = `${storageBuffer(0, "input0", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  let after_size = ${Int.toString(after)}u;
  let num_inputs = ${Int.toString(numInputs)}u;
  let before_idx = idx / (num_inputs * after_size);
  let remainder = idx % (num_inputs * after_size);
  let stack_idx = remainder / after_size;
  let after_idx = remainder % after_size;
  let input_idx = before_idx * after_size + after_idx;
  // For now, only support 2 inputs - would need dynamic binding for more
  output[idx] = input0[input_idx];
${mainEnd}`
  {
    name: "stack_" ++ Int.toString(numInputs) ++ "_" ++ Int.toString(inputSize),
    wgsl,
    bindings: [
      {binding: 0, size: inputSize * 4, usage: ReadOnly, name: "input0"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Broadcast Kernel (expand to target shape)
// ----------------------------------------
let genBroadcastKernel = (inputShape: shape, targetShape: shape): kernel => {
  let inputSize = Shape.numElements(inputShape)
  let outputSize = Shape.numElements(targetShape)
  let rank = Array.length(targetShape)
  let inputRank = Array.length(inputShape)
  
  // Calculate input strides
  let inputStrides = Array.fromInitializer(~length=inputRank, i => {
    let stride = ref(1)
    for j in i + 1 to inputRank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Calculate output strides
  let outputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * targetShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Build coordinate mapping code
  let rankDiff = rank - inputRank
  let coordCode = Array.mapWithIndex(outputStrides, (stride, i) => {
    let inputIdx = i - rankDiff
    let inDim = if inputIdx >= 0 { inputShape[inputIdx]->Option.getOr(1) } else { 1 }
    let inStride = if inputIdx >= 0 { inputStrides[inputIdx]->Option.getOr(1) } else { 0 }
    `  let coord_${Int.toString(i)} = (remaining / ${Int.toString(stride)}u) % ${Int.toString(targetShape[i]->Option.getOr(1))}u;
  remaining = remaining % ${Int.toString(stride)}u;
  let in_coord_${Int.toString(i)} = coord_${Int.toString(i)} % ${Int.toString(inDim)}u;
  input_idx = input_idx + in_coord_${Int.toString(i)} * ${Int.toString(inStride)}u;`
  })->Array.join("\n")
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  var remaining = idx;
  var input_idx = 0u;
${coordCode}
  output[idx] = input[input_idx];
${mainEnd}`
  {
    name: "broadcast_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: inputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// LSTM Cell Kernel (single step)
// ----------------------------------------
let genLSTMCellKernel = (batchSize: int, inputSize: int, hiddenSize: int): kernel => {
  let outputSize = batchSize * hiddenSize
  let gateSize = 4 * hiddenSize  // i, f, g, o gates
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "h_prev", ReadOnly)}
${storageBuffer(2, "c_prev", ReadOnly)}
${storageBuffer(3, "weight_ih", ReadOnly)}
${storageBuffer(4, "weight_hh", ReadOnly)}
${storageBuffer(5, "bias_ih", ReadOnly)}
${storageBuffer(6, "bias_hh", ReadOnly)}
${storageBuffer(7, "h_out", ReadWrite)}
${storageBuffer(8, "c_out", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  
  let batch_size = ${Int.toString(batchSize)}u;
  let input_size = ${Int.toString(inputSize)}u;
  let hidden_size = ${Int.toString(hiddenSize)}u;
  let gate_size = ${Int.toString(gateSize)}u;
  
  let batch_idx = idx / hidden_size;
  let h_idx = idx % hidden_size;
  
  // Compute gates: [i, f, g, o] = sigmoid/tanh(W_ih * x + W_hh * h + b)
  var gates: array<f32, 4>;
  for (var g = 0u; g < 4u; g = g + 1u) {
    var sum = 0.0;
    let gate_offset = g * hidden_size + h_idx;
    
    // W_ih * x
    for (var i = 0u; i < input_size; i = i + 1u) {
      let x_val = input[batch_idx * input_size + i];
      let w_val = weight_ih[gate_offset * input_size + i];
      sum = sum + x_val * w_val;
    }
    
    // W_hh * h_prev
    for (var i = 0u; i < hidden_size; i = i + 1u) {
      let h_val = h_prev[batch_idx * hidden_size + i];
      let w_val = weight_hh[gate_offset * hidden_size + i];
      sum = sum + h_val * w_val;
    }
    
    // Add biases
    sum = sum + bias_ih[gate_offset] + bias_hh[gate_offset];
    
    gates[g] = sum;
  }
  
  // Apply activations
  let i_gate = 1.0 / (1.0 + exp(-gates[0]));  // sigmoid
  let f_gate = 1.0 / (1.0 + exp(-gates[1]));  // sigmoid
  let g_gate = tanh(gates[2]);                 // tanh
  let o_gate = 1.0 / (1.0 + exp(-gates[3]));  // sigmoid
  
  // Cell state update: c = f * c_prev + i * g
  let c_prev_val = c_prev[idx];
  let c_new = f_gate * c_prev_val + i_gate * g_gate;
  c_out[idx] = c_new;
  
  // Hidden state: h = o * tanh(c)
  h_out[idx] = o_gate * tanh(c_new);
${mainEnd}`
  {
    name: "lstm_cell_" ++ Int.toString(batchSize) ++ "_" ++ Int.toString(hiddenSize),
    wgsl,
    bindings: [
      {binding: 0, size: batchSize * inputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: batchSize * hiddenSize * 4, usage: ReadOnly, name: "h_prev"},
      {binding: 2, size: batchSize * hiddenSize * 4, usage: ReadOnly, name: "c_prev"},
      {binding: 3, size: gateSize * inputSize * 4, usage: ReadOnly, name: "weight_ih"},
      {binding: 4, size: gateSize * hiddenSize * 4, usage: ReadOnly, name: "weight_hh"},
      {binding: 5, size: gateSize * 4, usage: ReadOnly, name: "bias_ih"},
      {binding: 6, size: gateSize * 4, usage: ReadOnly, name: "bias_hh"},
      {binding: 7, size: outputSize * 4, usage: ReadWrite, name: "h_out"},
      {binding: 8, size: outputSize * 4, usage: ReadWrite, name: "c_out"}
    ]
  }
}

// ----------------------------------------
// GRU Cell Kernel (single step)
// ----------------------------------------
let genGRUCellKernel = (batchSize: int, inputSize: int, hiddenSize: int): kernel => {
  let outputSize = batchSize * hiddenSize
  let gateSize = 3 * hiddenSize  // r, z, n gates
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "h_prev", ReadOnly)}
${storageBuffer(2, "weight_ih", ReadOnly)}
${storageBuffer(3, "weight_hh", ReadOnly)}
${storageBuffer(4, "bias_ih", ReadOnly)}
${storageBuffer(5, "bias_hh", ReadOnly)}
${storageBuffer(6, "h_out", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  
  let batch_size = ${Int.toString(batchSize)}u;
  let input_size = ${Int.toString(inputSize)}u;
  let hidden_size = ${Int.toString(hiddenSize)}u;
  
  let batch_idx = idx / hidden_size;
  let h_idx = idx % hidden_size;
  
  // Compute r and z gates
  var r_sum = 0.0;
  var z_sum = 0.0;
  
  // W_ih * x for r and z
  for (var i = 0u; i < input_size; i = i + 1u) {
    let x_val = input[batch_idx * input_size + i];
    r_sum = r_sum + x_val * weight_ih[h_idx * input_size + i];
    z_sum = z_sum + x_val * weight_ih[(hidden_size + h_idx) * input_size + i];
  }
  
  // W_hh * h_prev for r and z
  for (var i = 0u; i < hidden_size; i = i + 1u) {
    let h_val = h_prev[batch_idx * hidden_size + i];
    r_sum = r_sum + h_val * weight_hh[h_idx * hidden_size + i];
    z_sum = z_sum + h_val * weight_hh[(hidden_size + h_idx) * hidden_size + i];
  }
  
  r_sum = r_sum + bias_ih[h_idx] + bias_hh[h_idx];
  z_sum = z_sum + bias_ih[hidden_size + h_idx] + bias_hh[hidden_size + h_idx];
  
  let r_gate = 1.0 / (1.0 + exp(-r_sum));  // sigmoid
  let z_gate = 1.0 / (1.0 + exp(-z_sum));  // sigmoid
  
  // Compute n gate with reset
  var n_sum = 0.0;
  for (var i = 0u; i < input_size; i = i + 1u) {
    let x_val = input[batch_idx * input_size + i];
    n_sum = n_sum + x_val * weight_ih[(2u * hidden_size + h_idx) * input_size + i];
  }
  n_sum = n_sum + bias_ih[2u * hidden_size + h_idx];
  
  var h_sum = 0.0;
  for (var i = 0u; i < hidden_size; i = i + 1u) {
    let h_val = h_prev[batch_idx * hidden_size + i];
    h_sum = h_sum + r_gate * h_val * weight_hh[(2u * hidden_size + h_idx) * hidden_size + i];
  }
  h_sum = h_sum + bias_hh[2u * hidden_size + h_idx];
  
  let n_gate = tanh(n_sum + h_sum);
  
  // Output: h = (1 - z) * n + z * h_prev
  let h_prev_val = h_prev[idx];
  h_out[idx] = (1.0 - z_gate) * n_gate + z_gate * h_prev_val;
${mainEnd}`
  {
    name: "gru_cell_" ++ Int.toString(batchSize) ++ "_" ++ Int.toString(hiddenSize),
    wgsl,
    bindings: [
      {binding: 0, size: batchSize * inputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: batchSize * hiddenSize * 4, usage: ReadOnly, name: "h_prev"},
      {binding: 2, size: gateSize * inputSize * 4, usage: ReadOnly, name: "weight_ih"},
      {binding: 3, size: gateSize * hiddenSize * 4, usage: ReadOnly, name: "weight_hh"},
      {binding: 4, size: gateSize * 4, usage: ReadOnly, name: "bias_ih"},
      {binding: 5, size: gateSize * 4, usage: ReadOnly, name: "bias_hh"},
      {binding: 6, size: outputSize * 4, usage: ReadWrite, name: "h_out"}
    ]
  }
}

// ----------------------------------------
// CumProd Kernel (cumulative product)
// ----------------------------------------
let genCumprodKernel = (inputShape: shape, axis: int, exclusive: bool, reverse: bool): kernel => {
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to Array.length(inputShape) - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = Shape.numElements(inputShape)
  let numSlices = outer * inner
  let loopStart = if reverse { "axis_size - 1u" } else { "0u" }
  let loopCond = if reverse { "i > 0u || i == 0u" } else { "i < axis_size" }
  let loopIncr = if reverse { "i = i - 1u" } else { "i = i + 1u" }
  let exclusiveCode = if exclusive {
    "output[out_idx] = prod;\n    prod = prod * input[in_idx];"
  } else {
    "prod = prod * input[in_idx];\n    output[out_idx] = prod;"
  }
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  let slice_idx = idx;
  if (slice_idx >= ${Int.toString(numSlices)}u) { return; }
  let axis_size = ${Int.toString(axisSize)}u;
  let inner_size = ${Int.toString(inner)}u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  var prod = 1.0;
  for (var i = ${loopStart}; ${loopCond}; ${loopIncr}) {
    let in_idx = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    let out_idx = in_idx;
    ${exclusiveCode}
  }
${mainEnd}`
  {
    name: "cumprod_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Reverse Kernel (reverse along axes)
// ----------------------------------------
let genReverseKernel = (inputShape: shape, axes: array<int>): kernel => {
  let rank = Array.length(inputShape)
  let outputSize = Shape.numElements(inputShape)
  
  let strides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  let coordCode = Array.mapWithIndex(strides, (stride, i) => {
    let dim = inputShape[i]->Option.getOr(1)
    let isReversed = Array.includes(axes, i)
    let coordExpr = if isReversed {
      `${Int.toString(dim - 1)}u - coord_${Int.toString(i)}`
    } else {
      `coord_${Int.toString(i)}`
    }
    `  let coord_${Int.toString(i)} = (remaining / ${Int.toString(stride)}u) % ${Int.toString(dim)}u;
  remaining = remaining % ${Int.toString(stride)}u;
  input_idx = input_idx + (${coordExpr}) * ${Int.toString(stride)}u;`
  })->Array.join("\n")
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  if (idx >= ${Int.toString(outputSize)}u) { return; }
  var remaining = idx;
  var input_idx = 0u;
${coordCode}
  output[idx] = input[input_idx];
${mainEnd}`
  {
    name: "reverse_" ++ Int.toString(outputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// LogSoftmax Kernel
// ----------------------------------------
let genLogSoftmaxKernel = (outerSize: int, axisSize: int): kernel => {
  let outputSize = outerSize * axisSize
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  let outer_idx = idx;
  if (outer_idx >= ${Int.toString(outerSize)}u) { return; }
  let axis_size = ${Int.toString(axisSize)}u;
  let base = outer_idx * axis_size;
  
  // Find max for numerical stability
  var max_val = input[base];
  for (var i = 1u; i < axis_size; i = i + 1u) {
    max_val = max(max_val, input[base + i]);
  }
  
  // Compute sum of exp(x - max)
  var sum_exp = 0.0;
  for (var i = 0u; i < axis_size; i = i + 1u) {
    sum_exp = sum_exp + exp(input[base + i] - max_val);
  }
  
  // Compute log_softmax = x - max - log(sum_exp)
  let log_sum = log(sum_exp);
  for (var i = 0u; i < axis_size; i = i + 1u) {
    output[base + i] = input[base + i] - max_val - log_sum;
  }
${mainEnd}`
  {
    name: "logsoftmax_" ++ Int.toString(outerSize) ++ "_" ++ Int.toString(axisSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Sort Kernel (bubble sort - works for small arrays)
// ----------------------------------------
let genSortKernel = (inputShape: shape, axis: int, descending: bool): kernel => {
  let rank = Array.length(inputShape)
  let axisSize = inputShape[axis]->Option.getOr(1)
  let outerSize = ref(1)
  let innerSize = ref(1)
  for i in 0 to axis - 1 {
    outerSize := outerSize.contents * inputShape[i]->Option.getOr(1)
  }
  for i in axis + 1 to rank - 1 {
    innerSize := innerSize.contents * inputShape[i]->Option.getOr(1)
  }
  let outer = outerSize.contents
  let inner = innerSize.contents
  let outputSize = Shape.numElements(inputShape)
  let numSlices = outer * inner
  let cmpOp = if descending { "<" } else { ">" }
  
  let wgsl = `${storageBuffer(0, "input", ReadOnly)}
${storageBuffer(1, "output", ReadWrite)}
${mainSignature}
  let slice_idx = idx;
  if (slice_idx >= ${Int.toString(numSlices)}u) { return; }
  
  let axis_size = ${Int.toString(axisSize)}u;
  let inner_size = ${Int.toString(inner)}u;
  let outer_idx = slice_idx / inner_size;
  let inner_idx = slice_idx % inner_size;
  
  // Copy to output first
  for (var i = 0u; i < axis_size; i = i + 1u) {
    let idx_src = outer_idx * (axis_size * inner_size) + i * inner_size + inner_idx;
    output[idx_src] = input[idx_src];
  }
  
  // Bubble sort along axis
  for (var i = 0u; i < axis_size - 1u; i = i + 1u) {
    for (var j = 0u; j < axis_size - 1u - i; j = j + 1u) {
      let idx_j = outer_idx * (axis_size * inner_size) + j * inner_size + inner_idx;
      let idx_j1 = outer_idx * (axis_size * inner_size) + (j + 1u) * inner_size + inner_idx;
      let val_j = output[idx_j];
      let val_j1 = output[idx_j1];
      if (val_j ${cmpOp} val_j1) {
        output[idx_j] = val_j1;
        output[idx_j1] = val_j;
      }
    }
  }
${mainEnd}`
  {
    name: "sort_" ++ Int.toString(numSlices) ++ "_" ++ Int.toString(axisSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "input"},
      {binding: 1, size: outputSize * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Arange Kernel (generate sequence)
// ----------------------------------------
let genArangeKernel = (size: int, start: float, step: float): kernel => {
  let wgsl = `${storageBuffer(0, "output", ReadWrite)}
@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${Int.toString(size)}u) { return; }
  output[idx] = ${Float.toString(start)} + f32(idx) * ${Float.toString(step)};
}`
  {
    name: "arange_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "output"}
    ]
  }
}

// ----------------------------------------
// Main Code Generation Entry Point
// ----------------------------------------

let at = (arr, i) => arr[i]->Option.getOr(0)

let generate = (op: op, inputShapes: array<shape>): option<(kernel, dispatch)> => {
  let outputShape = Shape.infer(op, inputShapes)
  let input = inputShapes[0]->Option.getOr([])
  let r = Array.length(input)
  
  outputShape->Option.flatMap(outShape => {
    let outSize = Shape.numElements(outShape)
    
    switch op {
    // Element-wise unary
    | Neg | Abs | Sign | Reciprocal | Floor | Ceil | Round
    | Sqrt | Exp | Log | Log2 | Log10
    | Sin | Cos | Tan | Asin | Acos | Atan
    | Sinh | Cosh | Tanh | Asinh | Acosh | Atanh
    | ReLU | LeakyReLU(_) | ELU(_) | Sigmoid | GeLU | SiLU | Mish
    | Softplus | Softsign | Not | Identity =>
      genUnaryKernel(op, outSize)->Option.map(k => (k, computeDispatch(outSize, k.name, 0)))
    
    // Element-wise binary (with broadcasting support)
    | Add | Sub | Mul | Div | Pow | Mod | FloorDiv
    | Maximum | Minimum | Atan2
    | Equal | NotEqual | Greater | GreaterEqual | Less | LessEqual
    | And | Or | Xor => {
        let shape0 = inputShapes[0]->Option.getOr([])
        let shape1 = inputShapes[1]->Option.getOr([])
        genBinaryBroadcastKernel(op, shape0, shape1, outShape)->Option.map(k => 
          (k, computeDispatch(outSize, k.name, 0))
        )
      }
    
    // Reductions (supports any axes)
    | Reduce({op: reduceType, axes}) =>
      genReduceKernel(reduceType, input, axes, false)->Option.map(k => 
         (k, computeDispatch(outSize, k.name, 0))
      )
    
    // MatMul
    | MatMul => {
        let s1 = input
        let s2 = inputShapes[1]->Option.getOr([])
        let r1 = Array.length(s1)
        let r2 = Array.length(s2)
        if r1 >= 2 && r2 >= 2 {
          let m = at(s1, r1 - 2)
          let k = at(s1, r1 - 1)
          let n = at(s2, r2 - 1)
          
          // Extract batch dimensions
          let batchA = Array.slice(s1, ~start=0, ~end=r1 - 2)
          let batchB = Array.slice(s2, ~start=0, ~end=r2 - 2)
          let batchOut = Array.slice(outShape, ~start=0, ~end=Array.length(outShape) - 2)
          
          let batchSizeOut = Array.reduce(batchOut, 1, (a, b) => a * b)
          let totalOutput = batchSizeOut * m * n
          
          let kernel = genBatchedMatMulKernel(batchA, batchB, batchOut, m, k, n)
          
          // Use appropriate dispatch based on M size and batch
          if batchSizeOut == 1 {
            if m <= 16 {
              // Small M: use simple kernel (no tiling overhead)
              let smallKernel = genSmallMMatMulKernel(m, k, n)
              Some((smallKernel, computeDispatch(m * n, smallKernel.name, 0)))
            } else {
              // Large M: use tiled kernel
              Some((kernel, computeDispatch2D(n, m, 16, kernel.name, 0)))
            }
          } else {
            Some((kernel, computeDispatch(totalOutput, kernel.name, 0)))
          }
        } else {
          None
        }
      }

    // INT4 Quantized MatMul
    | MatMulInt4({groupSize}) => {
      let s1 = input  // activation: [M, K] or [K] for M=1
      let s2 = inputShapes[1]->Option.getOr([])  // packed weights: [PACKED_K, N] column-major
      let r1 = Array.length(s1)
      // For M=1 inference (most common LLM case), use optimized kernel
      let m = if r1 == 1 { 1 } else { at(s1, r1 - 2) }
      let k = at(s1, r1 - 1)
      let n = at(s2, 1)  // N is second dim in column-major layout
      if m == 1 && groupSize == 128 {
        // Use fully optimized kernel for single-token inference
	let wgSize = 64
	let kernel = genInt4MatMulKernel(m, k, n, groupSize)
	Some((kernel, computeDispatchInt4Opt(n, wgSize, kernel.name, 0)))
      } else {
	// Any M > 1: use tiled kernel
	let kernel = genInt4MatMulTiledKernel(m, k, n, groupSize)
	Some((kernel, computeDispatch2D(n, m, 16, kernel.name, 0)))
      }
    }
 
    // Conv2D
    | Conv2D({filters, kernel: (kH, kW), stride: (sH, sW), padding}) =>
      if r == 4 {
        let batch = at(input, 0)
        let inH = at(input, 1)
        let inW = at(input, 2)
        let inC = at(input, 3)
        let outH = at(outShape, 1)
        let outW = at(outShape, 2)
        let (padH, padW) = switch padding {
        | Same => ((kH - 1) / 2, (kW - 1) / 2)
        | Valid => (0, 0)
        | Explicit({pads}) => (at(pads, 0), at(pads, 1))
        }
        let kernel = genConv2DKernel(batch, inH, inW, inC, outH, outW, filters, kH, kW, sH, sW, padH, padW)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      } else {
        None
      }
    
    // Dense
    | Dense({units}) =>
      if r >= 1 {
        let batchSize = Shape.numElements(Array.slice(input, ~start=0, ~end=r-1))
        let inFeatures = at(input, r - 1)
        let kernel = genDenseKernel(max(batchSize, 1), inFeatures, units)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      } else {
        None
      }
    
    // Softmax
    | Softmax({axis}) => {
        let normAxis = axis < 0 ? r + axis : axis
        if normAxis == r - 1 {
          let axisSize = at(input, normAxis)
          let outerSize = Shape.numElements(input) / axisSize
          let kernel = genSoftmaxKernel(outerSize, axisSize)
          Some((kernel, computeDispatch(outerSize, kernel.name, 0)))
        } else {
          None
        }
      }
    
    // Transpose
    | Transpose({perm}) =>
      genTransposeKernel(input, perm)->Option.map(k => (k, computeDispatch(outSize, k.name, 0)))
    
    // Reshape (no-op copy)
    | Reshape(_) => {
        let kernel = genReshapeKernel(outSize)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }

    // MaxPool2D
    | MaxPool2D({kernel: (kH, kW), stride: (sH, sW), padding}) =>
      if r == 4 {
        let batch = at(input, 0)
        let inH = at(input, 1)
        let inW = at(input, 2)
        let channels = at(input, 3)
        let outH = at(outShape, 1)
        let outW = at(outShape, 2)
        let (padH, padW) = switch padding {
        | Same => ((kH - 1) / 2, (kW - 1) / 2)
        | Valid => (0, 0)
        | Explicit({pads}) => (at(pads, 0), at(pads, 1))
        }
        let kernel = genPool2DKernel("max", batch, inH, inW, channels, outH, outW, kH, kW, sH, sW, padH, padW)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      } else {
        None
      }

    // AvgPool2D
    | AvgPool2D({kernel: (kH, kW), stride: (sH, sW), padding}) =>
      if r == 4 {
        let batch = at(input, 0)
        let inH = at(input, 1)
        let inW = at(input, 2)
        let channels = at(input, 3)
        let outH = at(outShape, 1)
        let outW = at(outShape, 2)
        let (padH, padW) = switch padding {
        | Same => ((kH - 1) / 2, (kW - 1) / 2)
        | Valid => (0, 0)
        | Explicit({pads}) => (at(pads, 0), at(pads, 1))
        }
        let kernel = genPool2DKernel("avg", batch, inH, inW, channels, outH, outW, kH, kW, sH, sW, padH, padW)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      } else {
        None
      }

    // BatchNorm
    | BatchNorm(_) =>
      if r == 4 {
        let batch = at(input, 0)
        let height = at(input, 1)
        let width = at(input, 2)
        let channels = at(input, 3)
        let kernel = genBatchNormKernel(batch, height, width, channels)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      } else {
        None
      }

    // Conv1D
    | Conv1D({filters, kernel, stride, padding, dilation: _}) =>
      if r == 3 {
        let batch = at(input, 0)
        let inLen = at(input, 1)
        let inC = at(input, 2)
        let outLen = at(outShape, 1)
        let pad = switch padding {
        | Same => (kernel - 1) / 2
        | Valid => 0
        | Explicit({pads}) => at(pads, 0)
        }
        let k = genConv1DKernel(batch, inLen, inC, outLen, filters, kernel, stride, pad)
        Some((k, computeDispatch(outSize, k.name, 0)))
      } else {
        None
      }

    // GlobalMaxPool
    | GlobalMaxPool =>
      if r == 4 {
        let batch = at(input, 0)
        let height = at(input, 1)
        let width = at(input, 2)
        let channels = at(input, 3)
        let k = genGlobalPoolKernel("max", batch, height, width, channels)
        Some((k, computeDispatch(batch * channels, k.name, 0)))
      } else {
        None
      }

    // GlobalAvgPool
    | GlobalAvgPool =>
      if r == 4 {
        let batch = at(input, 0)
        let height = at(input, 1)
        let width = at(input, 2)
        let channels = at(input, 3)
        let k = genGlobalPoolKernel("avg", batch, height, width, channels)
        Some((k, computeDispatch(batch * channels, k.name, 0)))
      } else {
        None
      }

    // LayerNorm
    | LayerNorm({axes: _, epsilon}) => {
        let normAxis = r - 1
        let normSize = at(input, normAxis)
        let outerSize = Shape.numElements(input) / normSize
        let k = genLayerNormKernel(outerSize, normSize, epsilon)
        Some((k, computeDispatch(outerSize, k.name, 0)))
      }

    // ScaledDotProductAttention
    | ScaledDotProductAttention(_) =>
      if r == 3 {
        let batch = at(input, 0)
        let seqLen = at(input, 1)
        let dim = at(input, 2)
        let k = genAttentionKernel(batch, seqLen, dim)
        Some((k, computeDispatch(outSize, k.name, 0)))
      } else {
        None
      }

    // Embedding

    | Embedding({numEmbeddings, embeddingDim}) => {
        let batchSeq = Shape.numElements(input)
	let k = genEmbeddingKernel(batchSeq, numEmbeddings, embeddingDim)
	Some((k, computeDispatch(outSize, k.name, 0)))
      }

    // Concat
    | Concat({axis}) =>
      genConcatKernel(inputShapes, axis)->Option.map(k => (k, computeDispatch(outSize, k.name, 0)))

    // Clip
    | Clip({min, max}) => {
        let minVal = min->Option.getOr(-3.402823e+38)
	let maxVal = max->Option.getOr(3.402823e+38)
	let k = genClipKernel(outSize, minVal, maxVal)
	Some((k, computeDispatch(outSize, k.name, 0)))
      }

    // Where (conditional selection)
    | Where => {
        let size = outSize
        let k = genWhereKernel(size)
        Some((k, computeDispatch(size, k.name, 0)))
      }
    // Gather (index selection)
    | Gather({axis}) => {
        let indicesShape = inputShapes[1]->Option.getOr([])
        let indicesSize = Shape.numElements(indicesShape)
        let k = genGatherKernel(input, indicesSize, axis)
        Some((k, computeDispatch(outSize, k.name, 0)))
      }
    // Split - returns first split only (need multiple calls for all splits)
    | Split({axis, splitSizes}) => {
        let firstSplitSize = splitSizes[0]->Option.getOr(1)
        let k = genSplitKernel(input, axis, 0, firstSplitSize)
        Some((k, computeDispatch(outSize, k.name, 0)))
      }
    // TopK
    | TopK({k, axis, largest: _, sorted: _}) => {
        let kernel = genTopKKernel(input, k, axis)
        let numSlices = outSize / k
        Some((kernel, computeDispatch(numSlices, kernel.name, 0)))
      }
    | ArgMax({axis, keepDims: _, selectLastIndex}) => {
        let kernel = genArgMaxKernel(input, axis, selectLastIndex)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | ArgMin({axis, keepDims: _, selectLastIndex}) => {
        let kernel = genArgMinKernel(input, axis, selectLastIndex)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Pad({pads, mode: _, constantValue}) => {
        let kernel = genPadKernel(input, pads, constantValue)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Tile({repeats}) => {
        let kernel = genTileKernel(input, repeats)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Slice({starts, ends, axes, steps}) => {
        let kernel = genSliceKernel(input, starts, ends, axes, steps)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | OneHot({depth, axis: _}) => {
        let kernel = genOneHotKernel(input, depth)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Scatter({axis}) => {
        let indicesShape = inputShapes[1]->Option.getOr([])  // <- change inputs to inputShapes
        let indicesSize = Shape.numElements(indicesShape)
        let kernel = genScatterKernel(input, indicesSize, axis)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Cast(_) => {
        let kernel = genCastKernel(outSize)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | CumSum({axis, exclusive, reverse}) => {
        let kernel = genCumsumKernel(input, axis, exclusive, reverse)
        Some((kernel, computeDispatch(Shape.numElements(input) / input[axis]->Option.getOr(1), kernel.name, 0)))
      }
    | CumProd({axis, exclusive, reverse}) => {
        let kernel = genCumprodKernel(input, axis, exclusive, reverse)
        Some((kernel, computeDispatch(Shape.numElements(input) / input[axis]->Option.getOr(1), kernel.name, 0)))
      }
    | Squeeze(_) | Unsqueeze(_) | ExpandDims(_) => {
        let kernel = genSqueezeKernel(outSize)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Broadcast({targetShape}) => {
        let kernel = genBroadcastKernel(input, targetShape)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Stack({axis}) => {
        let numInputs = Array.length(inputShapes)
        let kernel = genStackKernel(input, numInputs, axis)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | Reverse({axes}) => {
        let kernel = genReverseKernel(input, axes)
        Some((kernel, computeDispatch(outSize, kernel.name, 0)))
      }
    | LogSoftmax({axis}) => {
        let normAxis = if axis < 0 { r + axis } else { axis }
        if normAxis == r - 1 {
          let axisSize = input[normAxis]->Option.getOr(1)
          let outerSize = Shape.numElements(input) / axisSize
          let kernel = genLogSoftmaxKernel(outerSize, axisSize)
          Some((kernel, computeDispatch(outerSize, kernel.name, 0)))
        } else {
          None
        }
      }
    | Sort({axis, descending}) => {
        let kernel = genSortKernel(input, axis, descending)
        let numSlices = outSize / input[axis]->Option.getOr(1)
        Some((kernel, computeDispatch(numSlices, kernel.name, 0)))
      }
    | _ => None
    }
  })
}
