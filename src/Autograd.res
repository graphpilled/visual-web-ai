// src/Autograd.res
// Automatic differentiation for training
//
// This module provides:
// 1. Backward kernel generators for computing gradients
// 2. Gradient reduction for broadcasting (sum gradients to original shape)
// 3. Optimizers (SGD, Adam)

open Types

// ============================================
// Helper Functions
// ============================================

let workgroupSize = 256

let storageBufferRO = (binding: int, name: string): string =>
  `@group(0) @binding(${Int.toString(binding)}) var<storage, read> ${name}: array<f32>;`

let storageBufferRW = (binding: int, name: string): string =>
  `@group(0) @binding(${Int.toString(binding)}) var<storage, read_write> ${name}: array<f32>;`

let mainSignature = `@compute @workgroup_size(${Int.toString(workgroupSize)})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;`

let mainEnd = "}"

let computeDispatch = (totalElements: int, kernelName: string, pipelineIndex: int): dispatch => {
  let workgroupCount = (totalElements + workgroupSize - 1) / workgroupSize
  {
    workgroupSize: (workgroupSize, 1, 1),
    workgroupCount: (workgroupCount, 1, 1),
    kernelName,
    pipelineIndex,
  }
}

// ============================================
// Gradient Reduction for Broadcasting
// When forward: [1,3] + [2,3] -> [2,3] (broadcast first input)
// Backward: grad_a needs to sum over axis 0 to go from [2,3] -> [1,3]
// ============================================

// Reduce gradient to match original shape (handles broadcasting)
let genGradReduceKernel = (
  gradShape: shape,      // Shape of incoming gradient
  targetShape: shape,    // Shape gradient needs to be reduced to
): kernel => {
  let gradSize = Shape.numElements(gradShape)
  let targetSize = Shape.numElements(targetShape)
  let gradRank = Array.length(gradShape)
  let targetRank = Array.length(targetShape)
  
  // Calculate which axes need reduction
  let maxRank = max(gradRank, targetRank)
  
  // Pad shapes to same rank (prepend 1s)
  let paddedGrad = Array.concat(
    Array.make(~length=maxRank - gradRank, 1),
    gradShape
  )
  let paddedTarget = Array.concat(
    Array.make(~length=maxRank - targetRank, 1),
    targetShape
  )
  
  // Find axes that were broadcast (target dim is 1, grad dim > 1)
  let _reduceAxes = Array.filterWithIndex(
    Array.mapWithIndex(paddedTarget, (dim, i) => (dim, i)),
    (elem, _idx) => { let (dim, i) = elem; dim == 1 && paddedGrad[i]->Option.getOr(1) > 1 }
  )->Array.map(((_, i)) => i)
  
  // Calculate strides for grad tensor
  let gradStrides = Array.fromInitializer(~length=maxRank, i => {
    let stride = ref(1)
    for j in i + 1 to maxRank - 1 {
      stride := stride.contents * paddedGrad[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Calculate strides for target tensor
  let targetStrides = Array.fromInitializer(~length=maxRank, i => {
    let stride = ref(1)
    for j in i + 1 to maxRank - 1 {
      stride := stride.contents * paddedTarget[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  let gradShapeStr = Array.map(paddedGrad, d => Int.toString(d))->Array.join(", ")
  let targetShapeStr = Array.map(paddedTarget, d => Int.toString(d))->Array.join(", ")
  let gradStridesStr = Array.map(gradStrides, d => Int.toString(d))->Array.join(", ")
  let targetStridesStr = Array.map(targetStrides, d => Int.toString(d))->Array.join(", ")
  
  let wgsl = if targetSize == gradSize {
    // No reduction needed - just copy
    `${storageBufferRO(0, "grad_in")}
${storageBufferRW(1, "grad_out")}
${mainSignature}
  if (idx >= ${Int.toString(targetSize)}u) { return; }
  grad_out[idx] = grad_in[idx];
${mainEnd}`
  } else {
    // Need to sum over broadcast dimensions
    `${storageBufferRO(0, "grad_in")}
${storageBufferRW(1, "grad_out")}
const RANK = ${Int.toString(maxRank)}u;
const GRAD_SIZE = ${Int.toString(gradSize)}u;
const TARGET_SIZE = ${Int.toString(targetSize)}u;
const GRAD_SHAPE = array<u32, ${Int.toString(maxRank)}>(${gradShapeStr});
const TARGET_SHAPE = array<u32, ${Int.toString(maxRank)}>(${targetShapeStr});
const GRAD_STRIDES = array<u32, ${Int.toString(maxRank)}>(${gradStridesStr});
const TARGET_STRIDES = array<u32, ${Int.toString(maxRank)}>(${targetStridesStr});
${mainSignature}
  if (idx >= TARGET_SIZE) { return; }
  
  // Convert target idx to coordinates
  var target_coords: array<u32, ${Int.toString(maxRank)}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    target_coords[d] = remaining / TARGET_STRIDES[d];
    remaining = remaining % TARGET_STRIDES[d];
  }
  
  // Sum over all grad elements that map to this target element
  var sum = 0.0;
  for (var g = 0u; g < GRAD_SIZE; g = g + 1u) {
    // Convert grad idx to coordinates
    var grad_coords: array<u32, ${Int.toString(maxRank)}>;
    var rem = g;
    for (var d = 0u; d < RANK; d = d + 1u) {
      grad_coords[d] = rem / GRAD_STRIDES[d];
      rem = rem % GRAD_STRIDES[d];
    }
    
    // Check if this grad element maps to our target element
    var matches = true;
    for (var d = 0u; d < RANK; d = d + 1u) {
      if (TARGET_SHAPE[d] > 1u && grad_coords[d] != target_coords[d]) {
        matches = false;
        break;
      }
    }
    
    if (matches) {
      sum = sum + grad_in[g];
    }
  }
  
  grad_out[idx] = sum;
${mainEnd}`
  }
  
  {
    name: "grad_reduce_" ++ Int.toString(gradSize) ++ "_to_" ++ Int.toString(targetSize),
    wgsl,
    bindings: [
      {binding: 0, size: gradSize * 4, usage: ReadOnly, name: "grad_in"},
      {binding: 1, size: targetSize * 4, usage: ReadWrite, name: "grad_out"},
    ],
  }
}

// ============================================
// Element-wise Unary Backward Kernels
// ============================================

// Neg backward: dL/dx = -dL/dout
let genNegBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = -grad_out[idx];
${mainEnd}`
  {
    name: "neg_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Abs backward: dL/dx = dL/dout * sign(x)
let genAbsBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = grad_out[idx] * sign(x[idx]);
${mainEnd}`
  {
    name: "abs_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Sqrt backward: dL/dx = dL/dout * 0.5 / sqrt(x) = dL/dout * 0.5 / out
let genSqrtBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "out")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let out_val = out[idx];
  grad_x[idx] = select(0.0, grad_out[idx] * 0.5 / out_val, out_val > 0.0);
${mainEnd}`
  {
    name: "sqrt_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "out"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Exp backward: dL/dx = dL/dout * exp(x) = dL/dout * out
let genExpBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "out")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = grad_out[idx] * out[idx];
${mainEnd}`
  {
    name: "exp_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "out"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Log backward: dL/dx = dL/dout / x
let genLogBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = grad_out[idx] / x[idx];
${mainEnd}`
  {
    name: "log_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Sin backward: dL/dx = dL/dout * cos(x)
let genSinBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = grad_out[idx] * cos(x[idx]);
${mainEnd}`
  {
    name: "sin_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Cos backward: dL/dx = -dL/dout * sin(x)
let genCosBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = -grad_out[idx] * sin(x[idx]);
${mainEnd}`
  {
    name: "cos_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Tanh backward: dL/dx = dL/dout * (1 - tanh(x)^2) = dL/dout * (1 - out^2)
let genTanhBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "out")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let t = out[idx];
  grad_x[idx] = grad_out[idx] * (1.0 - t * t);
${mainEnd}`
  {
    name: "tanh_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "out"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Sigmoid backward: dL/dx = dL/dout * sigmoid(x) * (1 - sigmoid(x)) = dL/dout * out * (1 - out)
let genSigmoidBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "out")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let s = out[idx];
  grad_x[idx] = grad_out[idx] * s * (1.0 - s);
${mainEnd}`
  {
    name: "sigmoid_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "out"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// ReLU backward: dL/dx = dL/dout if x > 0 else 0
let genReLUBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = select(0.0, grad_out[idx], x[idx] > 0.0);
${mainEnd}`
  {
    name: "relu_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// LeakyReLU backward: dL/dx = dL/dout if x > 0 else alpha * dL/dout
let genLeakyReLUBackwardKernel = (size: int, alpha: float): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  grad_x[idx] = select(${Float.toString(alpha)} * g, g, x[idx] > 0.0);
${mainEnd}`
  {
    name: "leaky_relu_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// GeLU backward: dL/dx = dL/dout * (0.5 * (1 + erf(x/sqrt(2))) + x * exp(-x^2/2) / sqrt(2*pi))
let genGeLUBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRW(2, "grad_x")}
const SQRT_2 = 1.4142135623730951;
const SQRT_2_PI = 0.7978845608028654;
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let x_val = x[idx];
  let cdf = 0.5 * (1.0 + tanh(SQRT_2 / 2.0 * (x_val + 0.044715 * x_val * x_val * x_val)));
  let pdf = SQRT_2_PI * exp(-0.5 * x_val * x_val);
  grad_x[idx] = grad_out[idx] * (cdf + x_val * pdf);
${mainEnd}`
  {
    name: "gelu_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// ============================================
// Element-wise Binary Backward Kernels
// These compute local gradients; broadcasting reduction handled separately
// ============================================

// Add backward: dL/da = dL/dout, dL/db = dL/dout
let genAddBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_a")}
${storageBufferRW(2, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g;
  grad_b[idx] = g;
${mainEnd}`
  {
    name: "add_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Sub backward: dL/da = dL/dout, dL/db = -dL/dout
let genSubBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_a")}
${storageBufferRW(2, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g;
  grad_b[idx] = -g;
${mainEnd}`
  {
    name: "sub_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Mul backward: dL/da = dL/dout * b, dL/db = dL/dout * a
let genMulBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRO(2, "b")}
${storageBufferRW(3, "grad_a")}
${storageBufferRW(4, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  grad_a[idx] = g * b[idx];
  grad_b[idx] = g * a[idx];
${mainEnd}`
  {
    name: "mul_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "b"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 4, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Div backward: dL/da = dL/dout / b, dL/db = -dL/dout * a / b^2
let genDivBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRO(2, "b")}
${storageBufferRW(3, "grad_a")}
${storageBufferRW(4, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  let b_val = b[idx];
  grad_a[idx] = g / b_val;
  grad_b[idx] = -g * a[idx] / (b_val * b_val);
${mainEnd}`
  {
    name: "div_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "b"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 4, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Pow backward: dL/da = dL/dout * b * a^(b-1), dL/db = dL/dout * a^b * ln(a)
let genPowBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRO(2, "b")}
${storageBufferRO(3, "out")}
${storageBufferRW(4, "grad_a")}
${storageBufferRW(5, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  let out_val = out[idx];
  // dL/da = g * b * a^(b-1) = g * b * out / a
  grad_a[idx] = select(0.0, g * b_val * out_val / a_val, a_val != 0.0);
  // dL/db = g * a^b * ln(a) = g * out * ln(a)
  grad_b[idx] = select(0.0, g * out_val * log(a_val), a_val > 0.0);
${mainEnd}`
  {
    name: "pow_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "b"},
      {binding: 3, size: size * 4, usage: ReadOnly, name: "out"},
      {binding: 4, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 5, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Maximum backward: gradient goes to the larger input
let genMaximumBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRO(2, "b")}
${storageBufferRW(3, "grad_a")}
${storageBufferRW(4, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  if (a_val > b_val) {
    grad_a[idx] = g;
    grad_b[idx] = 0.0;
  } else if (b_val > a_val) {
    grad_a[idx] = 0.0;
    grad_b[idx] = g;
  } else {
    // Equal: split gradient
    grad_a[idx] = g * 0.5;
    grad_b[idx] = g * 0.5;
  }
${mainEnd}`
  {
    name: "maximum_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "b"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 4, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Minimum backward: gradient goes to the smaller input
let genMinimumBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRO(2, "b")}
${storageBufferRW(3, "grad_a")}
${storageBufferRW(4, "grad_b")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let g = grad_out[idx];
  let a_val = a[idx];
  let b_val = b[idx];
  if (a_val < b_val) {
    grad_a[idx] = g;
    grad_b[idx] = 0.0;
  } else if (b_val < a_val) {
    grad_a[idx] = 0.0;
    grad_b[idx] = g;
  } else {
    grad_a[idx] = g * 0.5;
    grad_b[idx] = g * 0.5;
  }
${mainEnd}`
  {
    name: "minimum_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: size * 4, usage: ReadOnly, name: "b"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "grad_a"},
      {binding: 4, size: size * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// ============================================
// MatMul Backward
// C = A @ B where A: [M, K], B: [K, N], C: [M, N]
// dL/dA = dL/dC @ B^T  (shape: [M, N] @ [N, K] = [M, K])
// dL/dB = A^T @ dL/dC  (shape: [K, M] @ [M, N] = [K, N])
// ============================================

// MatMul backward for A: dL/dA = dL/dC @ B^T
let genMatMulBackwardAKernel = (m: int, k: int, n: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "b")}
${storageBufferRW(2, "grad_a")}
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= M || col >= K) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < N; i = i + 1u) {
    // grad_out[row, i] * b[col, i] (B transposed: b[k, n] -> b^T[n, k])
    sum = sum + grad_out[row * N + i] * b[col * N + i];
  }
  grad_a[row * K + col] = sum;
}`
  {
    name: "matmul_backward_a_" ++ Int.toString(m) ++ "x" ++ Int.toString(k),
    wgsl,
    bindings: [
      {binding: 0, size: m * n * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: k * n * 4, usage: ReadOnly, name: "b"},
      {binding: 2, size: m * k * 4, usage: ReadWrite, name: "grad_a"},
    ],
  }
}

// MatMul backward for B: dL/dB = A^T @ dL/dC
let genMatMulBackwardBKernel = (m: int, k: int, n: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRW(2, "grad_b")}
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= K || col >= N) { return; }
  
  var sum = 0.0;
  for (var i = 0u; i < M; i = i + 1u) {
    // a[i, row] (A transposed: a[m, k] -> a^T[k, m]) * grad_out[i, col]
    sum = sum + a[i * K + row] * grad_out[i * N + col];
  }
  grad_b[row * N + col] = sum;
}`
  {
    name: "matmul_backward_b_" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: m * n * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: k * n * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// Batched MatMul backward for A
let genBatchedMatMulBackwardAKernel = (batch: int, m: int, k: int, n: int): kernel => {
  let totalOutput = batch * m * k
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "b")}
${storageBufferRW(2, "grad_a")}
const BATCH = ${Int.toString(batch)}u;
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
${mainSignature}
  if (idx >= ${Int.toString(totalOutput)}u) { return; }
  
  let batch_idx = idx / (M * K);
  let matrix_idx = idx % (M * K);
  let row = matrix_idx / K;
  let col = matrix_idx % K;
  
  var sum = 0.0;
  let grad_base = batch_idx * M * N;
  let b_base = batch_idx * K * N;
  
  for (var i = 0u; i < N; i = i + 1u) {
    sum = sum + grad_out[grad_base + row * N + i] * b[b_base + col * N + i];
  }
  grad_a[idx] = sum;
${mainEnd}`
  {
    name: "batched_matmul_backward_a_" ++ Int.toString(batch) ++ "_" ++ Int.toString(m) ++ "x" ++ Int.toString(k),
    wgsl,
    bindings: [
      {binding: 0, size: batch * m * n * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: batch * k * n * 4, usage: ReadOnly, name: "b"},
      {binding: 2, size: batch * m * k * 4, usage: ReadWrite, name: "grad_a"},
    ],
  }
}

// Batched MatMul backward for B
let genBatchedMatMulBackwardBKernel = (batch: int, m: int, k: int, n: int): kernel => {
  let totalOutput = batch * k * n
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "a")}
${storageBufferRW(2, "grad_b")}
const BATCH = ${Int.toString(batch)}u;
const M = ${Int.toString(m)}u;
const K = ${Int.toString(k)}u;
const N = ${Int.toString(n)}u;
${mainSignature}
  if (idx >= ${Int.toString(totalOutput)}u) { return; }
  
  let batch_idx = idx / (K * N);
  let matrix_idx = idx % (K * N);
  let row = matrix_idx / N;
  let col = matrix_idx % N;
  
  var sum = 0.0;
  let grad_base = batch_idx * M * N;
  let a_base = batch_idx * M * K;
  
  for (var i = 0u; i < M; i = i + 1u) {
    sum = sum + a[a_base + i * K + row] * grad_out[grad_base + i * N + col];
  }
  grad_b[idx] = sum;
${mainEnd}`
  {
    name: "batched_matmul_backward_b_" ++ Int.toString(batch) ++ "_" ++ Int.toString(k) ++ "x" ++ Int.toString(n),
    wgsl,
    bindings: [
      {binding: 0, size: batch * m * n * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: batch * m * k * 4, usage: ReadOnly, name: "a"},
      {binding: 2, size: batch * k * n * 4, usage: ReadWrite, name: "grad_b"},
    ],
  }
}

// ============================================
// Reduction Backward
// Sum/Mean: gradient broadcasts back to input shape
// ============================================

// Sum backward: dL/dx[i] = dL/dout (broadcast)
let genSumBackwardKernel = (inputShape: shape, outputShape: shape, axes: array<int>): kernel => {
  let inputSize = Shape.numElements(inputShape)
  let outputSize = Shape.numElements(outputShape)
  let rank = Array.length(inputShape)
  
  // Calculate input strides
  let inputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  // Normalize axes
  let normAxes = Array.map(axes, a => a < 0 ? rank + a : a)
  let isReducedAxis = (i: int) => Array.includes(normAxes, i)
  
  // Calculate output strides (for non-reduced dimensions)
  let _outputStrides = {
    let _outRank = rank - Array.length(axes)
    let strides = ref([])
    let stride = ref(1)
    for i in rank - 1 downto 0 {
      if !isReducedAxis(i) {
        strides := Array.concat([stride.contents], strides.contents)
      }
      stride := stride.contents * inputShape[i]->Option.getOr(1)
    }
    strides.contents
  }
  
  let inputShapeStr = Array.map(inputShape, d => Int.toString(d))->Array.join(", ")
  let inputStridesStr = Array.map(inputStrides, d => Int.toString(d))->Array.join(", ")
  let axesStr = Array.map(normAxes, d => Int.toString(d))->Array.join(", ")
  let numAxes = Array.length(normAxes)
  
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_x")}
const RANK = ${Int.toString(rank)}u;
const INPUT_SIZE = ${Int.toString(inputSize)}u;
const OUTPUT_SIZE = ${Int.toString(outputSize)}u;
const NUM_AXES = ${Int.toString(numAxes)}u;
const INPUT_SHAPE = array<u32, ${Int.toString(rank)}>(${inputShapeStr});
const INPUT_STRIDES = array<u32, ${Int.toString(rank)}>(${inputStridesStr});
const AXES = array<u32, ${Int.toString(numAxes)}>(${axesStr});

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

${mainSignature}
  if (idx >= INPUT_SIZE) { return; }
  
  // Convert flat idx to coordinates
  var coords: array<u32, ${Int.toString(rank)}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / INPUT_STRIDES[d];
    remaining = remaining % INPUT_STRIDES[d];
  }
  
  // Compute output index (skip reduced dimensions)
  var out_idx = 0u;
  var out_stride = 1u;
  for (var d = i32(RANK) - 1; d >= 0; d = d - 1) {
    if (!isReduceAxis(u32(d))) {
      out_idx = out_idx + coords[d] * out_stride;
      out_stride = out_stride * INPUT_SHAPE[d];
    }
  }
  
  grad_x[idx] = grad_out[out_idx];
${mainEnd}`
  {
    name: "sum_backward_" ++ Int.toString(inputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: inputSize * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// Mean backward: dL/dx[i] = dL/dout / count
let genMeanBackwardKernel = (inputShape: shape, outputShape: shape, axes: array<int>): kernel => {
  let inputSize = Shape.numElements(inputShape)
  let outputSize = Shape.numElements(outputShape)
  let rank = Array.length(inputShape)
  
  // Calculate reduction count
  let normAxes = Array.map(axes, a => a < 0 ? rank + a : a)
  let reduceCount = Array.reduce(normAxes, 1, (acc, axis) => 
    acc * inputShape[axis]->Option.getOr(1)
  )
  
  let inputStrides = Array.fromInitializer(~length=rank, i => {
    let stride = ref(1)
    for j in i + 1 to rank - 1 {
      stride := stride.contents * inputShape[j]->Option.getOr(1)
    }
    stride.contents
  })
  
  let inputShapeStr = Array.map(inputShape, d => Int.toString(d))->Array.join(", ")
  let inputStridesStr = Array.map(inputStrides, d => Int.toString(d))->Array.join(", ")
  let axesStr = Array.map(normAxes, d => Int.toString(d))->Array.join(", ")
  let numAxes = Array.length(normAxes)
  
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_x")}
const RANK = ${Int.toString(rank)}u;
const INPUT_SIZE = ${Int.toString(inputSize)}u;
const REDUCE_COUNT = ${Float.toString(Float.fromInt(reduceCount))};
const NUM_AXES = ${Int.toString(numAxes)}u;
const INPUT_SHAPE = array<u32, ${Int.toString(rank)}>(${inputShapeStr});
const INPUT_STRIDES = array<u32, ${Int.toString(rank)}>(${inputStridesStr});
const AXES = array<u32, ${Int.toString(numAxes)}>(${axesStr});

fn isReduceAxis(axis: u32) -> bool {
  for (var i = 0u; i < NUM_AXES; i = i + 1u) {
    if (AXES[i] == axis) { return true; }
  }
  return false;
}

${mainSignature}
  if (idx >= INPUT_SIZE) { return; }
  
  var coords: array<u32, ${Int.toString(rank)}>;
  var remaining = idx;
  for (var d = 0u; d < RANK; d = d + 1u) {
    coords[d] = remaining / INPUT_STRIDES[d];
    remaining = remaining % INPUT_STRIDES[d];
  }
  
  var out_idx = 0u;
  var out_stride = 1u;
  for (var d = i32(RANK) - 1; d >= 0; d = d - 1) {
    if (!isReduceAxis(u32(d))) {
      out_idx = out_idx + coords[d] * out_stride;
      out_stride = out_stride * INPUT_SHAPE[d];
    }
  }
  
  grad_x[idx] = grad_out[out_idx] / REDUCE_COUNT;
${mainEnd}`
  {
    name: "mean_backward_" ++ Int.toString(inputSize),
    wgsl,
    bindings: [
      {binding: 0, size: outputSize * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: inputSize * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// ============================================
// Softmax Backward (Jacobian-vector product)
// ============================================

let genSoftmaxBackwardKernel = (outerSize: int, axisSize: int): kernel => {
  let totalSize = outerSize * axisSize
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "softmax_out")}
${storageBufferRW(2, "grad_x")}
const OUTER_SIZE = ${Int.toString(outerSize)}u;
const AXIS_SIZE = ${Int.toString(axisSize)}u;
${mainSignature}
  let outer_idx = idx;
  if (outer_idx >= OUTER_SIZE) { return; }
  
  let base = outer_idx * AXIS_SIZE;
  
  // Compute dot product: sum_j(grad_out[j] * softmax[j])
  var dot = 0.0;
  for (var j = 0u; j < AXIS_SIZE; j = j + 1u) {
    dot = dot + grad_out[base + j] * softmax_out[base + j];
  }
  
  // grad_x[i] = softmax[i] * (grad_out[i] - dot)
  for (var i = 0u; i < AXIS_SIZE; i = i + 1u) {
    let s = softmax_out[base + i];
    grad_x[base + i] = s * (grad_out[base + i] - dot);
  }
${mainEnd}`
  {
    name: "softmax_backward_" ++ Int.toString(outerSize) ++ "x" ++ Int.toString(axisSize),
    wgsl,
    bindings: [
      {binding: 0, size: totalSize * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: totalSize * 4, usage: ReadOnly, name: "softmax_out"},
      {binding: 2, size: totalSize * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// ============================================
// LayerNorm Backward
// ============================================

let genLayerNormBackwardKernel = (outerSize: int, normSize: int, epsilon: float): kernel => {
  let totalSize = outerSize * normSize
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRO(1, "x")}
${storageBufferRO(2, "gamma")}
${storageBufferRW(3, "grad_x")}
${storageBufferRW(4, "grad_gamma")}
${storageBufferRW(5, "grad_beta")}
const OUTER = ${Int.toString(outerSize)}u;
const NORM = ${Int.toString(normSize)}u;
const EPSILON = ${Float.toString(epsilon)};
${mainSignature}
  let outer_idx = idx;
  if (outer_idx >= OUTER) { return; }
  
  let base = outer_idx * NORM;
  
  // Recompute mean and variance
  var mean = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    mean = mean + x[base + i];
  }
  mean = mean / f32(NORM);
  
  var variance = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let diff = x[base + i] - mean;
    variance = variance + diff * diff;
  }
  variance = variance / f32(NORM);
  let inv_std = 1.0 / sqrt(variance + EPSILON);
  
  // Compute intermediate sums for gradient
  var sum_grad_out = 0.0;
  var sum_grad_out_x_centered = 0.0;
  for (var i = 0u; i < NORM; i = i + 1u) {
    let g = grad_out[base + i] * gamma[i];
    sum_grad_out = sum_grad_out + g;
    sum_grad_out_x_centered = sum_grad_out_x_centered + g * (x[base + i] - mean);
  }
  
  // Compute grad_x
  let norm_factor = 1.0 / f32(NORM);
  for (var i = 0u; i < NORM; i = i + 1u) {
    let x_centered = x[base + i] - mean;
    let x_norm = x_centered * inv_std;
    let g = grad_out[base + i] * gamma[i];
    grad_x[base + i] = inv_std * (g - norm_factor * sum_grad_out - norm_factor * x_norm * sum_grad_out_x_centered * inv_std);
  }
  
  // Accumulate grad_gamma and grad_beta (atomic would be better but this works for single outer)
  if (outer_idx == 0u) {
    for (var i = 0u; i < NORM; i = i + 1u) {
      var gg = 0.0;
      var gb = 0.0;
      for (var o = 0u; o < OUTER; o = o + 1u) {
        let b = o * NORM;
        var m = 0.0;
        for (var j = 0u; j < NORM; j = j + 1u) { m = m + x[b + j]; }
        m = m / f32(NORM);
        var v = 0.0;
        for (var j = 0u; j < NORM; j = j + 1u) { let d = x[b + j] - m; v = v + d * d; }
        v = v / f32(NORM);
        let istd = 1.0 / sqrt(v + EPSILON);
        let x_norm = (x[b + i] - m) * istd;
        gg = gg + grad_out[b + i] * x_norm;
        gb = gb + grad_out[b + i];
      }
      grad_gamma[i] = gg;
      grad_beta[i] = gb;
    }
  }
${mainEnd}`
  {
    name: "layernorm_backward_" ++ Int.toString(outerSize) ++ "x" ++ Int.toString(normSize),
    wgsl,
    bindings: [
      {binding: 0, size: totalSize * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: totalSize * 4, usage: ReadOnly, name: "x"},
      {binding: 2, size: normSize * 4, usage: ReadOnly, name: "gamma"},
      {binding: 3, size: totalSize * 4, usage: ReadWrite, name: "grad_x"},
      {binding: 4, size: normSize * 4, usage: ReadWrite, name: "grad_gamma"},
      {binding: 5, size: normSize * 4, usage: ReadWrite, name: "grad_beta"},
    ],
  }
}

// ============================================
// Reshape/Transpose Backward (just copy)
// ============================================

let genCopyBackwardKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRO(0, "grad_out")}
${storageBufferRW(1, "grad_x")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_x[idx] = grad_out[idx];
${mainEnd}`
  {
    name: "copy_backward_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadOnly, name: "grad_out"},
      {binding: 1, size: size * 4, usage: ReadWrite, name: "grad_x"},
    ],
  }
}

// ============================================
// Gradient Accumulation
// ============================================

let genGradAccumulateKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRW(0, "grad_acc")}
${storageBufferRO(1, "grad_new")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad_acc[idx] = grad_acc[idx] + grad_new[idx];
${mainEnd}`
  {
    name: "grad_accumulate_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "grad_acc"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "grad_new"},
    ],
  }
}

let genGradZeroKernel = (size: int): kernel => {
  let wgsl = `${storageBufferRW(0, "grad")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  grad[idx] = 0.0;
${mainEnd}`
  {
    name: "grad_zero_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "grad"},
    ],
  }
}

// ============================================
// Optimizers
// ============================================

// SGD: param = param - lr * grad
let genSGDKernel = (size: int, lr: float): kernel => {
  let wgsl = `${storageBufferRW(0, "param")}
${storageBufferRO(1, "grad")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  param[idx] = param[idx] - ${Float.toString(lr)} * grad[idx];
${mainEnd}`
  {
    name: "sgd_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "param"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "grad"},
    ],
  }
}

// SGD with momentum: v = momentum * v + grad; param = param - lr * v
let genSGDMomentumKernel = (size: int, lr: float, momentum: float): kernel => {
  let wgsl = `${storageBufferRW(0, "param")}
${storageBufferRO(1, "grad")}
${storageBufferRW(2, "velocity")}
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  let v = ${Float.toString(momentum)} * velocity[idx] + grad[idx];
  velocity[idx] = v;
  param[idx] = param[idx] - ${Float.toString(lr)} * v;
${mainEnd}`
  {
    name: "sgd_momentum_" ++ Int.toString(size),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "param"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "grad"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "velocity"},
    ],
  }
}

// Adam optimizer
let genAdamKernel = (size: int, lr: float, beta1: float, beta2: float, epsilon: float, t: int): kernel => {
  // Bias correction terms
  let beta1_t = Math.pow(beta1, ~exp=Float.fromInt(t))
  let beta2_t = Math.pow(beta2, ~exp=Float.fromInt(t))
  let lr_t = lr * Math.sqrt(1.0 -. beta2_t) /. (1.0 -. beta1_t)
  
  let wgsl = `${storageBufferRW(0, "param")}
${storageBufferRO(1, "grad")}
${storageBufferRW(2, "m")}
${storageBufferRW(3, "v")}
const LR_T = ${Float.toString(lr_t)};
const BETA1 = ${Float.toString(beta1)};
const BETA2 = ${Float.toString(beta2)};
const EPSILON = ${Float.toString(epsilon)};
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  
  let g = grad[idx];
  
  // Update biased first moment estimate
  let m_new = BETA1 * m[idx] + (1.0 - BETA1) * g;
  m[idx] = m_new;
  
  // Update biased second raw moment estimate  
  let v_new = BETA2 * v[idx] + (1.0 - BETA2) * g * g;
  v[idx] = v_new;
  
  // Update parameters (bias correction already in LR_T)
  param[idx] = param[idx] - LR_T * m_new / (sqrt(v_new) + EPSILON);
${mainEnd}`
  {
    name: "adam_" ++ Int.toString(size) ++ "_t" ++ Int.toString(t),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "param"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "grad"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "m"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "v"},
    ],
  }
}

// AdamW (Adam with decoupled weight decay)
let genAdamWKernel = (size: int, lr: float, beta1: float, beta2: float, epsilon: float, weightDecay: float, t: int): kernel => {
  let beta1_t = Math.pow(beta1, ~exp=Float.fromInt(t))
  let beta2_t = Math.pow(beta2, ~exp=Float.fromInt(t))
  let lr_t = lr * Math.sqrt(1.0 -. beta2_t) /. (1.0 -. beta1_t)
  
  let wgsl = `${storageBufferRW(0, "param")}
${storageBufferRO(1, "grad")}
${storageBufferRW(2, "m")}
${storageBufferRW(3, "v")}
const LR = ${Float.toString(lr)};
const LR_T = ${Float.toString(lr_t)};
const BETA1 = ${Float.toString(beta1)};
const BETA2 = ${Float.toString(beta2)};
const EPSILON = ${Float.toString(epsilon)};
const WEIGHT_DECAY = ${Float.toString(weightDecay)};
${mainSignature}
  if (idx >= ${Int.toString(size)}u) { return; }
  
  let g = grad[idx];
  let p = param[idx];
  
  // Update moments
  let m_new = BETA1 * m[idx] + (1.0 - BETA1) * g;
  m[idx] = m_new;
  let v_new = BETA2 * v[idx] + (1.0 - BETA2) * g * g;
  v[idx] = v_new;
  
  // Update with decoupled weight decay
  param[idx] = p - LR_T * m_new / (sqrt(v_new) + EPSILON) - LR * WEIGHT_DECAY * p;
${mainEnd}`
  {
    name: "adamw_" ++ Int.toString(size) ++ "_t" ++ Int.toString(t),
    wgsl,
    bindings: [
      {binding: 0, size: size * 4, usage: ReadWrite, name: "param"},
      {binding: 1, size: size * 4, usage: ReadOnly, name: "grad"},
      {binding: 2, size: size * 4, usage: ReadWrite, name: "m"},
      {binding: 3, size: size * 4, usage: ReadWrite, name: "v"},
    ],
  }
}
