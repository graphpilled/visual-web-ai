// src/nn.js
// High-level neural network API with automatic differentiation
//
// Usage:
//   import { Tensor, nn, optim } from './nn.js';
//   
//   const model = new nn.Sequential(
//     new nn.Linear(2, 4),
//     new nn.Tanh(),
//     new nn.Linear(4, 1),
//     new nn.Sigmoid()
//   );
//   
//   const optimizer = new optim.Adam(model.parameters(), 0.01);
//   
//   const y = await model.forward(x);
//   const loss = await nn.mseLoss(y, target);
//   await loss.backward();
//   nn.clipGradNorm(model.parameters(), 1.0);
//   await optimizer.step();

import { GPURuntime } from './runtime.js';
import { Autograd } from '../dist/bundle.js';

// ============================================================
// Global State
// ============================================================
let _runtime = null;
let _nextTensorId = 0;
const _tensors = new Map();
let _tape = [];
let _recording = true;

// ============================================================
// Initialize Runtime
// ============================================================
export async function init() {
  _runtime = new GPURuntime();
  await _runtime.init();
  return _runtime;
}

export function getRuntime() {
  return _runtime;
}

// ============================================================
// Tensor Class
// ============================================================
export class Tensor {
  constructor(data, shape, options = {}) {
    this.id = _nextTensorId++;
    this.shape = [...shape];
    this.size = shape.reduce((a, b) => a * b, 1);
    this.requiresGrad = options.requiresGrad || false;
    this.name = options.name || `t${this.id}`;
    this._data = data instanceof Float32Array ? data : new Float32Array(data);
    this._grad = null;
    this._creator = null;
    this._creatorInputs = null;
    _tensors.set(this.id, this);
  }
  
  // === Static Constructors ===
  static from(data, shape, options = {}) {
    const flat = Array.isArray(data) ? data.flat(Infinity) : data;
    return new Tensor(new Float32Array(flat), shape, options);
  }
  
  static zeros(shape, options = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape, options);
  }
  
  static ones(shape, options = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size).fill(1), shape, options);
  }
  
  static rand(shape, options = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) data[i] = Math.random() * 2 - 1;
    return new Tensor(data, shape, options);
  }
  
  static randn(shape, options = {}) {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      const u1 = Math.random(), u2 = Math.random();
      data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    return new Tensor(data, shape, options);
  }
  
  static xavier(shape, options = {}) {
    const [fanIn, fanOut] = shape.length >= 2 
      ? [shape[shape.length - 2], shape[shape.length - 1]]
      : [shape[0], shape[0]];
    const std = Math.sqrt(2.0 / (fanIn + fanOut));
    const t = Tensor.randn(shape, options);
    for (let i = 0; i < t.size; i++) t._data[i] *= std;
    return t;
  }
  
  static kaiming(shape, options = {}) {
    const fanIn = shape.length >= 2 ? shape[shape.length - 2] : shape[0];
    const std = Math.sqrt(2.0 / fanIn);
    const t = Tensor.randn(shape, options);
    for (let i = 0; i < t.size; i++) t._data[i] *= std;
    return t;
  }
  
  // === Data Access ===
  toArray() { return Array.from(this._data); }
  item() {
    if (this.size !== 1) throw new Error('item() only works on scalar tensors');
    return this._data[0];
  }
  grad() { return this._grad; }
  data() { return this._data; }
  
  // === Clone ===
  clone() {
    return new Tensor(new Float32Array(this._data), this.shape, {
      requiresGrad: this.requiresGrad,
      name: this.name + '_clone'
    });
  }
  
  // === Freeze/Unfreeze ===
  freeze() { this.requiresGrad = false; return this; }
  unfreeze() { this.requiresGrad = true; return this; }
  
  // === Recording Helper ===
  _recordOp(opName, inputs, output, extra = {}) {
    if (_recording && inputs.some(t => t.requiresGrad || _tensors.get(t.id)?._creator)) {
      output._creator = { op: opName, ...extra };
      output._creatorInputs = inputs;
      _tape.push({ output, op: opName, inputs, extra });
    }
  }
  
  // === Operations ===
  async neg() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => -x);
    this._recordOp('Neg', [this], result);
    return result;
  }
  
  async exp() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => Math.exp(x));
    this._recordOp('Exp', [this], result);
    return result;
  }
  
  async log() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => Math.log(x));
    this._recordOp('Log', [this], result);
    return result;
  }
  
  async tanh() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => Math.tanh(x));
    this._recordOp('Tanh', [this], result);
    return result;
  }
  
  async sigmoid() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => 1 / (1 + Math.exp(-x)));
    this._recordOp('Sigmoid', [this], result);
    return result;
  }
  
  async relu() {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => Math.max(0, x));
    this._recordOp('ReLU', [this], result);
    return result;
  }
  
  async leakyRelu(alpha = 0.01) {
    const result = Tensor.zeros(this.shape);
    result._data = this._data.map(x => x > 0 ? x : alpha * x);
    this._recordOp('LeakyReLU', [this], result, { alpha });
    return result;
  }
  
  async gelu() {
    const result = Tensor.zeros(this.shape);
    const sqrt2OverPi = Math.sqrt(2 / Math.PI);
    result._data = this._data.map(x => {
      return 0.5 * x * (1 + Math.tanh(sqrt2OverPi * (x + 0.044715 * x * x * x)));
    });
    this._recordOp('GeLU', [this], result);
    return result;
  }
  
  async add(other) {
    if (!(other instanceof Tensor)) other = Tensor.from([other], [1]);
    const result = Tensor.zeros(this.shape);
    if (other.size === 1) {
      const val = other._data[0];
      result._data = this._data.map(x => x + val);
    } else {
      for (let i = 0; i < this.size; i++) {
        result._data[i] = this._data[i] + other._data[i % other.size];
      }
    }
    this._recordOp('Add', [this, other], result);
    return result;
  }
  
  async sub(other) {
    if (!(other instanceof Tensor)) other = Tensor.from([other], [1]);
    const result = Tensor.zeros(this.shape);
    if (other.size === 1) {
      const val = other._data[0];
      result._data = this._data.map(x => x - val);
    } else {
      for (let i = 0; i < this.size; i++) {
        result._data[i] = this._data[i] - other._data[i % other.size];
      }
    }
    this._recordOp('Sub', [this, other], result);
    return result;
  }
  
  async mul(other) {
    if (!(other instanceof Tensor)) other = Tensor.from([other], [1]);
    const result = Tensor.zeros(this.shape);
    if (other.size === 1) {
      const val = other._data[0];
      result._data = this._data.map(x => x * val);
    } else {
      for (let i = 0; i < this.size; i++) {
        result._data[i] = this._data[i] * other._data[i % other.size];
      }
    }
    this._recordOp('Mul', [this, other], result);
    return result;
  }
  
  async div(other) {
    if (!(other instanceof Tensor)) other = Tensor.from([other], [1]);
    const result = Tensor.zeros(this.shape);
    if (other.size === 1) {
      const val = other._data[0];
      result._data = this._data.map(x => x / val);
    } else {
      for (let i = 0; i < this.size; i++) {
        result._data[i] = this._data[i] / other._data[i % other.size];
      }
    }
    this._recordOp('Div', [this, other], result);
    return result;
  }
  
  async matmul(other) {
    const aShape = this.shape;
    const bShape = other.shape;
    const m = aShape[aShape.length - 2] || 1;
    const k = aShape[aShape.length - 1];
    const n = bShape[bShape.length - 1];
    
    if (k !== (bShape[bShape.length - 2] || bShape[0])) {
      throw new Error(`MatMul shape mismatch: [${aShape}] @ [${bShape}]`);
    }
    
    const outShape = aShape.length > 1 ? [...aShape.slice(0, -1), n] : [n];
    const result = Tensor.zeros(outShape);
    
    const kernel = {
      name: `matmul_${m}x${k}x${n}`,
      wgsl: `
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.y;
  let col = gid.x;
  if (row >= ${m}u || col >= ${n}u) { return; }
  var sum = 0.0;
  for (var i = 0u; i < ${k}u; i = i + 1u) {
    sum = sum + a[row * ${k}u + i] * b[i * ${n}u + col];
  }
  c[row * ${n}u + col] = sum;
}`,
      bindings: [
        { binding: 0, size: m * k * 4, usage: 'ReadOnly', name: 'a' },
        { binding: 1, size: k * n * 4, usage: 'ReadOnly', name: 'b' },
        { binding: 2, size: m * n * 4, usage: 'ReadWrite', name: 'c' },
      ]
    };
    
    const dispatch = { workgroupCount: [Math.ceil(n / 16), Math.ceil(m / 16), 1] };
    const gpuResult = await _runtime.execute(kernel, dispatch, {
      a: this._data,
      b: other._data
    });
    
    result._data = gpuResult instanceof Float32Array ? gpuResult : new Float32Array(m * n);
    this._recordOp('MatMul', [this, other], result, { m, k, n });
    return result;
  }
  
  async sum(axis = null, keepDims = false) {
    let result;
    if (axis === null) {
      let total = 0;
      for (let i = 0; i < this.size; i++) total += this._data[i];
      result = Tensor.from([total], [1]);
    } else {
      let total = 0;
      for (let i = 0; i < this.size; i++) total += this._data[i];
      result = Tensor.from([total], [1]);
    }
    this._recordOp('Sum', [this], result, { axis, keepDims });
    return result;
  }
  
  async mean(axis = null, keepDims = false) {
    let result;
    if (axis === null) {
      let total = 0;
      for (let i = 0; i < this.size; i++) total += this._data[i];
      result = Tensor.from([total / this.size], [1]);
    } else {
      let total = 0;
      for (let i = 0; i < this.size; i++) total += this._data[i];
      result = Tensor.from([total / this.size], [1]);
    }
    this._recordOp('Mean', [this], result, { axis, keepDims, inputSize: this.size });
    return result;
  }
  
  async softmax(axis = -1) {
    const result = Tensor.zeros(this.shape);
    const axisSize = this.shape[this.shape.length - 1];
    const outerSize = this.size / axisSize;
    
    for (let i = 0; i < outerSize; i++) {
      const offset = i * axisSize;
      let max = this._data[offset];
      for (let j = 1; j < axisSize; j++) {
        if (this._data[offset + j] > max) max = this._data[offset + j];
      }
      let sum = 0;
      for (let j = 0; j < axisSize; j++) {
        result._data[offset + j] = Math.exp(this._data[offset + j] - max);
        sum += result._data[offset + j];
      }
      for (let j = 0; j < axisSize; j++) {
        result._data[offset + j] /= sum;
      }
    }
    
    this._recordOp('Softmax', [this], result, { axis });
    return result;
  }
  
  // ============================================================
  // BACKWARD PASS
  // ============================================================
  async backward() {
    if (this.size !== 1) {
      throw new Error('backward() can only be called on scalar tensors (loss)');
    }
    
    _recording = false;
    
    try {
      this._grad = new Float32Array([1.0]);
      
      for (let i = _tape.length - 1; i >= 0; i--) {
        const { output, op, inputs, extra } = _tape[i];
        const gradOut = output._grad;
        if (!gradOut) continue;
        await this._backwardOp(op, inputs, output, gradOut, extra);
      }
    } finally {
      _recording = true;
    }
  }
  
  async _backwardOp(op, inputs, output, gradOut, extra) {
    switch (op) {
      case 'Neg': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genNegBackwardKernel(x.size);
        const result = await _runtime.execute(kernel, 
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'Exp': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genExpBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, out: output._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'Log': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genLogBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, x: x._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'Tanh': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genTanhBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, out: output._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'Sigmoid': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genSigmoidBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, out: output._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'ReLU': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genReLUBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, x: x._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'LeakyReLU': {
        const [x] = inputs;
        const { alpha } = extra;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genLeakyReLUBackwardKernel(x.size, alpha);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, x: x._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'GeLU': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const kernel = Autograd.genGeLUBackwardKernel(x.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(x.size / 256), 1, 1] },
          { grad_out: gradOut, x: x._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      case 'Add': {
        const [a, b] = inputs;
        const kernel = Autograd.genAddBackwardKernel(output.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(output.size / 256), 1, 1] },
          { grad_out: gradOut }
        );
        if (!a._grad) a._grad = new Float32Array(a.size);
        if (!b._grad) b._grad = new Float32Array(b.size);
        accumulateGradWithBroadcast(a, result.grad_a, output.shape);
        accumulateGradWithBroadcast(b, result.grad_b, output.shape);
        break;
      }
      
      case 'Sub': {
        const [a, b] = inputs;
        const kernel = Autograd.genSubBackwardKernel(output.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(output.size / 256), 1, 1] },
          { grad_out: gradOut }
        );
        if (!a._grad) a._grad = new Float32Array(a.size);
        if (!b._grad) b._grad = new Float32Array(b.size);
        accumulateGradWithBroadcast(a, result.grad_a, output.shape);
        accumulateGradWithBroadcast(b, result.grad_b, output.shape);
        break;
      }
      
      case 'Mul': {
        const [a, b] = inputs;
        if (!a._grad) a._grad = new Float32Array(a.size);
        if (!b._grad) b._grad = new Float32Array(b.size);
        const aData = expandToSize(a._data, output.size);
        const bData = expandToSize(b._data, output.size);
        const kernel = Autograd.genMulBackwardKernel(output.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(output.size / 256), 1, 1] },
          { grad_out: gradOut, a: aData, b: bData }
        );
        accumulateGradWithBroadcast(a, result.grad_a, output.shape);
        accumulateGradWithBroadcast(b, result.grad_b, output.shape);
        break;
      }
      
      case 'Div': {
        const [a, b] = inputs;
        if (!a._grad) a._grad = new Float32Array(a.size);
        if (!b._grad) b._grad = new Float32Array(b.size);
        const aData = expandToSize(a._data, output.size);
        const bData = expandToSize(b._data, output.size);
        const kernel = Autograd.genDivBackwardKernel(output.size);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(output.size / 256), 1, 1] },
          { grad_out: gradOut, a: aData, b: bData }
        );
        accumulateGradWithBroadcast(a, result.grad_a, output.shape);
        accumulateGradWithBroadcast(b, result.grad_b, output.shape);
        break;
      }
      
      case 'MatMul': {
        const [a, b] = inputs;
        const { m, k, n } = extra;
        if (!a._grad) a._grad = new Float32Array(a.size);
        if (!b._grad) b._grad = new Float32Array(b.size);
        
        const kernelA = Autograd.genMatMulBackwardAKernel(m, k, n);
        const resultA = await _runtime.execute(kernelA,
          { workgroupCount: [Math.ceil(k / 16), Math.ceil(m / 16), 1] },
          { grad_out: gradOut, b: b._data }
        );
        accumulateGrad(a, resultA);
        
        const kernelB = Autograd.genMatMulBackwardBKernel(m, k, n);
        const resultB = await _runtime.execute(kernelB,
          { workgroupCount: [Math.ceil(n / 16), Math.ceil(k / 16), 1] },
          { grad_out: gradOut, a: a._data }
        );
        accumulateGrad(b, resultB);
        break;
      }
      
      case 'Sum': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        for (let i = 0; i < x.size; i++) {
          x._grad[i] += gradOut[0];
        }
        break;
      }
      
      case 'Mean': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const scale = 1 / x.size;
        for (let i = 0; i < x.size; i++) {
          x._grad[i] += gradOut[0] * scale;
        }
        break;
      }
      
      case 'Softmax': {
        const [x] = inputs;
        if (!x._grad) x._grad = new Float32Array(x.size);
        const axisSize = x.shape[x.shape.length - 1];
        const outerSize = x.size / axisSize;
        const kernel = Autograd.genSoftmaxBackwardKernel(outerSize, axisSize);
        const result = await _runtime.execute(kernel,
          { workgroupCount: [Math.ceil(outerSize / 256), 1, 1] },
          { grad_out: gradOut, softmax_out: output._data }
        );
        accumulateGrad(x, result);
        break;
      }
      
      default:
        console.warn(`No backward implementation for op: ${op}`);
    }
  }
}

// ============================================================
// Helper Functions
// ============================================================
function expandToSize(data, targetSize) {
  if (data.length === targetSize) return data;
  const expanded = new Float32Array(targetSize);
  for (let i = 0; i < targetSize; i++) {
    expanded[i] = data[i % data.length];
  }
  return expanded;
}

function accumulateGrad(tensor, grad) {
  const gradData = grad instanceof Float32Array ? grad : grad.grad_x || grad.grad_a || grad;
  if (!tensor._grad) tensor._grad = new Float32Array(tensor.size);
  for (let i = 0; i < tensor.size; i++) {
    tensor._grad[i] += gradData[i] || 0;
  }
}

function accumulateGradWithBroadcast(tensor, grad, outputShape) {
  const gradData = grad instanceof Float32Array ? grad : new Float32Array(grad);
  if (!tensor._grad) tensor._grad = new Float32Array(tensor.size);
  
  if (tensor.size === gradData.length) {
    for (let i = 0; i < tensor.size; i++) {
      tensor._grad[i] += gradData[i];
    }
  } else if (tensor.size === 1) {
    let sum = 0;
    for (let i = 0; i < gradData.length; i++) sum += gradData[i];
    tensor._grad[0] += sum;
  } else {
    const ratio = gradData.length / tensor.size;
    for (let i = 0; i < tensor.size; i++) {
      let sum = 0;
      for (let j = 0; j < ratio; j++) {
        sum += gradData[i + j * tensor.size] || gradData[i * ratio + j] || 0;
      }
      tensor._grad[i] += sum;
    }
  }
}

// ============================================================
// Gradient Utilities
// ============================================================
export function clearTape() {
  _tape = [];
}

export function clipGradNorm(parameters, maxNorm) {
  // Compute total gradient norm
  let totalNormSq = 0;
  for (const param of parameters) {
    if (param._grad) {
      for (let i = 0; i < param._grad.length; i++) {
        totalNormSq += param._grad[i] * param._grad[i];
      }
    }
  }
  const totalNorm = Math.sqrt(totalNormSq);
  
  // Clip if necessary
  if (totalNorm > maxNorm) {
    const scale = maxNorm / totalNorm;
    for (const param of parameters) {
      if (param._grad) {
        for (let i = 0; i < param._grad.length; i++) {
          param._grad[i] *= scale;
        }
      }
    }
  }
  
  return totalNorm;
}

export function clipGradValue(parameters, maxValue) {
  for (const param of parameters) {
    if (param._grad) {
      for (let i = 0; i < param._grad.length; i++) {
        param._grad[i] = Math.max(-maxValue, Math.min(maxValue, param._grad[i]));
      }
    }
  }
}

// ============================================================
// Module Base Class
// ============================================================
export class Module {
  constructor() {
    this._parameters = [];
    this._modules = [];
    this._training = true;
  }
  
  // Register a parameter
  registerParameter(name, tensor) {
    tensor.requiresGrad = true;
    tensor.name = name;
    this._parameters.push(tensor);
    this[name] = tensor;
    return tensor;
  }
  
  // Register a submodule
  registerModule(name, module) {
    this._modules.push(module);
    this[name] = module;
    return module;
  }
  
  // Get all parameters (recursive)
  parameters() {
    const params = [...this._parameters];
    for (const module of this._modules) {
      params.push(...module.parameters());
    }
    return params;
  }
  
  // Count total parameters
  numParameters() {
    return this.parameters().reduce((sum, p) => sum + p.size, 0);
  }
  
  // Training mode
  train() {
    this._training = true;
    for (const module of this._modules) module.train();
    return this;
  }
  
  // Evaluation mode
  eval() {
    this._training = false;
    for (const module of this._modules) module.eval();
    return this;
  }
  
  // Forward pass (to be overridden)
  async forward(x) {
    throw new Error('forward() must be implemented by subclass');
  }
  
  // Save state
  stateDict(prefix = '') {
    const state = {};
    for (const param of this._parameters) {
      const key = prefix ? `${prefix}.${param.name}` : param.name;
      state[key] = Array.from(param._data);
    }
    // Find registered module name
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Module && this._modules.includes(value)) {
        const modulePrefix = prefix ? `${prefix}.${name}` : name;
        const moduleState = value.stateDict(modulePrefix);
        Object.assign(state, moduleState);
      }
    }
    return state;
  }
  
  // Load state
  loadStateDict(state, prefix = '') {
    for (const param of this._parameters) {
      const key = prefix ? `${prefix}.${param.name}` : param.name;
      if (state[key]) {
        param._data = new Float32Array(state[key]);
      }
    }
    // Find registered module name
    for (const [name, value] of Object.entries(this)) {
      if (value instanceof Module && this._modules.includes(value)) {
        const modulePrefix = prefix ? `${prefix}.${name}` : name;
        value.loadStateDict(state, modulePrefix);
      }
    }
  } 
}

// ============================================================
// Layer Implementations
// ============================================================

// Linear (fully connected) layer
export class Linear extends Module {
  constructor(inFeatures, outFeatures, { bias = true } = {}) {
    super();
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    
    // Xavier initialization
    this.weight = this.registerParameter('weight', Tensor.xavier([inFeatures, outFeatures]));
    
    if (bias) {
      this.bias = this.registerParameter('bias', Tensor.zeros([outFeatures]));
    } else {
      this.bias = null;
    }
  }
  
  async forward(x) {
    let out = await x.matmul(this.weight);
    if (this.bias) {
      out = await out.add(this.bias);
    }
    return out;
  }
}

// ReLU activation
export class ReLU extends Module {
  async forward(x) {
    return await x.relu();
  }
}

// LeakyReLU activation
export class LeakyReLU extends Module {
  constructor(alpha = 0.01) {
    super();
    this.alpha = alpha;
  }
  
  async forward(x) {
    return await x.leakyRelu(this.alpha);
  }
}

// Tanh activation
export class Tanh extends Module {
  async forward(x) {
    return await x.tanh();
  }
}

// Sigmoid activation
export class Sigmoid extends Module {
  async forward(x) {
    return await x.sigmoid();
  }
}

// GeLU activation
export class GeLU extends Module {
  async forward(x) {
    return await x.gelu();
  }
}

// Softmax activation
export class Softmax extends Module {
  constructor(axis = -1) {
    super();
    this.axis = axis;
  }
  
  async forward(x) {
    return await x.softmax(this.axis);
  }
}

// Dropout (training only)
export class Dropout extends Module {
  constructor(p = 0.5) {
    super();
    this.p = p;
  }
  
  async forward(x) {
    if (!this._training || this.p === 0) {
      return x;
    }
    
    // Apply dropout mask
    const result = Tensor.zeros(x.shape);
    const scale = 1 / (1 - this.p);
    for (let i = 0; i < x.size; i++) {
      if (Math.random() > this.p) {
        result._data[i] = x._data[i] * scale;
      }
    }
    return result;
  }
}

// Sequential container
export class Sequential extends Module {
  constructor(...layers) {
    super();
    this.layers = [];
    for (let i = 0; i < layers.length; i++) {
      this.registerModule(`layer${i}`, layers[i]);
      this.layers.push(layers[i]);
    }
  }
  
  async forward(x) {
    let out = x;
    for (const layer of this.layers) {
      out = await layer.forward(out);
    }
    return out;
  }
  
  add(layer) {
    const idx = this.layers.length;
    this.registerModule(`layer${idx}`, layer);
    this.layers.push(layer);
    return this;
  }
}

// ============================================================
// Convolutional Layers
// ============================================================

// Conv2D layer
export class Conv2D extends Module {
  constructor(inChannels, outChannels, kernelSize, { stride = 1, padding = 0, bias = true } = {}) {
    super();
    this.inChannels = inChannels;
    this.outChannels = outChannels;
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = Array.isArray(stride) ? stride : [stride, stride];
    this.padding = Array.isArray(padding) ? padding : [padding, padding];
    
    // Kaiming initialization for conv weights
    const [kH, kW] = this.kernelSize;
    const fanIn = inChannels * kH * kW;
    const std = Math.sqrt(2.0 / fanIn);
    
    const weightData = new Float32Array(outChannels * inChannels * kH * kW);
    for (let i = 0; i < weightData.length; i++) {
      const u1 = Math.random(), u2 = Math.random();
      weightData[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * std;
    }
    
    this.weight = this.registerParameter('weight', 
      new Tensor(weightData, [outChannels, inChannels, kH, kW], { requiresGrad: true })
    );
    
    if (bias) {
      this.bias = this.registerParameter('bias', Tensor.zeros([outChannels]));
    } else {
      this.bias = null;
    }
  }
  
  async forward(x) {
    // x: [batch, inChannels, height, width]
    const [batch, , inH, inW] = x.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;
    
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;
    
    const result = Tensor.zeros([batch, this.outChannels, outH, outW]);
    
    // Naive conv2d implementation (CPU fallback)
    // For production, this would use the GPU kernel from Codegen.res
    for (let b = 0; b < batch; b++) {
      for (let oc = 0; oc < this.outChannels; oc++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = this.bias ? this.bias._data[oc] : 0;
            
            for (let ic = 0; ic < this.inChannels; ic++) {
              for (let kh = 0; kh < kH; kh++) {
                for (let kw = 0; kw < kW; kw++) {
                  const ih = oh * sH - pH + kh;
                  const iw = ow * sW - pW + kw;
                  
                  if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                    const xIdx = b * (this.inChannels * inH * inW) + ic * (inH * inW) + ih * inW + iw;
                    const wIdx = oc * (this.inChannels * kH * kW) + ic * (kH * kW) + kh * kW + kw;
                    sum += x._data[xIdx] * this.weight._data[wIdx];
                  }
                }
              }
            }
            
            const outIdx = b * (this.outChannels * outH * outW) + oc * (outH * outW) + oh * outW + ow;
            result._data[outIdx] = sum;
          }
        }
      }
    }
    
    // Record for backward (simplified - full implementation would record conv op)
    x._recordOp('Conv2D', [x, this.weight], result, {
      stride: this.stride, padding: this.padding, bias: this.bias
    });
    
    return result;
  }
}

// MaxPool2D layer
export class MaxPool2D extends Module {
  constructor(kernelSize, { stride = null, padding = 0 } = {}) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = Array.isArray(padding) ? padding : [padding, padding];
  }
  
  async forward(x) {
    const [batch, channels, inH, inW] = x.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;
    
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;
    
    const result = Tensor.zeros([batch, channels, outH, outW]);
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let maxVal = -Infinity;
            
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH - pH + kh;
                const iw = ow * sW - pW + kw;
                
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  const idx = b * (channels * inH * inW) + c * (inH * inW) + ih * inW + iw;
                  maxVal = Math.max(maxVal, x._data[idx]);
                }
              }
            }
            
            const outIdx = b * (channels * outH * outW) + c * (outH * outW) + oh * outW + ow;
            result._data[outIdx] = maxVal === -Infinity ? 0 : maxVal;
          }
        }
      }
    }
    
    return result;
  }
}

// AvgPool2D layer
export class AvgPool2D extends Module {
  constructor(kernelSize, { stride = null, padding = 0 } = {}) {
    super();
    this.kernelSize = Array.isArray(kernelSize) ? kernelSize : [kernelSize, kernelSize];
    this.stride = stride ? (Array.isArray(stride) ? stride : [stride, stride]) : this.kernelSize;
    this.padding = Array.isArray(padding) ? padding : [padding, padding];
  }
  
  async forward(x) {
    const [batch, channels, inH, inW] = x.shape;
    const [kH, kW] = this.kernelSize;
    const [sH, sW] = this.stride;
    const [pH, pW] = this.padding;
    
    const outH = Math.floor((inH + 2 * pH - kH) / sH) + 1;
    const outW = Math.floor((inW + 2 * pW - kW) / sW) + 1;
    
    const result = Tensor.zeros([batch, channels, outH, outW]);
    
    for (let b = 0; b < batch; b++) {
      for (let c = 0; c < channels; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = 0;
            let count = 0;
            
            for (let kh = 0; kh < kH; kh++) {
              for (let kw = 0; kw < kW; kw++) {
                const ih = oh * sH - pH + kh;
                const iw = ow * sW - pW + kw;
                
                if (ih >= 0 && ih < inH && iw >= 0 && iw < inW) {
                  const idx = b * (channels * inH * inW) + c * (inH * inW) + ih * inW + iw;
                  sum += x._data[idx];
                  count++;
                }
              }
            }
            
            const outIdx = b * (channels * outH * outW) + c * (outH * outW) + oh * outW + ow;
            result._data[outIdx] = count > 0 ? sum / count : 0;
          }
        }
      }
    }
    
    return result;
  }
}

// Flatten layer
export class Flatten extends Module {
  constructor(startDim = 1, endDim = -1) {
    super();
    this.startDim = startDim;
    this.endDim = endDim;
  }
  
  async forward(x) {
    const shape = x.shape;
    const rank = shape.length;
    
    const start = this.startDim >= 0 ? this.startDim : rank + this.startDim;
    const end = this.endDim >= 0 ? this.endDim : rank + this.endDim;
    
    // Calculate new shape
    const newShape = [];
    for (let i = 0; i < start; i++) {
      newShape.push(shape[i]);
    }
    
    let flatSize = 1;
    for (let i = start; i <= end; i++) {
      flatSize *= shape[i];
    }
    newShape.push(flatSize);
    
    for (let i = end + 1; i < rank; i++) {
      newShape.push(shape[i]);
    }
    
    const result = new Tensor(new Float32Array(x._data), newShape);
    return result;
  }
}

// ============================================================
// Normalization Layers
// ============================================================

// LayerNorm
export class LayerNorm extends Module {
  constructor(normalizedShape, { eps = 1e-5, elementwiseAffine = true } = {}) {
    super();
    this.normalizedShape = Array.isArray(normalizedShape) ? normalizedShape : [normalizedShape];
    this.eps = eps;
    this.elementwiseAffine = elementwiseAffine;
    
    const size = this.normalizedShape.reduce((a, b) => a * b, 1);
    
    if (elementwiseAffine) {
      this.gamma = this.registerParameter('gamma', Tensor.ones([size]));
      this.beta = this.registerParameter('beta', Tensor.zeros([size]));
    }
  }
  
  async forward(x) {
    const shape = x.shape;
    const normSize = this.normalizedShape.reduce((a, b) => a * b, 1);
    const outerSize = x.size / normSize;
    
    const result = Tensor.zeros(shape);
    
    for (let i = 0; i < outerSize; i++) {
      const offset = i * normSize;
      
      // Compute mean
      let mean = 0;
      for (let j = 0; j < normSize; j++) {
        mean += x._data[offset + j];
      }
      mean /= normSize;
      
      // Compute variance
      let variance = 0;
      for (let j = 0; j < normSize; j++) {
        const diff = x._data[offset + j] - mean;
        variance += diff * diff;
      }
      variance /= normSize;
      
      // Normalize
      const invStd = 1 / Math.sqrt(variance + this.eps);
      for (let j = 0; j < normSize; j++) {
        let normalized = (x._data[offset + j] - mean) * invStd;
        if (this.elementwiseAffine) {
          normalized = normalized * this.gamma._data[j] + this.beta._data[j];
        }
        result._data[offset + j] = normalized;
      }
    }
    
    x._recordOp('LayerNorm', [x], result, { 
      normalizedShape: this.normalizedShape, 
      eps: this.eps,
      gamma: this.gamma,
      beta: this.beta
    });
    
    return result;
  }
}

// BatchNorm1D
export class BatchNorm1D extends Module {
  constructor(numFeatures, { eps = 1e-5, momentum = 0.1, affine = true, trackRunningStats = true } = {}) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.momentum = momentum;
    this.affine = affine;
    this.trackRunningStats = trackRunningStats;
    
    if (affine) {
      this.gamma = this.registerParameter('gamma', Tensor.ones([numFeatures]));
      this.beta = this.registerParameter('beta', Tensor.zeros([numFeatures]));
    }
    
    if (trackRunningStats) {
      this.runningMean = Tensor.zeros([numFeatures]);
      this.runningVar = Tensor.ones([numFeatures]);
      this.numBatchesTracked = 0;
    }
  }
  
  async forward(x) {
    // x: [batch, features] or [batch, features, length]
    const shape = x.shape;
    const batch = shape[0];
    const features = shape[1];
    const length = shape.length > 2 ? shape.slice(2).reduce((a, b) => a * b, 1) : 1;
    
    const result = Tensor.zeros(shape);
    
    if (this._training) {
      // Compute batch statistics
      const mean = new Float32Array(features);
      const variance = new Float32Array(features);
      
      for (let f = 0; f < features; f++) {
        let sum = 0;
        for (let b = 0; b < batch; b++) {
          for (let l = 0; l < length; l++) {
            const idx = b * features * length + f * length + l;
            sum += x._data[idx];
          }
        }
        mean[f] = sum / (batch * length);
        
        let varSum = 0;
        for (let b = 0; b < batch; b++) {
          for (let l = 0; l < length; l++) {
            const idx = b * features * length + f * length + l;
            const diff = x._data[idx] - mean[f];
            varSum += diff * diff;
          }
        }
        variance[f] = varSum / (batch * length);
      }
      
      // Update running stats
      if (this.trackRunningStats) {
        for (let f = 0; f < features; f++) {
          this.runningMean._data[f] = (1 - this.momentum) * this.runningMean._data[f] + this.momentum * mean[f];
          this.runningVar._data[f] = (1 - this.momentum) * this.runningVar._data[f] + this.momentum * variance[f];
        }
        this.numBatchesTracked++;
      }
      
      // Normalize
      for (let b = 0; b < batch; b++) {
        for (let f = 0; f < features; f++) {
          const invStd = 1 / Math.sqrt(variance[f] + this.eps);
          for (let l = 0; l < length; l++) {
            const idx = b * features * length + f * length + l;
            let normalized = (x._data[idx] - mean[f]) * invStd;
            if (this.affine) {
              normalized = normalized * this.gamma._data[f] + this.beta._data[f];
            }
            result._data[idx] = normalized;
          }
        }
      }
    } else {
      // Use running stats
      for (let b = 0; b < batch; b++) {
        for (let f = 0; f < features; f++) {
          const invStd = 1 / Math.sqrt(this.runningVar._data[f] + this.eps);
          for (let l = 0; l < length; l++) {
            const idx = b * features * length + f * length + l;
            let normalized = (x._data[idx] - this.runningMean._data[f]) * invStd;
            if (this.affine) {
              normalized = normalized * this.gamma._data[f] + this.beta._data[f];
            }
            result._data[idx] = normalized;
          }
        }
      }
    }
    
    return result;
  }
}

// BatchNorm2D
export class BatchNorm2D extends Module {
  constructor(numFeatures, { eps = 1e-5, momentum = 0.1, affine = true, trackRunningStats = true } = {}) {
    super();
    this.numFeatures = numFeatures;
    this.eps = eps;
    this.momentum = momentum;
    this.affine = affine;
    this.trackRunningStats = trackRunningStats;
    
    if (affine) {
      this.gamma = this.registerParameter('gamma', Tensor.ones([numFeatures]));
      this.beta = this.registerParameter('beta', Tensor.zeros([numFeatures]));
    }
    
    if (trackRunningStats) {
      this.runningMean = Tensor.zeros([numFeatures]);
      this.runningVar = Tensor.ones([numFeatures]);
      this.numBatchesTracked = 0;
    }
  }
  
  async forward(x) {
    // x: [batch, channels, height, width]
    const [batch, channels, height, width] = x.shape;
    const spatialSize = height * width;
    
    const result = Tensor.zeros(x.shape);
    
    if (this._training) {
      const mean = new Float32Array(channels);
      const variance = new Float32Array(channels);
      
      for (let c = 0; c < channels; c++) {
        let sum = 0;
        for (let b = 0; b < batch; b++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const idx = b * channels * spatialSize + c * spatialSize + h * width + w;
              sum += x._data[idx];
            }
          }
        }
        mean[c] = sum / (batch * spatialSize);
        
        let varSum = 0;
        for (let b = 0; b < batch; b++) {
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const idx = b * channels * spatialSize + c * spatialSize + h * width + w;
              const diff = x._data[idx] - mean[c];
              varSum += diff * diff;
            }
          }
        }
        variance[c] = varSum / (batch * spatialSize);
      }
      
      if (this.trackRunningStats) {
        for (let c = 0; c < channels; c++) {
          this.runningMean._data[c] = (1 - this.momentum) * this.runningMean._data[c] + this.momentum * mean[c];
          this.runningVar._data[c] = (1 - this.momentum) * this.runningVar._data[c] + this.momentum * variance[c];
        }
        this.numBatchesTracked++;
      }
      
      for (let b = 0; b < batch; b++) {
        for (let c = 0; c < channels; c++) {
          const invStd = 1 / Math.sqrt(variance[c] + this.eps);
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const idx = b * channels * spatialSize + c * spatialSize + h * width + w;
              let normalized = (x._data[idx] - mean[c]) * invStd;
              if (this.affine) {
                normalized = normalized * this.gamma._data[c] + this.beta._data[c];
              }
              result._data[idx] = normalized;
            }
          }
        }
      }
    } else {
      for (let b = 0; b < batch; b++) {
        for (let c = 0; c < channels; c++) {
          const invStd = 1 / Math.sqrt(this.runningVar._data[c] + this.eps);
          for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
              const idx = b * channels * spatialSize + c * spatialSize + h * width + w;
              let normalized = (x._data[idx] - this.runningMean._data[c]) * invStd;
              if (this.affine) {
                normalized = normalized * this.gamma._data[c] + this.beta._data[c];
              }
              result._data[idx] = normalized;
            }
          }
        }
      }
    }
    
    return result;
  }
}

// ============================================================
// Embedding Layer
// ============================================================

export class Embedding extends Module {
  constructor(numEmbeddings, embeddingDim, { paddingIdx = null } = {}) {
    super();
    this.numEmbeddings = numEmbeddings;
    this.embeddingDim = embeddingDim;
    this.paddingIdx = paddingIdx;
    
    // Initialize with normal distribution
    this.weight = this.registerParameter('weight', Tensor.randn([numEmbeddings, embeddingDim]));
    
    // Zero out padding embedding if specified
    if (paddingIdx !== null) {
      for (let i = 0; i < embeddingDim; i++) {
        this.weight._data[paddingIdx * embeddingDim + i] = 0;
      }
    }
  }
  
  async forward(x) {
    // x: [batch, seqLen] of integer indices
    const shape = x.shape;
    const batch = shape[0];
    const seqLen = shape.length > 1 ? shape[1] : 1;
    
    const result = Tensor.zeros([batch, seqLen, this.embeddingDim]);
    
    for (let b = 0; b < batch; b++) {
      for (let s = 0; s < seqLen; s++) {
        const idx = Math.floor(x._data[b * seqLen + s]);
        if (idx >= 0 && idx < this.numEmbeddings) {
          for (let e = 0; e < this.embeddingDim; e++) {
            result._data[b * seqLen * this.embeddingDim + s * this.embeddingDim + e] = 
              this.weight._data[idx * this.embeddingDim + e];
          }
        }
      }
    }
    
    return result;
  }
}

// ============================================================
// RNN Layers
// ============================================================

// LSTM Cell
export class LSTMCell extends Module {
  constructor(inputSize, hiddenSize, { bias = true } = {}) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    
    // Combined weight matrices for all 4 gates: input, forget, cell, output
    // W_ih: [4*hiddenSize, inputSize]
    // W_hh: [4*hiddenSize, hiddenSize]
    this.W_ih = this.registerParameter('W_ih', Tensor.xavier([inputSize, 4 * hiddenSize]));
    this.W_hh = this.registerParameter('W_hh', Tensor.xavier([hiddenSize, 4 * hiddenSize]));
    
    if (bias) {
      this.b_ih = this.registerParameter('b_ih', Tensor.zeros([4 * hiddenSize]));
      this.b_hh = this.registerParameter('b_hh', Tensor.zeros([4 * hiddenSize]));
    } else {
      this.b_ih = null;
      this.b_hh = null;
    }
  }
  
  async forward(x, state = null) {
    // x: [batch, inputSize]
    // state: [h, c] where h, c: [batch, hiddenSize]
    const batch = x.shape[0];
    
    let h, c;
    if (state) {
      [h, c] = state;
    } else {
      h = Tensor.zeros([batch, this.hiddenSize]);
      c = Tensor.zeros([batch, this.hiddenSize]);
    }
    
    // Compute gates
    const gates_i = await x.matmul(this.W_ih);
    const gates_h = await h.matmul(this.W_hh);
    let gates = await gates_i.add(gates_h);
    
    if (this.b_ih) gates = await gates.add(this.b_ih);
    if (this.b_hh) gates = await gates.add(this.b_hh);
    
    // Split into 4 gates
    const H = this.hiddenSize;
    const newH = Tensor.zeros([batch, H]);
    const newC = Tensor.zeros([batch, H]);
    
    for (let b = 0; b < batch; b++) {
      for (let j = 0; j < H; j++) {
        const i_gate = 1 / (1 + Math.exp(-gates._data[b * 4 * H + j]));           // input gate
        const f_gate = 1 / (1 + Math.exp(-gates._data[b * 4 * H + H + j]));       // forget gate
        const g_gate = Math.tanh(gates._data[b * 4 * H + 2 * H + j]);             // cell gate
        const o_gate = 1 / (1 + Math.exp(-gates._data[b * 4 * H + 3 * H + j]));   // output gate
        
        const newCVal = f_gate * c._data[b * H + j] + i_gate * g_gate;
        newC._data[b * H + j] = newCVal;
        newH._data[b * H + j] = o_gate * Math.tanh(newCVal);
      }
    }
    
    return [newH, [newH, newC]];
  }
}

// LSTM (multi-layer)
export class LSTM extends Module {
  constructor(inputSize, hiddenSize, { numLayers = 1, bias = true, batchFirst = true, dropout = 0, bidirectional = false } = {}) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.batchFirst = batchFirst;
    this.dropout = dropout;
    this.bidirectional = bidirectional;
    
    this.cells = [];
    for (let i = 0; i < numLayers; i++) {
      const cellInputSize = i === 0 ? inputSize : hiddenSize * (bidirectional ? 2 : 1);
      const cell = new LSTMCell(cellInputSize, hiddenSize, { bias });
      this.registerModule(`cell${i}`, cell);
      this.cells.push(cell);
    }
  }
  
  async forward(x, state = null) {
    // x: [batch, seqLen, inputSize] if batchFirst, else [seqLen, batch, inputSize]
    let input = x;
    if (!this.batchFirst) {
      // Transpose to batch first
      const [seqLen, batch, features] = x.shape;
      input = Tensor.zeros([batch, seqLen, features]);
      for (let s = 0; s < seqLen; s++) {
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < features; f++) {
            input._data[b * seqLen * features + s * features + f] = 
              x._data[s * batch * features + b * features + f];
          }
        }
      }
    }
    
    const [batch, seqLen, _] = input.shape;
    
    // Initialize states for all layers
    let layerStates = [];
    if (state) {
      layerStates = state;
    } else {
      for (let l = 0; l < this.numLayers; l++) {
        layerStates.push([
          Tensor.zeros([batch, this.hiddenSize]),
          Tensor.zeros([batch, this.hiddenSize])
        ]);
      }
    }
    
    // Process sequence through each layer
    let currentInput = input;
    const allOutputs = [];
    
    for (let l = 0; l < this.numLayers; l++) {
      const outputs = [];
      let [h, c] = layerStates[l];
      
      for (let t = 0; t < seqLen; t++) {
        // Extract input at timestep t
        const xt = Tensor.zeros([batch, currentInput.shape[2]]);
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < currentInput.shape[2]; f++) {
            xt._data[b * currentInput.shape[2] + f] = 
              currentInput._data[b * seqLen * currentInput.shape[2] + t * currentInput.shape[2] + f];
          }
        }
        
        const [newH, newState] = await this.cells[l].forward(xt, [h, c]);
        [h, c] = newState;
        outputs.push(newH);
      }
      
      layerStates[l] = [h, c];
      
      // Stack outputs for next layer
      currentInput = Tensor.zeros([batch, seqLen, this.hiddenSize]);
      for (let t = 0; t < seqLen; t++) {
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < this.hiddenSize; f++) {
            currentInput._data[b * seqLen * this.hiddenSize + t * this.hiddenSize + f] = 
              outputs[t]._data[b * this.hiddenSize + f];
          }
        }
      }
    }
    
    return [currentInput, layerStates];
  }
}

// GRU Cell
export class GRUCell extends Module {
  constructor(inputSize, hiddenSize, { bias = true } = {}) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    
    // Combined weight matrices for 3 gates: reset, update, new
    this.W_ih = this.registerParameter('W_ih', Tensor.xavier([inputSize, 3 * hiddenSize]));
    this.W_hh = this.registerParameter('W_hh', Tensor.xavier([hiddenSize, 3 * hiddenSize]));
    
    if (bias) {
      this.b_ih = this.registerParameter('b_ih', Tensor.zeros([3 * hiddenSize]));
      this.b_hh = this.registerParameter('b_hh', Tensor.zeros([3 * hiddenSize]));
    } else {
      this.b_ih = null;
      this.b_hh = null;
    }
  }
  
  async forward(x, h = null) {
    const batch = x.shape[0];
    
    if (!h) {
      h = Tensor.zeros([batch, this.hiddenSize]);
    }
    
    const gates_i = await x.matmul(this.W_ih);
    const gates_h = await h.matmul(this.W_hh);
    
    let gi = gates_i;
    let gh = gates_h;
    if (this.b_ih) gi = await gi.add(this.b_ih);
    if (this.b_hh) gh = await gh.add(this.b_hh);
    
    const H = this.hiddenSize;
    const newH = Tensor.zeros([batch, H]);
    
    for (let b = 0; b < batch; b++) {
      for (let j = 0; j < H; j++) {
        const r_gate = 1 / (1 + Math.exp(-(gi._data[b * 3 * H + j] + gh._data[b * 3 * H + j])));
        const z_gate = 1 / (1 + Math.exp(-(gi._data[b * 3 * H + H + j] + gh._data[b * 3 * H + H + j])));
        const n_gate = Math.tanh(gi._data[b * 3 * H + 2 * H + j] + r_gate * gh._data[b * 3 * H + 2 * H + j]);
        
        newH._data[b * H + j] = (1 - z_gate) * n_gate + z_gate * h._data[b * H + j];
      }
    }
    
    return [newH, newH];
  }
}

// GRU (multi-layer)
export class GRU extends Module {
  constructor(inputSize, hiddenSize, { numLayers = 1, bias = true, batchFirst = true, dropout = 0, bidirectional = false } = {}) {
    super();
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.numLayers = numLayers;
    this.batchFirst = batchFirst;
    this.dropout = dropout;
    this.bidirectional = bidirectional;
    
    this.cells = [];
    for (let i = 0; i < numLayers; i++) {
      const cellInputSize = i === 0 ? inputSize : hiddenSize * (bidirectional ? 2 : 1);
      const cell = new GRUCell(cellInputSize, hiddenSize, { bias });
      this.registerModule(`cell${i}`, cell);
      this.cells.push(cell);
    }
  }
  
  async forward(x, state = null) {
    let input = x;
    if (!this.batchFirst) {
      const [seqLen, batch, features] = x.shape;
      input = Tensor.zeros([batch, seqLen, features]);
      for (let s = 0; s < seqLen; s++) {
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < features; f++) {
            input._data[b * seqLen * features + s * features + f] = 
              x._data[s * batch * features + b * features + f];
          }
        }
      }
    }
    
    const [batch, seqLen, _] = input.shape;
    
    let layerStates = state || [];
    if (!state) {
      for (let l = 0; l < this.numLayers; l++) {
        layerStates.push(Tensor.zeros([batch, this.hiddenSize]));
      }
    }
    
    let currentInput = input;
    
    for (let l = 0; l < this.numLayers; l++) {
      const outputs = [];
      let h = layerStates[l];
      
      for (let t = 0; t < seqLen; t++) {
        const xt = Tensor.zeros([batch, currentInput.shape[2]]);
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < currentInput.shape[2]; f++) {
            xt._data[b * currentInput.shape[2] + f] = 
              currentInput._data[b * seqLen * currentInput.shape[2] + t * currentInput.shape[2] + f];
          }
        }
        
        const [newH, _] = await this.cells[l].forward(xt, h);
        h = newH;
        outputs.push(newH);
      }
      
      layerStates[l] = h;
      
      currentInput = Tensor.zeros([batch, seqLen, this.hiddenSize]);
      for (let t = 0; t < seqLen; t++) {
        for (let b = 0; b < batch; b++) {
          for (let f = 0; f < this.hiddenSize; f++) {
            currentInput._data[b * seqLen * this.hiddenSize + t * this.hiddenSize + f] = 
              outputs[t]._data[b * this.hiddenSize + f];
          }
        }
      }
    }
    
    return [currentInput, layerStates];
  }
}

// ============================================================
// Loss Functions
// ============================================================
export async function mseLoss(predictions, targets) {
  const diff = await predictions.sub(targets);
  const squared = await diff.mul(diff);
  return await squared.mean();
}

export async function crossEntropyLoss(logits, targets) {
  const probs = await logits.softmax(-1);
  const logProbs = await probs.log();
  const weighted = await logProbs.mul(targets);
  const summed = await weighted.sum();
  const negated = await summed.neg();
  const batchSize = logits.shape[0] || 1;
  return await negated.div(Tensor.from([batchSize], [1]));
}

export async function bceLoss(predictions, targets) {
  const eps = Tensor.from([1e-7], [1]);
  const one = Tensor.from([1], [1]);
  
  const pSafe = await predictions.add(eps);
  const oneMinusP = await one.sub(predictions);
  const oneMinusPSafe = await oneMinusP.add(eps);
  
  const logP = await pSafe.log();
  const logOneMinusP = await oneMinusPSafe.log();
  
  const term1 = await targets.mul(logP);
  const oneMinusY = await one.sub(targets);
  const term2 = await oneMinusY.mul(logOneMinusP);
  
  const sum = await term1.add(term2);
  const neg = await sum.neg();
  return await neg.mean();
}

// ============================================================
// Optimizers
// ============================================================
export class SGD {
  constructor(parameters, lr = 0.01, { momentum = 0, weightDecay = 0 } = {}) {
    this.parameters = parameters;
    this.lr = lr;
    this.momentum = momentum;
    this.weightDecay = weightDecay;
    
    if (momentum > 0) {
      this.velocities = new Map();
      for (const param of parameters) {
        this.velocities.set(param.id, new Float32Array(param.size));
      }
    }
  }
  
  async step() {
    for (const param of this.parameters) {
      if (param._grad && param.requiresGrad) {
        if (this.momentum > 0) {
          const v = this.velocities.get(param.id);
          const kernel = Autograd.genSGDMomentumKernel(param.size, this.lr, this.momentum);
          const result = await _runtime.execute(kernel,
            { workgroupCount: [Math.ceil(param.size / 256), 1, 1] },
            { param: param._data, grad: param._grad, velocity: v }
          );
          param._data = result.param;
          this.velocities.set(param.id, result.velocity);
        } else {
          const kernel = Autograd.genSGDKernel(param.size, this.lr);
          const result = await _runtime.execute(kernel,
            { workgroupCount: [Math.ceil(param.size / 256), 1, 1] },
            { param: param._data, grad: param._grad }
          );
          param._data = result instanceof Float32Array ? result : result.param;
        }
      }
    }
  }
  
  zeroGrad() {
    for (const param of this.parameters) {
      param._grad = null;
    }
    clearTape();
  }
}

export class Adam {
  constructor(parameters, lr = 0.001, { beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, weightDecay = 0 } = {}) {
    this.parameters = parameters;
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.epsilon = epsilon;
    this.weightDecay = weightDecay;
    this.t = 0;
    
    this.m = new Map();
    this.v = new Map();
    for (const param of parameters) {
      this.m.set(param.id, new Float32Array(param.size));
      this.v.set(param.id, new Float32Array(param.size));
    }
  }
  
  async step() {
    this.t += 1;
    
    for (const param of this.parameters) {
      if (!param._grad || !param.requiresGrad) continue;
      
      const kernelFn = this.weightDecay > 0 ? Autograd.genAdamWKernel : Autograd.genAdamKernel;
      const kernel = this.weightDecay > 0
        ? kernelFn(param.size, this.lr, this.beta1, this.beta2, this.epsilon, this.weightDecay, this.t)
        : kernelFn(param.size, this.lr, this.beta1, this.beta2, this.epsilon, this.t);
      
      const result = await _runtime.execute(kernel,
        { workgroupCount: [Math.ceil(param.size / 256), 1, 1] },
        {
          param: param._data,
          grad: param._grad,
          m: this.m.get(param.id),
          v: this.v.get(param.id)
        }
      );
      
      param._data = result.param;
      this.m.set(param.id, result.m);
      this.v.set(param.id, result.v);
    }
  }
  
  zeroGrad() {
    for (const param of this.parameters) {
      param._grad = null;
    }
    clearTape();
  }
}

// ============================================================
// Exports
// ============================================================
export const nn = {
  // Core

  Tensor,
  Module,
  Linear,
  ReLU,
  LeakyReLU,
  Tanh,
  Sigmoid,
  GeLU,
  Softmax,
  Dropout,
  Sequential,
  // Convolutional
  Conv2D,
  MaxPool2D,
  AvgPool2D,
  Flatten,
  // Normalization
  LayerNorm,
  BatchNorm1D,
  BatchNorm2D,
  // Embedding
  Embedding,
  // RNN
  LSTMCell,
  LSTM,
  GRUCell,
  GRU,
  init,
  clearTape,
  clipGradNorm,
  clipGradValue,
  mseLoss,
  crossEntropyLoss,
  bceLoss,
};

export const optim = {
  SGD,
  Adam,
};

// ============================================================
// Save/Load Utilities
// ============================================================
export function saveModel(model, filename = 'model.json') {
  const state = model.stateDict();
  const json = JSON.stringify(state);
  
  // Create download link
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
  
  return state;
}

export async function loadModel(model, file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const state = JSON.parse(e.target.result);
        model.loadStateDict(state);
        resolve(state);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

export function modelSummary(model) {
  const params = model.parameters();
  let summary = `Model Summary\n${'='.repeat(50)}\n`;
  summary += `Total parameters: ${model.numParameters()}\n`;
  summary += `${'='.repeat(50)}\n`;
  
  for (const param of params) {
    summary += `${param.name}: [${param.shape}] = ${param.size} params\n`;
  }
  
  return summary;
}
