// src/trainer.js
// High-level training API that wraps GradTape and Autograd
//
// Usage:
//   const trainer = new Trainer(runtime);
//   
//   // Mark parameters
//   trainer.markParameter(weightNodeId, weightBuffer);
//   
//   // Forward pass (records to tape automatically via graph execution)
//   const loss = await executor.execute(inputs);
//   
//   // Backward pass
//   await trainer.backward(compiledGraph, lossNodeId);
//   
//   // Update parameters
//   await trainer.step();

import { Autograd, GradTape } from '../dist/bundle.js';

export class Trainer {
  constructor(runtime, options = {}) {
    this.runtime = runtime;
    this.lr = options.lr || 0.001;
    this.optimizer = options.optimizer || 'adam';
    this.beta1 = options.beta1 || 0.9;
    this.beta2 = options.beta2 || 0.999;
    this.epsilon = options.epsilon || 1e-8;
    this.weightDecay = options.weightDecay || 0;
    this.t = 0; // Adam timestep
    
    // Parameter tracking
    this.parameters = new Map(); // nodeId -> { buffer, shape, name }
    this.gradients = new Map();  // nodeId -> gradient buffer
    this.adamState = new Map();  // nodeId -> { m, v } for Adam
    
    // Forward pass cache (for backward)
    this.forwardCache = new Map(); // nodeId -> buffer (activations/outputs)
    
    // Gradient tape
    this.tape = GradTape.create();
  }
  
  // Mark a node as a trainable parameter
  markParameter(nodeId, buffer, shape, name = null) {
    this.parameters.set(nodeId, { buffer, shape, name: name || `param_${nodeId}` });
    GradTape.markParameter(this.tape, nodeId);
    
    // Initialize Adam state if needed
    if (this.optimizer === 'adam' || this.optimizer === 'adamw') {
      const size = shape.reduce((a, b) => a * b, 1);
      this.adamState.set(nodeId, {
        m: this.runtime.createBuffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, new Float32Array(size)),
        v: this.runtime.createBuffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, new Float32Array(size)),
      });
    }
  }
  
  // Cache forward pass output for use in backward
  cacheForward(nodeId, buffer) {
    this.forwardCache.set(nodeId, buffer);
  }
  
  // Record an operation to the tape
  recordOp(nodeId, op, inputIds, inputShapes, outputShape) {
    GradTape.recordOp(this.tape, nodeId, op, inputIds, inputShapes, outputShape);
  }
  
  // Zero all gradients
  async zeroGrad() {
    for (const [nodeId, param] of this.parameters) {
      const size = param.shape.reduce((a, b) => a * b, 1);
      
      if (!this.gradients.has(nodeId)) {
        // Create gradient buffer
        this.gradients.set(nodeId, 
          this.runtime.createBuffer(size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC, new Float32Array(size))
        );
      }
      
      // Zero the gradient
      const kernel = Autograd.genGradZeroKernel(size);
      const dispatch = { workgroupCount: [Math.ceil(size / 256), 1, 1] };
      await this.runtime.execute(kernel, dispatch, { grad: this.gradients.get(nodeId) });
    }
  }
  
  // Compute backward pass for a simple computation
  // For now, this handles simple cases like: loss = f(x, params)
  async backward(lossBuffer, lossShape, compiledGraph) {
    const lossSize = lossShape.reduce((a, b) => a * b, 1);
    
    // Initialize loss gradient to 1.0
    const onesData = new Float32Array(lossSize).fill(1.0);
    const lossGrad = this.runtime.createBuffer(
      lossSize * 4,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      onesData
    );
    
    // Process operations in reverse order
    const ops = [...compiledGraph.ops].reverse();
    
    let currentGrad = lossGrad;
    let currentGradShape = lossShape;
    
    for (const op of ops) {
      const nodeId = op.nodeId;
      
      // Get backward kernels for this op
      // This is simplified - in practice we'd use the tape recording
      const backwardKernels = this.getBackwardKernelsForOp(op, currentGradShape);
      
      for (const { kernel, inputBindings } of backwardKernels) {
        // Build input map for the kernel
        const inputs = {};
        
        // Map gradient and cached values
        for (const bindingName of Object.keys(inputBindings)) {
          const binding = inputBindings[bindingName];
          if (binding.type === 'grad_out') {
            inputs[bindingName] = currentGrad;
          } else if (binding.type === 'cached') {
            inputs[bindingName] = this.forwardCache.get(binding.nodeId);
          }
        }
        
        const dispatch = { workgroupCount: [Math.ceil(kernel.bindings[0].size / 4 / 256), 1, 1] };
        const results = await this.runtime.execute(kernel, dispatch, inputs);
        
        // Accumulate gradients for parameters
        for (const [name, buffer] of Object.entries(results)) {
          if (name.startsWith('grad_')) {
            const targetNodeId = inputBindings[name]?.nodeId;
            if (targetNodeId && this.parameters.has(targetNodeId)) {
              await this.accumulateGradient(targetNodeId, buffer);
            }
          }
        }
      }
    }
    
    lossGrad.destroy();
  }
  
  // Simple backward for common patterns
  // loss = mean((pred - target)^2) for MSE
  // This computes gradients for all parameters
  async backwardMSE(predictions, targets, predShape) {
    const size = predShape.reduce((a, b) => a * b, 1);
    
    // d(MSE)/d(pred) = 2 * (pred - target) / n
    // For simplicity, we'll compute this directly
    const kernel = {
      name: 'mse_backward',
      wgsl: `
@group(0) @binding(0) var<storage, read> pred: array<f32>;
@group(0) @binding(1) var<storage, read> target: array<f32>;
@group(0) @binding(2) var<storage, read_write> grad: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${size}u) { return; }
  grad[idx] = 2.0 * (pred[idx] - target[idx]) / f32(${size}u);
}`,
      bindings: [
        { binding: 0, size: size * 4, usage: 'ReadOnly', name: 'pred' },
        { binding: 1, size: size * 4, usage: 'ReadOnly', name: 'target' },
        { binding: 2, size: size * 4, usage: 'ReadWrite', name: 'grad' },
      ]
    };
    
    const dispatch = { workgroupCount: [Math.ceil(size / 256), 1, 1] };
    const result = await this.runtime.execute(kernel, dispatch, {
      pred: predictions,
      target: targets
    });
    
    return result;
  }
  
  // Accumulate gradient into parameter's gradient buffer
  async accumulateGradient(nodeId, gradBuffer) {
    if (!this.gradients.has(nodeId)) return;
    
    const param = this.parameters.get(nodeId);
    const size = param.shape.reduce((a, b) => a * b, 1);
    
    const kernel = Autograd.genGradAccumulateKernel(size);
    const dispatch = { workgroupCount: [Math.ceil(size / 256), 1, 1] };
    
    await this.runtime.execute(kernel, dispatch, {
      grad_acc: this.gradients.get(nodeId),
      grad_new: gradBuffer
    });
  }
  
  // Apply gradients to parameters (optimizer step)
  async step() {
    this.t += 1;
    
    for (const [nodeId, param] of this.parameters) {
      const size = param.shape.reduce((a, b) => a * b, 1);
      const gradBuffer = this.gradients.get(nodeId);
      
      if (!gradBuffer) continue;
      
      let kernel;
      let inputs;
      
      switch (this.optimizer) {
        case 'sgd':
          kernel = Autograd.genSGDKernel(size, this.lr);
          inputs = { param: param.buffer, grad: gradBuffer };
          break;
          
        case 'sgd_momentum':
          kernel = Autograd.genSGDMomentumKernel(size, this.lr, this.momentum || 0.9);
          const velocity = this.adamState.get(nodeId)?.m; // Reuse m buffer for velocity
          inputs = { param: param.buffer, grad: gradBuffer, velocity };
          break;
          
        case 'adam':
          kernel = Autograd.genAdamKernel(size, this.lr, this.beta1, this.beta2, this.epsilon, this.t);
          const adamState = this.adamState.get(nodeId);
          inputs = { param: param.buffer, grad: gradBuffer, m: adamState.m, v: adamState.v };
          break;
          
        case 'adamw':
          kernel = Autograd.genAdamWKernel(size, this.lr, this.beta1, this.beta2, this.epsilon, this.weightDecay, this.t);
          const adamwState = this.adamState.get(nodeId);
          inputs = { param: param.buffer, grad: gradBuffer, m: adamwState.m, v: adamwState.v };
          break;
          
        default:
          throw new Error(`Unknown optimizer: ${this.optimizer}`);
      }
      
      const dispatch = { workgroupCount: [Math.ceil(size / 256), 1, 1] };
      await this.runtime.execute(kernel, dispatch, inputs);
    }
  }
  
  // Get current learning rate (for LR schedulers)
  getLR() {
    return this.lr;
  }
  
  // Set learning rate
  setLR(lr) {
    this.lr = lr;
  }
  
  // Helper to get backward kernels for an op (simplified version)
  getBackwardKernelsForOp(op, gradShape) {
    // This would be more sophisticated in practice
    // For now, return empty - the tape-based approach handles this
    return [];
  }
  
  // Cleanup
  destroy() {
    for (const buffer of this.gradients.values()) {
      buffer.destroy();
    }
    for (const state of this.adamState.values()) {
      state.m.destroy();
      state.v.destroy();
    }
    this.forwardCache.clear();
  }
}

// Convenience function to compute MSE loss
export async function mseLoss(runtime, predictions, targets, shape) {
  const size = shape.reduce((a, b) => a * b, 1);
  
  const kernel = {
    name: 'mse_loss',
    wgsl: `
@group(0) @binding(0) var<storage, read> pred: array<f32>;
@group(0) @binding(1) var<storage, read> target: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
var<workgroup> shared: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = gid.x;
  var diff_sq = 0.0;
  if (idx < ${size}u) {
    let diff = pred[idx] - target[idx];
    diff_sq = diff * diff;
  }
  shared[lid.x] = diff_sq;
  workgroupBarrier();
  
  // Parallel reduction
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (lid.x < s) {
      shared[lid.x] = shared[lid.x] + shared[lid.x + s];
    }
    workgroupBarrier();
  }
  
  if (lid.x == 0u) {
    loss[0] = shared[0] / f32(${size}u);
  }
}`,
    bindings: [
      { binding: 0, size: size * 4, usage: 'ReadOnly', name: 'pred' },
      { binding: 1, size: size * 4, usage: 'ReadOnly', name: 'target' },
      { binding: 2, size: 4, usage: 'ReadWrite', name: 'loss' },
    ]
  };
  
  const dispatch = { workgroupCount: [1, 1, 1] };
  const result = await runtime.execute(kernel, dispatch, {
    pred: predictions,
    target: targets
  });
  
  return result;
}

// Cross-entropy loss (for classification)
export async function crossEntropyLoss(runtime, logits, targets, numClasses, batchSize) {
  const kernel = {
    name: 'cross_entropy_loss',
    wgsl: `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> targets: array<f32>;
@group(0) @binding(2) var<storage, read_write> loss: array<f32>;
const NUM_CLASSES = ${numClasses}u;
const BATCH_SIZE = ${batchSize}u;
@compute @workgroup_size(1)
fn main() {
  var total_loss = 0.0;
  for (var b = 0u; b < BATCH_SIZE; b = b + 1u) {
    let base = b * NUM_CLASSES;
    let target_class = u32(targets[b]);
    
    // Compute log-softmax for numerical stability
    var max_logit = logits[base];
    for (var c = 1u; c < NUM_CLASSES; c = c + 1u) {
      max_logit = max(max_logit, logits[base + c]);
    }
    
    var sum_exp = 0.0;
    for (var c = 0u; c < NUM_CLASSES; c = c + 1u) {
      sum_exp = sum_exp + exp(logits[base + c] - max_logit);
    }
    
    let log_softmax = logits[base + target_class] - max_logit - log(sum_exp);
    total_loss = total_loss - log_softmax;
  }
  
  loss[0] = total_loss / f32(BATCH_SIZE);
}`,
    bindings: [
      { binding: 0, size: batchSize * numClasses * 4, usage: 'ReadOnly', name: 'logits' },
      { binding: 1, size: batchSize * 4, usage: 'ReadOnly', name: 'targets' },
      { binding: 2, size: 4, usage: 'ReadWrite', name: 'loss' },
    ]
  };
  
  const dispatch = { workgroupCount: [1, 1, 1] };
  return await runtime.execute(kernel, dispatch, { logits, targets });
}
