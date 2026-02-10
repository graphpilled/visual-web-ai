/**
 * LM Head v2 - GPU-Only Top-K
 * 
 * Previous approach was slow due to:
 * 1. Multiple GPU→CPU round-trips per chunk
 * 2. CPU-side merging overhead
 * 
 * New approach:
 * 1. Compute ALL logits on GPU (single dispatch)
 * 2. GPU-based top-k reduction (no CPU round-trips)
 * 3. Only final top-k transferred to CPU
 * 
 * For sampling, we don't need all 152K logits - just top-k candidates.
 * We can use a two-phase approach:
 * - Phase 1: Each workgroup finds local max
 * - Phase 2: Reduce local maxes to global top-k
 */

/**
 * Single-pass LM Head that computes logits and finds top-k in one go
 * Uses workgroup-level reduction to find candidates
 */
function genLMHeadWithTopKKernel(vocabSize, hiddenSize, topK) {
  const wgSize = 256;
  const hiddenSizePacked = hiddenSize / 2;
  const numWorkgroups = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const WG_SIZE = ${wgSize}u;

var<workgroup> wg_partial: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let vocab_idx = wgid.x;
  let tid = lid.x;
  
  if (vocab_idx >= VOCAB_SIZE) { return; }
  
  var partial_sum = 0.0;
  let weight_base = vocab_idx * HIDDEN_SIZE_PACKED;
  
  for (var d = tid; d < HIDDEN_SIZE_PACKED; d = d + WG_SIZE) {
    let packed = weight[weight_base + d];
    let unpacked = unpack2x16float(packed);
    let h_idx = d * 2u;
    partial_sum = partial_sum + hidden[h_idx] * unpacked.x;
    partial_sum = partial_sum + hidden[h_idx + 1u] * unpacked.y;
  }
  
  wg_partial[tid] = partial_sum;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_partial[tid] = wg_partial[tid] + wg_partial[tid + stride];
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    logits[vocab_idx] = wg_partial[0];
  }
}`;
}

/**
 * GPU-based argmax kernel
 * Finds the index of maximum value
 */
function genArgmaxKernel(size) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result_value: array<f32>;
@group(0) @binding(2) var<storage, read_write> result_index: array<u32>;

const SIZE = ${size}u;
const WG_SIZE = ${wgSize}u;

var<workgroup> wg_values: array<f32, ${wgSize}>;
var<workgroup> wg_indices: array<u32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  
  // Each thread finds local max in its portion
  var local_max = -1e30;
  var local_idx = 0u;
  
  for (var i = tid; i < SIZE; i = i + WG_SIZE) {
    let val = values[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }
  
  wg_values[tid] = local_max;
  wg_indices[tid] = local_idx;
  workgroupBarrier();
  
  // Parallel reduction to find global max
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      if (wg_values[tid + stride] > wg_values[tid]) {
        wg_values[tid] = wg_values[tid + stride];
        wg_indices[tid] = wg_indices[tid + stride];
      }
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    result_value[0] = wg_values[0];
    result_index[0] = wg_indices[0];
  }
}`;
}

/**
 * GPU-based top-k kernel using iterative approach
 * Finds top-k by repeatedly finding max and masking
 */
function genTopKIterativeKernel(size, k) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read_write> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> top_values: array<f32>;
@group(0) @binding(2) var<storage, read_write> top_indices: array<u32>;
@group(0) @binding(3) var<storage, read> k_idx: array<u32>;

const SIZE = ${size}u;
const WG_SIZE = ${wgSize}u;
const NEG_INF = -1e30;

var<workgroup> wg_values: array<f32, ${wgSize}>;
var<workgroup> wg_indices: array<u32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let current_k = k_idx[0];
  
  // Find max
  var local_max = NEG_INF;
  var local_idx = 0u;
  
  for (var i = tid; i < SIZE; i = i + WG_SIZE) {
    let val = values[i];
    if (val > local_max) {
      local_max = val;
      local_idx = i;
    }
  }
  
  wg_values[tid] = local_max;
  wg_indices[tid] = local_idx;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      if (wg_values[tid + stride] > wg_values[tid]) {
        wg_values[tid] = wg_values[tid + stride];
        wg_indices[tid] = wg_indices[tid + stride];
      }
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    let max_idx = wg_indices[0];
    top_values[current_k] = wg_values[0];
    top_indices[current_k] = max_idx;
    // Mask out the found max for next iteration
    values[max_idx] = NEG_INF;
  }
}`;
}

/**
 * Softmax kernel for top-k values (for sampling)
 */
function genSoftmaxTopKKernel(k) {
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<storage, read> temperature: array<f32>;

const K = ${k}u;

@compute @workgroup_size(${Math.min(k, 256)})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let temp = temperature[0];
  
  // Find max for numerical stability (single thread for small k)
  var max_val = input[0];
  for (var i = 1u; i < K; i = i + 1u) {
    max_val = max(max_val, input[i]);
  }
  
  // Compute exp and sum
  var sum_exp = 0.0;
  for (var i = 0u; i < K; i = i + 1u) {
    sum_exp = sum_exp + exp((input[i] - max_val) / temp);
  }
  
  // Normalize
  if (tid < K) {
    output[tid] = exp((input[tid] - max_val) / temp) / sum_exp;
  }
}`;
}

/**
 * LM Head v2 - Compute all logits, then GPU argmax/top-k
 */
class LMHeadV2 {
  constructor(device, vocabSize, hiddenSize) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.weightBuffer = null;
    
    this.logitsPipeline = null;
    this.argmaxPipeline = null;
    this.logitsBuffer = null;
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    // Logits computation pipeline
    const logitsShader = genLMHeadWithTopKKernel(this.vocabSize, this.hiddenSize, 256);
    const logitsModule = this.device.createShaderModule({ code: logitsShader });
    this.logitsPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: logitsModule, entryPoint: 'main' }
    });
    
    // Argmax pipeline
    const argmaxShader = genArgmaxKernel(this.vocabSize);
    const argmaxModule = this.device.createShaderModule({ code: argmaxShader });
    this.argmaxPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: argmaxModule, entryPoint: 'main' }
    });
    
    // Logits buffer
    this.logitsBuffer = this.device.createBuffer({
      size: this.vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'logits'
    });
    
    // Argmax result buffers
    this.argmaxValueBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'argmax_value'
    });
    
    this.argmaxIndexBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'argmax_index'
    });
    
    console.log(`LM Head V2 initialized: [${this.hiddenSize}] -> [${this.vocabSize}]`);
  }
  
  /**
   * Compute logits and return argmax (greedy decoding)
   * Single GPU round-trip
   */
  async forwardArgmax(hiddenBuffer) {
    const commandEncoder = this.device.createCommandEncoder();
    
    // Phase 1: Compute all logits
    const logitsBindGroup = this.device.createBindGroup({
      layout: this.logitsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
    
    const logitsPass = commandEncoder.beginComputePass();
    logitsPass.setPipeline(this.logitsPipeline);
    logitsPass.setBindGroup(0, logitsBindGroup);
    logitsPass.dispatchWorkgroups(this.vocabSize);
    logitsPass.end();
    
    // Phase 2: Find argmax on GPU
    const argmaxBindGroup = this.device.createBindGroup({
      layout: this.argmaxPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.logitsBuffer } },
        { binding: 1, resource: { buffer: this.argmaxValueBuffer } },
        { binding: 2, resource: { buffer: this.argmaxIndexBuffer } }
      ]
    });
    
    const argmaxPass = commandEncoder.beginComputePass();
    argmaxPass.setPipeline(this.argmaxPipeline);
    argmaxPass.setBindGroup(0, argmaxBindGroup);
    argmaxPass.dispatchWorkgroups(1);
    argmaxPass.end();
    
    // Copy results to staging
    const readValueBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const readIndexBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    
    commandEncoder.copyBufferToBuffer(this.argmaxValueBuffer, 0, readValueBuffer, 0, 4);
    commandEncoder.copyBufferToBuffer(this.argmaxIndexBuffer, 0, readIndexBuffer, 0, 4);
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Single GPU→CPU transfer of just 8 bytes!
    await readValueBuffer.mapAsync(GPUMapMode.READ);
    await readIndexBuffer.mapAsync(GPUMapMode.READ);
    
    const value = new Float32Array(readValueBuffer.getMappedRange())[0];
    const index = new Uint32Array(readIndexBuffer.getMappedRange())[0];
    
    readValueBuffer.unmap();
    readIndexBuffer.unmap();
    readValueBuffer.destroy();
    readIndexBuffer.destroy();
    
    return { value, index };
  }
  
  /**
   * Get logits buffer for external sampling
   */
  getLogitsBuffer() {
    return this.logitsBuffer;
  }
  
  destroy() {
    if (this.logitsBuffer) this.logitsBuffer.destroy();
    if (this.argmaxValueBuffer) this.argmaxValueBuffer.destroy();
    if (this.argmaxIndexBuffer) this.argmaxIndexBuffer.destroy();
  }
}

/**
 * LM Head V3 - With GPU-based Top-K for sampling
 */
class LMHeadV3 {
  constructor(device, vocabSize, hiddenSize, topK = 50) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.topK = topK;
    this.weightBuffer = null;
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    // Logits pipeline
    const logitsShader = genLMHeadWithTopKKernel(this.vocabSize, this.hiddenSize, this.topK);
    const logitsModule = this.device.createShaderModule({ code: logitsShader });
    this.logitsPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: logitsModule, entryPoint: 'main' }
    });
    
    // Top-k iterative pipeline
    const topkShader = genTopKIterativeKernel(this.vocabSize, this.topK);
    const topkModule = this.device.createShaderModule({ code: topkShader });
    this.topkPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: topkModule, entryPoint: 'main' }
    });
    
    // Buffers
    this.logitsBuffer = this.device.createBuffer({
      size: this.vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'logits'
    });
    
    this.logitsCopyBuffer = this.device.createBuffer({
      size: this.vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'logits_copy'
    });
    
    this.topKValuesBuffer = this.device.createBuffer({
      size: this.topK * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'topk_values'
    });
    
    this.topKIndicesBuffer = this.device.createBuffer({
      size: this.topK * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'topk_indices'
    });
    
    this.kIdxBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'k_idx'
    });
    
    console.log(`LM Head V3 initialized: vocab=${this.vocabSize}, topK=${this.topK}`);
  }
  
  /**
   * Compute logits and extract top-k on GPU
   */
  async forwardTopK(hiddenBuffer) {
    // Phase 1: Compute logits
    const logitsBindGroup = this.device.createBindGroup({
      layout: this.logitsPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
    
    let commandEncoder = this.device.createCommandEncoder();
    const logitsPass = commandEncoder.beginComputePass();
    logitsPass.setPipeline(this.logitsPipeline);
    logitsPass.setBindGroup(0, logitsBindGroup);
    logitsPass.dispatchWorkgroups(this.vocabSize);
    logitsPass.end();
    
    // Copy logits (we'll modify them during top-k)
    commandEncoder.copyBufferToBuffer(this.logitsBuffer, 0, this.logitsCopyBuffer, 0, this.vocabSize * 4);
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Phase 2: Iterative top-k extraction
    const topkBindGroup = this.device.createBindGroup({
      layout: this.topkPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.logitsCopyBuffer } },
        { binding: 1, resource: { buffer: this.topKValuesBuffer } },
        { binding: 2, resource: { buffer: this.topKIndicesBuffer } },
        { binding: 3, resource: { buffer: this.kIdxBuffer } }
      ]
    });
    
    // Run top-k iterations (each finds next max)
    for (let k = 0; k < this.topK; k++) {
      this.device.queue.writeBuffer(this.kIdxBuffer, 0, new Uint32Array([k]));
      
      commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.topkPipeline);
      pass.setBindGroup(0, topkBindGroup);
      pass.dispatchWorkgroups(1);
      pass.end();
      this.device.queue.submit([commandEncoder.finish()]);
    }
    
    // Read results
    const readValues = this.device.createBuffer({
      size: this.topK * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    const readIndices = this.device.createBuffer({
      size: this.topK * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });
    
    commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(this.topKValuesBuffer, 0, readValues, 0, this.topK * 4);
    commandEncoder.copyBufferToBuffer(this.topKIndicesBuffer, 0, readIndices, 0, this.topK * 4);
    this.device.queue.submit([commandEncoder.finish()]);
    
    await readValues.mapAsync(GPUMapMode.READ);
    await readIndices.mapAsync(GPUMapMode.READ);
    
    const values = new Float32Array(readValues.getMappedRange().slice(0));
    const indices = new Uint32Array(readIndices.getMappedRange().slice(0));
    
    readValues.unmap();
    readIndices.unmap();
    readValues.destroy();
    readIndices.destroy();
    
    return { values, indices };
  }
  
  destroy() {
    if (this.logitsBuffer) this.logitsBuffer.destroy();
    if (this.logitsCopyBuffer) this.logitsCopyBuffer.destroy();
    if (this.topKValuesBuffer) this.topKValuesBuffer.destroy();
    if (this.topKIndicesBuffer) this.topKIndicesBuffer.destroy();
    if (this.kIdxBuffer) this.kIdxBuffer.destroy();
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LMHeadV2, LMHeadV3, genArgmaxKernel, genTopKIterativeKernel, genSoftmaxTopKKernel };
}
if (typeof window !== 'undefined') {
  window.LMHeadV2 = LMHeadV2;
  window.LMHeadV3 = LMHeadV3;
  window.genArgmaxKernel = genArgmaxKernel;
  window.genTopKIterativeKernel = genTopKIterativeKernel;
  window.genSoftmaxTopKKernel = genSoftmaxTopKKernel;
}
