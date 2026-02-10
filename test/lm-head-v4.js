/**
 * LM Head V4 - Batched Logits Per Workgroup
 * 
 * Previous versions: 1 workgroup per vocab token = 32K-152K workgroups
 * This version: Each workgroup computes multiple logits
 * 
 * Key insight: Workgroup dispatch overhead is significant.
 * By batching, we reduce dispatch overhead and improve cache locality.
 */

/**
 * Batched LM Head kernel - each workgroup computes BATCH_SIZE logits
 */
function genLMHeadBatchedKernel(vocabSize, hiddenSize, batchSize) {
  const wgSize = 256;
  const hiddenSizePacked = hiddenSize / 2;
  const numWorkgroups = Math.ceil(vocabSize / batchSize);
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const BATCH_SIZE = ${batchSize}u;
const WG_SIZE = ${wgSize}u;

// Shared memory for hidden state (loaded once per workgroup)
var<workgroup> wg_hidden: array<f32, ${hiddenSize}>;
var<workgroup> wg_partial: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let batch_idx = wgid.x;
  let tid = lid.x;
  let vocab_start = batch_idx * BATCH_SIZE;
  
  // Load hidden state into shared memory (all threads cooperate)
  for (var d = tid; d < HIDDEN_SIZE; d = d + WG_SIZE) {
    wg_hidden[d] = hidden[d];
  }
  workgroupBarrier();
  
  // Each thread computes one logit in the batch
  // If BATCH_SIZE > WG_SIZE, threads loop
  for (var b = tid; b < BATCH_SIZE; b = b + WG_SIZE) {
    let vocab_idx = vocab_start + b;
    if (vocab_idx >= VOCAB_SIZE) { continue; }
    
    var sum = 0.0;
    let weight_base = vocab_idx * HIDDEN_SIZE_PACKED;
    
    // Compute dot product using shared hidden state
    for (var d = 0u; d < HIDDEN_SIZE_PACKED; d = d + 1u) {
      let packed = weight[weight_base + d];
      let unpacked = unpack2x16float(packed);
      let h_idx = d * 2u;
      sum = sum + wg_hidden[h_idx] * unpacked.x;
      sum = sum + wg_hidden[h_idx + 1u] * unpacked.y;
    }
    
    logits[vocab_idx] = sum;
  }
}`;
}

/**
 * Alternative: Tile-based approach with better memory access
 * Processes vocab in tiles, with hidden dimension split across threads
 */
function genLMHeadTiledKernel(vocabSize, hiddenSize, tileSize) {
  const wgSize = 256;
  const hiddenSizePacked = hiddenSize / 2;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const TILE_SIZE = ${tileSize}u;
const WG_SIZE = ${wgSize}u;

var<workgroup> wg_hidden: array<f32, ${hiddenSize}>;
var<workgroup> wg_sums: array<f32, ${tileSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tile_idx = wgid.x;
  let tid = lid.x;
  let vocab_start = tile_idx * TILE_SIZE;
  
  // Cooperatively load hidden state
  for (var d = tid; d < HIDDEN_SIZE; d = d + WG_SIZE) {
    wg_hidden[d] = hidden[d];
  }
  workgroupBarrier();
  
  // Process tile - each thread handles subset of vocab tokens
  let tokens_per_thread = (TILE_SIZE + WG_SIZE - 1u) / WG_SIZE;
  
  for (var t = 0u; t < tokens_per_thread; t = t + 1u) {
    let local_idx = tid * tokens_per_thread + t;
    if (local_idx >= TILE_SIZE) { continue; }
    
    let vocab_idx = vocab_start + local_idx;
    if (vocab_idx >= VOCAB_SIZE) { continue; }
    
    var sum = 0.0;
    let weight_base = vocab_idx * HIDDEN_SIZE_PACKED;
    
    for (var d = 0u; d < HIDDEN_SIZE_PACKED; d = d + 1u) {
      let packed = weight[weight_base + d];
      let unpacked = unpack2x16float(packed);
      sum = sum + wg_hidden[d * 2u] * unpacked.x;
      sum = sum + wg_hidden[d * 2u + 1u] * unpacked.y;
    }
    
    logits[vocab_idx] = sum;
  }
}`;
}

/**
 * Vectorized LM Head - process 4 vocab tokens per thread using vec4
 */
function genLMHeadVec4Kernel(vocabSize, hiddenSize) {
  const wgSize = 64; // Smaller workgroup, more work per thread
  const hiddenSizePacked = hiddenSize / 2;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const WG_SIZE = ${wgSize}u;
const TOKENS_PER_WG = ${wgSize * 4}u;

var<workgroup> wg_hidden: array<f32, ${hiddenSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let wg_idx = wgid.x;
  let tid = lid.x;
  let vocab_start = wg_idx * TOKENS_PER_WG;
  
  // Load hidden into shared memory
  for (var d = tid; d < HIDDEN_SIZE; d = d + WG_SIZE) {
    wg_hidden[d] = hidden[d];
  }
  workgroupBarrier();
  
  // Each thread processes 4 vocab tokens
  for (var t = 0u; t < 4u; t = t + 1u) {
    let vocab_idx = vocab_start + tid * 4u + t;
    if (vocab_idx >= VOCAB_SIZE) { continue; }
    
    var sum = 0.0;
    let weight_base = vocab_idx * HIDDEN_SIZE_PACKED;
    
    for (var d = 0u; d < HIDDEN_SIZE_PACKED; d = d + 1u) {
      let packed = weight[weight_base + d];
      let unpacked = unpack2x16float(packed);
      sum = sum + wg_hidden[d * 2u] * unpacked.x;
      sum = sum + wg_hidden[d * 2u + 1u] * unpacked.y;
    }
    
    logits[vocab_idx] = sum;
  }
}`;
}

/**
 * LM Head V4 class with batched processing
 */
class LMHeadV4 {
  constructor(device, vocabSize, hiddenSize, options = {}) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.batchSize = options.batchSize || 256; // Vocab tokens per workgroup
    this.weightBuffer = null;
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    const shader = genLMHeadBatchedKernel(this.vocabSize, this.hiddenSize, this.batchSize);
    const module = this.device.createShaderModule({ code: shader });
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
    
    this.logitsBuffer = this.device.createBuffer({
      size: this.vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'logits'
    });
    
    this.numWorkgroups = Math.ceil(this.vocabSize / this.batchSize);
    console.log(`LM Head V4: ${this.numWorkgroups} workgroups (batch=${this.batchSize})`);
  }
  
  forward(hiddenBuffer) {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.numWorkgroups);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    
    return this.logitsBuffer;
  }
  
  getLogitsBuffer() {
    return this.logitsBuffer;
  }
  
  destroy() {
    if (this.logitsBuffer) this.logitsBuffer.destroy();
  }
}

/**
 * LM Head with vec4 processing
 */
class LMHeadVec4 {
  constructor(device, vocabSize, hiddenSize) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.weightBuffer = null;
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    const shader = genLMHeadVec4Kernel(this.vocabSize, this.hiddenSize);
    const module = this.device.createShaderModule({ code: shader });
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
    
    this.logitsBuffer = this.device.createBuffer({
      size: this.vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'logits'
    });
    
    const tokensPerWg = 64 * 4; // wgSize * 4
    this.numWorkgroups = Math.ceil(this.vocabSize / tokensPerWg);
    console.log(`LM Head Vec4: ${this.numWorkgroups} workgroups`);
  }
  
  forward(hiddenBuffer) {
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
    
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(this.numWorkgroups);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    
    return this.logitsBuffer;
  }
  
  getLogitsBuffer() {
    return this.logitsBuffer;
  }
  
  destroy() {
    if (this.logitsBuffer) this.logitsBuffer.destroy();
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LMHeadV4, LMHeadVec4, genLMHeadBatchedKernel, genLMHeadVec4Kernel };
}
if (typeof window !== 'undefined') {
  window.LMHeadV4 = LMHeadV4;
  window.LMHeadVec4 = LMHeadVec4;
  window.genLMHeadBatchedKernel = genLMHeadBatchedKernel;
  window.genLMHeadVec4Kernel = genLMHeadVec4Kernel;
}
