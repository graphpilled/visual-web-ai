/**
 * LM Head - Final Projection to Vocabulary Logits
 * 
 * For Qwen2.5-7B:
 *   - hidden_size: 3584
 *   - vocab_size: 152064
 *   - Weights stored as F16 (same as embedding, often tied/shared)
 *   - Weight shape: [vocab_size, hidden_size] = [152064, 3584]
 *   - Size: 152064 × 3584 × 2 = 1.04 GB (F16)
 * 
 * Forward pass:
 *   logits[v] = sum(hidden[d] * weight[v, d]) for d in 0..hidden_size
 */

/**
 * Ultra-optimized LM Head kernel
 * - Each workgroup computes TOKENS_PER_WG output logits
 * - Uses vec4 loads for hidden state (cached in workgroup memory)
 * - Processes 8 F16 weights per thread per iteration
 * - Reduces workgroup count from 152K to ~2400
 */
function genLMHeadKernelV2(vocabSize, hiddenSize) {
  const wgSize = 256;
  const tokensPerWG = 64;  // Each workgroup computes 64 vocab logits
  const hiddenVec4 = hiddenSize / 4;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // F16 packed as u32
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_VEC4 = ${hiddenVec4}u;
const WG_SIZE = ${wgSize}u;
const TOKENS_PER_WG = ${tokensPerWG}u;

// Cache hidden state in workgroup memory (3584 floats = 14KB)
var<workgroup> wg_hidden: array<vec4<f32>, ${hiddenVec4}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tid = lid.x;
  let base_vocab = wgid.x * TOKENS_PER_WG;
  
  // Collaboratively load hidden state into workgroup memory
  for (var i = tid; i < HIDDEN_VEC4; i = i + WG_SIZE) {
    wg_hidden[i] = hidden[i];
  }
  workgroupBarrier();
  
  // Each thread computes ceil(TOKENS_PER_WG / WG_SIZE) tokens
  // With 64 tokens and 256 threads: threads 0-63 each compute 1 token
  if (tid < TOKENS_PER_WG) {
    let vocab_idx = base_vocab + tid;
    if (vocab_idx < VOCAB_SIZE) {
      var sum = 0.0;
      let weight_base = vocab_idx * (HIDDEN_SIZE / 2u);
      
      // Process hidden in chunks of 4 (8 F16 weights = 4 u32)
      for (var i = 0u; i < HIDDEN_VEC4; i = i + 1u) {
        let h = wg_hidden[i];
        let w_idx = i * 2u;
        
        // Load 4 F16 values (2 u32s)
        let packed0 = weight[weight_base + w_idx];
        let packed1 = weight[weight_base + w_idx + 1u];
        let w01 = unpack2x16float(packed0);
        let w23 = unpack2x16float(packed1);
        
        sum = sum + h.x * w01.x + h.y * w01.y + h.z * w23.x + h.w * w23.y;
      }
      
      logits[vocab_idx] = sum;
    }
  }
}`;
}

/**
 * Alternative: Even more parallel version
 * Each thread works on part of dot product, then atomic add
 * Better for very wide hidden sizes
 */
function genLMHeadKernelV3(vocabSize, hiddenSize) {
  const wgSize = 128;
  const chunksPerToken = 4;  // Split hidden into 4 chunks
  const chunkSize = hiddenSize / chunksPerToken;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<atomic<u32>>;  // Atomic for accumulation

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const WG_SIZE = ${wgSize}u;
const CHUNK_SIZE = ${chunkSize}u;
const CHUNKS_PER_TOKEN = ${chunksPerToken}u;

var<workgroup> wg_partial: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let vocab_idx = wgid.x;
  let chunk_idx = wgid.y;
  let tid = lid.x;
  
  if (vocab_idx >= VOCAB_SIZE) { return; }
  
  let chunk_start = chunk_idx * CHUNK_SIZE;
  let chunk_end = chunk_start + CHUNK_SIZE;
  
  var partial = 0.0;
  let weight_base = vocab_idx * (HIDDEN_SIZE / 2u);
  
  for (var d = chunk_start + tid * 2u; d < chunk_end; d = d + WG_SIZE * 2u) {
    let packed = weight[weight_base + d / 2u];
    let w = unpack2x16float(packed);
    partial = partial + hidden[d] * w.x + hidden[d + 1u] * w.y;
  }
  
  wg_partial[tid] = partial;
  workgroupBarrier();
  
  // Reduce within workgroup
  for (var s = WG_SIZE / 2u; s > 0u; s = s / 2u) {
    if (tid < s) {
      wg_partial[tid] = wg_partial[tid] + wg_partial[tid + s];
    }
    workgroupBarrier();
  }
  
  // Atomic add to final result (only thread 0)
  if (tid == 0u) {
    // Convert float to bits for atomic add workaround
    let bits = bitcast<u32>(wg_partial[0]);
    atomicAdd(&logits[vocab_idx], bits);
  }
}`;
}

/**
 * Simple but effective optimized kernel
 * - Processes multiple tokens per workgroup
 * - Caches hidden state
 * - Uses parallel reduction
 */
function genLMHeadKernelFast(vocabSize, hiddenSize) {
  const wgSize = 256;
  const hiddenPacked = hiddenSize / 2;
  
  // Each workgroup handles 4 vocab tokens, using 64 threads per token
  const tokensPerWG = 4;
  const threadsPerToken = wgSize / tokensPerWG;  // 64
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${vocabSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_PACKED = ${hiddenPacked}u;
const WG_SIZE = ${wgSize}u;
const TOKENS_PER_WG = ${tokensPerWG}u;
const THREADS_PER_TOKEN = ${threadsPerToken}u;

var<workgroup> wg_sums: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tid = lid.x;
  let token_in_wg = tid / THREADS_PER_TOKEN;  // 0-3
  let tid_in_token = tid % THREADS_PER_TOKEN;  // 0-63
  
  let vocab_idx = wgid.x * TOKENS_PER_WG + token_in_wg;
  
  var sum = 0.0;
  
  if (vocab_idx < VOCAB_SIZE) {
    let weight_base = vocab_idx * HIDDEN_PACKED;
    
    // Each thread processes HIDDEN_PACKED / THREADS_PER_TOKEN elements
    // = 1792 / 64 = 28 packed values = 56 elements
    for (var i = tid_in_token; i < HIDDEN_PACKED; i = i + THREADS_PER_TOKEN) {
      let packed = weight[weight_base + i];
      let w = unpack2x16float(packed);
      let h_idx = i * 2u;
      sum = sum + hidden[h_idx] * w.x + hidden[h_idx + 1u] * w.y;
    }
  }
  
  wg_sums[tid] = sum;
  workgroupBarrier();
  
  // Parallel reduction within each token's 64 threads
  // Threads 0-63 reduce to slot 0, threads 64-127 reduce to slot 64, etc.
  let base = token_in_wg * THREADS_PER_TOKEN;
  
  for (var s = THREADS_PER_TOKEN / 2u; s > 0u; s = s / 2u) {
    if (tid_in_token < s) {
      wg_sums[base + tid_in_token] = wg_sums[base + tid_in_token] + wg_sums[base + tid_in_token + s];
    }
    workgroupBarrier();
  }
  
  // Write results (4 threads write, one per token)
  if (tid_in_token == 0u && vocab_idx < VOCAB_SIZE) {
    logits[vocab_idx] = wg_sums[base];
  }
}`;
}

/**
 * LM Head class for managing the final projection
 */
class LMHead {
  constructor(device, vocabSize, hiddenSize) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.weightBuffer = null;
    this.pipeline = null;
    this.logitsBuffer = null;
    this.bindGroup = null;
    
    // With 4 tokens per WG, we need vocabSize/4 workgroups
    this.tokensPerWG = 4;
    this.numWorkgroups = Math.ceil(vocabSize / this.tokensPerWG);
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    // Use the optimized kernel
    const shader = genLMHeadKernelFast(this.vocabSize, this.hiddenSize);
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
    
    console.log(`LM Head initialized: [${this.hiddenSize}] -> [${this.vocabSize}], workgroups: ${this.numWorkgroups} (was ${this.vocabSize})`);
  }
  
  /**
   * Pre-create bind group (call after init and after hidden buffer is created)
   */
  createBindGroup(hiddenBuffer) {
    this.bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
  }
  
  /**
   * Compute logits - now returns encoder for batching
   */
  forward(hiddenBuffer, encoder = null) {
    // Create bind group if not pre-created or if hidden buffer changed
    const bg = this.bindGroup || this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.logitsBuffer } }
      ]
    });
    
    const ownEncoder = encoder === null;
    if (ownEncoder) {
      encoder = this.device.createCommandEncoder();
    }
    
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(this.numWorkgroups);
    pass.end();
    
    if (ownEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
    
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
  module.exports = { LMHead, genLMHeadKernelFast, genLMHeadKernelV2 };
}
if (typeof window !== 'undefined') {
  window.LMHead = LMHead;
}
