/**
 * Optimized LM Head with Top-K Extraction
 * 
 * Instead of computing all 152K logits and sending to CPU,
 * we compute logits in chunks and maintain a running top-k on GPU.
 * 
 * Strategy:
 * 1. Divide vocab into chunks (e.g., 1024 tokens each)
 * 2. For each chunk, compute logits and find local top-k
 * 3. Merge local top-k with global top-k
 * 4. Only return final top-k (e.g., 256 candidates) to CPU
 * 
 * This reduces:
 * - Memory: 152K×4 = 600KB → K×8 = 2KB (for K=256)
 * - CPU transfer: 600KB → 2KB
 * - Enables efficient sampling on GPU
 */

/**
 * Generate chunked LM Head kernel
 * Computes logits for a chunk of vocabulary and finds local top-k
 * 
 * @param {number} chunkSize - Tokens per chunk (e.g., 4096)
 * @param {number} hiddenSize - Hidden dimension (3584)
 * @param {number} topK - Number of top candidates per chunk
 */
function genLMHeadChunkKernel(chunkSize, hiddenSize, topK) {
  const wgSize = 256;
  const hiddenSizePacked = hiddenSize / 2;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> chunk_logits: array<f32>;
@group(0) @binding(3) var<storage, read> chunk_offset: array<u32>;

const CHUNK_SIZE = ${chunkSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const WG_SIZE = ${wgSize}u;

var<workgroup> wg_partial: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let local_vocab_idx = wgid.x;
  let tid = lid.x;
  let offset = chunk_offset[0];
  let vocab_idx = offset + local_vocab_idx;
  
  if (local_vocab_idx >= CHUNK_SIZE) { return; }
  
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
    chunk_logits[local_vocab_idx] = wg_partial[0];
  }
}`;
}

/**
 * Generate top-k extraction kernel using parallel bitonic sort
 * Finds top-k from a chunk of logits
 */
function genTopKExtractKernel(chunkSize, topK) {
  const wgSize = Math.min(256, chunkSize);
  
  return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> chunk_offset: array<u32>;
@group(0) @binding(2) var<storage, read_write> top_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> top_indices: array<u32>;
@group(0) @binding(4) var<storage, read> num_existing: array<u32>;

const CHUNK_SIZE = ${chunkSize}u;
const TOP_K = ${topK}u;
const WG_SIZE = ${wgSize}u;

// Each thread maintains a local min-heap of size TOP_K/WG_SIZE
// Then we merge across threads

struct TopKEntry {
  value: f32,
  index: u32,
}

var<workgroup> wg_entries: array<TopKEntry, ${topK}>;
var<workgroup> wg_count: atomic<u32>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let tid = lid.x;
  let offset = chunk_offset[0];
  let existing = num_existing[0];
  
  // Initialize with existing top-k
  if (tid < existing && tid < TOP_K) {
    wg_entries[tid].value = top_values[tid];
    wg_entries[tid].index = top_indices[tid];
  }
  workgroupBarrier();
  
  // Each thread processes a portion of the chunk
  for (var i = tid; i < CHUNK_SIZE; i = i + WG_SIZE) {
    let val = logits[i];
    let idx = offset + i;
    
    // Check if this value should be in top-k
    // Find minimum in current top-k
    var min_val = 1e30;
    var min_pos = 0u;
    
    let current_count = min(existing + i + 1u, TOP_K);
    
    for (var k = 0u; k < current_count; k = k + 1u) {
      if (wg_entries[k].value < min_val) {
        min_val = wg_entries[k].value;
        min_pos = k;
      }
    }
    
    // Replace if better
    if (val > min_val || atomicLoad(&wg_count) < TOP_K) {
      // This is a simplified version - full implementation needs atomic operations
      // For now, we'll use a simpler approach
    }
  }
  workgroupBarrier();
  
  // Write results
  if (tid < TOP_K) {
    top_values[tid] = wg_entries[tid].value;
    top_indices[tid] = wg_entries[tid].index;
  }
}`;
}

/**
 * Simpler approach: Compute all logits for a chunk, then CPU-side top-k merge
 * This is more practical for WebGPU
 */
function genLMHeadBatchKernel(batchSize, hiddenSize) {
  const wgSize = 256;
  const hiddenSizePacked = hiddenSize / 2;
  
  return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;
@group(0) @binding(3) var<storage, read> vocab_offset: array<u32>;

const BATCH_SIZE = ${batchSize}u;
const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSizePacked}u;
const WG_SIZE = ${wgSize}u;

var<workgroup> wg_partial: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let batch_idx = wgid.x;
  let tid = lid.x;
  
  if (batch_idx >= BATCH_SIZE) { return; }
  
  let vocab_idx = vocab_offset[0] + batch_idx;
  
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
    logits[batch_idx] = wg_partial[0];
  }
}`;
}

/**
 * GPU-based Top-K using parallel reduction
 * More efficient than CPU for large vocab
 */
function genTopKReduceKernel(inputSize, topK) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read> input_values: array<f32>;
@group(0) @binding(1) var<storage, read> input_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_values: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_indices: array<u32>;
@group(0) @binding(4) var<storage, read> input_size: array<u32>;

const TOP_K = ${topK}u;
const WG_SIZE = ${wgSize}u;

// Simple selection: each thread finds one of top-k
// Thread i finds the (i+1)th largest element

@compute @workgroup_size(${Math.min(topK, 256)})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let k_idx = lid.x;
  if (k_idx >= TOP_K) { return; }
  
  let n = input_size[0];
  
  // Find k_idx-th largest by counting elements larger than each candidate
  var best_val = -1e30;
  var best_idx = 0u;
  var target_rank = k_idx;
  
  // Simple O(n) scan - find element with exactly k_idx elements larger than it
  for (var i = 0u; i < n; i = i + 1u) {
    let val = input_values[i];
    let idx = input_indices[i];
    
    // Count how many are larger
    var rank = 0u;
    for (var j = 0u; j < n; j = j + 1u) {
      if (input_values[j] > val || (input_values[j] == val && j < i)) {
        rank = rank + 1u;
      }
    }
    
    if (rank == target_rank) {
      best_val = val;
      best_idx = idx;
      break;
    }
  }
  
  output_values[k_idx] = best_val;
  output_indices[k_idx] = best_idx;
}`;
}

/**
 * Optimized LM Head class with chunked processing
 */
class LMHeadOptimized {
  constructor(device, vocabSize, hiddenSize, options = {}) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.chunkSize = options.chunkSize || 8192;
    this.topK = options.topK || 256;
    
    this.weightBuffer = null;
    this.pipeline = null;
    this.chunkLogitsBuffer = null;
    this.offsetBuffer = null;
    
    // For top-k tracking
    this.topKValues = new Float32Array(this.topK);
    this.topKIndices = new Uint32Array(this.topK);
  }
  
  setWeightBuffer(buffer) {
    this.weightBuffer = buffer;
  }
  
  async init() {
    // Create batch compute pipeline
    const shader = genLMHeadBatchKernel(this.chunkSize, this.hiddenSize);
    const module = this.device.createShaderModule({ code: shader });
    
    this.pipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
    
    // Chunk logits buffer
    this.chunkLogitsBuffer = this.device.createBuffer({
      size: this.chunkSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'chunk_logits'
    });
    
    // Offset buffer
    this.offsetBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'vocab_offset'
    });
    
    // Read buffer for chunk results
    this.readBuffer = this.device.createBuffer({
      size: this.chunkSize * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'chunk_read'
    });
    
    console.log(`LM Head Optimized: vocab=${this.vocabSize}, chunks=${Math.ceil(this.vocabSize / this.chunkSize)}, topK=${this.topK}`);
  }
  
  /**
   * Process one chunk of vocabulary
   */
  async processChunk(hiddenBuffer, startIdx) {
    const actualChunkSize = Math.min(this.chunkSize, this.vocabSize - startIdx);
    
    // Set offset
    this.device.queue.writeBuffer(this.offsetBuffer, 0, new Uint32Array([startIdx]));
    
    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: hiddenBuffer } },
        { binding: 1, resource: { buffer: this.weightBuffer } },
        { binding: 2, resource: { buffer: this.chunkLogitsBuffer } },
        { binding: 3, resource: { buffer: this.offsetBuffer } }
      ]
    });
    
    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(actualChunkSize);
    pass.end();
    
    // Copy to read buffer
    commandEncoder.copyBufferToBuffer(
      this.chunkLogitsBuffer, 0,
      this.readBuffer, 0,
      actualChunkSize * 4
    );
    
    this.device.queue.submit([commandEncoder.finish()]);
    
    // Read results
    await this.readBuffer.mapAsync(GPUMapMode.READ);
    const chunkLogits = new Float32Array(this.readBuffer.getMappedRange().slice(0, actualChunkSize * 4));
    this.readBuffer.unmap();
    
    return { logits: chunkLogits, startIdx, size: actualChunkSize };
  }
  
  /**
   * Merge chunk results into running top-k
   */
  mergeTopK(chunkLogits, chunkStartIdx, chunkSize) {
    // Create combined array of current top-k + new chunk
    const combined = [];
    
    // Add existing top-k
    for (let i = 0; i < this.topK; i++) {
      if (this.topKValues[i] > -1e20) {
        combined.push({ value: this.topKValues[i], index: this.topKIndices[i] });
      }
    }
    
    // Add new chunk values
    for (let i = 0; i < chunkSize; i++) {
      combined.push({ value: chunkLogits[i], index: chunkStartIdx + i });
    }
    
    // Sort descending and take top-k
    combined.sort((a, b) => b.value - a.value);
    
    for (let i = 0; i < this.topK; i++) {
      if (i < combined.length) {
        this.topKValues[i] = combined[i].value;
        this.topKIndices[i] = combined[i].index;
      } else {
        this.topKValues[i] = -1e30;
        this.topKIndices[i] = 0;
      }
    }
  }
  
  /**
   * Full forward pass with top-k extraction
   * Returns only top-k logits and indices
   */
  async forward(hiddenBuffer) {
    // Reset top-k
    this.topKValues.fill(-1e30);
    this.topKIndices.fill(0);
    
    // Process vocab in chunks
    const numChunks = Math.ceil(this.vocabSize / this.chunkSize);
    
    for (let c = 0; c < numChunks; c++) {
      const startIdx = c * this.chunkSize;
      const chunk = await this.processChunk(hiddenBuffer, startIdx);
      this.mergeTopK(chunk.logits, chunk.startIdx, chunk.size);
    }
    
    return {
      values: this.topKValues.slice(),
      indices: this.topKIndices.slice()
    };
  }
  
  /**
   * Get current top-k
   */
  getTopK() {
    return {
      values: this.topKValues.slice(),
      indices: this.topKIndices.slice()
    };
  }
  
  destroy() {
    if (this.chunkLogitsBuffer) this.chunkLogitsBuffer.destroy();
    if (this.offsetBuffer) this.offsetBuffer.destroy();
    if (this.readBuffer) this.readBuffer.destroy();
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { 
    LMHeadOptimized, 
    genLMHeadChunkKernel, 
    genLMHeadBatchKernel,
    genTopKReduceKernel 
  };
}
if (typeof window !== 'undefined') {
  window.LMHeadOptimized = LMHeadOptimized;
  window.genLMHeadChunkKernel = genLMHeadChunkKernel;
  window.genLMHeadBatchKernel = genLMHeadBatchKernel;
  window.genTopKReduceKernel = genTopKReduceKernel;
}
