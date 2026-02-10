/**
 * KV Cache for Autoregressive Generation
 * 
 * For Qwen2.5-7B:
 *   - 28 layers
 *   - 4 KV heads per layer
 *   - 128 head_dim
 *   - KV cache shape per layer: [max_seq_len, num_kv_heads, head_dim]
 * 
 * Memory per layer: max_seq_len * 4 * 128 * 4 bytes (K) + same for V
 * For 2048 tokens: 2048 * 4 * 128 * 4 * 2 = 8 MB per layer
 * Total for 28 layers: 224 MB
 */

class KVCache {
  constructor(device, config) {
    this.device = device;
    this.numLayers = config.numLayers || 28;
    this.numKVHeads = config.numKVHeads || 4;
    this.headDim = config.headDim || 128;
    this.maxSeqLen = config.maxSeqLen || 2048;
    
    // Current sequence length (number of tokens processed)
    this.seqLen = 0;
    
    // GPU buffers for K and V cache per layer
    this.kCacheBuffers = [];
    this.vCacheBuffers = [];
    
    // Size per layer in bytes
    this.cacheSize = this.maxSeqLen * this.numKVHeads * this.headDim * 4;
    
    this._initBuffers();
  }
  
  _initBuffers() {
    for (let layer = 0; layer < this.numLayers; layer++) {
      // K cache buffer
      const kBuffer = this.device.createBuffer({
        size: this.cacheSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: `k_cache_layer_${layer}`
      });
      
      // V cache buffer
      const vBuffer = this.device.createBuffer({
        size: this.cacheSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: `v_cache_layer_${layer}`
      });
      
      this.kCacheBuffers.push(kBuffer);
      this.vCacheBuffers.push(vBuffer);
    }
    
    // Sequence length buffer (shared across layers)
    this.seqLenBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'seq_len'
    });
    
    console.log(`KV Cache initialized: ${this.numLayers} layers, ${(this.cacheSize * 2 * this.numLayers / 1024 / 1024).toFixed(1)} MB total`);
  }
  
  /**
   * Reset cache for new generation
   */
  reset() {
    this.seqLen = 0;
    // No need to clear GPU buffers - we just track seqLen
  }
  
  /**
   * Get current sequence length
   */
  getSeqLen() {
    return this.seqLen;
  }
  
  /**
   * Update sequence length on GPU
   */
  updateSeqLenBuffer() {
    this.device.queue.writeBuffer(
      this.seqLenBuffer,
      0,
      new Uint32Array([this.seqLen])
    );
  }
  
  /**
   * Append new K/V vectors for a layer
   * 
   * @param {number} layer - Layer index (0-27)
   * @param {GPUBuffer} newK - New K vector [num_kv_heads, head_dim]
   * @param {GPUBuffer} newV - New V vector [num_kv_heads, head_dim]
   */
  append(layer, newK, newV) {
    const kvSize = this.numKVHeads * this.headDim * 4; // bytes
    const offset = this.seqLen * this.numKVHeads * this.headDim * 4;
    
    // Copy new K to cache at current position
    const commandEncoder = this.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(newK, 0, this.kCacheBuffers[layer], offset, kvSize);
    commandEncoder.copyBufferToBuffer(newV, 0, this.vCacheBuffers[layer], offset, kvSize);
    this.device.queue.submit([commandEncoder.finish()]);
  }
  
  /**
   * Append K/V from CPU arrays
   */
  appendFromCPU(layer, kArray, vArray) {
    const kvSize = this.numKVHeads * this.headDim * 4;
    const offset = this.seqLen * this.numKVHeads * this.headDim * 4;
    
    this.device.queue.writeBuffer(this.kCacheBuffers[layer], offset, kArray);
    this.device.queue.writeBuffer(this.vCacheBuffers[layer], offset, vArray);
  }
  
  /**
   * Increment sequence length after processing a token
   */
  incrementSeqLen() {
    this.seqLen++;
    if (this.seqLen > this.maxSeqLen) {
      throw new Error(`Sequence length ${this.seqLen} exceeds max ${this.maxSeqLen}`);
    }
    this.updateSeqLenBuffer();
  }
  
  /**
   * Get buffers for a specific layer (for attention computation)
   */
  getLayerBuffers(layer) {
    return {
      kCache: this.kCacheBuffers[layer],
      vCache: this.vCacheBuffers[layer],
      seqLen: this.seqLenBuffer
    };
  }
  
  /**
   * Get memory usage in bytes
   */
  getMemoryUsage() {
    return this.cacheSize * 2 * this.numLayers;
  }
  
  /**
   * Destroy all buffers
   */
  destroy() {
    for (let i = 0; i < this.numLayers; i++) {
      this.kCacheBuffers[i].destroy();
      this.vCacheBuffers[i].destroy();
    }
    this.seqLenBuffer.destroy();
  }
}

/**
 * WGSL Kernel to append K/V to cache
 * This is an alternative to buffer copies - useful if K/V need transformation
 */
function genKVCacheAppendKernel(numKVHeads, headDim) {
  const size = numKVHeads * headDim;
  
  return `
@group(0) @binding(0) var<storage, read> new_k: array<f32>;
@group(0) @binding(1) var<storage, read> new_v: array<f32>;
@group(0) @binding(2) var<storage, read_write> k_cache: array<f32>;
@group(0) @binding(3) var<storage, read_write> v_cache: array<f32>;
@group(0) @binding(4) var<storage, read> seq_pos: array<u32>;

const NUM_KV_HEADS = ${numKVHeads}u;
const HEAD_DIM = ${headDim}u;
const KV_SIZE = ${size}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= KV_SIZE) { return; }
  
  let pos = seq_pos[0];
  let cache_idx = pos * KV_SIZE + idx;
  
  k_cache[cache_idx] = new_k[idx];
  v_cache[cache_idx] = new_v[idx];
}`;
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { KVCache, genKVCacheAppendKernel };
}
if (typeof window !== 'undefined') {
  window.KVCache = KVCache;
  window.genKVCacheAppendKernel = genKVCacheAppendKernel;
}
