/**
 * Embedding Lookup Kernel (Optimized)
 */

function genEmbeddingLookupSingleShader(vocabSize, hiddenSize) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read> token_id: array<u32>;
@group(0) @binding(1) var<storage, read> embedding_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const HIDDEN_SIZE = ${hiddenSize}u;
const HIDDEN_SIZE_PACKED = ${hiddenSize / 2}u;
const WG_SIZE = ${wgSize}u;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tok_id = token_id[0];
  
  for (var d = tid; d < HIDDEN_SIZE_PACKED; d = d + WG_SIZE) {
    let emb_idx = tok_id * HIDDEN_SIZE_PACKED + d;
    let packed = embedding_table[emb_idx];
    let unpacked = unpack2x16float(packed);
    output[d * 2u] = unpacked.x;
    output[d * 2u + 1u] = unpacked.y;
  }
}`;
}

class EmbeddingLookup {
  constructor(device, vocabSize, hiddenSize) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.hiddenSize = hiddenSize;
    this.embeddingBuffer = null;
    this.singlePipeline = null;
    // Pre-allocated buffers
    this.tokenIdBuffer = null;
    this.bindGroup = null;
  }
  
  setEmbeddingBuffer(buffer) {
    this.embeddingBuffer = buffer;
  }
  
  async init() {
    const singleShader = genEmbeddingLookupSingleShader(this.vocabSize, this.hiddenSize);
    const singleModule = this.device.createShaderModule({ code: singleShader });
    this.singlePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: singleModule, entryPoint: 'main' }
    });
    
    // Pre-allocate token ID buffer (reused every call)
    this.tokenIdBuffer = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'embed_token_id'
    });
  }
  
  /**
   * Create bind group for a specific output buffer (call once during init)
   */
  createBindGroup(outputBuffer) {
    this.bindGroup = this.device.createBindGroup({
      layout: this.singlePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tokenIdBuffer } },
        { binding: 1, resource: { buffer: this.embeddingBuffer } },
        { binding: 2, resource: { buffer: outputBuffer } }
      ]
    });
  }
  
  /**
   * Look up embedding - can be integrated into existing encoder
   * @param {number} tokenId 
   * @param {GPUCommandEncoder} encoder - optional, if null creates own submission
   */
  lookupSingle(tokenId, encoder = null) {
    // Write token ID
    this.device.queue.writeBuffer(this.tokenIdBuffer, 0, new Uint32Array([tokenId]));
    
    const ownEncoder = encoder === null;
    if (ownEncoder) {
      encoder = this.device.createCommandEncoder();
    }
    
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.singlePipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    if (ownEncoder) {
      this.device.queue.submit([encoder.finish()]);
    }
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { EmbeddingLookup, genEmbeddingLookupSingleShader };
}
if (typeof window !== 'undefined') {
  window.EmbeddingLookup = EmbeddingLookup;
}
