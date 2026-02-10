/**
 * Draft Model Inference Engine for Qwen2.5-0.5B
 * 
 * Fast FP32 inference for draft token generation in speculative decoding.
 * Optimized for single-token generation speed.
 */

class DraftModelInference {
  constructor(device, config) {
    this.device = device;
    this.config = config;
    
    // Config values
    this.H = config.hidden_size;        // 896
    this.I = config.intermediate_size;  // 4864
    this.numLayers = config.num_hidden_layers;  // 24
    this.numQHeads = config.num_attention_heads; // 14
    this.numKVHeads = config.num_key_value_heads; // 2
    this.headDim = config.head_dim;     // 64
    this.vocabSize = config.vocab_size; // 151936
    this.kvSize = this.numKVHeads * this.headDim;  // 128
    
    this.pipelines = {};
    this.workBuffers = null;
    this.modelBuffers = null;
    this.kvCache = null;
    this.initialized = false;
  }
  
  async init(modelBuffers) {
    this.modelBuffers = modelBuffers;
    
    // Create work buffers
    this.workBuffers = {
      hidden: this.device.createBuffer({
        size: this.H * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        label: 'draft_hidden'
      }),
      normed: this.device.createBuffer({
        size: this.H * 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'draft_normed'
      }),
      residual: this.device.createBuffer({
        size: this.H * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        label: 'draft_residual'
      }),
      q: this.device.createBuffer({
        size: this.H * 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'draft_q'
      }),
      k: this.device.createBuffer({
        size: this.kvSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: 'draft_k'
      }),
      v: this.device.createBuffer({
        size: this.kvSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: 'draft_v'
      }),
      attnOut: this.device.createBuffer({
        size: this.H * 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'draft_attn_out'
      }),
      gate: this.device.createBuffer({
        size: this.I * 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'draft_gate'
      }),
      up: this.device.createBuffer({
        size: this.I * 4,
        usage: GPUBufferUsage.STORAGE,
        label: 'draft_up'
      }),
      position: this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'draft_position'
      }),
      logits: this.device.createBuffer({
        size: this.vocabSize * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: 'draft_logits'
      }),
      argmaxResult: this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        label: 'draft_argmax'
      }),
      argmaxRead: this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        label: 'draft_argmax_read'
      })
    };
    
    // Create KV cache
    await this.initKVCache();
    
    // Create pipelines
    await this.createPipelines();
    
    this.initialized = true;
    console.log('Draft model inference initialized');
  }
  
  async initKVCache(maxSeqLen = 2048) {
    const kvEntrySize = this.numKVHeads * this.headDim * 4;
    
    this.kvCache = {
      maxSeqLen,
      seqLen: 0,
      k: [],
      v: [],
      seqLenBuffer: this.device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'draft_kv_seq_len'
      })
    };
    
    for (let i = 0; i < this.numLayers; i++) {
      this.kvCache.k.push(this.device.createBuffer({
        size: maxSeqLen * kvEntrySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: `draft_k_cache_${i}`
      }));
      this.kvCache.v.push(this.device.createBuffer({
        size: maxSeqLen * kvEntrySize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: `draft_v_cache_${i}`
      }));
    }
  }
  
  resetKVCache() {
    this.kvCache.seqLen = 0;
  }
  
  updateSeqLen(len) {
    this.kvCache.seqLen = len;
    this.device.queue.writeBuffer(this.kvCache.seqLenBuffer, 0, new Uint32Array([len]));
  }
  
  async createPipelines() {
    // RMSNorm
    this.pipelines.rmsNorm = await this.createPipeline(this.genRMSNormShader());
    
    // FP32 MatMul (for FP32 weights)
    this.pipelines.matmul = {};
    for (const [inF, outF] of [[this.H, this.H], [this.H, this.kvSize], [this.H, this.I], [this.I, this.H]]) {
      const key = `${inF}_${outF}`;
      this.pipelines.matmul[key] = await this.createPipeline(this.genMatmulShader(inF, outF));
    }
    
    // RoPE
    this.pipelines.rope = await this.createPipeline(this.genRoPEShader());
    
    // Attention
    this.pipelines.attention = await this.createPipeline(this.genAttentionShader());
    
    // SiLU-Mul
    this.pipelines.siluMul = await this.createPipeline(this.genSiLUMulShader());
    
    // Add
    this.pipelines.add = await this.createPipeline(this.genAddShader());
    
    // Bias Add
    this.pipelines.biasAddH = await this.createPipeline(this.genBiasAddShader(this.H));
    this.pipelines.biasAddKV = await this.createPipeline(this.genBiasAddShader(this.kvSize));
    
    // Embedding lookup (from packed FP16)
    this.pipelines.embedding = await this.createPipeline(this.genEmbeddingShader());
    
    // LM Head (from packed FP16)
    this.pipelines.lmHead = await this.createPipeline(this.genLMHeadShader());
    
    // Argmax
    this.pipelines.argmax = await this.createPipeline(this.genArgmaxShader());
  }
  
  async createPipeline(code) {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
  }
  
  // ============================================
  // Shader Generators
  // ============================================
  
  genRMSNormShader() {
    return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const H = ${this.H}u;
const EPS = ${this.config.rms_norm_eps}f;

var<workgroup> wg_sum: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  
  var local_sum = 0.0;
  for (var i = tid; i < H; i = i + 256u) {
    let v = input[i];
    local_sum = local_sum + v * v;
  }
  wg_sum[tid] = local_sum;
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s = s / 2u) {
    if (tid < s) { wg_sum[tid] = wg_sum[tid] + wg_sum[tid + s]; }
    workgroupBarrier();
  }
  
  let rms = sqrt(wg_sum[0] / f32(H) + EPS);
  let scale = 1.0 / rms;
  
  for (var i = tid; i < H; i = i + 256u) {
    output[i] = input[i] * scale * weight[i];
  }
}`;
  }
  
  genMatmulShader(inFeatures, outFeatures) {
    // FP16-packed matmul: output = input @ weight.T
    // weight is [outFeatures, inFeatures] stored as packed FP16 (2 values per u32)
    const inPacked = inFeatures / 2;
    
    return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;  // Packed FP16
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const IN_F = ${inFeatures}u;
const OUT_F = ${outFeatures}u;
const IN_PACKED = ${inPacked}u;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  if (col >= OUT_F) { return; }
  
  var sum = 0.0;
  let w_base = col * IN_PACKED;
  
  // Process 8 input elements per iteration (4 packed u32s) - unrolled
  let iters = IN_PACKED / 4u;
  for (var i = 0u; i < iters; i = i + 1u) {
    let base = w_base + i * 4u;
    let in_base = i * 8u;
    
    let w0 = unpack2x16float(weight[base + 0u]);
    let w1 = unpack2x16float(weight[base + 1u]);
    let w2 = unpack2x16float(weight[base + 2u]);
    let w3 = unpack2x16float(weight[base + 3u]);
    
    sum += input[in_base + 0u] * w0.x + input[in_base + 1u] * w0.y;
    sum += input[in_base + 2u] * w1.x + input[in_base + 3u] * w1.y;
    sum += input[in_base + 4u] * w2.x + input[in_base + 5u] * w2.y;
    sum += input[in_base + 6u] * w3.x + input[in_base + 7u] * w3.y;
  }
  
  // Handle remainder
  let remainder_start = iters * 4u;
  for (var i = remainder_start; i < IN_PACKED; i = i + 1u) {
    let w = unpack2x16float(weight[w_base + i]);
    let in_idx = i * 2u;
    sum += input[in_idx] * w.x + input[in_idx + 1u] * w.y;
  }
  
  output[col] = sum;
}`;
  }
  
  genRoPEShader() {
    const halfDim = this.headDim / 2;
    return `
@group(0) @binding(0) var<storage, read_write> q: array<f32>;
@group(0) @binding(1) var<storage, read_write> k: array<f32>;
@group(0) @binding(2) var<storage, read> position: array<u32>;

const NUM_Q_HEADS = ${this.numQHeads}u;
const NUM_KV_HEADS = ${this.numKVHeads}u;
const HEAD_DIM = ${this.headDim}u;
const HALF_DIM = ${halfDim}u;
const ROPE_THETA = ${this.config.rope_theta}f;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let pos = f32(position[0]);
  
  // Q heads
  let total_q = NUM_Q_HEADS * HALF_DIM;
  if (idx < total_q) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(pair * 2u) / f32(HEAD_DIM));
    let angle = pos * freq;
    let c = cos(angle);
    let s = sin(angle);
    
    let base = head * HEAD_DIM + pair * 2u;
    let q0 = q[base];
    let q1 = q[base + 1u];
    q[base] = q0 * c - q1 * s;
    q[base + 1u] = q0 * s + q1 * c;
  }
  
  // K heads
  let total_k = NUM_KV_HEADS * HALF_DIM;
  if (idx < total_k) {
    let head = idx / HALF_DIM;
    let pair = idx % HALF_DIM;
    
    let freq = 1.0 / pow(ROPE_THETA, f32(pair * 2u) / f32(HEAD_DIM));
    let angle = pos * freq;
    let c = cos(angle);
    let s = sin(angle);
    
    let base = head * HEAD_DIM + pair * 2u;
    let k0 = k[base];
    let k1 = k[base + 1u];
    k[base] = k0 * c - k1 * s;
    k[base + 1u] = k0 * s + k1 * c;
  }
}`;
  }
  
  genAttentionShader(maxSeqLen = 2048) {
    const headsPerGroup = this.numQHeads / this.numKVHeads;
    const scale = 1.0 / Math.sqrt(this.headDim);
    
    return `
@group(0) @binding(0) var<storage, read> q: array<f32>;
@group(0) @binding(1) var<storage, read> k_cache: array<f32>;
@group(0) @binding(2) var<storage, read> v_cache: array<f32>;
@group(0) @binding(3) var<storage, read> seq_len: array<u32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

const NUM_Q_HEADS = ${this.numQHeads}u;
const NUM_KV_HEADS = ${this.numKVHeads}u;
const HEADS_PER_GROUP = ${headsPerGroup}u;
const HEAD_DIM = ${this.headDim}u;
const SCALE = ${scale}f;
const MAX_SEQ = ${maxSeqLen}u;

var<workgroup> wg_q: array<f32, ${this.headDim}>;
var<workgroup> wg_scores: array<f32, ${maxSeqLen}>;
var<workgroup> wg_reduce: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let q_head = wgid.x;
  let tid = lid.x;
  let kv_head = q_head / HEADS_PER_GROUP;
  let cur_len = seq_len[0];
  
  // Load Q
  let q_base = q_head * HEAD_DIM;
  for (var d = tid; d < HEAD_DIM; d = d + 256u) {
    wg_q[d] = q[q_base + d];
  }
  workgroupBarrier();
  
  // Compute scores
  for (var pos = tid; pos < cur_len; pos = pos + 256u) {
    var score = 0.0;
    let k_base = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM;
    for (var d = 0u; d < HEAD_DIM; d++) {
      score = score + wg_q[d] * k_cache[k_base + d];
    }
    wg_scores[pos] = score * SCALE;
  }
  workgroupBarrier();
  
  // Max
  var local_max = -1e30f;
  for (var pos = tid; pos < cur_len; pos = pos + 256u) {
    local_max = max(local_max, wg_scores[pos]);
  }
  wg_reduce[tid] = local_max;
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s = s / 2u) {
    if (tid < s) { wg_reduce[tid] = max(wg_reduce[tid], wg_reduce[tid + s]); }
    workgroupBarrier();
  }
  let max_val = wg_reduce[0];
  
  // Exp and sum
  var local_sum = 0.0f;
  for (var pos = tid; pos < cur_len; pos = pos + 256u) {
    let e = exp(wg_scores[pos] - max_val);
    wg_scores[pos] = e;
    local_sum = local_sum + e;
  }
  wg_reduce[tid] = local_sum;
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s = s / 2u) {
    if (tid < s) { wg_reduce[tid] = wg_reduce[tid] + wg_reduce[tid + s]; }
    workgroupBarrier();
  }
  let sum_val = wg_reduce[0];
  
  // Normalize
  for (var pos = tid; pos < cur_len; pos = pos + 256u) {
    wg_scores[pos] = wg_scores[pos] / sum_val;
  }
  workgroupBarrier();
  
  // Output
  let out_base = q_head * HEAD_DIM;
  for (var d = tid; d < HEAD_DIM; d = d + 256u) {
    var out_val = 0.0;
    for (var pos = 0u; pos < cur_len; pos++) {
      let v_idx = pos * NUM_KV_HEADS * HEAD_DIM + kv_head * HEAD_DIM + d;
      out_val = out_val + wg_scores[pos] * v_cache[v_idx];
    }
    output[out_base + d] = out_val;
  }
}`;
  }
  
  genSiLUMulShader() {
    return `
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;

const SIZE = ${this.I}u;

fn silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  gate[idx] = silu(gate[idx]) * up[idx];
}`;
  }
  
  genAddShader() {
    return `
@group(0) @binding(0) var<storage, read_write> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;

const SIZE = ${this.H}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  a[idx] = a[idx] + b[idx];
}`;
  }
  
  genBiasAddShader(size) {
    return `
@group(0) @binding(0) var<storage, read_write> x: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;

const SIZE = ${size}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  x[idx] = x[idx] + bias[idx];
}`;
  }
  
  genEmbeddingShader() {
    return `
@group(0) @binding(0) var<storage, read> token_id: array<u32>;
@group(0) @binding(1) var<storage, read> embed_table: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const H = ${this.H}u;
const H_PACKED = ${this.H / 2}u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= H_PACKED) { return; }
  
  let tid = token_id[0];
  let packed = embed_table[tid * H_PACKED + idx];
  let unpacked = unpack2x16float(packed);
  
  output[idx * 2u] = unpacked.x;
  output[idx * 2u + 1u] = unpacked.y;
}`;
  }
  
  genLMHeadShader() {
    // weight is [vocab_size, hidden_size] packed as FP16
    const tokensPerWG = 4;
    return `
@group(0) @binding(0) var<storage, read> hidden: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> logits: array<f32>;

const VOCAB_SIZE = ${this.vocabSize}u;
const H = ${this.H}u;
const H_PACKED = ${this.H / 2}u;
const TOKENS_PER_WG = ${tokensPerWG}u;

var<workgroup> wg_sums: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wgid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>
) {
  let base_vocab = wgid.x * TOKENS_PER_WG;
  let tid = lid.x;
  
  let token_in_wg = tid / 64u;
  let tid_in_token = tid % 64u;
  let vocab_idx = base_vocab + token_in_wg;
  
  var sum = 0.0f;
  
  if (vocab_idx < VOCAB_SIZE) {
    let w_base = vocab_idx * H_PACKED;
    
    for (var i = tid_in_token; i < H_PACKED; i = i + 64u) {
      let packed = weight[w_base + i];
      let w = unpack2x16float(packed);
      sum = sum + hidden[i * 2u] * w.x + hidden[i * 2u + 1u] * w.y;
    }
  }
  
  wg_sums[tid] = sum;
  workgroupBarrier();
  
  // Reduce within token
  let base = token_in_wg * 64u;
  for (var s = 32u; s > 0u; s = s / 2u) {
    if (tid_in_token < s) {
      wg_sums[base + tid_in_token] = wg_sums[base + tid_in_token] + wg_sums[base + tid_in_token + s];
    }
    workgroupBarrier();
  }
  
  if (tid_in_token == 0u && vocab_idx < VOCAB_SIZE) {
    logits[vocab_idx] = wg_sums[base];
  }
}`;
  }
  
  genArgmaxShader() {
    return `
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

const VOCAB_SIZE = ${this.vocabSize}u;

var<workgroup> wg_max: array<f32, 256>;
var<workgroup> wg_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  
  var local_max = -1e30f;
  var local_idx = 0u;
  
  for (var i = tid; i < VOCAB_SIZE; i = i + 256u) {
    let v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
  }
  
  wg_max[tid] = local_max;
  wg_idx[tid] = local_idx;
  workgroupBarrier();
  
  for (var s = 128u; s > 0u; s = s / 2u) {
    if (tid < s) {
      if (wg_max[tid + s] > wg_max[tid]) {
        wg_max[tid] = wg_max[tid + s];
        wg_idx[tid] = wg_idx[tid + s];
      }
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    result[0] = wg_idx[0];
  }
}`;
  }
  
  // ============================================
  // Forward Pass
  // ============================================
  
  async forward(tokenId, position) {
    const encoder = this.device.createCommandEncoder();
    
    // Update position
    this.device.queue.writeBuffer(this.workBuffers.position, 0, new Uint32Array([position]));
    
    // Embedding lookup
    let bg = this.device.createBindGroup({
      layout: this.pipelines.embedding.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workBuffers.position } },  // reuse as token_id holder
        { binding: 1, resource: { buffer: this.modelBuffers.embed_tokens } },
        { binding: 2, resource: { buffer: this.workBuffers.hidden } }
      ]
    });
    // Actually need separate token ID buffer
    const tokenBuf = this.device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true
    });
    new Uint32Array(tokenBuf.getMappedRange()).set([tokenId]);
    tokenBuf.unmap();
    
    bg = this.device.createBindGroup({
      layout: this.pipelines.embedding.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: tokenBuf } },
        { binding: 1, resource: { buffer: this.modelBuffers.embed_tokens } },
        { binding: 2, resource: { buffer: this.workBuffers.hidden } }
      ]
    });
    
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.embedding);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(this.H / 2 / 256));
    pass.end();
    
    // Process layers
    for (let layer = 0; layer < this.numLayers; layer++) {
      const layerBufs = this.modelBuffers.layers[layer];
      
      // Save residual
      encoder.copyBufferToBuffer(this.workBuffers.hidden, 0, this.workBuffers.residual, 0, this.H * 4);
      
      // RMSNorm
      bg = this.device.createBindGroup({
        layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: layerBufs.input_layernorm } },
          { binding: 2, resource: { buffer: this.workBuffers.normed } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.rmsNorm);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();
      
      // Q projection
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.H}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: layerBufs.q_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.q } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.H}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.H / 64));
      pass.end();
      
      // K projection
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.kvSize}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: layerBufs.k_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.k } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.kvSize}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.kvSize / 64));
      pass.end();
      
      // V projection
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.kvSize}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: layerBufs.v_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.v } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.kvSize}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.kvSize / 64));
      pass.end();
      
      // Biases
      if (layerBufs.q_proj.bias) {
        bg = this.device.createBindGroup({
          layout: this.pipelines.biasAddH.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.q } },
            { binding: 1, resource: { buffer: layerBufs.q_proj.bias } }
          ]
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.biasAddH);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.H / 256));
        pass.end();
      }
      
      if (layerBufs.k_proj.bias) {
        bg = this.device.createBindGroup({
          layout: this.pipelines.biasAddKV.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.k } },
            { binding: 1, resource: { buffer: layerBufs.k_proj.bias } }
          ]
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.biasAddKV);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.kvSize / 256));
        pass.end();
      }
      
      if (layerBufs.v_proj.bias) {
        bg = this.device.createBindGroup({
          layout: this.pipelines.biasAddKV.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.workBuffers.v } },
            { binding: 1, resource: { buffer: layerBufs.v_proj.bias } }
          ]
        });
        pass = encoder.beginComputePass();
        pass.setPipeline(this.pipelines.biasAddKV);
        pass.setBindGroup(0, bg);
        pass.dispatchWorkgroups(Math.ceil(this.kvSize / 256));
        pass.end();
      }
      
      // RoPE
      bg = this.device.createBindGroup({
        layout: this.pipelines.rope.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.q } },
          { binding: 1, resource: { buffer: this.workBuffers.k } },
          { binding: 2, resource: { buffer: this.workBuffers.position } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.rope);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.numQHeads * this.headDim / 2 / 256));
      pass.end();
      
      // Update KV cache
      const kvEntrySize = this.kvSize * 4;
      encoder.copyBufferToBuffer(
        this.workBuffers.k, 0,
        this.kvCache.k[layer], position * kvEntrySize,
        kvEntrySize
      );
      encoder.copyBufferToBuffer(
        this.workBuffers.v, 0,
        this.kvCache.v[layer], position * kvEntrySize,
        kvEntrySize
      );
      
      // Attention
      bg = this.device.createBindGroup({
        layout: this.pipelines.attention.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.q } },
          { binding: 1, resource: { buffer: this.kvCache.k[layer] } },
          { binding: 2, resource: { buffer: this.kvCache.v[layer] } },
          { binding: 3, resource: { buffer: this.kvCache.seqLenBuffer } },
          { binding: 4, resource: { buffer: this.workBuffers.attnOut } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.attention);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(this.numQHeads);
      pass.end();
      
      // O projection
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.H}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.attnOut } },
          { binding: 1, resource: { buffer: layerBufs.o_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.hidden } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.H}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.H / 64));
      pass.end();
      
      // Residual add
      bg = this.device.createBindGroup({
        layout: this.pipelines.add.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: this.workBuffers.residual } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.add);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.H / 256));
      pass.end();
      
      // Save residual
      encoder.copyBufferToBuffer(this.workBuffers.hidden, 0, this.workBuffers.residual, 0, this.H * 4);
      
      // Post-attention RMSNorm
      bg = this.device.createBindGroup({
        layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: layerBufs.post_attention_layernorm } },
          { binding: 2, resource: { buffer: this.workBuffers.normed } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.rmsNorm);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(1);
      pass.end();
      
      // MLP gate
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.I}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: layerBufs.gate_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.gate } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.I}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.I / 64));
      pass.end();
      
      // MLP up
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.H}_${this.I}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.normed } },
          { binding: 1, resource: { buffer: layerBufs.up_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.up } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.H}_${this.I}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.I / 64));
      pass.end();
      
      // SiLU-Mul
      bg = this.device.createBindGroup({
        layout: this.pipelines.siluMul.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.gate } },
          { binding: 1, resource: { buffer: this.workBuffers.up } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.siluMul);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.I / 256));
      pass.end();
      
      // MLP down
      bg = this.device.createBindGroup({
        layout: this.pipelines.matmul[`${this.I}_${this.H}`].getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.gate } },
          { binding: 1, resource: { buffer: layerBufs.down_proj.weight } },
          { binding: 2, resource: { buffer: this.workBuffers.hidden } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.matmul[`${this.I}_${this.H}`]);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.H / 64));
      pass.end();
      
      // Final residual add
      bg = this.device.createBindGroup({
        layout: this.pipelines.add.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.workBuffers.hidden } },
          { binding: 1, resource: { buffer: this.workBuffers.residual } }
        ]
      });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.add);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.H / 256));
      pass.end();
    }
    
    // Final RMSNorm
    bg = this.device.createBindGroup({
      layout: this.pipelines.rmsNorm.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workBuffers.hidden } },
        { binding: 1, resource: { buffer: this.modelBuffers.norm } },
        { binding: 2, resource: { buffer: this.workBuffers.normed } }
      ]
    });
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.rmsNorm);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    // LM Head
    const numVocabWGs = Math.ceil(this.vocabSize / 4);
    bg = this.device.createBindGroup({
      layout: this.pipelines.lmHead.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workBuffers.normed } },
        { binding: 1, resource: { buffer: this.modelBuffers.lm_head } },
        { binding: 2, resource: { buffer: this.workBuffers.logits } }
      ]
    });
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.lmHead);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(numVocabWGs);
    pass.end();
    
    this.device.queue.submit([encoder.finish()]);
    tokenBuf.destroy();
    
    return this.workBuffers.logits;
  }
  
  /**
   * Fast greedy sampling - returns argmax token
   */
  async sampleGreedy() {
    const encoder = this.device.createCommandEncoder();
    
    const bg = this.device.createBindGroup({
      layout: this.pipelines.argmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.workBuffers.logits } },
        { binding: 1, resource: { buffer: this.workBuffers.argmaxResult } }
      ]
    });
    
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.argmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    encoder.copyBufferToBuffer(
      this.workBuffers.argmaxResult, 0,
      this.workBuffers.argmaxRead, 0, 4
    );
    
    this.device.queue.submit([encoder.finish()]);
    
    await this.workBuffers.argmaxRead.mapAsync(GPUMapMode.READ);
    const tokenId = new Uint32Array(this.workBuffers.argmaxRead.getMappedRange())[0];
    this.workBuffers.argmaxRead.unmap();
    
    return tokenId;
  }
  
  destroy() {
    if (this.workBuffers) {
      for (const buf of Object.values(this.workBuffers)) {
        if (buf && buf.destroy) buf.destroy();
      }
    }
    if (this.kvCache) {
      for (const buf of this.kvCache.k) buf.destroy();
      for (const buf of this.kvCache.v) buf.destroy();
      this.kvCache.seqLenBuffer.destroy();
    }
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DraftModelInference };
}
if (typeof window !== 'undefined') {
  window.DraftModelInference = DraftModelInference;
}
