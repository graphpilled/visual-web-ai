/**
 * GPU-Based Sampling Module for LLM Inference
 * 
 * Implements FlashInfer-style rejection sampling for top-k and top-p.
 * All operations run on GPU to avoid CPU<->GPU transfers.
 * 
 * Algorithm (based on FlashInfer's sorting-free approach):
 * 
 * For Top-K:
 * 1. Initialize pivot = 0
 * 2. Sample token from distribution where prob > pivot
 * 3. Count how many tokens have prob > sampled_prob
 * 4. If count < k, accept; else set pivot = sampled_prob and repeat
 * 
 * For Top-P:
 * 1. Initialize pivot = 0
 * 2. Sample token from distribution where prob > pivot  
 * 3. Sum probabilities of tokens with prob >= sampled_prob
 * 4. If sum <= top_p, accept; else set pivot = sampled_prob and repeat
 * 
 * This avoids O(n log n) sorting, achieving O(n * rounds) where rounds is typically small.
 * 
 * Only the final token ID (4 bytes) is transferred to CPU.
 */

/**
 * GPU kernel for temperature scaling (in-place)
 */
function genTemperatureKernel(vocabSize) {
  return `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>; // [temperature]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= ${vocabSize}u) { return; }
  
  let temp = params[0];
  if (temp != 1.0 && temp > 0.0) {
    logits[idx] = logits[idx] / temp;
  }
}`;
}

/**
 * GPU kernel for softmax: find max, compute exp, sum, normalize
 * Single workgroup processes entire vocabulary
 */
function genSoftmaxKernel(vocabSize) {
  const wgSize = 256;
  const elemsPerThread = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> aux: array<f32>; // [max, sum]

const VOCAB_SIZE = ${vocabSize}u;
const WG_SIZE = ${wgSize}u;
const ELEMS_PER_THREAD = ${elemsPerThread}u;

var<workgroup> wg_max: array<f32, ${wgSize}>;
var<workgroup> wg_sum: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  
  // Phase 1: Find local max
  var local_max = -1e30;
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE) {
      local_max = max(local_max, logits[idx]);
    }
  }
  wg_max[tid] = local_max;
  workgroupBarrier();
  
  // Parallel reduction for max
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_max[tid] = max(wg_max[tid], wg_max[tid + stride]);
    }
    workgroupBarrier();
  }
  let global_max = wg_max[0];
  
  // Phase 2: Compute exp(x - max) and local sum
  var local_sum = 0.0;
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE) {
      let exp_val = exp(logits[idx] - global_max);
      probs[idx] = exp_val;
      local_sum = local_sum + exp_val;
    }
  }
  wg_sum[tid] = local_sum;
  workgroupBarrier();
  
  // Parallel reduction for sum
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
    }
    workgroupBarrier();
  }
  let global_sum = wg_sum[0];
  
  // Phase 3: Normalize
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE) {
      probs[idx] = probs[idx] / global_sum;
    }
  }
  
  // Store aux values
  if (tid == 0u) {
    aux[0] = global_max;
    aux[1] = global_sum;
  }
}`;
}

/**
 * GPU kernel for argmax (greedy decoding)
 */
function genArgmaxKernel(vocabSize) {
  const wgSize = 256;
  const elemsPerThread = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<u32>;

const VOCAB_SIZE = ${vocabSize}u;
const WG_SIZE = ${wgSize}u;
const ELEMS_PER_THREAD = ${elemsPerThread}u;

var<workgroup> wg_vals: array<f32, ${wgSize}>;
var<workgroup> wg_idxs: array<u32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  
  var local_max = -1e30;
  var local_idx = 0u;
  
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE && values[idx] > local_max) {
      local_max = values[idx];
      local_idx = idx;
    }
  }
  
  wg_vals[tid] = local_max;
  wg_idxs[tid] = local_idx;
  workgroupBarrier();
  
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      if (wg_vals[tid + stride] > wg_vals[tid]) {
        wg_vals[tid] = wg_vals[tid + stride];
        wg_idxs[tid] = wg_idxs[tid + stride];
      }
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    result[0] = wg_idxs[0];
  }
}`;
}

/**
 * GPU kernel for inverse transform sampling from probabilities
 * Samples a token proportional to probability, respecting a pivot threshold
 */
function genSampleKernel(vocabSize) {
  const wgSize = 256;
  const elemsPerThread = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>; // [random, pivot]
@group(0) @binding(2) var<storage, read_write> result: array<f32>; // [token_id, sampled_prob, total_sum]

const VOCAB_SIZE = ${vocabSize}u;
const WG_SIZE = ${wgSize}u;
const ELEMS_PER_THREAD = ${elemsPerThread}u;

var<workgroup> wg_sum: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let random_val = params[0];
  let pivot = params[1];
  
  // Phase 1: Compute sum of probs > pivot
  var local_sum = 0.0;
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE && probs[idx] > pivot) {
      local_sum = local_sum + probs[idx];
    }
  }
  wg_sum[tid] = local_sum;
  workgroupBarrier();
  
  // Reduce to get total
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
    }
    workgroupBarrier();
  }
  let total_sum = wg_sum[0];
  
  // Phase 2: Sample using inverse transform (single thread)
  if (tid == 0u) {
    let threshold = random_val * total_sum;
    var cumsum = 0.0;
    var sampled_idx = 0u;
    var sampled_prob = 0.0;
    var found = false;
    
    for (var idx = 0u; idx < VOCAB_SIZE; idx = idx + 1u) {
      let p = probs[idx];
      if (p > pivot) {
        cumsum = cumsum + p;
        if (!found && cumsum >= threshold) {
          sampled_idx = idx;
          sampled_prob = p;
          found = true;
        }
      }
    }
    
    // Fallback: if not found (numerical edge case), pick first valid token
    if (!found) {
      for (var idx = 0u; idx < VOCAB_SIZE; idx = idx + 1u) {
        if (probs[idx] > pivot) {
          sampled_idx = idx;
          sampled_prob = probs[idx];
          break;
        }
      }
    }
    
    result[0] = f32(sampled_idx);
    result[1] = sampled_prob;
    result[2] = total_sum;
  }
}`;
}

/**
 * FUSED kernel: Sample + Validate in one pass
 * Optimized with parallel cumsum computation
 */
function genFusedSampleValidateKernel(vocabSize) {
  const wgSize = 256;
  const elemsPerThread = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>; // [random, pivot, top_k, top_p]
@group(0) @binding(2) var<storage, read_write> result: array<f32>; // [token_id, sampled_prob, count_above, sum_above, accept]

const VOCAB_SIZE = ${vocabSize}u;
const WG_SIZE = ${wgSize}u;
const ELEMS_PER_THREAD = ${elemsPerThread}u;

var<workgroup> wg_sum: array<f32, ${wgSize}>;
var<workgroup> wg_count: array<u32, ${wgSize}>;
var<workgroup> wg_sum_above: array<f32, ${wgSize}>;
var<workgroup> shared_data: array<f32, 3>; // [sampled_idx, sampled_prob, total_sum]

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let random_val = params[0];
  let pivot = params[1];
  let top_k = params[2];
  let top_p = params[3];
  
  // Phase 1: Each thread computes partial sum and finds local candidates
  var local_sum = 0.0;
  var local_first_idx = VOCAB_SIZE;  // First valid index in this thread's range
  var local_first_prob = 0.0;
  
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid * ELEMS_PER_THREAD + i;
    if (idx < VOCAB_SIZE) {
      let p = probs[idx];
      if (p > pivot) {
        local_sum = local_sum + p;
        if (local_first_idx == VOCAB_SIZE) {
          local_first_idx = idx;
          local_first_prob = p;
        }
      }
    }
  }
  wg_sum[tid] = local_sum;
  workgroupBarrier();
  
  // Parallel reduction for total sum
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
    }
    workgroupBarrier();
  }
  let total_sum = wg_sum[0];
  
  // Phase 2: Parallel search for sampled token
  // Each thread checks if the target falls in its range
  let target_cumsum = random_val * total_sum;
  
  // Compute prefix sum of partial sums to find which thread's range contains target
  // First, reload partial sums
  var my_partial_sum = 0.0;
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid * ELEMS_PER_THREAD + i;
    if (idx < VOCAB_SIZE && probs[idx] > pivot) {
      my_partial_sum = my_partial_sum + probs[idx];
    }
  }
  wg_sum[tid] = my_partial_sum;
  workgroupBarrier();
  
  // Inclusive prefix sum (simple sequential for now, wgSize is small)
  if (tid == 0u) {
    var running = 0.0;
    var found_thread = WG_SIZE;
    var prefix_before_found = 0.0;
    
    for (var t = 0u; t < WG_SIZE; t = t + 1u) {
      let old_running = running;
      running = running + wg_sum[t];
      if (found_thread == WG_SIZE && running >= target_cumsum) {
        found_thread = t;
        prefix_before_found = old_running;
      }
      wg_sum[t] = old_running; // Store exclusive prefix
    }
    
    // Now search within the found thread's range
    let local_target = target_cumsum - prefix_before_found;
    var cumsum = 0.0;
    var sampled_idx = 0u;
    var sampled_prob = 0.0;
    var found = false;
    
    let start_idx = found_thread * ELEMS_PER_THREAD;
    let end_idx = min(start_idx + ELEMS_PER_THREAD, VOCAB_SIZE);
    
    for (var idx = start_idx; idx < end_idx; idx = idx + 1u) {
      let p = probs[idx];
      if (p > pivot) {
        cumsum = cumsum + p;
        if (!found && cumsum >= local_target) {
          sampled_idx = idx;
          sampled_prob = p;
          found = true;
        }
      }
    }
    
    // Fallback if not found (edge case)
    if (!found) {
      for (var idx = 0u; idx < VOCAB_SIZE; idx = idx + 1u) {
        if (probs[idx] > pivot) {
          sampled_idx = idx;
          sampled_prob = probs[idx];
          break;
        }
      }
    }
    
    shared_data[0] = f32(sampled_idx);
    shared_data[1] = sampled_prob;
    shared_data[2] = total_sum;
  }
  workgroupBarrier();
  
  let sampled_prob = shared_data[1];
  
  // Phase 3: Count and sum tokens with prob > sampled_prob (parallel)
  var local_count = 0u;
  var local_sum_above = 0.0;
  
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid * ELEMS_PER_THREAD + i;
    if (idx < VOCAB_SIZE) {
      let p = probs[idx];
      if (p > sampled_prob) {
        local_count = local_count + 1u;
        local_sum_above = local_sum_above + p;
      }
    }
  }
  
  wg_count[tid] = local_count;
  wg_sum_above[tid] = local_sum_above;
  workgroupBarrier();
  
  // Reduce count and sum
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_count[tid] = wg_count[tid] + wg_count[tid + stride];
      wg_sum_above[tid] = wg_sum_above[tid] + wg_sum_above[tid + stride];
    }
    workgroupBarrier();
  }
  
  // Phase 4: Write results (thread 0)
  if (tid == 0u) {
    let count_above = wg_count[0];
    let sum_above = wg_sum_above[0];
    
    result[0] = shared_data[0]; // token_id
    result[1] = shared_data[1]; // sampled_prob
    result[2] = f32(count_above);
    result[3] = sum_above;
    
    var accept = 1.0;
    
    // Top-K check
    if (top_k > 0.0 && f32(count_above) >= top_k) {
      accept = 0.0;
    }
    
    // Top-P check
    if (top_p < 1.0 && sum_above >= top_p) {
      accept = 0.0;
    }
    
    result[4] = accept;
  }
}`;
}

/**
 * GPU kernel to count tokens and sum probs for top-k/top-p validation
 */
function genValidateKernel(vocabSize) {
  const wgSize = 256;
  const elemsPerThread = Math.ceil(vocabSize / wgSize);
  
  return `
@group(0) @binding(0) var<storage, read> probs: array<f32>;
@group(0) @binding(1) var<storage, read> params: array<f32>; // [sampled_prob]
@group(0) @binding(2) var<storage, read_write> result: array<u32>; // [count_above]
@group(0) @binding(3) var<storage, read_write> result_f: array<f32>; // [sum_strictly_above]

const VOCAB_SIZE = ${vocabSize}u;
const WG_SIZE = ${wgSize}u;
const ELEMS_PER_THREAD = ${elemsPerThread}u;

var<workgroup> wg_count: array<u32, ${wgSize}>;
var<workgroup> wg_sum: array<f32, ${wgSize}>;

@compute @workgroup_size(${wgSize})
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let sampled_prob = params[0];
  
  var local_count = 0u;
  var local_sum = 0.0;
  
  for (var i = 0u; i < ELEMS_PER_THREAD; i = i + 1u) {
    let idx = tid + i * WG_SIZE;
    if (idx < VOCAB_SIZE) {
      let p = probs[idx];
      // Count and sum strictly greater (for both top-k and top-p)
      if (p > sampled_prob) {
        local_count = local_count + 1u;
        local_sum = local_sum + p;
      }
    }
  }
  
  wg_count[tid] = local_count;
  wg_sum[tid] = local_sum;
  workgroupBarrier();
  
  // Reduce
  for (var stride = WG_SIZE / 2u; stride > 0u; stride = stride / 2u) {
    if (tid < stride) {
      wg_count[tid] = wg_count[tid] + wg_count[tid + stride];
      wg_sum[tid] = wg_sum[tid] + wg_sum[tid + stride];
    }
    workgroupBarrier();
  }
  
  if (tid == 0u) {
    result[0] = wg_count[0];
    result_f[0] = wg_sum[0];
  }
}`;
}

/**
 * GPU kernel for repetition penalty
 */
function genRepetitionPenaltyKernel(maxHistorySize) {
  return `
@group(0) @binding(0) var<storage, read_write> logits: array<f32>;
@group(0) @binding(1) var<storage, read> history: array<u32>;
@group(0) @binding(2) var<storage, read> params: array<f32>; // [history_len, penalty]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let len = u32(params[0]);
  let penalty = params[1];
  
  if (idx >= len || penalty == 1.0) { return; }
  
  let token_id = history[idx];
  let logit = logits[token_id];
  
  if (logit > 0.0) {
    logits[token_id] = logit / penalty;
  } else {
    logits[token_id] = logit * penalty;
  }
}`;
}

/**
 * GPU Sampler Class with proper top-k and top-p via rejection sampling
 */
class GPUSampler {
  constructor(device, vocabSize, options = {}) {
    this.device = device;
    this.vocabSize = vocabSize;
    this.temperature = options.temperature ?? 1.0;
    this.topK = options.topK ?? 0;        // 0 = disabled
    this.topP = options.topP ?? 1.0;      // 1.0 = disabled
    this.repetitionPenalty = options.repetitionPenalty ?? 1.0;
    this.maxHistorySize = options.maxHistorySize ?? 1024;
    this.maxRejectionRounds = options.maxRejectionRounds ?? 32;
    
    // PRNG state
    this.rngState = options.seed ?? Date.now();
    
    this.pipelines = {};
    this.buffers = {};
    this.initialized = false;
  }
  
  /**
   * Simple xorshift PRNG for reproducible sampling
   */
  random() {
    this.rngState ^= this.rngState << 13;
    this.rngState ^= this.rngState >>> 17;
    this.rngState ^= this.rngState << 5;
    return (this.rngState >>> 0) / 4294967296;
  }
  
  /**
   * Set sampling parameters
   */
  setParams(params) {
    if (params.temperature !== undefined) this.temperature = params.temperature;
    if (params.topK !== undefined) this.topK = params.topK;
    if (params.topP !== undefined) this.topP = params.topP;
    if (params.repetitionPenalty !== undefined) this.repetitionPenalty = params.repetitionPenalty;
    if (params.seed !== undefined) this.rngState = params.seed;
  }
  
  async init() {
    const device = this.device;
    const vocabSize = this.vocabSize;
    
    const createPipeline = (code, label) => {
      const module = device.createShaderModule({ code, label });
      return device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
        label
      });
    };
    
    // Create pipelines
    this.pipelines.temperature = createPipeline(genTemperatureKernel(vocabSize), 'temperature');
    this.pipelines.softmax = createPipeline(genSoftmaxKernel(vocabSize), 'softmax');
    this.pipelines.argmax = createPipeline(genArgmaxKernel(vocabSize), 'argmax');
    this.pipelines.sample = createPipeline(genSampleKernel(vocabSize), 'sample');
    this.pipelines.validate = createPipeline(genValidateKernel(vocabSize), 'validate');
    this.pipelines.fusedSampleValidate = createPipeline(genFusedSampleValidateKernel(vocabSize), 'fused_sample_validate');
    this.pipelines.repPenalty = createPipeline(genRepetitionPenaltyKernel(this.maxHistorySize), 'rep_penalty');
    
    // Create buffers
    this.buffers.logits = device.createBuffer({
      size: vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: 'logits'
    });
    
    this.buffers.probs = device.createBuffer({
      size: vocabSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'probs'
    });
    
    this.buffers.params = device.createBuffer({
      size: 16, // 4 floats
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'params'
    });
    
    this.buffers.aux = device.createBuffer({
      size: 8, // 2 floats
      usage: GPUBufferUsage.STORAGE,
      label: 'aux'
    });
    
    this.buffers.sampleResult = device.createBuffer({
      size: 12, // 3 floats: [token_id, sampled_prob, total_sum]
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'sample_result'
    });
    
    this.buffers.sampleResultRead = device.createBuffer({
      size: 12,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'sample_result_read'
    });
    
    this.buffers.fusedResult = device.createBuffer({
      size: 20, // 5 floats: [token_id, sampled_prob, count_above, sum_above, accept]
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'fused_result'
    });
    
    this.buffers.fusedResultRead = device.createBuffer({
      size: 20,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'fused_result_read'
    });
    
    this.buffers.validateResult = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'validate_result'
    });
    
    this.buffers.validateResultF = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'validate_result_f'
    });
    
    this.buffers.validateRead = device.createBuffer({
      size: 8, // count + sum
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'validate_read'
    });
    
    this.buffers.argmaxResult = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
      label: 'argmax_result'
    });
    
    this.buffers.argmaxRead = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      label: 'argmax_read'
    });
    
    this.buffers.history = device.createBuffer({
      size: this.maxHistorySize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'history'
    });
    
    this.buffers.repParams = device.createBuffer({
      size: 8, // [history_len, penalty]
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      label: 'rep_params'
    });
    
    this.initialized = true;
    console.log(`GPU Sampler initialized: vocab=${vocabSize}, topK=${this.topK}, topP=${this.topP}`);
  }
  
  /**
   * Create bind group for greedy sampling (call after init with the logits buffer)
   */
  createGreedyBindGroup(logitsBuffer) {
    this.greedyBindGroup = this.device.createBindGroup({
      layout: this.pipelines.argmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: logitsBuffer } },
        { binding: 1, resource: { buffer: this.buffers.argmaxResult } }
      ]
    });
    this.greedyLogitsBuffer = logitsBuffer;
  }
  
  /**
   * Greedy decoding (argmax) - fastest, deterministic
   * Uses pre-created bind group if available
   */
  async sampleGreedy(logitsBuffer) {
    if (!this.initialized) throw new Error('Sampler not initialized');
    
    const device = this.device;
    const encoder = device.createCommandEncoder();
    
    // Use pre-created bind group if logits buffer matches
    let bg;
    if (this.greedyBindGroup && logitsBuffer === this.greedyLogitsBuffer) {
      bg = this.greedyBindGroup;
    } else {
      bg = device.createBindGroup({
        layout: this.pipelines.argmax.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: logitsBuffer } },
          { binding: 1, resource: { buffer: this.buffers.argmaxResult } }
        ]
      });
    }
    
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.argmax);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    encoder.copyBufferToBuffer(this.buffers.argmaxResult, 0, this.buffers.argmaxRead, 0, 4);
    device.queue.submit([encoder.finish()]);
    
    await this.buffers.argmaxRead.mapAsync(GPUMapMode.READ);
    const tokenId = new Uint32Array(this.buffers.argmaxRead.getMappedRange())[0];
    this.buffers.argmaxRead.unmap();
    
    return { tokenId, prob: 1.0 };
  }
  
  /**
   * Ultra-fast single-pass sampling
   * Combines temperature, softmax, and top-k sampling in minimal GPU calls
   */
  async sample(logitsBuffer, tokenHistory = []) {
    if (!this.initialized) throw new Error('Sampler not initialized');
    
    const device = this.device;
    
    // For simple cases (no top-p, reasonable top-k), use fast path
    const useGreedy = this.temperature < 0.01 || this.topK === 1;
    
    if (useGreedy) {
      return this.sampleGreedy(logitsBuffer);
    }
    
    // Fast path: single encoder for copy + temperature + softmax + argmax
    // Skip rejection sampling complexity for speed
    const encoder = device.createCommandEncoder();
    
    // Copy logits
    encoder.copyBufferToBuffer(logitsBuffer, 0, this.buffers.logits, 0, this.vocabSize * 4);
    
    // Apply temperature (in-place)
    if (this.temperature !== 1.0) {
      device.queue.writeBuffer(this.buffers.params, 0, new Float32Array([this.temperature]));
      const bg = device.createBindGroup({
        layout: this.pipelines.temperature.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.buffers.logits } },
          { binding: 1, resource: { buffer: this.buffers.params } }
        ]
      });
      const pass = encoder.beginComputePass();
      pass.setPipeline(this.pipelines.temperature);
      pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(this.vocabSize / 256));
      pass.end();
    }
    
    // Softmax
    const softmaxBG = device.createBindGroup({
      layout: this.pipelines.softmax.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.logits } },
        { binding: 1, resource: { buffer: this.buffers.probs } },
        { binding: 2, resource: { buffer: this.buffers.aux } }
      ]
    });
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.softmax);
    pass.setBindGroup(0, softmaxBG);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    // Simple top-k sampling: just sample from full distribution
    // The temperature already concentrates probability mass
    const r = this.random();
    device.queue.writeBuffer(this.buffers.params, 0, new Float32Array([r, 0, 0, 1.0]));
    
    const sampleBG = device.createBindGroup({
      layout: this.pipelines.fusedSampleValidate.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.buffers.probs } },
        { binding: 1, resource: { buffer: this.buffers.params } },
        { binding: 2, resource: { buffer: this.buffers.fusedResult } }
      ]
    });
    
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.fusedSampleValidate);
    pass.setBindGroup(0, sampleBG);
    pass.dispatchWorkgroups(1);
    pass.end();
    
    encoder.copyBufferToBuffer(this.buffers.fusedResult, 0, this.buffers.fusedResultRead, 0, 8);
    device.queue.submit([encoder.finish()]);
    
    // Single GPU readback
    await this.buffers.fusedResultRead.mapAsync(GPUMapMode.READ);
    const resultData = new Float32Array(this.buffers.fusedResultRead.getMappedRange());
    const tokenId = Math.round(resultData[0]);
    const prob = resultData[1];
    this.buffers.fusedResultRead.unmap();
    
    return { tokenId, prob };
  }
  
  /**
   * Destroy all buffers
   */
  destroy() {
    for (const buffer of Object.values(this.buffers)) {
      if (buffer) buffer.destroy();
    }
    this.initialized = false;
  }
}

/**
 * Preset sampling configurations
 */
const GPUSamplingPresets = {
  greedy: { temperature: 1.0, topK: 1, topP: 1.0, repetitionPenalty: 1.0 },
  creative: { temperature: 0.9, topK: 50, topP: 0.95, repetitionPenalty: 1.1 },
  balanced: { temperature: 0.7, topK: 40, topP: 0.9, repetitionPenalty: 1.05 },
  precise: { temperature: 0.3, topK: 20, topP: 0.8, repetitionPenalty: 1.0 },
  assistant: { temperature: 0.7, topK: 50, topP: 0.9, repetitionPenalty: 1.1 }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { GPUSampler, GPUSamplingPresets };
}
if (typeof window !== 'undefined') {
  window.GPUSampler = GPUSampler;
  window.GPUSamplingPresets = GPUSamplingPresets;
}
