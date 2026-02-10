/**
 * LoRA Loader for Qwen2.5-7B WebGPU Inference
 * 
 * Loads LoRA adapters and merges them with the base INT4 model.
 * 
 * LoRA formula: W_effective = W_base + (B @ A) * scale
 * where scale = lora_alpha / r
 * 
 * For INT4 base models, we have two options:
 * 1. Merge at runtime (compute LoRA contribution separately)
 * 2. Dequantize → merge → requantize (expensive, one-time)
 * 
 * This implementation uses runtime merging for flexibility.
 */

class LoRALoader {
  constructor(device) {
    this.device = device;
    this.config = null;
    this.weights = null;
    this.loaded = false;
  }
  
  /**
   * Load LoRA config from adapter_config.json
   */
  async loadConfig(configJson) {
    this.config = JSON.parse(configJson);
    
    // Extract key parameters
    this.rank = this.config.r;
    this.alpha = this.config.lora_alpha;
    this.scale = this.alpha / this.rank;
    this.targetModules = this.config.target_modules;
    
    console.log('LoRA Config loaded:');
    console.log(`  Rank: ${this.rank}`);
    console.log(`  Alpha: ${this.alpha}`);
    console.log(`  Scale: ${this.scale}`);
    console.log(`  Target modules: ${this.targetModules.join(', ')}`);
    
    return this.config;
  }
  
  /**
   * Parse safetensors file
   */
  parseSafetensors(buffer) {
    const view = new DataView(buffer);
    const headerSize = Number(view.getBigUint64(0, true));
    const headerJson = new TextDecoder().decode(
      new Uint8Array(buffer, 8, headerSize)
    );
    const header = JSON.parse(headerJson);
    
    const dataOffset = 8 + headerSize;
    const tensors = {};
    
    for (const [name, info] of Object.entries(header)) {
      if (name === '__metadata__') continue;
      
      const dtype = info.dtype;
      const shape = info.shape;
      const [start, end] = info.data_offsets;
      
      const tensorData = new Uint8Array(buffer, dataOffset + start, end - start);
      tensors[name] = { dtype, shape, data: tensorData };
    }
    
    return tensors;
  }
  
  /**
   * Convert tensor data to Float32Array
   */
  tensorToFloat32(tensor) {
    const { dtype, data } = tensor;
    
    if (dtype === 'F32') {
      return new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
    } else if (dtype === 'F16') {
      const f16View = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
      const f32 = new Float32Array(f16View.length);
      for (let i = 0; i < f16View.length; i++) {
        f32[i] = this.fp16ToFp32(f16View[i]);
      }
      return f32;
    } else if (dtype === 'BF16') {
      const bf16View = new Uint16Array(data.buffer, data.byteOffset, data.byteLength / 2);
      const f32 = new Float32Array(bf16View.length);
      for (let i = 0; i < bf16View.length; i++) {
        // BF16: just shift left by 16 bits
        const u32 = bf16View[i] << 16;
        const f32View = new Float32Array(new Uint32Array([u32]).buffer);
        f32[i] = f32View[0];
      }
      return f32;
    }
    
    throw new Error(`Unsupported dtype: ${dtype}`);
  }
  
  fp16ToFp32(h) {
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1F;
    const mant = h & 0x03FF;
    
    let result;
    if (exp === 0) {
      if (mant === 0) {
        result = sign;
      } else {
        let e = -1;
        let m = mant;
        while ((m & 0x0400) === 0) {
          m <<= 1;
          e--;
        }
        result = sign | ((127 - 15 + e) << 23) | ((m & 0x03FF) << 13);
      }
    } else if (exp === 31) {
      result = sign | 0x7F800000 | (mant << 13);
    } else {
      result = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    
    return new Float32Array(new Uint32Array([result]).buffer)[0];
  }
  
  /**
   * Load LoRA weights from safetensors file
   */
  async loadWeights(file, progressCallback) {
    const log = progressCallback || console.log;
    
    log('Loading LoRA weights...');
    
    let buffer;
    if (file instanceof File) {
      buffer = await file.arrayBuffer();
    } else if (file instanceof ArrayBuffer) {
      buffer = file;
    } else {
      throw new Error('Expected File or ArrayBuffer');
    }
    
    const tensors = this.parseSafetensors(buffer);
    log(`Parsed ${Object.keys(tensors).length} tensors`);
    
    // Organize weights by layer and module
    this.weights = {
      layers: []
    };
    
    // Initialize layer structure
    for (let i = 0; i < 28; i++) {
      this.weights.layers.push({});
    }
    
    // Parse tensor names like:
    // base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
    // base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight
    
    for (const [name, tensor] of Object.entries(tensors)) {
      // Extract layer number and module name
      const layerMatch = name.match(/layers\.(\d+)\./);
      if (!layerMatch) continue;
      
      const layerIdx = parseInt(layerMatch[1]);
      if (layerIdx >= 28) continue;
      
      // Extract module name (q_proj, v_proj, etc.)
      let moduleName = null;
      for (const target of this.targetModules) {
        if (name.includes(`.${target}.`)) {
          moduleName = target;
          break;
        }
      }
      if (!moduleName) continue;
      
      // Extract A or B
      const isA = name.includes('lora_A');
      const isB = name.includes('lora_B');
      if (!isA && !isB) continue;
      
      // Convert to Float32
      const f32 = this.tensorToFloat32(tensor);
      
      // Store in structure
      if (!this.weights.layers[layerIdx][moduleName]) {
        this.weights.layers[layerIdx][moduleName] = { A: null, B: null };
      }
      
      if (isA) {
        this.weights.layers[layerIdx][moduleName].A = {
          data: f32,
          shape: tensor.shape  // [r, in_features] or [in_features, r] depending on convention
        };
      } else {
        this.weights.layers[layerIdx][moduleName].B = {
          data: f32,
          shape: tensor.shape  // [out_features, r] or [r, out_features]
        };
      }
    }
    
    // Count loaded modules
    let moduleCount = 0;
    for (let i = 0; i < 28; i++) {
      moduleCount += Object.keys(this.weights.layers[i]).length;
    }
    log(`Loaded LoRA for ${moduleCount} modules across 28 layers`);
    
    this.loaded = true;
    return this.weights;
  }
  
  /**
   * Create GPU buffers for LoRA weights
   * Returns structure ready for runtime application
   */
  createGPUBuffers() {
    if (!this.loaded) {
      throw new Error('LoRA weights not loaded');
    }
    
    const gpuWeights = {
      layers: [],
      scale: this.scale
    };
    
    for (let i = 0; i < 28; i++) {
      const layerWeights = {};
      
      for (const [moduleName, lora] of Object.entries(this.weights.layers[i])) {
        if (!lora.A || !lora.B) continue;
        
        // Create GPU buffers for A and B matrices
        const bufferA = this.device.createBuffer({
          size: lora.A.data.byteLength,
          usage: GPUBufferUsage.STORAGE,
          mappedAtCreation: true
        });
        new Float32Array(bufferA.getMappedRange()).set(lora.A.data);
        bufferA.unmap();
        
        const bufferB = this.device.createBuffer({
          size: lora.B.data.byteLength,
          usage: GPUBufferUsage.STORAGE,
          mappedAtCreation: true
        });
        new Float32Array(bufferB.getMappedRange()).set(lora.B.data);
        bufferB.unmap();
        
        layerWeights[moduleName] = {
          A: bufferA,
          B: bufferB,
          shapeA: lora.A.shape,
          shapeB: lora.B.shape
        };
      }
      
      gpuWeights.layers.push(layerWeights);
    }
    
    console.log('Created GPU buffers for LoRA weights');
    return gpuWeights;
  }
  
  /**
   * Get info about the loaded LoRA
   */
  getInfo() {
    return {
      rank: this.rank,
      alpha: this.alpha,
      scale: this.scale,
      targetModules: this.targetModules,
      loaded: this.loaded
    };
  }
}


/**
 * LoRA Runtime Applicator
 * 
 * Applies LoRA at runtime during forward pass.
 * For each linear layer: output = base_output + (input @ A.T @ B.T) * scale
 */
class LoRAApplicator {
  constructor(device, loraWeights, config) {
    this.device = device;
    this.loraWeights = loraWeights;
    this.scale = loraWeights.scale;
    this.config = config;
    
    this.pipelines = {};
    this.initialized = false;
  }
  
  async init() {
    // Create pipelines for LoRA computation
    // We need: input @ A.T @ B.T * scale
    // A is [r, in_features], B is [out_features, r]
    // So: input[1, in] @ A.T[in, r] = temp[1, r]
    //     temp[1, r] @ B.T[r, out] = output[1, out]
    
    // Create pipeline for first matmul (input @ A.T)
    this.pipelines.matmulA = await this.createMatmulPipeline('A');
    
    // Create pipeline for second matmul (temp @ B.T) with scale and add
    this.pipelines.matmulBAdd = await this.createMatmulAddPipeline();
    
    // Create intermediate buffer for rank-sized vector
    const rank = 32; // From config
    this.tempBuffer = this.device.createBuffer({
      size: rank * 4,
      usage: GPUBufferUsage.STORAGE,
      label: 'lora_temp'
    });
    
    this.initialized = true;
    console.log('LoRA applicator initialized');
  }
  
  async createMatmulPipeline(name) {
    // Generic matmul: output = input @ weight.T
    const code = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> dims: vec3<u32>;  // [inFeatures, outFeatures, 0]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let inF = dims.x;
  let outF = dims.y;
  
  if (col >= outF) { return; }
  
  var sum = 0.0;
  // weight is [outFeatures, inFeatures], we want row 'col'
  let w_base = col * inF;
  
  for (var i = 0u; i < inF; i = i + 1u) {
    sum = sum + input[i] * weight[w_base + i];
  }
  
  output[col] = sum;
}`;
    
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
  }
  
  async createMatmulAddPipeline() {
    // Matmul with scale and add to existing output
    // output += (input @ weight.T) * scale
    const code = `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: vec4<f32>;  // [inFeatures, outFeatures, scale, 0]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let col = gid.x;
  let inF = u32(params.x);
  let outF = u32(params.y);
  let scale = params.z;
  
  if (col >= outF) { return; }
  
  var sum = 0.0;
  let w_base = col * inF;
  
  for (var i = 0u; i < inF; i = i + 1u) {
    sum = sum + input[i] * weight[w_base + i];
  }
  
  output[col] = output[col] + sum * scale;
}`;
    
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
  }
  
  /**
   * Apply LoRA to a layer's output
   * Call this AFTER the base INT4 matmul for each target module
   * 
   * @param encoder - GPU command encoder
   * @param layerIdx - Layer index (0-27)
   * @param moduleName - Module name (q_proj, v_proj, etc.)
   * @param inputBuffer - Input to the linear layer
   * @param outputBuffer - Output buffer (will be modified in-place)
   * @param inFeatures - Input feature size
   * @param outFeatures - Output feature size
   */
  applyLoRA(encoder, layerIdx, moduleName, inputBuffer, outputBuffer, inFeatures, outFeatures) {
    const loraLayer = this.loraWeights.layers[layerIdx];
    if (!loraLayer || !loraLayer[moduleName]) {
      return; // No LoRA for this module
    }
    
    const lora = loraLayer[moduleName];
    const rank = 32; // From config
    
    // Step 1: temp = input @ A.T
    // A is [r, in_features], A.T is [in_features, r]
    // But stored as [r, in_features], so we read rows
    
    // Create dims uniform buffer
    const dimsBuffer1 = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true
    });
    new Uint32Array(dimsBuffer1.getMappedRange()).set([inFeatures, rank, 0, 0]);
    dimsBuffer1.unmap();
    
    const bg1 = this.device.createBindGroup({
      layout: this.pipelines.matmulA.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: inputBuffer } },
        { binding: 1, resource: { buffer: lora.A } },
        { binding: 2, resource: { buffer: this.tempBuffer } },
        { binding: 3, resource: { buffer: dimsBuffer1 } }
      ]
    });
    
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.matmulA);
    pass.setBindGroup(0, bg1);
    pass.dispatchWorkgroups(1); // rank is small, one workgroup enough
    pass.end();
    
    // Step 2: output += (temp @ B.T) * scale
    // B is [out_features, r], B.T is [r, out_features]
    
    const paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM,
      mappedAtCreation: true
    });
    new Float32Array(paramsBuffer.getMappedRange()).set([rank, outFeatures, this.scale, 0]);
    paramsBuffer.unmap();
    
    const bg2 = this.device.createBindGroup({
      layout: this.pipelines.matmulBAdd.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.tempBuffer } },
        { binding: 1, resource: { buffer: lora.B } },
        { binding: 2, resource: { buffer: outputBuffer } },
        { binding: 3, resource: { buffer: paramsBuffer } }
      ]
    });
    
    pass = encoder.beginComputePass();
    pass.setPipeline(this.pipelines.matmulBAdd);
    pass.setBindGroup(0, bg2);
    pass.dispatchWorkgroups(Math.ceil(outFeatures / 64));
    pass.end();
    
    // Clean up uniform buffers (in real impl, pre-create these)
    // dimsBuffer1.destroy();
    // paramsBuffer.destroy();
  }
}


// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LoRALoader, LoRAApplicator };
}
if (typeof window !== 'undefined') {
  window.LoRALoader = LoRALoader;
  window.LoRAApplicator = LoRAApplicator;
}
