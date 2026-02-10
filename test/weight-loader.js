/**
 * GPTQ Weight Loader for Qwen2.5-7B-Instruct-GPTQ-Int4
 * 
 * Version 3: Memory-efficient - keeps embeddings in F16, streams to GPU
 */

const QWEN2_7B_CONFIG = {
  vocab_size: 152064,
  hidden_size: 3584,
  intermediate_size: 18944,
  num_hidden_layers: 28,
  num_attention_heads: 28,
  num_key_value_heads: 4,
  head_dim: 128,
  rms_norm_eps: 1e-6,
  rope_theta: 1000000.0,
  max_position_embeddings: 32768,
  bits: 4,
  group_size: 128,
};

/**
 * Parse safetensors header only
 */
async function parseSafetensorsFile(file) {
  const headerLenBuffer = await file.slice(0, 8).arrayBuffer();
  const headerLen = Number(new DataView(headerLenBuffer).getBigUint64(0, true));
  
  const headerBuffer = await file.slice(8, 8 + headerLen).arrayBuffer();
  const headerStr = new TextDecoder().decode(new Uint8Array(headerBuffer));
  const header = JSON.parse(headerStr);
  
  const dataOffset = 8 + headerLen;
  
  return {
    header,
    dataOffset,
    file,
    
    // Read tensor as raw bytes (no conversion)
    async readTensorRaw(name) {
      const tensorInfo = header[name];
      if (!tensorInfo) return null;
      
      const { dtype, shape, data_offsets } = tensorInfo;
      const [start, end] = data_offsets;
      const tensorBuffer = await file.slice(dataOffset + start, dataOffset + end).arrayBuffer();
      
      return { buffer: tensorBuffer, shape, dtype };
    },
    
    // Read and convert tensor (for small tensors only)
    async readTensor(name) {
      const tensorInfo = header[name];
      if (!tensorInfo) return null;
      
      const { dtype, shape, data_offsets } = tensorInfo;
      const [start, end] = data_offsets;
      const tensorBuffer = await file.slice(dataOffset + start, dataOffset + end).arrayBuffer();
      
      let data;
      switch (dtype) {
        case 'F32':
          data = new Float32Array(tensorBuffer);
          break;
        case 'F16':
          // Keep as Uint16Array - convert on GPU or when needed
          data = new Uint16Array(tensorBuffer);
          data._isF16 = true;
          break;
        case 'I32':
          data = new Int32Array(tensorBuffer);
          break;
        case 'U32':
          data = new Uint32Array(tensorBuffer);
          break;
        case 'BF16':
          data = new Uint16Array(tensorBuffer);
          data._isBF16 = true;
          break;
        default:
          throw new Error(`Unsupported dtype: ${dtype}`);
      }
      
      return { data, shape, dtype };
    },
    
    hasTensor(name) {
      return name in header && name !== '__metadata__';
    },
    
    getTensorInfo(name) {
      return header[name] || null;
    },
    
    getTensorNames() {
      return Object.keys(header).filter(k => k !== '__metadata__');
    }
  };
}

/**
 * Convert Float16 to Float32 - for small arrays only
 */
function float16ToFloat32(uint16Array) {
  const float32 = new Float32Array(uint16Array.length);
  for (let i = 0; i < uint16Array.length; i++) {
    const h = uint16Array[i];
    const sign = (h >> 15) & 0x1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;
    
    let f;
    if (exponent === 0) {
      f = mantissa * Math.pow(2, -24);
    } else if (exponent === 31) {
      f = mantissa ? NaN : Infinity;
    } else {
      f = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
    }
    
    float32[i] = sign ? -f : f;
  }
  return float32;
}

/**
 * Convert GPTQ format to column-major INT4
 */
function convertGPTQToColumnMajor(qweightData, qzerosData, scalesData, inFeatures, outFeatures, groupSize) {
  const numGroups = Math.ceil(inFeatures / groupSize);
  const packedPerGroup = groupSize / 8;
  
  console.log(`convertGPTQ: in=${inFeatures}, out=${outFeatures}, groups=${numGroups}, packedPerGroup=${packedPerGroup}`);
  console.log(`  qweight len=${qweightData.length}, scales len=${scalesData.length}`);
  console.log(`  output packed size=${numGroups * packedPerGroup * outFeatures}, scales size=${numGroups * outFeatures}`);
  
  const packedOut = new Uint32Array(numGroups * packedPerGroup * outFeatures);
  const scalesOut = new Float32Array(numGroups * outFeatures);
  
  // Convert scales from F16 if needed
  let scales = scalesData;
  if (scalesData._isF16) {
    scales = float16ToFloat32(scalesData);
  }
  
  for (let col = 0; col < outFeatures; col++) {
    for (let g = 0; g < numGroups; g++) {
      const scale = scales[g * outFeatures + col];
      scalesOut[g * outFeatures + col] = scale;
      
      let zero = 8;
      if (qzerosData) {
        const zerosPackedCols = Math.ceil(outFeatures / 8);
        const zeroPackedCol = Math.floor(col / 8);
        const zeroSubIdx = col % 8;
        if (g * zerosPackedCols + zeroPackedCol < qzerosData.length) {
          const zeroPacked = qzerosData[g * zerosPackedCols + zeroPackedCol];
          zero = ((zeroPacked >> (zeroSubIdx * 4)) & 0xF) + 1;
        }
      }
      
      for (let p = 0; p < packedPerGroup; p++) {
        let packed = 0;
        
        for (let sub = 0; sub < 8; sub++) {
          const row = g * groupSize + p * 8 + sub;
          
          if (row < inFeatures) {
            const packedRow = Math.floor(row / 8);
            const packedSub = row % 8;
            const origPacked = qweightData[packedRow * outFeatures + col];
            const origVal = (origPacked >> (packedSub * 4)) & 0xF;
            
            let newVal = origVal - zero + 8;
            newVal = Math.max(0, Math.min(15, newVal));
            
            packed |= (newVal << (sub * 4));
          }
        }
        
        packedOut[(g * packedPerGroup + p) * outFeatures + col] = packed;
      }
    }
  }
  
  return { packed: packedOut, scales: scalesOut };
}

function getLayerWeightNames(layerIdx) {
  const prefix = `model.layers.${layerIdx}`;
  return {
    q_proj: `${prefix}.self_attn.q_proj`,
    k_proj: `${prefix}.self_attn.k_proj`,
    v_proj: `${prefix}.self_attn.v_proj`,
    o_proj: `${prefix}.self_attn.o_proj`,
    gate_proj: `${prefix}.mlp.gate_proj`,
    up_proj: `${prefix}.mlp.up_proj`,
    down_proj: `${prefix}.mlp.down_proj`,
    input_layernorm: `${prefix}.input_layernorm.weight`,
    post_attention_layernorm: `${prefix}.post_attention_layernorm.weight`,
  };
}

/**
 * Main loader class - V3 with direct GPU streaming
 */
class Qwen2WeightLoader {
  constructor(config = QWEN2_7B_CONFIG) {
    this.config = config;
    this.safetensorsFiles = [];
    this.onProgress = null;
    
    // Store minimal metadata, not full weights
    this.layerInfo = [];
    this.embedInfo = null;
    this.normInfo = null;
    this.lmHeadInfo = null;
  }
  
  setProgressCallback(callback) {
    this.onProgress = callback;
  }
  
  _progress(msg) {
    if (this.onProgress) this.onProgress(msg);
  }
  
  async loadFiles(files) {
    const fileArray = Array.isArray(files) ? files : [files];
    
    for (const file of fileArray) {
      this._progress(`Parsing header: ${file.name}...`);
      const st = await parseSafetensorsFile(file);
      this.safetensorsFiles.push(st);
      this._progress(`Found ${st.getTensorNames().length} tensors in ${file.name}`);
    }
  }
  
  _findTensorFile(name) {
    for (const st of this.safetensorsFiles) {
      if (st.hasTensor(name)) return st;
    }
    return null;
  }
  
  /**
   * Load GPTQ linear layer
   */
  async _loadGPTQLinear(baseName) {
    this._progress(`  Loading ${baseName}...`);
    
    // Each tensor might be in a different file - search for each separately
    const qweightSt = this._findTensorFile(`${baseName}.qweight`);
    if (!qweightSt) throw new Error(`Cannot find ${baseName}.qweight`);
    
    const qzerosSt = this._findTensorFile(`${baseName}.qzeros`);
    const scalesSt = this._findTensorFile(`${baseName}.scales`);
    const biasSt = this._findTensorFile(`${baseName}.bias`);
    
    if (!scalesSt) throw new Error(`Cannot find ${baseName}.scales`);
    
    const qweight = await qweightSt.readTensor(`${baseName}.qweight`);
    const qzeros = qzerosSt ? await qzerosSt.readTensor(`${baseName}.qzeros`) : null;
    const scales = await scalesSt.readTensor(`${baseName}.scales`);
    const bias = biasSt ? await biasSt.readTensor(`${baseName}.bias`) : null;
    
    const [packedIn, outFeatures] = qweight.shape;
    const inFeatures = packedIn * 8;
    
    const { packed, scales: scalesOut } = convertGPTQToColumnMajor(
      new Uint32Array(qweight.data.buffer, qweight.data.byteOffset, qweight.data.length),
      qzeros ? new Uint32Array(qzeros.data.buffer, qzeros.data.byteOffset, qzeros.data.length) : null,
      scales.data,
      inFeatures,
      outFeatures,
      this.config.group_size
    );
    
    return {
      packed,
      scales: scalesOut,
      bias: bias ? (bias.data._isF16 ? float16ToFloat32(bias.data) : bias.data) : null,
      shape: [inFeatures, outFeatures],
    };
  }
  
  /**
   * Load norm layer (small, always convert to F32)
   */
  async _loadNorm(name) {
    const st = this._findTensorFile(name);
    if (!st) throw new Error(`Cannot find ${name}`);
    
    this._progress(`  Loading ${name}...`);
    const tensor = await st.readTensor(name);
    
    // Convert F16 to F32 for norms (they're small)
    if (tensor.data._isF16) {
      tensor.data = float16ToFloat32(tensor.data);
    }
    
    return tensor;
  }
  
  /**
   * Analyze model structure without loading weights
   */
  async analyzeModel() {
    this._progress('Analyzing model structure...');
    
    // Check embedding
    const embedSt = this._findTensorFile('model.embed_tokens.weight');
    if (embedSt) {
      const info = embedSt.getTensorInfo('model.embed_tokens.weight');
      this.embedInfo = {
        shape: info.shape,
        dtype: info.dtype,
        sizeMB: (info.data_offsets[1] - info.data_offsets[0]) / 1024 / 1024
      };
      this._progress(`  Embedding: [${info.shape.join(', ')}] ${info.dtype} (${this.embedInfo.sizeMB.toFixed(1)} MB)`);
    }
    
    // Check layers
    for (let i = 0; i < this.config.num_hidden_layers; i++) {
      const names = getLayerWeightNames(i);
      const st = this._findTensorFile(`${names.q_proj}.qweight`);
      if (st) {
        const qInfo = st.getTensorInfo(`${names.q_proj}.qweight`);
        this.layerInfo.push({
          index: i,
          qProjShape: qInfo.shape,
        });
      }
    }
    this._progress(`  Found ${this.layerInfo.length} transformer layers`);
    
    // Check final layers
    const normSt = this._findTensorFile('model.norm.weight');
    if (normSt) {
      this.normInfo = normSt.getTensorInfo('model.norm.weight');
    }
    
    const lmHeadSt = this._findTensorFile('lm_head.weight');
    if (lmHeadSt) {
      this.lmHeadInfo = lmHeadSt.getTensorInfo('lm_head.weight');
      this._progress(`  LM Head: [${this.lmHeadInfo.shape.join(', ')}] ${this.lmHeadInfo.dtype}`);
    }
    
    return {
      numLayers: this.layerInfo.length,
      hasEmbed: !!this.embedInfo,
      hasLmHead: !!this.lmHeadInfo,
    };
  }
  
  /**
   * Load weights and create GPU buffers in one pass (memory efficient)
   * This streams weights directly to GPU without keeping CPU copies
   */
  async loadToGPU(device) {
    this._progress('Loading weights directly to GPU...\n');
    
    const buffers = { layers: [] };
    
    const createBuffer = (data, name) => {
      const buffer = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: name,
      });
      device.queue.writeBuffer(buffer, 0, data);
      return buffer;
    };
    
    // 1. Load embedding - keep as F16, we'll convert on GPU during lookup
    this._progress('Loading embedding (keeping as F16)...');
    const embedSt = this._findTensorFile('model.embed_tokens.weight');
    if (embedSt) {
      const embedRaw = await embedSt.readTensorRaw('model.embed_tokens.weight');
      // Store as F16 (half the memory)
      buffers.embed_tokens = device.createBuffer({
        size: embedRaw.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'embed_tokens_f16',
      });
      device.queue.writeBuffer(buffers.embed_tokens, 0, embedRaw.buffer);
      buffers.embed_tokens_dtype = embedRaw.dtype;
      buffers.embed_tokens_shape = embedRaw.shape;
      this._progress(`  ✅ Embedding: ${(embedRaw.buffer.byteLength / 1024 / 1024).toFixed(1)} MB (${embedRaw.dtype})`);
    }
    
    // 2. Load layers one at a time
    this._progress(`\nLoading ${this.config.num_hidden_layers} transformer layers...`);
    
    for (let i = 0; i < this.config.num_hidden_layers; i++) {
      this._progress(`\nLayer ${i}/${this.config.num_hidden_layers - 1}:`);
      
      const names = getLayerWeightNames(i);
      const layerBuffers = {};
      
      if (!this._findTensorFile(`${names.q_proj}.qweight`)) {
        this._progress(`  ⚠️ Layer ${i} not found, skipping`);
        continue;
      }
      
      // Attention projections
      for (const proj of ['q_proj', 'k_proj', 'v_proj', 'o_proj']) {
        const weights = await this._loadGPTQLinear(names[proj]);
        layerBuffers[proj] = {
          packed: createBuffer(weights.packed, `layer${i}.${proj}.packed`),
          scales: createBuffer(weights.scales, `layer${i}.${proj}.scales`),
          shape: weights.shape,
        };
        if (weights.bias) {
          layerBuffers[proj].bias = createBuffer(weights.bias, `layer${i}.${proj}.bias`);
        }
        // Let GC clean up
        weights.packed = null;
        weights.scales = null;
      }
      
      // FFN projections
      for (const proj of ['gate_proj', 'up_proj', 'down_proj']) {
        const weights = await this._loadGPTQLinear(names[proj]);
        layerBuffers[proj] = {
          packed: createBuffer(weights.packed, `layer${i}.${proj}.packed`),
          scales: createBuffer(weights.scales, `layer${i}.${proj}.scales`),
          shape: weights.shape,
        };
        weights.packed = null;
        weights.scales = null;
      }
      
      // Norms
      const inputLn = await this._loadNorm(names.input_layernorm);
      layerBuffers.input_layernorm = createBuffer(inputLn.data, `layer${i}.input_ln`);
      
      const postAttnLn = await this._loadNorm(names.post_attention_layernorm);
      layerBuffers.post_attention_layernorm = createBuffer(postAttnLn.data, `layer${i}.post_attn_ln`);
      
      buffers.layers.push(layerBuffers);
      
      // Force GC opportunity
      await new Promise(r => setTimeout(r, 10));
    }
    
    // 3. Final norm
    this._progress('\nLoading final layers...');
    const norm = await this._loadNorm('model.norm.weight');
    buffers.norm = createBuffer(norm.data, 'final_norm');
    
    // 4. LM head - also keep as F16 (it's huge: 152064 × 3584)
    const lmHeadSt = this._findTensorFile('lm_head.weight');
    if (lmHeadSt) {
      this._progress('  Loading lm_head (keeping as F16)...');
      const lmHeadRaw = await lmHeadSt.readTensorRaw('lm_head.weight');
      buffers.lm_head = device.createBuffer({
        size: lmHeadRaw.buffer.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        label: 'lm_head_f16',
      });
      device.queue.writeBuffer(buffers.lm_head, 0, lmHeadRaw.buffer);
      buffers.lm_head_dtype = lmHeadRaw.dtype;
      buffers.lm_head_shape = lmHeadRaw.shape;
      this._progress(`  ✅ LM Head: ${(lmHeadRaw.buffer.byteLength / 1024 / 1024).toFixed(1)} MB (${lmHeadRaw.dtype})`);
    }
    
    // Calculate total GPU memory
    let totalSize = 0;
    const countBuffer = (buf) => {
      if (!buf) return;
      if (buf.size !== undefined) totalSize += buf.size;
    };
    
    countBuffer(buffers.embed_tokens);
    for (const layer of buffers.layers) {
      for (const key of Object.keys(layer)) {
        const item = layer[key];
        if (item.packed) {
          countBuffer(item.packed);
          countBuffer(item.scales);
          countBuffer(item.bias);
        } else {
          countBuffer(item);
        }
      }
    }
    countBuffer(buffers.norm);
    countBuffer(buffers.lm_head);
    
    this._progress(`\n✅ Total GPU memory: ${(totalSize / 1024 / 1024).toFixed(1)} MB`);
    this._progress(`✅ ${buffers.layers.length} layers loaded`);
    
    return buffers;
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Qwen2WeightLoader, QWEN2_7B_CONFIG };
}
if (typeof window !== 'undefined') {
  window.Qwen2WeightLoader = Qwen2WeightLoader;
  window.QWEN2_7B_CONFIG = QWEN2_7B_CONFIG;
}
