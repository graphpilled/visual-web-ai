/**
 * Draft Model Loader for Qwen2.5-0.5B
 * 
 * Loads the smaller 0.5B model in FP16 format for fast draft generation
 * in speculative decoding.
 * 
 * Model size: ~1GB in FP16 (vs 5.4GB for quantized 7B)
 */

class DraftModelLoader {
  constructor(device, baseUrl) {
    this.device = device;
    this.baseUrl = baseUrl || 'http://localhost:8000/Qwen2.5-0.5B';
    
    // Draft model config
    this.config = {
      vocab_size: 151936,
      hidden_size: 896,
      intermediate_size: 4864,
      num_hidden_layers: 24,
      num_attention_heads: 14,
      num_key_value_heads: 2,
      head_dim: 64,  // 896 / 14
      rms_norm_eps: 1e-6,
      rope_theta: 1000000.0,
      tie_word_embeddings: true
    };
    
    this.buffers = null;
    this.loaded = false;
  }
  
  getConfig() {
    return this.config;
  }
  
  /**
   * Load safetensors index to understand file structure
   */
  async loadIndex() {
    try {
      const response = await fetch(`${this.baseUrl}/model.safetensors.index.json`);
      if (!response.ok) {
        // Single file model, no index - this is expected for small models
        return null;
      }
      return await response.json();
    } catch (err) {
      // Network error, CORS, or file not found - assume single file model
      console.log('No index.json found, assuming single-file model');
      return null;
    }
  }
  
  /**
   * Load a safetensors file and parse its structure
   */
  async loadSafetensors(filename, progressCallback) {
    const log = progressCallback || console.log;
    const url = `${this.baseUrl}/${filename}`;
    
    log(`Fetching ${filename}...`);
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load ${filename}: ${response.status}`);
    }
    
    // Get content length for progress
    const contentLength = response.headers.get('content-length');
    const totalBytes = contentLength ? parseInt(contentLength) : 0;
    
    if (totalBytes > 0) {
      log(`Downloading ${(totalBytes / 1024 / 1024).toFixed(1)} MB...`);
    }
    
    // Read with progress
    const reader = response.body.getReader();
    const chunks = [];
    let receivedBytes = 0;
    let lastProgress = 0;
    
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      
      chunks.push(value);
      receivedBytes += value.length;
      
      // Report progress every 10%
      if (totalBytes > 0) {
        const progress = Math.floor(receivedBytes / totalBytes * 100);
        if (progress >= lastProgress + 10) {
          log(`Download: ${progress}% (${(receivedBytes / 1024 / 1024).toFixed(1)} MB)`);
          lastProgress = progress;
          // Yield to prevent freezing
          await new Promise(r => setTimeout(r, 0));
        }
      }
    }
    
    log('Download complete, parsing...');
    await new Promise(r => setTimeout(r, 0)); // Yield
    
    // Combine chunks into single ArrayBuffer
    const buffer = new ArrayBuffer(receivedBytes);
    const bufferView = new Uint8Array(buffer);
    let offset = 0;
    for (const chunk of chunks) {
      bufferView.set(chunk, offset);
      offset += chunk.length;
    }
    
    return await this.parseSafetensors(buffer, log);
  }
  
  /**
   * Parse safetensors binary format (non-blocking)
   */
  async parseSafetensors(buffer, progressCallback) {
    const log = progressCallback || console.log;
    
    const view = new DataView(buffer);
    const headerSize = Number(view.getBigUint64(0, true));
    const headerJson = new TextDecoder().decode(
      new Uint8Array(buffer, 8, headerSize)
    );
    const header = JSON.parse(headerJson);
    
    const dataOffset = 8 + headerSize;
    const tensors = {};
    
    const entries = Object.entries(header).filter(([name]) => name !== '__metadata__');
    const total = entries.length;
    let processed = 0;
    
    for (const [name, info] of entries) {
      const dtype = info.dtype;
      const shape = info.shape;
      const [start, end] = info.data_offsets;
      
      const tensorData = new Uint8Array(buffer, dataOffset + start, end - start);
      tensors[name] = { dtype, shape, data: tensorData };
      
      processed++;
      
      // Yield every 10 tensors to prevent freezing
      if (processed % 10 === 0) {
        log(`Parsing tensors: ${processed}/${total}`);
        await new Promise(r => setTimeout(r, 0));
      }
    }
    
    log(`Parsed ${total} tensors`);
    return tensors;
  }
  
  /**
   * Convert tensor to Float32Array
   */
  tensorToFloat32(tensor) {
    if (tensor.dtype === 'F32') {
      return new Float32Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 4);
    } else if (tensor.dtype === 'F16' || tensor.dtype === 'BF16') {
      // Convert from FP16/BF16 to FP32
      const f16View = new Uint16Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength / 2);
      const f32 = new Float32Array(f16View.length);
      
      if (tensor.dtype === 'BF16') {
        // BF16: just shift left by 16 bits
        for (let i = 0; i < f16View.length; i++) {
          const u32 = f16View[i] << 16;
          const f32View = new Float32Array(new Uint32Array([u32]).buffer);
          f32[i] = f32View[0];
        }
      } else {
        // FP16 conversion
        for (let i = 0; i < f16View.length; i++) {
          f32[i] = this.fp16ToFp32(f16View[i]);
        }
      }
      return f32;
    }
    throw new Error(`Unsupported dtype: ${tensor.dtype}`);
  }
  
  /**
   * Convert FP16 to FP32
   */
  fp16ToFp32(h) {
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1F;
    const mant = h & 0x03FF;
    
    let result;
    if (exp === 0) {
      if (mant === 0) {
        result = sign;
      } else {
        // Subnormal
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
    
    const f32arr = new Float32Array(new Uint32Array([result]).buffer);
    return f32arr[0];
  }
  
  /**
   * Create GPU buffer from tensor - store as packed FP16 for efficiency
   * Two FP16 values packed into one u32, aligned for vec4 access
   */
  createBuffer(tensor, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const f32 = this.tensorToFloat32(tensor);
    
    // Pack as FP16 pairs (2 values per u32)
    // Align to 16 bytes (4 u32s) for vec4 access
    const packedSize = Math.ceil(f32.length / 2);
    const alignedSize = Math.ceil(packedSize / 4) * 4;  // Align to vec4
    
    const buffer = this.device.createBuffer({
      size: alignedSize * 4,
      usage,
      mappedAtCreation: true
    });
    
    const u32View = new Uint32Array(buffer.getMappedRange());
    for (let i = 0; i < packedSize; i++) {
      const v0 = f32[i * 2] || 0;
      const v1 = f32[i * 2 + 1] || 0;
      const h0 = this.fp32ToFp16(v0);
      const h1 = this.fp32ToFp16(v1);
      u32View[i] = h0 | (h1 << 16);
    }
    // Pad with zeros
    for (let i = packedSize; i < alignedSize; i++) {
      u32View[i] = 0;
    }
    buffer.unmap();
    
    // Store metadata
    buffer._originalSize = f32.length;
    buffer._packedSize = packedSize;
    buffer._isPacked = true;
    
    return buffer;
  }
  
  /**
   * Create GPU buffer as raw FP32 (for small buffers like biases, norms)
   */
  createBufferF32(tensor, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) {
    const f32 = this.tensorToFloat32(tensor);
    const buffer = this.device.createBuffer({
      size: f32.byteLength,
      usage,
      mappedAtCreation: true
    });
    new Float32Array(buffer.getMappedRange()).set(f32);
    buffer.unmap();
    buffer._isPacked = false;
    return buffer;
  }
  
  /**
   * Pack embeddings to FP16 for storage efficiency
   */
  createEmbeddingBuffer(tensor) {
    // Store as packed FP16 (2 values per u32)
    const f32 = this.tensorToFloat32(tensor);
    const packedSize = Math.ceil(f32.length / 2);
    
    const buffer = this.device.createBuffer({
      size: packedSize * 4,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true
    });
    
    const u32View = new Uint32Array(buffer.getMappedRange());
    for (let i = 0; i < packedSize; i++) {
      const v0 = f32[i * 2] || 0;
      const v1 = f32[i * 2 + 1] || 0;
      const h0 = this.fp32ToFp16(v0);
      const h1 = this.fp32ToFp16(v1);
      u32View[i] = h0 | (h1 << 16);
    }
    buffer.unmap();
    return buffer;
  }
  
  /**
   * Convert FP32 to FP16
   */
  fp32ToFp16(f) {
    const f32 = new Float32Array([f]);
    const u32 = new Uint32Array(f32.buffer)[0];
    
    const sign = (u32 >> 16) & 0x8000;
    const exp = (u32 >> 23) & 0xFF;
    const mant = u32 & 0x007FFFFF;
    
    if (exp === 0) {
      return sign;
    } else if (exp === 255) {
      return sign | 0x7C00 | (mant ? 0x0200 : 0);
    }
    
    const newExp = exp - 127 + 15;
    if (newExp >= 31) {
      return sign | 0x7C00;
    } else if (newExp <= 0) {
      return sign;
    }
    
    return sign | (newExp << 10) | (mant >> 13);
  }
  
  /**
   * Load all model weights
   */
  async load(progressCallback) {
    const log = progressCallback || console.log;
    
    log('Loading draft model (Qwen2.5-0.5B)...');
    
    // Check if single file or sharded
    const index = await this.loadIndex();
    
    let tensors = {};
    if (index) {
      // Sharded model
      const files = [...new Set(Object.values(index.weight_map))];
      for (const file of files) {
        log(`Loading ${file}...`);
        const fileTensors = await this.loadSafetensors(file, log);
        Object.assign(tensors, fileTensors);
      }
    } else {
      // Single file
      log('Loading model.safetensors...');
      tensors = await this.loadSafetensors('model.safetensors', log);
    }
    
    log('Creating GPU buffers...');
    await new Promise(r => setTimeout(r, 0));
    
    this.buffers = {
      embed_tokens: null,
      layers: [],
      norm: null,
      lm_head: null  // May be tied to embed_tokens
    };
    
    // Embedding table
    log('Creating embedding buffer...');
    const embedTensor = tensors['model.embed_tokens.weight'];
    if (embedTensor) {
      this.buffers.embed_tokens = this.createEmbeddingBuffer(embedTensor);
      log(`Embeddings: ${embedTensor.shape.join('x')}`);
    }
    await new Promise(r => setTimeout(r, 0));
    
    // Final layer norm (small, keep FP32)
    const normTensor = tensors['model.norm.weight'];
    if (normTensor) {
      this.buffers.norm = this.createBufferF32(normTensor);
    }
    
    // LM head (may be tied)
    const lmHeadTensor = tensors['lm_head.weight'];
    if (lmHeadTensor) {
      this.buffers.lm_head = this.createEmbeddingBuffer(lmHeadTensor);
      log('LM head: separate weights');
    } else if (this.config.tie_word_embeddings) {
      this.buffers.lm_head = this.buffers.embed_tokens;
      log('LM head: tied to embeddings');
    }
    
    // Load transformer layers
    for (let i = 0; i < this.config.num_hidden_layers; i++) {
      const prefix = `model.layers.${i}`;
      
      const layer = {
        // Norms are small - keep as FP32
        input_layernorm: this.createBufferF32(tensors[`${prefix}.input_layernorm.weight`]),
        post_attention_layernorm: this.createBufferF32(tensors[`${prefix}.post_attention_layernorm.weight`]),
        
        // Large weight matrices - pack as FP16
        q_proj: {
          weight: this.createBuffer(tensors[`${prefix}.self_attn.q_proj.weight`]),
          bias: tensors[`${prefix}.self_attn.q_proj.bias`] ? 
                this.createBufferF32(tensors[`${prefix}.self_attn.q_proj.bias`]) : null
        },
        k_proj: {
          weight: this.createBuffer(tensors[`${prefix}.self_attn.k_proj.weight`]),
          bias: tensors[`${prefix}.self_attn.k_proj.bias`] ?
                this.createBufferF32(tensors[`${prefix}.self_attn.k_proj.bias`]) : null
        },
        v_proj: {
          weight: this.createBuffer(tensors[`${prefix}.self_attn.v_proj.weight`]),
          bias: tensors[`${prefix}.self_attn.v_proj.bias`] ?
                this.createBufferF32(tensors[`${prefix}.self_attn.v_proj.bias`]) : null
        },
        o_proj: {
          weight: this.createBuffer(tensors[`${prefix}.self_attn.o_proj.weight`]),
          bias: null
        },
        
        // MLP weights - pack as FP16
        gate_proj: {
          weight: this.createBuffer(tensors[`${prefix}.mlp.gate_proj.weight`])
        },
        up_proj: {
          weight: this.createBuffer(tensors[`${prefix}.mlp.up_proj.weight`])
        },
        down_proj: {
          weight: this.createBuffer(tensors[`${prefix}.mlp.down_proj.weight`])
        }
      };
      
      this.buffers.layers.push(layer);
      
      // Yield every layer to prevent UI freezing
      log(`Creating GPU buffers: layer ${i + 1}/${this.config.num_hidden_layers}`);
      await new Promise(r => setTimeout(r, 0));
    }
    
    this.loaded = true;
    log('Draft model loaded successfully!');
    
    return this.buffers;
  }
  
  getBuffers() {
    return this.buffers;
  }
  
  destroy() {
    if (this.buffers) {
      if (this.buffers.embed_tokens) this.buffers.embed_tokens.destroy();
      if (this.buffers.norm) this.buffers.norm.destroy();
      if (this.buffers.lm_head && this.buffers.lm_head !== this.buffers.embed_tokens) {
        this.buffers.lm_head.destroy();
      }
      for (const layer of this.buffers.layers) {
        for (const key of Object.keys(layer)) {
          const item = layer[key];
          if (item && item.destroy) {
            item.destroy();
          } else if (item && item.weight && item.weight.destroy) {
            item.weight.destroy();
            if (item.bias) item.bias.destroy();
          }
        }
      }
    }
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DraftModelLoader };
}
if (typeof window !== 'undefined') {
  window.DraftModelLoader = DraftModelLoader;
}
