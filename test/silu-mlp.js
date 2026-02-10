/**
 * SiLU Activation and MLP Block for Qwen2.5
 * 
 * Qwen2.5-7B MLP architecture:
 *   - gate_proj: [3584, 18944] INT4
 *   - up_proj:   [3584, 18944] INT4
 *   - down_proj: [18944, 3584] INT4
 * 
 * Forward pass:
 *   hidden = SiLU(gate_proj(x)) * up_proj(x)
 *   output = down_proj(hidden)
 * 
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 */

/**
 * Generate SiLU activation kernel
 * Applies SiLU element-wise: y = x / (1 + exp(-x))
 */
function genSiLUKernel(size) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

const SIZE = ${size}u;
const WG_SIZE = ${wgSize}u;

@compute @workgroup_size(${wgSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  
  let x = input[idx];
  // SiLU: x * sigmoid(x) = x / (1 + exp(-x))
  output[idx] = x / (1.0 + exp(-x));
}`;
}

/**
 * Generate fused SiLU + element-wise multiply kernel
 * Computes: output = SiLU(gate) * up
 * This fuses two operations into one kernel for efficiency
 */
function genSiLUMulKernel(size) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

const SIZE = ${size}u;
const WG_SIZE = ${wgSize}u;

@compute @workgroup_size(${wgSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  
  let g = gate[idx];
  let u = up[idx];
  
  // SiLU(gate) * up
  let silu_g = g / (1.0 + exp(-g));
  output[idx] = silu_g * u;
}`;
}

/**
 * Generate in-place SiLU + multiply kernel
 * Computes: gate = SiLU(gate) * up
 * Saves memory by reusing gate buffer
 */
function genSiLUMulInPlaceKernel(size) {
  const wgSize = 256;
  
  return `
@group(0) @binding(0) var<storage, read_write> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;

const SIZE = ${size}u;
const WG_SIZE = ${wgSize}u;

@compute @workgroup_size(${wgSize})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= SIZE) { return; }
  
  let g = gate[idx];
  let u = up[idx];
  
  // SiLU(gate) * up, stored back to gate
  let silu_g = g / (1.0 + exp(-g));
  gate[idx] = silu_g * u;
}`;
}

/**
 * MLP Block class
 * Manages the three projections and SiLU activation
 */
class MLPBlock {
  constructor(device, config) {
    this.device = device;
    this.hiddenSize = config.hiddenSize || 3584;
    this.intermediateSize = config.intermediateSize || 18944;
    
    // Buffers for intermediate results
    this.gateBuffer = null;
    this.upBuffer = null;
    this.hiddenBuffer = null;
    
    // Pipelines
    this.siluMulPipeline = null;
    
    // Weight buffers (set externally)
    this.gateProjWeights = null;
    this.gateProjScales = null;
    this.upProjWeights = null;
    this.upProjScales = null;
    this.downProjWeights = null;
    this.downProjScales = null;
  }
  
  /**
   * Initialize buffers and pipelines
   */
  async init() {
    // Create intermediate buffers
    this.gateBuffer = this.device.createBuffer({
      size: this.intermediateSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'mlp_gate'
    });
    
    this.upBuffer = this.device.createBuffer({
      size: this.intermediateSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'mlp_up'
    });
    
    this.hiddenBuffer = this.device.createBuffer({
      size: this.intermediateSize * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      label: 'mlp_hidden'
    });
    
    // Create SiLU + Mul pipeline
    const siluMulShader = genSiLUMulKernel(this.intermediateSize);
    const module = this.device.createShaderModule({ code: siluMulShader });
    this.siluMulPipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module, entryPoint: 'main' }
    });
    
    console.log(`MLP Block initialized: ${this.hiddenSize} -> ${this.intermediateSize} -> ${this.hiddenSize}`);
  }
  
  /**
   * Set weight buffers for the projections
   */
  setWeights(weights) {
    this.gateProjWeights = weights.gate_proj_weights;
    this.gateProjScales = weights.gate_proj_scales;
    this.upProjWeights = weights.up_proj_weights;
    this.upProjScales = weights.up_proj_scales;
    this.downProjWeights = weights.down_proj_weights;
    this.downProjScales = weights.down_proj_scales;
  }
  
  /**
   * Forward pass through MLP
   * 
   * @param {GPUBuffer} input - Input tensor [hidden_size]
   * @param {GPUBuffer} output - Output tensor [hidden_size]
   * @param {Function} int4MatMul - INT4 matmul function from your Codegen
   */
  forward(input, output, int4MatMul) {
    const commandEncoder = this.device.createCommandEncoder();
    
    // Step 1: gate = gate_proj(input)  [3584] -> [18944]
    // Step 2: up = up_proj(input)      [3584] -> [18944]
    // (These use your INT4 matmul kernel)
    
    // Step 3: hidden = SiLU(gate) * up
    const siluMulBindGroup = this.device.createBindGroup({
      layout: this.siluMulPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.gateBuffer } },
        { binding: 1, resource: { buffer: this.upBuffer } },
        { binding: 2, resource: { buffer: this.hiddenBuffer } }
      ]
    });
    
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.siluMulPipeline);
    pass.setBindGroup(0, siluMulBindGroup);
    pass.dispatchWorkgroups(Math.ceil(this.intermediateSize / 256));
    pass.end();
    
    // Step 4: output = down_proj(hidden)  [18944] -> [3584]
    // (This uses your INT4 matmul kernel)
    
    this.device.queue.submit([commandEncoder.finish()]);
  }
  
  /**
   * Get intermediate buffers for external INT4 matmul
   */
  getBuffers() {
    return {
      gate: this.gateBuffer,
      up: this.upBuffer,
      hidden: this.hiddenBuffer
    };
  }
  
  /**
   * Destroy buffers
   */
  destroy() {
    if (this.gateBuffer) this.gateBuffer.destroy();
    if (this.upBuffer) this.upBuffer.destroy();
    if (this.hiddenBuffer) this.hiddenBuffer.destroy();
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MLPBlock, genSiLUKernel, genSiLUMulKernel, genSiLUMulInPlaceKernel };
}
if (typeof window !== 'undefined') {
  window.MLPBlock = MLPBlock;
  window.genSiLUKernel = genSiLUKernel;
  window.genSiLUMulKernel = genSiLUMulKernel;
  window.genSiLUMulInPlaceKernel = genSiLUMulInPlaceKernel;
}
