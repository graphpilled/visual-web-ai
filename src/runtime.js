// src/runtime.js

/**
 * WebGPU Runtime for executing generated WGSL shaders
 * Supports multi-output operations (TopK, Split, etc.)
 */

class GPURuntime {
  constructor() {
    this.device = null;
    this.pipelineCache = new Map();
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error("WebGPU not supported in this browser");
    }

    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      throw new Error("No WebGPU adapter found");
    }

    // Request higher limits if available (needed for large matmuls)
    const requiredLimits = {};
    const adapterLimits = adapter.limits;
    
    // Request max available storage buffer size (default is 128MB, need more for LLM layers)
    if (adapterLimits.maxStorageBufferBindingSize > 134217728) {
      requiredLimits.maxStorageBufferBindingSize = adapterLimits.maxStorageBufferBindingSize;
    }
    
    // Request max buffer size
    if (adapterLimits.maxBufferSize > 268435456) {
      requiredLimits.maxBufferSize = adapterLimits.maxBufferSize;
    }

    this.device = await adapter.requestDevice({
      requiredLimits: Object.keys(requiredLimits).length > 0 ? requiredLimits : undefined
    });
    
    console.log("WebGPU initialized");
    console.log(`  maxStorageBufferBindingSize: ${(this.device.limits.maxStorageBufferBindingSize / 1024 / 1024).toFixed(0)} MB`);
    console.log(`  maxBufferSize: ${(this.device.limits.maxBufferSize / 1024 / 1024).toFixed(0)} MB`);
    return this;
  }

  createBuffer(size, usage, data = null) {
    const buffer = this.device.createBuffer({
      size,
      usage,
      mappedAtCreation: data !== null
    });

    if (data !== null) {
      new Float32Array(buffer.getMappedRange()).set(data);
      buffer.unmap();
    }

    return buffer;
  }

  createPipeline(wgsl, label = "compute") {
    if (this.pipelineCache.has(wgsl)) {
      return this.pipelineCache.get(wgsl);
    }

    const shaderModule = this.device.createShaderModule({
      label: label + "_shader",
      code: wgsl
    });

    const pipeline = this.device.createComputePipeline({
      label: label + "_pipeline",
      layout: "auto",
      compute: {
        module: shaderModule,
        entryPoint: "main"
      }
    });

    this.pipelineCache.set(wgsl, pipeline);
    return pipeline;
  }

  async execute(kernel, dispatch, inputs) {
    const { wgsl, bindings } = kernel;
    const { workgroupCount } = dispatch;

    const pipeline = this.createPipeline(wgsl, kernel.name);

    const gpuBuffers = {};
    const outputBindings = bindings.filter(b => b.usage === 'ReadWrite');

    for (const binding of bindings) {
      const isOutput = binding.usage === 'ReadWrite';
      const inputData = inputs[binding.name];

      let usage = GPUBufferUsage.STORAGE;
      if (isOutput) {
        usage |= GPUBufferUsage.COPY_SRC;
      }
      if (inputData) {
        usage |= GPUBufferUsage.COPY_DST;
      }

      const buffer = this.createBuffer(
        binding.size,
        usage,
        inputData || null
      );

      gpuBuffers[binding.name] = buffer;
    }

    const bindGroupEntries = bindings.map(b => ({
      binding: b.binding,
      resource: { buffer: gpuBuffers[b.name] }
    }));

    const bindGroup = this.device.createBindGroup({
      label: kernel.name + "_bindgroup",
      layout: pipeline.getBindGroupLayout(0),
      entries: bindGroupEntries
    });

    const commandEncoder = this.device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();

    passEncoder.setPipeline(pipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(
      workgroupCount[0],
      workgroupCount[1],
      workgroupCount[2]
    );
    passEncoder.end();

    // Handle multiple outputs
    const results = {};
    for (const outputBinding of outputBindings) {
      const stagingBuffer = this.device.createBuffer({
        size: outputBinding.size,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      });

      commandEncoder.copyBufferToBuffer(
        gpuBuffers[outputBinding.name],
        0,
        stagingBuffer,
        0,
        outputBinding.size
      );

      results[outputBinding.name] = { staging: stagingBuffer, size: outputBinding.size };
    }

    this.device.queue.submit([commandEncoder.finish()]);

    // Read back all outputs
    const output = {};
    for (const [name, { staging }] of Object.entries(results)) {
      await staging.mapAsync(GPUMapMode.READ);
      output[name] = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
    }

    // Cleanup
    for (const buffer of Object.values(gpuBuffers)) {
      buffer.destroy();
    }

    // Return single output for backwards compatibility, or object for multiple
    const outputNames = Object.keys(output);
    if (outputNames.length === 1) {
      return output[outputNames[0]];
    }
    return output;
  }

  async runOp(kernel, dispatch, ...inputArrays) {
    const inputs = {};
    const inputBindings = kernel.bindings.filter(b => b.usage !== 'ReadWrite');

    inputBindings.forEach((binding, i) => {
      if (inputArrays[i]) {
        inputs[binding.name] = inputArrays[i];
      }
    });

    return this.execute(kernel, dispatch, inputs);
  }

  destroy() {
    this.pipelineCache.clear();
    this.device = null;
  }
}

/**
 * Graph Executor - runs a compiled graph on the GPU
 * Supports multi-output operations
 */
class GraphExecutor {
  constructor(runtime, compiledGraph) {
    this.runtime = runtime;
    this.graph = compiledGraph;
    this.buffers = new Map(); // bufferId -> GPUBuffer
    this.pipelines = new Map(); // wgsl -> GPUComputePipeline
    this.initialized = false;
  }

  async init() {
    const device = this.runtime.device;

    // Create GPU buffers for all buffers in the graph
    for (const bufInfo of this.graph.buffers) {
      let usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

      // Output buffers need COPY_SRC for readback
      if (this.graph.outputBufferIds.includes(bufInfo.id)) {
        usage |= GPUBufferUsage.COPY_SRC;
      }

      const buffer = device.createBuffer({
        label: `buffer_${bufInfo.id}_node${bufInfo.nodeId}_out${bufInfo.outputIndex}`,
        size: Math.max(bufInfo.size, 4),
        usage
      });

      this.buffers.set(bufInfo.id, buffer);

      // Initialize weights and constants with their data
      if (bufInfo.data && (bufInfo.kind === "Weight" || bufInfo.kind === "Constant")) {
        const data = new Float32Array(bufInfo.data);
        device.queue.writeBuffer(buffer, 0, data);
      }
    }

    // Compile pipelines for each op
    for (const op of this.graph.ops) {
      if (!this.pipelines.has(op.kernel.wgsl)) {
        const pipeline = this.runtime.createPipeline(op.kernel.wgsl, op.kernel.name);
        this.pipelines.set(op.kernel.wgsl, pipeline);
      }
    }

    this.initialized = true;
    return this;
  }

  setInput(inputIndex, data) {
    const bufferId = this.graph.inputBufferIds[inputIndex];
    const buffer = this.buffers.get(bufferId);

    if (!buffer) {
      throw new Error(`Input buffer ${inputIndex} not found`);
    }

    this.runtime.device.queue.writeBuffer(buffer, 0, data);
  }

  setWeight(weightIndex, data) {
    const bufferId = this.graph.weightBufferIds[weightIndex];
    const buffer = this.buffers.get(bufferId);

    if (!buffer) {
      throw new Error(`Weight buffer ${weightIndex} not found`);
    }

    this.runtime.device.queue.writeBuffer(buffer, 0, data);
  }

  setWeightByName(name, data) {
    const weightIndex = this.graph.weightNames.indexOf(name);
    if (weightIndex === -1) {
      throw new Error(`Weight "${name}" not found`);
    }
    this.setWeight(weightIndex, data);
  }

  loadWeights(weightDict) {
    for (const [name, data] of Object.entries(weightDict)) {
      try {
        this.setWeightByName(name, data);
      } catch (e) {
        console.warn(`Warning: ${e.message}`);
      }
    }
  }

  getWeightInfo() {
    return this.graph.weightNames.map((name, i) => ({
      name,
      shape: this.graph.weightShapes[i],
      bufferId: this.graph.weightBufferIds[i]
    }));
  }

  async run() {
    if (!this.initialized) {
      throw new Error('Executor not initialized. Call init() first.');
    }

    const device = this.runtime.device;

    // Execute each op sequentially
    for (const op of this.graph.ops) {
      const pipeline = this.pipelines.get(op.kernel.wgsl);

      // Separate input and output bindings from the kernel
      const inputBindings = op.kernel.bindings.filter(b => b.usage === 'ReadOnly');
      const outputBindings = op.kernel.bindings.filter(b => b.usage === 'ReadWrite');

      // Build bind group entries
      const entries = [];

      // Add input buffers (matched by position)
      inputBindings.forEach((binding, i) => {
        const bufferId = op.inputBufferIds[i];
        const buffer = this.buffers.get(bufferId);
        if (buffer) {
          entries.push({
            binding: binding.binding,
            resource: { buffer }
          });
        }
      });

      // Add output buffers (matched by position in outputBufferIds)
      outputBindings.forEach((binding, i) => {
        const bufferId = op.outputBufferIds[i];
        const buffer = this.buffers.get(bufferId);
        if (buffer) {
          entries.push({
            binding: binding.binding,
            resource: { buffer }
          });
        }
      });

      const bindGroup = device.createBindGroup({
        label: `${op.kernel.name}_bindgroup`,
        layout: pipeline.getBindGroupLayout(0),
        entries
      });

      const commandEncoder = device.createCommandEncoder();
      const passEncoder = commandEncoder.beginComputePass();
      passEncoder.setPipeline(pipeline);
      passEncoder.setBindGroup(0, bindGroup);
      passEncoder.dispatchWorkgroups(
        op.dispatch.workgroupCount[0],
        op.dispatch.workgroupCount[1],
        op.dispatch.workgroupCount[2]
      );
      passEncoder.end();

      device.queue.submit([commandEncoder.finish()]);
      await device.queue.onSubmittedWorkDone();
    }
  }

  async getOutput(outputIndex = 0) {
    const bufferId = this.graph.outputBufferIds[outputIndex];
    const buffer = this.buffers.get(bufferId);
    const bufInfo = this.graph.buffers.find(b => b.id === bufferId);

    if (!buffer || !bufInfo) {
      throw new Error(`Output buffer ${outputIndex} not found`);
    }

    const stagingBuffer = this.runtime.device.createBuffer({
      size: bufInfo.size,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
    });

    const commandEncoder = this.runtime.device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, bufInfo.size);
    this.runtime.device.queue.submit([commandEncoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const result = new Float32Array(stagingBuffer.getMappedRange().slice(0));
    stagingBuffer.unmap();
    stagingBuffer.destroy();

    return result;
  }

  async getAllOutputs() {
    const outputs = [];
    for (let i = 0; i < this.graph.outputBufferIds.length; i++) {
      outputs.push(await this.getOutput(i));
    }
    return outputs;
  }

  async execute(...inputs) {
    inputs.forEach((data, i) => this.setInput(i, data));
    await this.run();

    if (this.graph.outputBufferIds.length === 1) {
      return this.getOutput(0);
    } else {
      return this.getAllOutputs();
    }
  }

  destroy() {
    for (const buffer of this.buffers.values()) {
      buffer.destroy();
    }
    this.buffers.clear();
    this.pipelines.clear();
    this.initialized = false;
  }
}

export { GPURuntime, GraphExecutor };
