// src/bridge.js
// Bridge to expose ReScript Codegen to browser

import * as Codegen from './Codegen.res.mjs';
import * as Types from './Types.res.mjs';
import * as Shape from './Shape.res.mjs';
import * as Compiler from './Compiler.res.mjs';
import * as AutogradRes from './Autograd.res.mjs';

export { Codegen, Types, Shape, Compiler };

// Autograd exports
export const Autograd = {
  // Gradient reduction for broadcasting
  genGradReduceKernel: (gradShape, targetShape) => 
    AutogradRes.genGradReduceKernel(gradShape, targetShape),
  
  // Unary backward kernels
  genNegBackwardKernel: AutogradRes.genNegBackwardKernel,
  genAbsBackwardKernel: AutogradRes.genAbsBackwardKernel,
  genSqrtBackwardKernel: AutogradRes.genSqrtBackwardKernel,
  genExpBackwardKernel: AutogradRes.genExpBackwardKernel,
  genLogBackwardKernel: AutogradRes.genLogBackwardKernel,
  genSinBackwardKernel: AutogradRes.genSinBackwardKernel,
  genCosBackwardKernel: AutogradRes.genCosBackwardKernel,
  genTanhBackwardKernel: AutogradRes.genTanhBackwardKernel,
  genSigmoidBackwardKernel: AutogradRes.genSigmoidBackwardKernel,
  genReLUBackwardKernel: AutogradRes.genReLUBackwardKernel,
  genLeakyReLUBackwardKernel: AutogradRes.genLeakyReLUBackwardKernel,
  genGeLUBackwardKernel: AutogradRes.genGeLUBackwardKernel,
  
  // Binary backward kernels
  genAddBackwardKernel: AutogradRes.genAddBackwardKernel,
  genSubBackwardKernel: AutogradRes.genSubBackwardKernel,
  genMulBackwardKernel: AutogradRes.genMulBackwardKernel,
  genDivBackwardKernel: AutogradRes.genDivBackwardKernel,
  genPowBackwardKernel: AutogradRes.genPowBackwardKernel,
  genMaximumBackwardKernel: AutogradRes.genMaximumBackwardKernel,
  genMinimumBackwardKernel: AutogradRes.genMinimumBackwardKernel,
  
  // MatMul backward
  genMatMulBackwardAKernel: AutogradRes.genMatMulBackwardAKernel,
  genMatMulBackwardBKernel: AutogradRes.genMatMulBackwardBKernel,
  genBatchedMatMulBackwardAKernel: AutogradRes.genBatchedMatMulBackwardAKernel,
  genBatchedMatMulBackwardBKernel: AutogradRes.genBatchedMatMulBackwardBKernel,
  
  // Reduction backward
  genSumBackwardKernel: AutogradRes.genSumBackwardKernel,
  genMeanBackwardKernel: AutogradRes.genMeanBackwardKernel,
  
  // Softmax backward
  genSoftmaxBackwardKernel: AutogradRes.genSoftmaxBackwardKernel,
  
  // LayerNorm backward
  genLayerNormBackwardKernel: AutogradRes.genLayerNormBackwardKernel,
  
  // Utility
  genCopyBackwardKernel: AutogradRes.genCopyBackwardKernel,
  genGradAccumulateKernel: AutogradRes.genGradAccumulateKernel,
  genGradZeroKernel: AutogradRes.genGradZeroKernel,
  
  // Optimizers
  genSGDKernel: AutogradRes.genSGDKernel,
  genSGDMomentumKernel: AutogradRes.genSGDMomentumKernel,
  genAdamKernel: AutogradRes.genAdamKernel,
  genAdamWKernel: AutogradRes.genAdamWKernel,
};

// GradTape exports
import * as GradTapeRes from './GradTape.res.mjs';
export const GradTape = {
  create: GradTapeRes.create,
  markParameter: GradTapeRes.markParameter,
  isParameter: GradTapeRes.isParameter,
  recordOp: GradTapeRes.recordOp,
  reset: GradTapeRes.reset,
  getBackwardKernels: GradTapeRes.getBackwardKernels,
  supportsGradient: GradTapeRes.supportsGradient,
  compileBackward: GradTapeRes.compileBackward,
};

// AutogradEngine exports
import * as AutogradEngineRes from './AutogradEngine.res.mjs';
export const AutogradEngine = {
  create: AutogradEngineRes.create,
  markRequiresGrad: AutogradEngineRes.markRequiresGrad,
  needsGrad: AutogradEngineRes.needsGrad,
  record: AutogradEngineRes.record,
  clearTape: AutogradEngineRes.clearTape,
  generateBackwardOps: AutogradEngineRes.generateBackwardOps,
  getParameterGradients: AutogradEngineRes.getParameterGradients,
  generateOptimizerOps: AutogradEngineRes.generateOptimizerOps,
  generateZeroGradOps: AutogradEngineRes.generateZeroGradOps,
};
