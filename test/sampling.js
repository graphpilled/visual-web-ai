/**
 * Sampling Module for LLM Inference
 * 
 * Supports:
 * - Temperature scaling
 * - Top-k filtering
 * - Top-p (nucleus) sampling
 * - Greedy decoding (argmax)
 * - Repetition penalty
 * 
 * All operations on CPU for simplicity - sampling is not the bottleneck.
 * The logits array is only 152K × 4 = 600KB, fast to transfer and process.
 */

class Sampler {
  constructor(options = {}) {
    this.temperature = options.temperature ?? 1.0;
    this.topK = options.topK ?? 50;
    this.topP = options.topP ?? 0.9;
    this.repetitionPenalty = options.repetitionPenalty ?? 1.0;
    this.seed = options.seed ?? null;
    
    // Simple PRNG for reproducibility
    this.rngState = this.seed ?? Date.now();
  }
  
  /**
   * Set sampling parameters
   */
  setParams(params) {
    if (params.temperature !== undefined) this.temperature = params.temperature;
    if (params.topK !== undefined) this.topK = params.topK;
    if (params.topP !== undefined) this.topP = params.topP;
    if (params.repetitionPenalty !== undefined) this.repetitionPenalty = params.repetitionPenalty;
    if (params.seed !== undefined) {
      this.seed = params.seed;
      this.rngState = params.seed;
    }
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
   * Apply repetition penalty to logits
   * Reduces probability of tokens that appeared in context
   * 
   * @param {Float32Array} logits - Raw logits
   * @param {number[]} tokenHistory - Previously generated token IDs
   * @returns {Float32Array} - Modified logits
   */
  applyRepetitionPenalty(logits, tokenHistory) {
    if (this.repetitionPenalty === 1.0 || tokenHistory.length === 0) {
      return logits;
    }
    
    const result = new Float32Array(logits);
    const seen = new Set(tokenHistory);
    
    for (const tokenId of seen) {
      if (tokenId < result.length) {
        if (result[tokenId] > 0) {
          result[tokenId] /= this.repetitionPenalty;
        } else {
          result[tokenId] *= this.repetitionPenalty;
        }
      }
    }
    
    return result;
  }
  
  /**
   * Apply temperature scaling
   * Higher temperature = more random, lower = more deterministic
   * 
   * @param {Float32Array} logits - Raw logits
   * @returns {Float32Array} - Scaled logits
   */
  applyTemperature(logits) {
    if (this.temperature === 1.0) {
      return logits;
    }
    
    const result = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      result[i] = logits[i] / this.temperature;
    }
    return result;
  }
  
  /**
   * Apply top-k filtering
   * Keep only the k highest probability tokens
   * 
   * @param {Float32Array} logits - Logits (will be modified)
   * @returns {Float32Array} - Filtered logits (others set to -Infinity)
   */
  applyTopK(logits) {
    if (this.topK <= 0 || this.topK >= logits.length) {
      return logits;
    }
    
    // Find top-k values
    const indexed = Array.from(logits).map((v, i) => ({ value: v, index: i }));
    indexed.sort((a, b) => b.value - a.value);
    
    const result = new Float32Array(logits.length).fill(-Infinity);
    for (let i = 0; i < this.topK; i++) {
      result[indexed[i].index] = indexed[i].value;
    }
    
    return result;
  }
  
  /**
   * Apply top-p (nucleus) filtering
   * Keep smallest set of tokens whose cumulative probability >= p
   * 
   * @param {Float32Array} logits - Logits
   * @returns {Float32Array} - Filtered logits
   */
  applyTopP(logits) {
    if (this.topP >= 1.0) {
      return logits;
    }
    
    // Convert to probabilities
    const probs = this.softmax(logits);
    
    // Sort by probability descending
    const indexed = Array.from(probs).map((p, i) => ({ prob: p, index: i, logit: logits[i] }));
    indexed.sort((a, b) => b.prob - a.prob);
    
    // Find cutoff
    let cumSum = 0;
    let cutoffIdx = indexed.length;
    
    for (let i = 0; i < indexed.length; i++) {
      cumSum += indexed[i].prob;
      if (cumSum >= this.topP) {
        cutoffIdx = i + 1;
        break;
      }
    }
    
    // Keep only tokens within nucleus
    const result = new Float32Array(logits.length).fill(-Infinity);
    for (let i = 0; i < cutoffIdx; i++) {
      result[indexed[i].index] = indexed[i].logit;
    }
    
    return result;
  }
  
  /**
   * Softmax function
   * 
   * @param {Float32Array} logits - Input logits
   * @returns {Float32Array} - Probabilities
   */
  softmax(logits) {
    // Find max for numerical stability
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    
    // Compute exp and sum
    const probs = new Float32Array(logits.length);
    let sum = 0;
    
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] === -Infinity) {
        probs[i] = 0;
      } else {
        probs[i] = Math.exp(logits[i] - maxLogit);
        sum += probs[i];
      }
    }
    
    // Normalize
    if (sum > 0) {
      for (let i = 0; i < probs.length; i++) {
        probs[i] /= sum;
      }
    }
    
    return probs;
  }
  
  /**
   * Greedy decoding - return token with highest logit
   * 
   * @param {Float32Array} logits - Raw logits
   * @returns {number} - Token ID
   */
  argmax(logits) {
    let maxIdx = 0;
    let maxVal = logits[0];
    
    for (let i = 1; i < logits.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    
    return maxIdx;
  }
  
  /**
   * Sample from probability distribution
   * 
   * @param {Float32Array} probs - Probability distribution
   * @returns {number} - Sampled token ID
   */
  sampleFromProbs(probs) {
    const r = this.random();
    let cumSum = 0;
    
    for (let i = 0; i < probs.length; i++) {
      cumSum += probs[i];
      if (r < cumSum) {
        return i;
      }
    }
    
    // Fallback (shouldn't happen with valid probs)
    return probs.length - 1;
  }
  
  /**
   * Full sampling pipeline
   * 
   * @param {Float32Array} logits - Raw logits from LM head
   * @param {number[]} tokenHistory - Previously generated tokens (for repetition penalty)
   * @returns {{tokenId: number, prob: number}} - Sampled token and its probability
   */
  sample(logits, tokenHistory = []) {
    // 1. Apply repetition penalty
    let processed = this.applyRepetitionPenalty(logits, tokenHistory);
    
    // 2. Apply temperature
    processed = this.applyTemperature(processed);
    
    // 3. Apply top-k
    processed = this.applyTopK(processed);
    
    // 4. Apply top-p
    processed = this.applyTopP(processed);
    
    // 5. Convert to probabilities
    const probs = this.softmax(processed);
    
    // 6. Sample
    const tokenId = this.sampleFromProbs(probs);
    
    return {
      tokenId,
      prob: probs[tokenId]
    };
  }
  
  /**
   * Greedy sampling (temperature=0 equivalent)
   * 
   * @param {Float32Array} logits - Raw logits
   * @param {number[]} tokenHistory - For repetition penalty
   * @returns {{tokenId: number, prob: number}}
   */
  sampleGreedy(logits, tokenHistory = []) {
    let processed = this.applyRepetitionPenalty(logits, tokenHistory);
    const tokenId = this.argmax(processed);
    const probs = this.softmax(processed);
    
    return {
      tokenId,
      prob: probs[tokenId]
    };
  }
  
  /**
   * Get top-k tokens with probabilities (for debugging/UI)
   * 
   * @param {Float32Array} logits - Raw logits
   * @param {number} k - Number of top tokens to return
   * @returns {{tokenId: number, prob: number}[]}
   */
  getTopTokens(logits, k = 10) {
    const probs = this.softmax(this.applyTemperature(logits));
    
    const indexed = Array.from(probs).map((p, i) => ({ tokenId: i, prob: p }));
    indexed.sort((a, b) => b.prob - a.prob);
    
    return indexed.slice(0, k);
  }
}

/**
 * Preset sampling configurations
 */
const SamplingPresets = {
  // Deterministic, always pick highest probability
  greedy: {
    temperature: 1.0,
    topK: 1,
    topP: 1.0,
    repetitionPenalty: 1.0
  },
  
  // Good for creative writing
  creative: {
    temperature: 0.9,
    topK: 50,
    topP: 0.95,
    repetitionPenalty: 1.1
  },
  
  // Balanced, good default
  balanced: {
    temperature: 0.7,
    topK: 40,
    topP: 0.9,
    repetitionPenalty: 1.05
  },
  
  // Precise, good for code/factual
  precise: {
    temperature: 0.3,
    topK: 20,
    topP: 0.8,
    repetitionPenalty: 1.0
  },
  
  // Chat assistant style
  assistant: {
    temperature: 0.7,
    topK: 50,
    topP: 0.9,
    repetitionPenalty: 1.1
  }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Sampler, SamplingPresets };
}
if (typeof window !== 'undefined') {
  window.Sampler = Sampler;
  window.SamplingPresets = SamplingPresets;
}
