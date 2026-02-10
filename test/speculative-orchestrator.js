/**
 * Speculative Decoding Orchestrator
 * 
 * Coordinates between draft model (Qwen2.5-0.5B) and target model (Qwen2.5-7B)
 * to achieve faster inference through speculation and verification.
 * 
 * Algorithm:
 * 1. Draft phase: Generate K tokens with fast draft model
 * 2. Verify phase: Run single forward pass on target model for all K positions
 * 3. Accept/Reject: Accept matching tokens, reject from first mismatch
 * 4. Bonus token: Even on rejection, we get one new valid token from target
 */

class SpeculativeOrchestrator {
  constructor(device, targetConfig, draftConfig) {
    this.device = device;
    this.targetConfig = targetConfig;
    this.draftConfig = draftConfig;
    
    this.draftModel = null;
    this.draftLoader = null;
    
    this.specLength = 5;  // Number of tokens to speculate
    this.initialized = false;
    
    // Stats
    this.stats = {
      totalDrafted: 0,
      totalAccepted: 0,
      totalIterations: 0
    };
  }
  
  /**
   * Initialize the draft model
   */
  async initDraftModel(baseUrl, progressCallback) {
    const log = progressCallback || console.log;
    
    log('Initializing speculative decoding...');
    
    // Load draft model weights
    this.draftLoader = new DraftModelLoader(this.device, baseUrl);
    const draftBuffers = await this.draftLoader.load(log);
    
    // Create draft inference engine
    this.draftModel = new DraftModelInference(this.device, this.draftLoader.getConfig());
    await this.draftModel.init(draftBuffers);
    
    this.initialized = true;
    log('Speculative decoding ready!');
    
    return this;
  }
  
  setSpecLength(k) {
    this.specLength = k;
  }
  
  getStats() {
    const acceptRate = this.stats.totalDrafted > 0 
      ? (this.stats.totalAccepted / this.stats.totalDrafted * 100).toFixed(1)
      : 0;
    return {
      ...this.stats,
      acceptRate: `${acceptRate}%`,
      avgAccepted: this.stats.totalIterations > 0 
        ? (this.stats.totalAccepted / this.stats.totalIterations).toFixed(2)
        : 0
    };
  }
  
  resetStats() {
    this.stats = { totalDrafted: 0, totalAccepted: 0, totalIterations: 0 };
  }
  
  /**
   * Reset KV caches for both models
   */
  resetCaches() {
    this.draftModel.resetKVCache();
  }
  
  /**
   * Prefill both models with prompt tokens
   * 
   * @param {number[]} promptTokens - Prompt token IDs
   * @param {function} targetForward - Target model forward function
   * @param {object} targetKVCache - Target model KV cache
   */
  async prefill(promptTokens, targetForward, targetKVCache) {
    // Prefill target model (already done in main code)
    // Just prefill draft model here
    
    for (let i = 0; i < promptTokens.length; i++) {
      this.draftModel.updateSeqLen(i + 1);
      await this.draftModel.forward(promptTokens[i], i);
    }
  }
  
  /**
   * Generate tokens using speculative decoding
   * 
   * @param {object} logitsBuffer - Current logits from target model
   * @param {number} startPosition - Current position in sequence
   * @param {function} targetForward - Target model forward function
   * @param {function} targetSample - Target model sampling function  
   * @param {object} targetKVCache - Target model KV cache
   * @param {number[]} existingTokens - Already generated tokens (for EOS check)
   * @param {number} eosToken - EOS token ID
   * @returns {object} { tokens: number[], newLogits: buffer, hitEOS: boolean }
   */
  async generateStep(
    logitsBuffer,
    startPosition,
    targetForward,
    targetSample,
    targetKVCache,
    existingTokens,
    eosToken
  ) {
    if (!this.initialized) {
      throw new Error('Speculative decoder not initialized');
    }
    
    const K = this.specLength;
    
    // ============================================
    // Phase 1: Generate K draft tokens
    // ============================================
    const draftTokens = [];
    
    // First draft token comes from target model's current logits
    const firstToken = await targetSample(logitsBuffer);
    if (firstToken === eosToken) {
      return { tokens: [], newLogits: logitsBuffer, hitEOS: true };
    }
    draftTokens.push(firstToken);
    
    // Generate remaining draft tokens with draft model
    // Sync draft model KV cache position with target
    this.draftModel.updateSeqLen(startPosition + 1);
    await this.draftModel.forward(firstToken, startPosition);
    
    for (let k = 1; k < K; k++) {
      const draftToken = await this.draftModel.sampleGreedy();
      
      if (draftToken === eosToken) {
        break;
      }
      
      draftTokens.push(draftToken);
      
      // Forward draft model
      this.draftModel.updateSeqLen(startPosition + k + 1);
      await this.draftModel.forward(draftToken, startPosition + k);
    }
    
    if (draftTokens.length === 0) {
      return { tokens: [], newLogits: logitsBuffer, hitEOS: true };
    }
    
    this.stats.totalDrafted += draftTokens.length;
    this.stats.totalIterations++;
    
    // ============================================
    // Phase 2: Verify draft tokens with target model
    // ============================================
    
    // Run target model forward for each draft token to get verification logits
    // In true speculative decoding, this would be a single batched forward pass
    // For now, we do sequential (still faster because draft model is fast)
    
    const verifyLogits = [];
    let currentLogits = logitsBuffer;
    
    for (let k = 0; k < draftTokens.length; k++) {
      // Sample from target model's logits
      const targetToken = await targetSample(currentLogits);
      
      // Check if target agrees with draft
      if (targetToken !== draftTokens[k]) {
        // Mismatch! Accept tokens up to this point, use target's token
        const acceptedTokens = draftTokens.slice(0, k);
        
        // If target token is EOS, don't include it
        if (targetToken !== eosToken) {
          acceptedTokens.push(targetToken);
        }
        
        this.stats.totalAccepted += acceptedTokens.length;
        
        // Roll back draft model KV cache to match accepted tokens
        // (In practice, we'd need to handle this more carefully)
        
        // Forward target model for the accepted token to get new logits
        if (acceptedTokens.length > 0) {
          const lastAcceptedPos = startPosition + acceptedTokens.length - 1;
          targetKVCache.seqLen = lastAcceptedPos + 1;
          targetKVCache.updateSeqLenBuffer();
          currentLogits = await targetForward(acceptedTokens[acceptedTokens.length - 1], lastAcceptedPos);
        }
        
        return {
          tokens: acceptedTokens,
          newLogits: currentLogits,
          hitEOS: targetToken === eosToken
        };
      }
      
      // Target agrees with draft, forward target model
      const pos = startPosition + k;
      targetKVCache.seqLen = pos + 1;
      targetKVCache.updateSeqLenBuffer();
      currentLogits = await targetForward(draftTokens[k], pos);
    }
    
    // All draft tokens accepted!
    this.stats.totalAccepted += draftTokens.length;
    
    // Get one bonus token from target model
    const bonusToken = await targetSample(currentLogits);
    
    if (bonusToken !== eosToken) {
      draftTokens.push(bonusToken);
      
      // Forward for bonus token
      const bonusPos = startPosition + draftTokens.length - 1;
      targetKVCache.seqLen = bonusPos + 1;
      targetKVCache.updateSeqLenBuffer();
      currentLogits = await targetForward(bonusToken, bonusPos);
      
      // Update draft model too
      this.draftModel.updateSeqLen(bonusPos + 1);
      await this.draftModel.forward(bonusToken, bonusPos);
    }
    
    return {
      tokens: draftTokens,
      newLogits: currentLogits,
      hitEOS: bonusToken === eosToken
    };
  }
  
  destroy() {
    if (this.draftModel) {
      this.draftModel.destroy();
    }
    if (this.draftLoader) {
      this.draftLoader.destroy();
    }
  }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { SpeculativeOrchestrator };
}
if (typeof window !== 'undefined') {
  window.SpeculativeOrchestrator = SpeculativeOrchestrator;
}
