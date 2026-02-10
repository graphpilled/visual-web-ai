/**
 * Qwen2.5 BPE Tokenizer
 * 
 * A JavaScript implementation of the Qwen2/Qwen2.5 tokenizer.
 * Uses vocab.json + merges.txt format (standard HuggingFace BPE).
 * 
 * Files needed from the model repo:
 * - vocab.json: token -> id mapping
 * - merges.txt: BPE merge rules
 * - tokenizer.json (optional): for special tokens
 */

// Qwen2.5 Special Token IDs
const QWEN2_SPECIAL_TOKENS = {
  '<|endoftext|>': 151643,
  '<|im_start|>': 151644,
  '<|im_end|>': 151645,
  // Additional special tokens from tokenizer.json
  '<|object_ref_start|>': 151646,
  '<|object_ref_end|>': 151647,
  '<|box_start|>': 151648,
  '<|box_end|>': 151649,
  '<|quad_start|>': 151650,
  '<|quad_end|>': 151651,
  '<|vision_start|>': 151652,
  '<|vision_end|>': 151653,
  '<|vision_pad|>': 151654,
  '<|image_pad|>': 151655,
  '<|video_pad|>': 151656,
};

// Byte-to-unicode mapping (GPT-2 style)
function bytesToUnicode() {
  const bs = [];
  // Printable ASCII
  for (let i = 33; i <= 126; i++) bs.push(i);    // ! to ~
  for (let i = 161; i <= 172; i++) bs.push(i);   // ¡ to ¬
  for (let i = 174; i <= 255; i++) bs.push(i);   // ® to ÿ
  
  const cs = [...bs];
  let n = 0;
  
  // Map remaining bytes to unicode chars starting at 256
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  
  const byteToChar = {};
  const charToByte = {};
  
  for (let i = 0; i < bs.length; i++) {
    byteToChar[bs[i]] = String.fromCharCode(cs[i]);
    charToByte[String.fromCharCode(cs[i])] = bs[i];
  }
  
  return { byteToChar, charToByte };
}

// Pre-tokenization regex (GPT-2/Qwen style)
// This splits text into chunks before BPE
const PRETOKENIZE_REGEX = /(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

// Simpler fallback regex for environments without full Unicode support
const PRETOKENIZE_REGEX_SIMPLE = /'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+/g;

/**
 * Get pairs of consecutive symbols in a word
 */
function getPairs(word) {
  const pairs = new Set();
  let prev = word[0];
  for (let i = 1; i < word.length; i++) {
    pairs.add(prev + '\0' + word[i]);
    prev = word[i];
  }
  return pairs;
}

/**
 * Main Qwen2 Tokenizer class
 */
class Qwen2Tokenizer {
  constructor() {
    this.vocab = null;          // token string -> id
    this.vocabReverse = null;   // id -> token string
    this.merges = null;         // merge rules as Map
    this.bpeRanks = null;       // pair -> rank
    this.byteEncoder = null;
    this.byteDecoder = null;
    this.specialTokens = { ...QWEN2_SPECIAL_TOKENS };
    this.specialTokensReverse = {};
    this.cache = new Map();     // BPE cache for performance
    
    // Build reverse special tokens map
    for (const [token, id] of Object.entries(this.specialTokens)) {
      this.specialTokensReverse[id] = token;
    }
  }
  
  /**
   * Load tokenizer from vocab.json and merges.txt
   */
  async loadFromFiles(vocabJson, mergesTxt) {
    // Parse vocab
    if (typeof vocabJson === 'string') {
      this.vocab = JSON.parse(vocabJson);
    } else {
      this.vocab = vocabJson;
    }
    
    // Build reverse vocab
    this.vocabReverse = {};
    for (const [token, id] of Object.entries(this.vocab)) {
      this.vocabReverse[id] = token;
    }
    
    // Parse merges
    const mergeLines = mergesTxt.split('\n');
    this.bpeRanks = new Map();
    
    let rank = 0;
    for (const line of mergeLines) {
      // Skip header line and empty lines
      if (line.startsWith('#') || line.trim() === '') continue;
      
      const parts = line.split(' ');
      if (parts.length >= 2) {
        const pair = parts[0] + '\0' + parts[1];
        this.bpeRanks.set(pair, rank++);
      }
    }
    
    // Setup byte encoder/decoder
    const { byteToChar, charToByte } = bytesToUnicode();
    this.byteEncoder = byteToChar;
    this.byteDecoder = charToByte;
    
    console.log(`Loaded tokenizer: ${Object.keys(this.vocab).length} tokens, ${this.bpeRanks.size} merges`);
  }
  
  /**
   * Convert text to BPE-encoded unicode string
   */
  _textToBpeString(text) {
    const bytes = new TextEncoder().encode(text);
    let result = '';
    for (const b of bytes) {
      result += this.byteEncoder[b];
    }
    return result;
  }
  
  /**
   * Convert BPE-encoded unicode string back to text
   */
  _bpeStringToText(bpeStr) {
    const bytes = [];
    for (const char of bpeStr) {
      if (this.byteDecoder[char] !== undefined) {
        bytes.push(this.byteDecoder[char]);
      }
    }
    return new TextDecoder().decode(new Uint8Array(bytes));
  }
  
  /**
   * Apply BPE to a single word
   */
  _bpe(token) {
    // Check cache
    if (this.cache.has(token)) {
      return this.cache.get(token);
    }
    
    let word = token.split('');
    
    if (word.length <= 1) {
      return word;
    }
    
    while (true) {
      // Find the pair with lowest rank
      let minPair = null;
      let minRank = Infinity;
      
      for (let i = 0; i < word.length - 1; i++) {
        const pair = word[i] + '\0' + word[i + 1];
        const rank = this.bpeRanks.get(pair);
        if (rank !== undefined && rank < minRank) {
          minRank = rank;
          minPair = [word[i], word[i + 1], i];
        }
      }
      
      if (minPair === null) {
        break;
      }
      
      // Merge the pair
      const [first, second, idx] = minPair;
      const newWord = [];
      
      let i = 0;
      while (i < word.length) {
        if (i === idx) {
          newWord.push(first + second);
          i += 2;
        } else {
          newWord.push(word[i]);
          i++;
        }
      }
      
      word = newWord;
      
      if (word.length === 1) {
        break;
      }
    }
    
    this.cache.set(token, word);
    return word;
  }
  
  /**
   * Encode text to token IDs
   */
  encode(text, options = {}) {
    const { addSpecialTokens = false } = options;
    
    if (!this.vocab) {
      throw new Error('Tokenizer not loaded. Call loadFromFiles() first.');
    }
    
    const tokens = [];
    
    // Check for special tokens first
    let remaining = text;
    const specialPattern = new RegExp(
      Object.keys(this.specialTokens)
        .map(t => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'))
        .join('|'),
      'g'
    );
    
    let lastIdx = 0;
    let match;
    
    while ((match = specialPattern.exec(text)) !== null) {
      // Encode text before special token
      if (match.index > lastIdx) {
        const chunk = text.slice(lastIdx, match.index);
        tokens.push(...this._encodeChunk(chunk));
      }
      
      // Add special token
      tokens.push(this.specialTokens[match[0]]);
      lastIdx = match.index + match[0].length;
    }
    
    // Encode remaining text
    if (lastIdx < text.length) {
      tokens.push(...this._encodeChunk(text.slice(lastIdx)));
    }
    
    return tokens;
  }
  
  /**
   * Encode a text chunk (no special tokens)
   */
  _encodeChunk(text) {
    const tokens = [];
    
    // Pre-tokenize
    let matches;
    try {
      matches = text.match(PRETOKENIZE_REGEX) || [];
    } catch (e) {
      // Fallback for environments without full Unicode regex support
      matches = text.match(PRETOKENIZE_REGEX_SIMPLE) || [];
    }
    
    for (const chunk of matches) {
      // Convert to BPE string
      const bpeStr = this._textToBpeString(chunk);
      
      // Apply BPE
      const bpeTokens = this._bpe(bpeStr);
      
      // Convert to IDs
      for (const token of bpeTokens) {
        const id = this.vocab[token];
        if (id !== undefined) {
          tokens.push(id);
        } else {
          // Unknown token - encode bytes individually
          for (const char of token) {
            const byteId = this.vocab[char];
            if (byteId !== undefined) {
              tokens.push(byteId);
            }
          }
        }
      }
    }
    
    return tokens;
  }
  
  /**
   * Decode token IDs to text
   */
  decode(tokenIds, options = {}) {
    const { skipSpecialTokens = false } = options;
    
    if (!this.vocabReverse) {
      throw new Error('Tokenizer not loaded. Call loadFromFiles() first.');
    }
    
    let bpeStr = '';
    
    for (const id of tokenIds) {
      // Check for special token
      if (this.specialTokensReverse[id]) {
        if (!skipSpecialTokens) {
          // Special tokens are already proper strings, add them directly
          // First decode what we have so far
          if (bpeStr) {
            // This is a simplification - in reality we'd need to handle this better
          }
          bpeStr += this._textToBpeString(this.specialTokensReverse[id]);
        }
        continue;
      }
      
      const token = this.vocabReverse[id];
      if (token !== undefined) {
        bpeStr += token;
      }
    }
    
    return this._bpeStringToText(bpeStr);
  }
  
  /**
   * Get vocabulary size
   */
  get vocabSize() {
    return this.vocab ? Object.keys(this.vocab).length : 0;
  }
  
  /**
   * Get special token ID
   */
  getSpecialTokenId(name) {
    return this.specialTokens[name];
  }
  
  /**
   * Apply chat template to messages
   */
  applyChatTemplate(messages, options = {}) {
    const {
      addGenerationPrompt = true,
      tokenize = true,
    } = options;
    
    let text = '';
    
    for (const msg of messages) {
      const role = msg.role;
      const content = msg.content;
      
      text += `<|im_start|>${role}\n${content}<|im_end|>\n`;
    }
    
    if (addGenerationPrompt) {
      text += '<|im_start|>assistant\n';
    }
    
    if (tokenize) {
      return this.encode(text);
    }
    
    return text;
  }
}

/**
 * Helper to load tokenizer from File objects (browser)
 */
async function loadTokenizerFromFiles(vocabFile, mergesFile, tokenizerJsonFile = null) {
  const tokenizer = new Qwen2Tokenizer();
  
  const vocabText = await vocabFile.text();
  const mergesText = await mergesFile.text();
  
  await tokenizer.loadFromFiles(vocabText, mergesText);
  
  // Load additional special tokens from tokenizer.json if provided
  if (tokenizerJsonFile) {
    try {
      const tokenizerJson = JSON.parse(await tokenizerJsonFile.text());
      if (tokenizerJson.added_tokens) {
        for (const token of tokenizerJson.added_tokens) {
          if (token.special && token.content && token.id !== undefined) {
            tokenizer.specialTokens[token.content] = token.id;
            tokenizer.specialTokensReverse[token.id] = token.content;
          }
        }
        console.log(`Loaded ${tokenizerJson.added_tokens.length} special tokens from tokenizer.json`);
      }
    } catch (e) {
      console.warn('Could not parse tokenizer.json:', e.message);
    }
  }
  
  return tokenizer;
}

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { Qwen2Tokenizer, loadTokenizerFromFiles, QWEN2_SPECIAL_TOKENS };
}
if (typeof window !== 'undefined') {
  window.Qwen2Tokenizer = Qwen2Tokenizer;
  window.loadTokenizerFromFiles = loadTokenizerFromFiles;
  window.QWEN2_SPECIAL_TOKENS = QWEN2_SPECIAL_TOKENS;
}
