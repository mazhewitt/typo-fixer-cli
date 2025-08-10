# CoreML Tensor Shape Mismatch Analysis

## Problem Summary

The typo-fixer-cli project experiences CoreML tensor shape mismatches when running tests with the `-- --ignored` flag. The core error message is:

```
MultiArray shape (64) does not match the shape (1) specified in the model description
```

## Root Cause Analysis

### 1. **Model Configuration vs Actual Model Files**

The built-in configuration expects different tensor shapes than what the actual CoreML model files require:

**Configuration Expectations (builtin_configs.rs:208-209):**
```json
"position_ids": {
  "shape": [64],  // Expects 64 elements
  "data_type": "INT32"
}
```

**Actual Model File Requirements:**
- The CoreML model file actually expects `[1]` shape for position_ids during prefill operations
- This creates a fundamental mismatch between configuration and reality

### 2. **Pattern Matching vs Explicit Paths Issue**

**Original Dangerous Pattern Matching:**
```json
"ffn_infer": {
  "file_pattern": "qwen-typo-fixer_FFN_PF_lut*"  // WRONG: Points to prefill file
}
```

**Fixed with Explicit Paths:**
```json
"ffn_infer": {
  "file_path": "qwen-typo-fixer_FFN_lut4_chunk_01of01.mlpackage"  // CORRECT: Points to actual infer file
}
```

## Model Files Architecture

### Model File Locations

The typo-fixer model consists of multiple CoreML model files located at:
```
/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane/
```

### Model File Breakdown

The Qwen model is split into **4 separate CoreML model files**, each handling a specific part of the inference pipeline:

#### 1. **Embeddings Model**
- **File:** `qwen-typo-fixer_embeddings.mlpackage`
- **Purpose:** Converts token IDs to hidden state embeddings
- **Input:** `input_ids` with shape `[1, 64]`
- **Output:** `hidden_states` with shape `[1, 64, 1024]`

#### 2. **FFN Prefill Model** 
- **File:** `qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage`
- **Purpose:** Process full sequence to populate KV cache (attention states)
- **Function:** `prefill`
- **Inputs:**
  - `hidden_states`: `[1, 64, 1024]`
  - `position_ids`: `[64]` ← **MISMATCH: Model expects `[1]`**
  - `causal_mask`: `[1, 1, 64, 256]` 
  - `current_pos`: `[1]`
- **Output:** `output_hidden_states` `[1, 1, 1024]`

#### 3. **FFN Infer Model**
- **File:** `qwen-typo-fixer_FFN_lut4_chunk_01of01.mlpackage` 
- **Purpose:** Generate next token using populated KV cache
- **Function:** `infer`
- **Inputs:**
  - `hidden_states`: `[1, 1, 1024]`
  - `position_ids`: `[1]` ← **Correct shape**
  - `causal_mask`: `[1, 1, 1, 256]`
  - `current_pos`: `[1]`
- **Output:** `output_hidden_states` `[1, 1, 1024]`

#### 4. **LM Head Model**
- **File:** `qwen-typo-fixer_lm_head_lut6.mlpackage`
- **Purpose:** Convert hidden states to vocabulary logits for token selection
- **Input:** `hidden_states` with shape `[1, 1, 1024]`
- **Output:** `logits1` with shape `[1, 1, 9496]` (vocabulary size)

### Pipeline Flow

```
Input Text → Tokenizer → [Token IDs]
    ↓
[Token IDs] → Embeddings Model → [Hidden States] 
    ↓
[Hidden States] → FFN Prefill Model → [Output Hidden States] + KV Cache
    ↓ (for next token generation)
[Token Embedding] → FFN Infer Model → [Hidden States] (using KV Cache)
    ↓
[Hidden States] → LM Head Model → [Logits] → Next Token
```

### The Split Architecture Reason

The model is split into separate files because:

1. **Memory Optimization** - Each component can be loaded/unloaded as needed
2. **Function Specialization** - Prefill vs Infer have different computational patterns
3. **CoreML Limitations** - Large models may exceed single-file size limits
4. **Caching Strategy** - Prefill populates KV cache, Infer reuses it efficiently

## Technical Details

### Shape Expectations by Phase

#### Prefill Phase
- **Code Creates:** `position_ids` with shape `[64]` (based on batch_size)
- **Model Expects:** `position_ids` with shape `[1]` (single position)
- **Result:** Shape mismatch error during prefill

#### Infer Phase  
- **Code Creates:** `position_ids` with shape `[1]` (single token)
- **Model Expects:** `position_ids` with shape `[1]` (matches)
- **Result:** Should work correctly

### Error Location

The shape mismatch occurs specifically during the **prefill phase** when:

1. Code tokenizes input text (e.g., "a" → `[tokens]`)
2. Code creates position_ids tensor with shape `[64]` based on batch_size configuration
3. CoreML prefill model expects position_ids with shape `[1]`
4. **MISMATCH:** `[64]` provided vs `[1]` expected

## Files Affected

### Configuration Files
- `/Users/mazdahewitt/projects/candle-coreml/src/builtin_configs.rs:208-209`
  - Contains the mismatch between expected `[64]` and actual `[1]` shapes

### Model Loading Logic
- `/Users/mazdahewitt/projects/candle-coreml/src/qwen/model.rs:260-340`
  - Updated to prefer explicit file paths over dangerous pattern matching
  - Now safely loads correct model files for prefill vs infer operations

### Test Files
- `/Users/mazdahewitt/projects/typo-fixer-cli/tests/debug_model_issue.rs`
- `/Users/mazdahewitt/projects/typo-fixer-cli/tests/isolate_shape_mismatch_test.rs`  
- `/Users/mazdahewitt/projects/typo-fixer-cli/tests/tensor_validation_test.rs`

## Solutions Implemented

### 1. **Explicit File Paths (COMPLETED)**
Replaced dangerous glob patterns with explicit file paths to ensure correct model files are loaded:

```json
{
  "ffn_prefill": {
    "file_path": "qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage"
  },
  "ffn_infer": {  
    "file_path": "qwen-typo-fixer_FFN_lut4_chunk_01of01.mlpackage"
  }
}
```

### 2. **Model Loading Safety (COMPLETED)**
Updated model loading logic to prefer explicit paths over pattern matching:

```rust
if let Some(file_path) = &embeddings_component.file_path {
    // SAFE: Use explicit file path - no pattern matching risk
    actual_model_dir.join(file_path)
} else if let Some(pattern) = &embeddings_component.file_pattern {
    // FALLBACK: Use pattern matching (legacy support)
    Self::find_model_file_with_pattern(actual_model_dir, pattern)?
}
```

## Outstanding Issues

### 1. **Shape Configuration Mismatch (PENDING)**
The built-in configuration still specifies incorrect shapes that don't match the actual CoreML model requirements:

**Current Config:**
```json
"position_ids": {
  "shape": [64]  // Incorrect
}
```

**Should Be:**
```json  
"position_ids": {
  "shape": [1]   // Matches actual model
}
```

### 2. **Batch Size vs Single Token Confusion**
The configuration conflates "batch processing" with "model input shape":

- **batch_size: 64** ≠ **input tensor shape [64]**
- CoreML models may process batches internally but expect single-token inputs
- The prefill model appears to expect `[1]` shape regardless of batch processing

## Recommendations

### Immediate Fixes Needed

1. **Update Built-in Config Shapes**
   ```diff
   "position_ids": {
   -  "shape": [64],
   +  "shape": [1],
     "data_type": "INT32"  
   }
   ```

2. **Verify All Shape Configurations**
   - Run tensor validation tests to confirm what shapes the actual model files expect
   - Update configuration to match reality, not assumptions

3. **Add Shape Discovery**
   - Implement runtime shape discovery from actual CoreML model files
   - Use discovered shapes instead of hardcoded configuration values

### Testing Strategy

1. **Tensor Validation Tests** - Validate what shapes models actually expect
2. **Shape Discovery Tests** - Test runtime detection of model requirements  
3. **Integration Tests** - Ensure full pipeline works with corrected shapes

## Key Insights

1. **Configuration ≠ Reality** - Built-in configs contained assumptions that didn't match actual model files
2. **Pattern Matching is Dangerous** - Glob patterns loaded wrong files; explicit paths are safer
3. **Prefill vs Infer Distinction** - Different model files have different shape requirements
4. **Batch Size Confusion** - batch_size config parameter doesn't directly translate to tensor shapes

## Status

- ✅ **Fixed:** Explicit file paths prevent wrong model loading
- ✅ **Fixed:** Model loading safety with path preference  
- ⚠️  **Pending:** Shape configuration corrections
- ⚠️  **Pending:** Runtime shape discovery implementation
- ❓ **Testing:** Tensor validation tests need completion

---

*Generated during investigation of CoreML tensor shape mismatches in typo-fixer-cli project*