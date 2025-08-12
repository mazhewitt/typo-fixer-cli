# Model Configuration Files

This directory contains validated model configuration files for use with the typo-fixer CLI. These configurations define the shapes, components, and file paths needed to load multi-component CoreML models with candle-coreml.

## Working Configurations

- `qwen-typo-fixer-ane.json` - **Production-ready** configuration for the fine-tuned typo-fixer model with single-token sequential architecture

## Usage Examples

### 1. Using Explicit Configuration (Recommended)
```bash
./target/release/typo-fixer-cli \
  --local-path "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane" \
  --config "./configs/qwen-typo-fixer-ane.json" \
  --verbose "helo wrold, how ar you?"
```

### 2. Auto-Detection (Model Directory with config.json)
```bash
./target/release/typo-fixer-cli \
  --local-path "/path/to/model" \
  --verbose "fix this tyop"
```

### 3. Performance Tuning
```bash
# Deterministic generation
./target/release/typo-fixer-cli \
  --local-path "/path/to/model" \
  --config "./configs/qwen-typo-fixer-ane.json" \
  --temperature 0.0 --max-tokens 10 "test"

# Creative generation
./target/release/typo-fixer-cli \
  --local-path "/path/to/model" \
  --temperature 0.5 --max-tokens 25 "longer text here"
```

## Generating New Configurations

Use the discovery tool to generate configurations for your models:

```bash
python3 tools/discover_shapes.py --model-dir /path/to/your/model --output configs/your-model.json --verbose
```

## Configuration Architecture

### Single-Token Sequential Models
The `qwen-typo-fixer-ane.json` demonstrates a **single-token sequential architecture**:

- **Embeddings**: `[1, 1] -> [1, 1, 1024]` - Single token input/output
- **FFN Prefill**: Sequential token-by-token processing for context building
- **FFN Infer**: Optimized single-token inference with state management  
- **LM Head**: Multi-part logits (16 outputs) for large vocabulary handling

### Configuration Structure

```json
{
  "model_info": {
    "path": "/path/to/model",
    "model_type": "qwen", 
    "discovered_at": "ISO timestamp"
  },
  "shapes": {
    "batch_size": 1,              // Single batch processing
    "context_length": 256,        // Maximum context window
    "hidden_size": 1024,         // Model dimension
    "vocab_size": 151669         // Qwen vocabulary size
  },
  "components": {
    "embeddings": {
      "file_path": "path/to/embeddings.mlpackage",
      "inputs": {"input_ids": {"shape": [1, 1], "data_type": "INT32"}},
      "outputs": {"hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"}}
    },
    "ffn_prefill": {
      "file_path": "path/to/ffn_prefill.mlpackage", 
      "inputs": {
        "hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"},
        "position_ids": {"shape": [1], "data_type": "INT32"},
        "causal_mask": {"shape": [1, 1, 1, 256], "data_type": "FLOAT16"},
        "current_pos": {"shape": [1], "data_type": "INT32"}
      },
      "outputs": {"output_hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"}}
    },
    "ffn_infer": {
      "file_path": "path/to/ffn_infer.mlpackage",
      "inputs": {
        "hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"},
        "position_ids": {"shape": [1], "data_type": "INT32"},
        "update_mask": {"shape": [1, 1, 256, 1], "data_type": "FLOAT16"},
        "causal_mask": {"shape": [1, 1, 1, 256], "data_type": "FLOAT16"},
        "current_pos": {"shape": [1], "data_type": "INT32"}
      },
      "outputs": {"output_hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"}}
    },
    "lm_head": {
      "file_path": "path/to/lm_head.mlpackage",
      "inputs": {"hidden_states": {"shape": [1, 1, 1024], "data_type": "FLOAT16"}},
      "outputs": {
        "logits1": {"shape": [1, 1, 9480], "data_type": "FLOAT16"},
        "logits2": {"shape": [1, 1, 9480], "data_type": "FLOAT16"},
        // ... up to logits16 for full vocabulary coverage
      }
    }
  },
  "naming": {
    "embeddings_pattern": "model_embeddings.mlpackage",
    "ffn_infer_pattern": "model_FFN_lut*_chunk_*.mlpackage", 
    "ffn_prefill_pattern": "model_FFN_PF_lut*_chunk_*.mlpackage",
    "lm_head_pattern": "model_lm_head_lut*.mlpackage"
  }
}
```

### Performance Implications

**Single-Token Architecture Benefits**:
- ✅ **Memory Efficiency**: Lower peak memory usage
- ✅ **CoreML Optimization**: Better ANE utilization
- ✅ **Precision**: Higher numerical precision with FLOAT16

**Trade-offs**:
- ⚠️ **Sequential Processing**: Token-by-token prefill is slower
- ⚠️ **Latency**: Higher time-to-first-token for longer contexts
- ⚠️ **Throughput**: Lower tokens/second compared to batch models