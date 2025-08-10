# Model Configuration Files

This directory contains model configuration files for use with the typo-fixer CLI. These configurations define the shapes, components, and file paths needed to load CoreML models.

## Generated Configurations

- `qwen-typo-fixer-ane.json` - Configuration for the typo-fixer model, automatically generated from the actual model files

## How to Use

1. **With configuration file:**
   ```bash
   ./target/release/typo-fixer-cli --local-path /path/to/model --config configs/qwen-typo-fixer-ane.json "fix this tyop"
   ```

2. **Auto-detection (if model has config.json):**
   ```bash
   ./target/release/typo-fixer-cli --local-path /path/to/model "fix this tyop"
   ```

3. **Default configuration:**
   ```bash
   ./target/release/typo-fixer-cli --model mazhewitt/qwen-typo-fixer "fix this tyop"
   ```

## Generating New Configurations

Use the discovery tool to generate configurations for your models:

```bash
python3 tools/discover_shapes.py --model-dir /path/to/your/model --output configs/your-model.json --verbose
```

## Configuration Format

Configuration files follow the candle-coreml ModelConfig format:

```json
{
  "model_info": {
    "path": "/path/to/model",
    "model_type": "qwen",
    "discovered_at": "2025-08-09T23:09:34.776656"
  },
  "shapes": {
    "batch_size": 1,
    "context_length": 256,
    "hidden_size": 1024,
    "vocab_size": 151669
  },
  "components": {
    "embeddings": { ... },
    "ffn_infer": { ... },
    "ffn_prefill": { ... },
    "lm_head": { ... }
  },
  "naming": {
    "embeddings_pattern": "...",
    "ffn_infer_pattern": "...",
    "lm_head_pattern": "..."
  }
}
```