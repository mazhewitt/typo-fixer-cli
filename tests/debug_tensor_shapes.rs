//! Debug test to understand tensor shape retrieval

use anyhow::Result;
use candle_coreml::{get_builtin_config, QwenConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] 
async fn debug_tensor_shapes() -> Result<()> {
    println!("🔍 Debugging tensor shape retrieval for typo-fixer model");
    
    let model_config = get_builtin_config("mazhewitt/qwen-typo-fixer")
        .ok_or_else(|| anyhow::anyhow!("No built-in config found"))?;
        
    println!("📊 Model config components:");
    for (name, component) in &model_config.components {
        println!("  Component: {}", name);
        for (input_name, tensor_config) in &component.inputs {
            println!("    Input '{}': shape {:?}", input_name, tensor_config.shape);
        }
    }
    
    let qwen_config = QwenConfig::from_model_config(model_config);
    
    // Test the exact logic from create_position_ids_with_mode_detection
    println!("\n🧪 Testing mode detection logic:");
    
    // Check if ffn_infer component exists
    let has_ffn_infer = qwen_config.model_config.components.contains_key("ffn_infer");
    println!("Has ffn_infer component: {}", has_ffn_infer);
    
    if has_ffn_infer {
        // Check the tensor shape for ffn_infer position_ids
        let infer_shape = qwen_config
            .model_config
            .get_tensor_shape("ffn_infer", "position_ids", true);
            
        match infer_shape {
            Some(shape) => {
                println!("ffn_infer position_ids shape: {:?}", shape);
                println!("shape[0] = {}", shape[0]);
                println!("shape[0] == 1: {}", shape[0] == 1);
                
                if shape[0] == 1 {
                    println!("✅ Should use create_infer_position_ids_tensor");
                } else {
                    println!("❌ Will fall back to create_ffn_position_ids_tensor with shape: {:?}", shape);
                }
            }
            None => {
                println!("❌ No shape found for ffn_infer position_ids");
                println!("Will fall back to create_ffn_position_ids_tensor");
            }
        }
    }
    
    // Also check ffn_prefill for comparison
    let prefill_shape = qwen_config
        .model_config
        .get_tensor_shape("ffn_prefill", "position_ids", true);
        
    match prefill_shape {
        Some(shape) => {
            println!("ffn_prefill position_ids shape: {:?}", shape);
        }
        None => {
            println!("❌ No shape found for ffn_prefill position_ids");
        }
    }
    
    Ok(())
}