//! Debug test to understand tensor shape retrieval

use anyhow::Result;
use candle_coreml::{ConfigGenerator, QwenConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] 
async fn debug_tensor_shapes() -> Result<()> {
    println!("🔍 Debugging tensor shape retrieval for typo-fixer model");

    // Allow overriding the local model path via env; otherwise use the legacy default.
    let working_model_path = std::env::var("TYPO_FIXER_LOCAL_MODEL")
        .unwrap_or_else(|_| "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane".to_string());
    let model_path = std::path::Path::new(&working_model_path);
    if !model_path.exists() {
        println!(
            "⚠️  Local model directory not found: {}\n   Set TYPO_FIXER_LOCAL_MODEL to a valid path or skip this ignored test.",
            working_model_path
        );
        // Gracefully skip instead of failing hard when fixture isn't present
        return Ok(());
    }
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
        
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