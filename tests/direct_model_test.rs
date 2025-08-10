//! Direct test of individual CoreML model components

use anyhow::Result;
use candle_coreml::{CoreMLModel, Config as CoreMLConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] 
async fn test_individual_model_components() -> Result<()> {
    println!("🔍 Testing individual CoreML model components directly");
    
    let model_dir = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    
    // Test 1: Try to load the infer model directly
    println!("\n🧪 Test 1: Loading FFN infer model directly");
    let infer_path = std::path::Path::new(model_dir).join("qwen-typo-fixer_FFN_lut4_chunk_01of01.mlpackage");
    
    if infer_path.exists() {
        println!("✅ Infer model file exists: {:?}", infer_path);
        
        let infer_config = CoreMLConfig {
            input_names: vec![
                "hidden_states".to_string(),
                "position_ids".to_string(),
                "causal_mask".to_string(),
                "current_pos".to_string(),
            ],
            output_name: "output_hidden_states".to_string(),
            max_sequence_length: 1, // Single token for inference
            vocab_size: 1024,
            model_type: "qwen-ffn-infer".to_string(),
        };
        
        match CoreMLModel::load_with_function(&infer_path, &infer_config, "infer") {
            Ok(_model) => {
                println!("✅ FFN infer model loaded successfully");
            }
            Err(e) => {
                println!("❌ Failed to load FFN infer model: {}", e);
            }
        }
    } else {
        println!("❌ Infer model file does not exist");
    }
    
    // Test 2: Try to load the prefill model directly
    println!("\n🧪 Test 2: Loading FFN prefill model directly");
    let prefill_path = std::path::Path::new(model_dir).join("qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage");
    
    if prefill_path.exists() {
        println!("✅ Prefill model file exists: {:?}", prefill_path);
        
        let prefill_config = CoreMLConfig {
            input_names: vec![
                "hidden_states".to_string(),
                "position_ids".to_string(),
                "causal_mask".to_string(),
                "current_pos".to_string(),
            ],
            output_name: "output_hidden_states".to_string(),
            max_sequence_length: 64, // Batch processing
            vocab_size: 1024,
            model_type: "qwen-ffn-prefill".to_string(),
        };
        
        match CoreMLModel::load_with_function(&prefill_path, &prefill_config, "prefill") {
            Ok(_model) => {
                println!("✅ FFN prefill model loaded successfully");
            }
            Err(e) => {
                println!("❌ Failed to load FFN prefill model: {}", e);
            }
        }
    } else {
        println!("❌ Prefill model file does not exist");
    }
    
    // Test 3: Check if the issue is with the "function" approach
    println!("\n🧪 Test 3: Loading models without specific functions");
    
    match CoreMLModel::load(&infer_path) {
        Ok(_) => println!("✅ Infer model loads without function specification"),
        Err(e) => println!("❌ Infer model fails without function: {}", e),
    }
    
    Ok(())
}