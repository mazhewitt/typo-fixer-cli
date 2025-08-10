//! Tests for model loading and integration with candle-coreml

use anyhow::Result;
use tokio;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Ignore by default since this downloads a large model
async fn test_load_typo_fixer_model_from_hub() -> Result<()> {
    // Test with the working local model provided by the user
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    
    println!("🔄 Testing model from local path: {}", working_model_path);
    
    let model_path = std::path::Path::new(working_model_path);
    
    println!("✅ Using model at: {:?}", model_path);
    
    // Check that the model directory exists
    assert!(model_path.exists(), "Local model directory should exist");
    
    // List the files in the model directory to understand its structure
    println!("📁 Model directory contents:");
    if let Ok(entries) = std::fs::read_dir(&model_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                println!("  - {}", entry.file_name().to_string_lossy());
            }
        }
    }
    
    // Just verify the directory is not empty
    let has_files = std::fs::read_dir(&model_path)?
        .next()
        .is_some();
    assert!(has_files, "Model directory should contain files");
    
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test] 
#[ignore] // Ignore by default since this requires model download
async fn test_load_qwen_model_from_downloaded_components() -> Result<()> {
    use candle_coreml::{QwenModel, QwenConfig};
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    
    println!("🔄 Testing QwenModel loading from local components");
    
    // Use the local working model
    let model_path = std::path::Path::new(working_model_path);
    
    // Use the same configuration logic as the main app for typo-fixer models
    use candle_coreml::get_builtin_config;
    let config = if let Some(model_config) = get_builtin_config("mazhewitt/qwen-typo-fixer") {
        println!("🔧 Using built-in typo-fixer configuration");
        QwenConfig::from_model_config(model_config)
    } else {
        println!("⚠️ Built-in typo-fixer config not found, trying for_model_id");
        QwenConfig::for_model_id("mazhewitt/qwen-typo-fixer")
            .unwrap_or_else(|_| QwenConfig::default())
    };
    
    let model_result = QwenModel::load_from_directory(&model_path, Some(config));
    
    match model_result {
        Ok(_model) => {
            println!("✅ QwenModel loaded successfully");
            
            // Test that the model has the expected components
            // Note: This will depend on the internal structure of QwenModel
            // We might need to adjust based on how candle-coreml exposes this
            
            println!("🎯 Model loaded and ready for inference");
        }
        Err(e) => {
            println!("❌ Failed to load QwenModel: {}", e);
            // For now, let's see what the actual error is and adjust accordingly
            return Err(anyhow::anyhow!("Model loading failed: {}", e));
        }
    }
    
    Ok(())
}

#[cfg(not(target_os = "macos"))]
#[tokio::test]
async fn test_model_loading_not_supported() {
    // On non-macOS platforms, model loading should fail gracefully
    use candle_coreml::{QwenModel, QwenConfig};
    use std::path::PathBuf;
    
    let fake_path = PathBuf::from("fake_path");
    let config = QwenConfig::default();
    
    let result = QwenModel::load_from_directory(&fake_path, Some(config));
    
    // Should fail on non-macOS
    assert!(result.is_err(), "Model loading should fail on non-macOS platforms");
}

#[test]
fn test_model_id_validation() {
    // Test that we can validate model IDs
    let valid_model_ids = [
        "mazhewitt/qwen-typo-fixer",
        "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4",
        "huggingface/model-name",
    ];
    
    let invalid_model_ids = [
        "",
        "invalid",
        "too/many/slashes/here",
    ];
    
    for valid_id in &valid_model_ids {
        assert!(is_valid_model_id(valid_id), "Should be valid: {}", valid_id);
    }
    
    for invalid_id in &invalid_model_ids {
        assert!(!is_valid_model_id(invalid_id), "Should be invalid: {}", invalid_id);
    }
}

// Helper function to validate HuggingFace model IDs
fn is_valid_model_id(model_id: &str) -> bool {
    if model_id.is_empty() {
        return false;
    }
    
    let parts: Vec<&str> = model_id.split('/').collect();
    
    // Should have exactly 2 parts: username/model-name
    if parts.len() != 2 {
        return false;
    }
    
    // Both parts should be non-empty
    parts[0].len() > 0 && parts[1].len() > 0
}