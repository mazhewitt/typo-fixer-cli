//! Tests for model loading and integration with candle-coreml

use anyhow::Result;
use tokio;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Ignore by default since this downloads a large model
async fn test_load_typo_fixer_model_from_hub() -> Result<()> {
    use candle_coreml::model_downloader;
    
    // Test loading the actual fine-tuned typo fixer model
    let model_id = "mazhewitt/qwen-typo-fixer";
    
    println!("🔄 Testing model download from HuggingFace Hub: {}", model_id);
    
    // First, test that we can download the model
    let model_path = model_downloader::ensure_model_downloaded(model_id, true)
        .map_err(|e| anyhow::anyhow!("Failed to download model: {}", e))?;
    
    println!("✅ Model downloaded to: {:?}", model_path);
    
    // Check that the expected CoreML components exist
    let coreml_dir = model_path.join("coreml");
    assert!(coreml_dir.exists(), "CoreML directory should exist");
    
    // Check for the expected model files based on HuggingFace repo structure
    let expected_files = [
        "qwen-typo-fixer_embeddings.mlpackage",
        "qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage", 
        "qwen-typo-fixer_lm_head_lut6.mlpackage",
    ];
    
    for file in &expected_files {
        let file_path = coreml_dir.join(file);
        assert!(
            file_path.exists(), 
            "Expected CoreML file should exist: {:?}", 
            file_path
        );
        println!("✅ Found expected file: {}", file);
    }
    
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test] 
#[ignore] // Ignore by default since this requires model download
async fn test_load_qwen_model_from_downloaded_components() -> Result<()> {
    use candle_coreml::{QwenModel, QwenConfig, model_downloader};
    
    let model_id = "mazhewitt/qwen-typo-fixer";
    
    println!("🔄 Testing QwenModel loading from downloaded components");
    
    // Ensure model is downloaded
    let model_path = model_downloader::ensure_model_downloaded(model_id, true)?;
    
    // Try to load the QwenModel from the downloaded components
    let config = QwenConfig::default();
    
    let model_result = QwenModel::load_from_directory(&model_path, Some(config));
    
    match model_result {
        Ok(model) => {
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