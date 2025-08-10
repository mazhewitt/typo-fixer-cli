use anyhow::Result;
use candle_coreml::{QwenModel, QwenConfig, model_downloader};
use std::error::Error;

#[tokio::test]
async fn test_debug_typo_fixer_model_loading() -> Result<()> {
    println!("🔍 Debugging typo-fixer model loading issue...");
    
    let model_id = "mazhewitt/qwen-typo-fixer";
    
    // Step 1: Download model
    println!("📥 Step 1: Downloading model...");
    let model_path = model_downloader::ensure_model_downloaded(model_id, true)?;
    println!("✅ Model path: {:?}", model_path);
    
    // Step 2: Check files exist
    println!("📁 Step 2: Checking model files...");
    let coreml_dir = model_path.join("coreml");
    if !coreml_dir.exists() {
        println!("❌ CoreML directory doesn't exist at {:?}", coreml_dir);
        return Ok(());
    }
    
    // List all files
    let entries = std::fs::read_dir(&coreml_dir)?;
    println!("📋 Files in CoreML directory:");
    for entry in entries {
        let entry = entry?;
        println!("   - {}", entry.file_name().to_string_lossy());
    }
    
    // Step 3: Try with different configs
    println!("🔧 Step 3: Testing different naming configurations...");
    
    // Try with typo-fixer config
    println!("   Testing with QwenConfig::for_model_id()...");
    let config1 = QwenConfig::for_model_id(model_id).unwrap_or_else(|_| QwenConfig::default());
    match QwenModel::load_from_directory(&model_path, Some(config1)) {
        Ok(_) => println!("   ✅ SUCCESS with for_typo_fixer config!"),
        Err(e) => {
            println!("   ❌ Failed with for_typo_fixer config:");
            println!("      Error: {}", e);
            
            // Print the full error chain
            let mut source = e.source();
            while let Some(err) = source {
                println!("      Caused by: {}", err);
                source = err.source();
            }
        }
    }
    
    // Try with default config  
    println!("   Testing with QwenConfig::default()...");
    let config2 = QwenConfig::default();
    match QwenModel::load_from_directory(&model_path, Some(config2)) {
        Ok(_) => println!("   ✅ SUCCESS with default config!"),
        Err(e) => {
            println!("   ❌ Failed with default config:");
            println!("      Error: {}", e);
        }
    }
    
    // Try with custom config that matches the exact file names
    println!("   Testing with custom config for exact file names...");
    let config3 = QwenConfig::with_custom_naming("qwen-typo-fixer_", ".mlpackage");
    match QwenModel::load_from_directory(&model_path, Some(config3)) {
        Ok(_) => println!("   ✅ SUCCESS with custom config!"),
        Err(e) => {
            println!("   ❌ Failed with custom config:");
            println!("      Error: {}", e);
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_compare_working_vs_broken_model_files() -> Result<()> {
    println!("🔍 Comparing working vs broken model file structures...");
    
    let working_model = "anemll/anemll-Qwen-Qwen3-0.6B-LUT888-ctx512_0.3.4";
    let broken_model = "mazhewitt/qwen-typo-fixer";
    
    // Download both models
    let working_path = model_downloader::ensure_model_downloaded(working_model, false)?;
    let broken_path = model_downloader::ensure_model_downloaded(broken_model, false)?;
    
    println!("📁 Working model files:");
    let working_coreml = working_path.join("coreml");
    if working_coreml.exists() {
        let entries = std::fs::read_dir(&working_coreml)?;
        for entry in entries {
            let entry = entry?;
            println!("   ✅ {}", entry.file_name().to_string_lossy());
        }
    } else {
        println!("   No coreml directory found");
    }
    
    println!("📁 Broken model files:");
    let broken_coreml = broken_path.join("coreml");
    if broken_coreml.exists() {
        let entries = std::fs::read_dir(&broken_coreml)?;
        for entry in entries {
            let entry = entry?;
            println!("   ❌ {}", entry.file_name().to_string_lossy());
        }
    } else {
        println!("   No coreml directory found");
    }
    
    // Try loading the working model to confirm it works
    println!("🧪 Testing working model load:");
    let working_config = QwenConfig::default();
    match QwenModel::load_from_directory(&working_path, Some(working_config)) {
        Ok(_) => println!("   ✅ Working model loads successfully"),
        Err(e) => println!("   ❌ Even working model failed: {}", e),
    }
    
    Ok(())
}