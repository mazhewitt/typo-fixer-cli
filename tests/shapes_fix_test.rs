//! Test to implement and validate the fix for CoreML shapes issue

use anyhow::Result;
use candle_coreml::{ConfigGenerator, QwenModel, QwenConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually to test the fix
async fn test_shapes_fix_approach() -> Result<()> {
    println!("🔧 Testing potential fix for CoreML shapes issue");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    // Generate configuration from local directory
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    
    println!("📊 Model config analysis:");
    println!("  Batch size: {}", model_config.shapes.batch_size);
    println!("  Context length: {}", model_config.shapes.context_length);
    
    // Create config from the built-in model config
    let config = QwenConfig::from_model_config(model_config);
    println!("✅ Created QwenConfig from built-in model config");
    
    // Load the model
    let mut model = QwenModel::load_from_directory(model_path, Some(config))?;
    println!("✅ Model loaded successfully");
    
    // The issue appears to be that we're calling generate_text with inputs that are too long
    // for the inference mode, causing it to use prefill mode dimensions
    
    println!("\n🧪 Testing different approaches:");
    
    // Approach 1: Try with exactly 1 token input
    println!("\nApproach 1: Single token generation");
    test_single_token_generation(&mut model).await?;
    
    // Approach 2: Try with shorter context
    println!("\nApproach 2: Shorter context generation");
    test_short_context_generation(&mut model).await?;
    
    // Approach 3: Try with modified batch size
    println!("\nApproach 3: Modified configuration");
    test_modified_config_generation(model_path).await?;
    
    Ok(())
}

async fn test_single_token_generation(model: &mut QwenModel) -> Result<()> {
    println!("  Testing single token generation...");
    
    // Try generating just 1 token from minimal input
    let inputs = vec!["a", "the", "yes"];
    
    for input in inputs {
        println!("    Input: '{}'", input);
        match model.generate_text(input, 1, 0.0) {
            Ok(output) => {
                println!("    ✅ Success! Output: '{}'", output);
                return Ok(());
            }
            Err(e) => {
                println!("    ❌ Failed: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn test_short_context_generation(model: &mut QwenModel) -> Result<()> {
    println!("  Testing with very short generation limits...");
    
    let inputs = vec![
        ("a", 1),
        ("test", 1),
        ("hello", 2),
    ];
    
    for (input, max_tokens) in inputs {
        println!("    Input: '{}', max_tokens: {}", input, max_tokens);
        match model.generate_text(input, max_tokens, 0.0) {
            Ok(output) => {
                println!("    ✅ Success! Output: '{}'", output);
                return Ok(());
            }
            Err(e) => {
                println!("    ❌ Failed: {}", e);
            }
        }
    }
    
    Ok(())
}

async fn test_modified_config_generation(model_path: &std::path::Path) -> Result<()> {
    println!("  Testing with modified configuration...");
    
    // Get the original config via generator
    let generator = ConfigGenerator::new()?;
    let original_config = generator.generate_config_from_directory(
        model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    
    println!("    Original batch_size: {}", original_config.shapes.batch_size);
    
    // Try to create a config that might work better
    // The issue might be that batch_size=64 is causing problems
    
    // For now, let's try using for_model_id which might have different defaults
    match QwenConfig::for_model_id("mazhewitt/qwen-typo-fixer") {
        Ok(config) => {
            println!("    Trying for_model_id config...");
            let mut model = QwenModel::load_from_directory(model_path, Some(config))?;
            
            match model.generate_text("test", 1, 0.0) {
                Ok(output) => {
                    println!("    ✅ Success with for_model_id! Output: '{}'", output);
                }
                Err(e) => {
                    println!("    ❌ for_model_id also failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("    ❌ Could not create for_model_id config: {}", e);
        }
    }
    
    // Try with a completely custom config
    println!("    Trying custom config approach...");
    let naming = candle_coreml::ModelNamingConfig::custom("qwen-typo-fixer", "mlpackage");
    let custom_config = QwenConfig::default().with_naming(naming);
    
    match QwenModel::load_from_directory(model_path, Some(custom_config)) {
        Ok(mut model) => {
            match model.generate_text("test", 1, 0.0) {
                Ok(output) => {
                    println!("    ✅ Success with custom config! Output: '{}'", output);
                }
                Err(e) => {
                    println!("    ❌ Custom config also failed: {}", e);
                }
            }
        }
        Err(e) => {
            println!("    ❌ Could not load with custom config: {}", e);
        }
    }
    
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]  
#[ignore] // Run manually to test potential workaround
async fn test_potential_workaround() -> Result<()> {
    println!("🔄 Testing potential workaround approaches");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    // The core issue might be that the model expects different tensor shapes
    // for different operations. Let's see if we can work around this by:
    
    // 1. Using a working model from cache for comparison
    println!("\n📂 Trying cached working models for comparison:");
    
    let cache_models = vec![
        "/Users/mazdahewitt/Library/Caches/candle-coreml/clean-anemll--anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4",
        "/Users/mazdahewitt/Library/Caches/candle-coreml/clean-anemll--anemll-Qwen-Qwen3-0.6B-LUT888-ctx512_0.3.4",
    ];
    
    for cached_model_path in cache_models {
        let cached_path = std::path::Path::new(cached_model_path);
        if cached_path.exists() {
            println!("  Testing cached model: {}", cached_model_path);
            
            // Try with default config for this model
            let config = QwenConfig::default();
            match QwenModel::load_from_directory(cached_path, Some(config)) {
                Ok(mut model) => {
                    println!("    ✅ Cached model loaded successfully");
                    
                    match model.generate_text("hello", 5, 0.0) {
                        Ok(output) => {
                            println!("    ✅ Cached model generation works! Output: '{}'", output);
                            println!("    🎯 This proves the candle-coreml library works with compatible models");
                            
                            // Now let's see what's different about our problematic model
                            return analyze_model_differences(cached_path, model_path).await;
                        }
                        Err(e) => {
                            println!("    ❌ Cached model generation failed: {}", e);
                        }
                    }
                }
                Err(e) => {
                    println!("    ❌ Could not load cached model: {}", e);
                }
            }
        } else {
            println!("  Cached model not found: {}", cached_model_path);
        }
    }
    
    Ok(())
}

async fn analyze_model_differences(working_model: &std::path::Path, problematic_model: &std::path::Path) -> Result<()> {
    println!("\n🔍 Analyzing differences between working and problematic models");
    
    println!("Working model: {:?}", working_model);
    println!("Problematic model: {:?}", problematic_model);
    
    // Compare file structures
    println!("\n📂 File structure comparison:");
    compare_directory_structure(working_model, "WORKING")?;
    compare_directory_structure(problematic_model, "PROBLEMATIC")?;
    
    Ok(())
}

fn compare_directory_structure(path: &std::path::Path, label: &str) -> Result<()> {
    println!("\n{} model structure:", label);
    
    if let Ok(entries) = std::fs::read_dir(path) {
        let mut files: Vec<_> = entries.filter_map(|e| e.ok()).collect();
        files.sort_by_key(|e| e.file_name());
        
        for entry in files {
            let name = entry.file_name();
            let file_name = name.to_string_lossy();
            
            if entry.path().is_dir() {
                println!("  📁 {}/", file_name);
            } else {
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                println!("  📄 {} ({} bytes)", file_name, size);
            }
        }
    }
    
    Ok(())
}