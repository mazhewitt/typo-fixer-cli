//! Targeted test to isolate and debug the CoreML shapes mismatch issue

use anyhow::Result;
use candle_coreml::{ConfigGenerator, QwenModel, QwenConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually to debug shapes issue
async fn test_isolate_shapes_issue() -> Result<()> {
    println!("🔍 Debugging CoreML shapes issue");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    println!("📁 Model path: {:?}", model_path);
    
    // Step 1: Check if model directory exists and list contents
    println!("\n📂 Step 1: Model directory analysis");
    if !model_path.exists() {
        return Err(anyhow::anyhow!("Model path does not exist: {:?}", model_path));
    }
    
    println!("📋 Contents of model directory:");
    if let Ok(entries) = std::fs::read_dir(&model_path) {
        for entry in entries {
            if let Ok(entry) = entry {
                let name = entry.file_name();
                let metadata = entry.metadata().unwrap_or_else(|_| panic!("Failed to get metadata"));
                println!("  - {} ({})", name.to_string_lossy(), if metadata.is_dir() { "dir" } else { "file" });
            }
        }
    }
    
    // Step 2: Generate config from local model directory
    println!("\n🔧 Step 2: Configuration analysis");
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        &model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    let qwen_config = QwenConfig::from_model_config(model_config);
    println!("   QwenConfig created from generated model config");
    test_model_loading_and_shapes(&model_path, qwen_config, "generated config").await?;
    
    // Try for_model_id
    println!("\n🔧 Trying QwenConfig::for_model_id");
    match QwenConfig::for_model_id("mazhewitt/qwen-typo-fixer") {
        Ok(config) => {
            println!("✅ Created config using for_model_id");
            test_model_loading_and_shapes(&model_path, config, "for_model_id config").await?;
        }
        Err(e) => {
            println!("❌ Failed to create config using for_model_id: {}", e);
        }
    }
    
    // Try default config
    println!("\n🔧 Trying QwenConfig::default");
    let default_config = QwenConfig::default();
    test_model_loading_and_shapes(&model_path, default_config, "default config").await?;
    
    Ok(())
}

async fn test_model_loading_and_shapes(
    model_path: &std::path::Path, 
    config: QwenConfig, 
    config_name: &str
) -> Result<()> {
    println!("\n🧪 Testing with {}", config_name);
    
    // Step 1: Try to load the model
    println!("  Loading model...");
    let model_result = QwenModel::load_from_directory(model_path, Some(config));
    
    match model_result {
        Ok(mut model) => {
            println!("  ✅ Model loaded successfully with {}", config_name);
            
            // Step 2: Try minimal generation to trigger the shapes issue
            println!("  🔬 Testing minimal generation...");
            
            // Try with the shortest possible input
            let minimal_inputs = vec![
                "a",           // Single character
                "test",        // Single word
                "hello world", // Two words
            ];
            
            for (i, input) in minimal_inputs.iter().enumerate() {
                println!("    Test {}: Input '{}' (length: {})", i+1, input, input.len());
                
                // Try with minimal token generation
                let result = model.generate_text(input, 1, 0.0);
                match result {
                    Ok(output) => {
                        println!("    ✅ Success! Output: '{}'", output);
                        return Ok(()); // If one succeeds, we're good
                    }
                    Err(e) => {
                        println!("    ❌ Failed: {}", e);
                        
                        // Check if it's the shapes error we're looking for
                        let error_string = e.to_string();
                        if error_string.contains("MultiArray shape") {
                            println!("    🎯 Found the shapes error!");
                            println!("    Error details: {}", error_string);
                            
                            // Try to extract shape information
                            if error_string.contains("shape (") {
                                let parts: Vec<&str> = error_string.split("shape (").collect();
                                if parts.len() > 1 {
                                    let shape_part = parts[1].split(")").next().unwrap_or("unknown");
                                    println!("    📐 Actual shape: ({})", shape_part);
                                }
                                if let Some(expected_part) = parts.get(2) {
                                    let expected_shape = expected_part.split("shape (").nth(1)
                                        .and_then(|s| s.split(")").next())
                                        .unwrap_or("unknown");
                                    println!("    📐 Expected shape: ({})", expected_shape);
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(e) => {
            println!("  ❌ Failed to load model with {}: {}", config_name, e);
        }
    }
    
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually to debug shapes issue
async fn test_compare_working_vs_broken_model() -> Result<()> {
    println!("🔍 Comparing working vs problematic model structures");
    
    // We know the typo-fixer model has issues, but let's see if we can find a working one
    // This test will help us understand what a working model looks like
    
    let problematic_model = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    
    println!("📁 Problematic model: {}", problematic_model);
    if std::path::Path::new(problematic_model).exists() {
        analyze_model_structure(problematic_model, "Problematic").await?;
    } else {
        println!("❌ Problematic model path does not exist");
    }
    
    // Try to find any other models in the system for comparison
    let cache_dir = "/Users/mazdahewitt/Library/Caches/candle-coreml";
    if std::path::Path::new(cache_dir).exists() {
        println!("\n📂 Checking cached models for comparison:");
        if let Ok(entries) = std::fs::read_dir(cache_dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let name = entry.file_name();
                    println!("  - {}", name.to_string_lossy());
                }
            }
        }
    }
    
    Ok(())
}

async fn analyze_model_structure(model_path: &str, label: &str) -> Result<()> {
    println!("\n🔬 Analyzing {} model structure", label);
    let path = std::path::Path::new(model_path);
    
    if !path.exists() {
        println!("❌ Path does not exist: {}", model_path);
        return Ok(());
    }
    
    // Recursively list all files to understand structure
    fn list_files_recursive(dir: &std::path::Path, prefix: &str) -> Result<()> {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries {
                if let Ok(entry) = entry {
                    let entry_path = entry.path();
                    let name = entry.file_name();
                    
                    if entry_path.is_dir() {
                        println!("{}📁 {}/", prefix, name.to_string_lossy());
                        list_files_recursive(&entry_path, &format!("{}  ", prefix))?;
                    } else {
                        let size = entry.metadata()
                            .map(|m| m.len())
                            .unwrap_or(0);
                        println!("{}📄 {} ({} bytes)", prefix, name.to_string_lossy(), size);
                        
                        // If it's a .mlpackage or .mlmodelc file, note it specifically
                        if name.to_string_lossy().ends_with(".mlpackage") || 
                           name.to_string_lossy().ends_with(".mlmodelc") {
                            println!("{}   🎯 CoreML model file!", prefix);
                        }
                    }
                }
            }
        }
        Ok(())
    }
    
    list_files_recursive(path, "")?;
    
    Ok(())
}