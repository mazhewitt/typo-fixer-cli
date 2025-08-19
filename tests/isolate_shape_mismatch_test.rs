//! Precise test to isolate exactly where and why the shape mismatch occurs

use anyhow::Result;
use candle_coreml::{ConfigGenerator, QwenModel, QwenConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] 
async fn isolate_exact_shape_mismatch() -> Result<()> {
    println!("🔍 ISOLATION TEST: Finding exact location of shape mismatch");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    // Generate config from the local directory
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    let config = QwenConfig::from_model_config(model_config);
    
    let mut model = QwenModel::load_from_directory(model_path, Some(config))?;
    println!("✅ Model loaded successfully");
    
    // ISOLATION STEP 1: Test the minimal possible input
    println!("\n🧪 STEP 1: Testing minimal tokenization");
    let test_text = "a";
    let tokens = model.tokenize(test_text)?;
    println!("✅ Tokenized '{}' to {:?} (length: {})", test_text, tokens, tokens.len());
    
    // ISOLATION STEP 2: Test state initialization
    println!("\n🧪 STEP 2: Testing state initialization");
    if model.unified_state.is_none() || model.cached_causal_mask.is_none() {
        match model.initialize_states() {
            Ok(_) => println!("✅ States initialized successfully"),
            Err(e) => {
                println!("❌ State initialization failed: {}", e);
                return Ok(());
            }
        }
    }
    
    // ISOLATION STEP 3: Test embeddings computation
    println!("\n🧪 STEP 3: Testing embeddings computation");
    match model.compute_embeddings(&tokens) {
        Ok(embeddings) => {
            println!("✅ Embeddings computed successfully");
            println!("   Embeddings shape: {:?}", embeddings.shape());
        }
        Err(e) => {
            println!("❌ Embeddings computation failed: {}", e);
            if e.to_string().contains("MultiArray shape") {
                println!("🎯 FOUND IT: Shape mismatch occurs during embeddings computation!");
            }
            return Ok(());
        }
    }
    
    // ISOLATION STEP 4: Test prefill phase
    println!("\n🧪 STEP 4: Testing prefill phase");
    let context_pos = tokens.len();
    match model.run_chatpy_prefill(&tokens, context_pos) {
        Ok(_) => println!("✅ Prefill completed successfully"),
        Err(e) => {
            println!("❌ Prefill failed: {}", e);
            if e.to_string().contains("MultiArray shape") {
                println!("🎯 FOUND IT: Shape mismatch occurs during prefill phase!");
                analyze_shape_error(&e.to_string());
            }
            return Ok(());
        }
    }
    
    // ISOLATION STEP 5: Test infer phase
    println!("\n🧪 STEP 5: Testing infer phase");
    match model.run_chatpy_infer(&tokens, context_pos) {
        Ok(token) => {
            println!("✅ Infer completed successfully!");
            println!("   Generated token: {}", token);
        }
        Err(e) => {
            println!("❌ Infer failed: {}", e);
            if e.to_string().contains("MultiArray shape") {
                println!("🎯 FOUND IT: Shape mismatch occurs during infer phase!");
                analyze_shape_error(&e.to_string());
            }
            return Ok(());
        }
    }
    
    println!("\n🎉 ALL PHASES COMPLETED SUCCESSFULLY!");
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn isolate_tensor_creation_shapes() -> Result<()> {
    println!("🔍 ISOLATION TEST: Testing individual tensor creation");
    
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        std::path::Path::new("/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane"),
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    let qwen_config = QwenConfig::from_model_config(model_config);
    
    // Test position_ids creation for different scenarios
    println!("\n🧪 Testing position_ids tensor creation:");
    
    // Test 1: Infer mode (should create shape [1])
    println!("  Test 1: Infer mode position_ids");
    match qwen_config.create_position_ids_with_mode_detection(&[0], false) {
        Ok(tensor) => {
            println!("    ✅ Infer position_ids shape: {:?}", tensor.shape());
            println!("    ✅ Infer position_ids values: {:?}", tensor.to_vec1::<i64>().unwrap_or_default());
        }
        Err(e) => println!("    ❌ Infer position_ids failed: {}", e),
    }
    
    // Test 2: Prefill mode (should create shape [64])  
    println!("  Test 2: Prefill mode position_ids");
    let positions: Vec<i64> = (0..5).collect(); // 5 positions
    match qwen_config.create_position_ids_with_mode_detection(&positions, true) {
        Ok(tensor) => {
            println!("    ✅ Prefill position_ids shape: {:?}", tensor.shape());
            let values = tensor.to_vec1::<i64>().unwrap_or_default();
            println!("    ✅ Prefill position_ids values (first 10): {:?}", &values[..10.min(values.len())]);
        }
        Err(e) => println!("    ❌ Prefill position_ids failed: {}", e),
    }
    
    // Test causal mask creation
    println!("\n🧪 Testing causal_mask tensor creation:");
    
    // Test 3: Infer mode causal mask
    println!("  Test 3: Infer mode causal_mask");
    match qwen_config.create_causal_mask_with_mode_detection(0, 256, false) {
        Ok(tensor) => {
            println!("    ✅ Infer causal_mask shape: {:?}", tensor.shape());
        }
        Err(e) => println!("    ❌ Infer causal_mask failed: {}", e),
    }
    
    // Test 4: Prefill mode causal mask
    println!("  Test 4: Prefill mode causal_mask");
    match qwen_config.create_causal_mask_with_mode_detection(5, 256, true) {
        Ok(tensor) => {
            println!("    ✅ Prefill causal_mask shape: {:?}", tensor.shape());
        }
        Err(e) => println!("    ❌ Prefill causal_mask failed: {}", e),
    }
    
    Ok(())
}

fn analyze_shape_error(error_msg: &str) {
    println!("\n🔍 ANALYZING SHAPE ERROR:");
    println!("Full error: {}", error_msg);
    
    // Try to extract the specific shapes mentioned
    if let Some(actual_start) = error_msg.find("MultiArray shape (") {
        if let Some(actual_end) = error_msg[actual_start..].find(")") {
            let actual_shape = &error_msg[actual_start + 18..actual_start + actual_end];
            println!("📐 ACTUAL shape provided: ({})", actual_shape);
        }
    }
    
    if let Some(expected_start) = error_msg.find("shape (") {
        let search_start = expected_start + 7;
        if let Some(expected_end) = error_msg[search_start..].find(" specified") {
            let expected_shape = &error_msg[search_start..search_start + expected_end - 1]; // -1 to remove closing paren
            println!("📐 EXPECTED shape by model: ({})", expected_shape);
        }
    }
    
    // Analyze what this means
    if error_msg.contains("(64)") && error_msg.contains("(1)") {
        println!("\n💡 ANALYSIS:");
        println!("   - Code is providing a tensor with 64 elements");
        println!("   - CoreML model expects a tensor with 1 element");
        println!("   - This suggests we're using PREFILL tensor shapes for INFER operation");
        println!("   - OR there's a mismatch in the model configuration vs actual model file");
    }
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn test_step_by_step_inference_with_logging() -> Result<()> {
    println!("🔍 STEP-BY-STEP INFERENCE TEST with detailed logging");
    
    // Enable tracing for detailed logs
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .try_init();
        
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    let generator = ConfigGenerator::new()?;
    let model_config = generator.generate_config_from_directory(
        model_path,
        "mazhewitt/qwen-typo-fixer",
        "qwen",
    )?;
    let config = QwenConfig::from_model_config(model_config);
    
    let mut model = QwenModel::load_from_directory(model_path, Some(config))?;
    
    // Try the absolute minimum: single character, single token generation
    println!("\n🧪 MINIMAL TEST: Single character 'a' with max_tokens=1");
    
    match model.generate_text("a", 1, 0.0) {
        Ok(result) => {
            println!("✅ SUCCESS! Generated: '{}'", result);
        }
        Err(e) => {
            println!("❌ FAILED: {}", e);
            analyze_shape_error(&e.to_string());
        }
    }
    
    Ok(())
}