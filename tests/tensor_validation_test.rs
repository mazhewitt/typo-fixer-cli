//! Test to validate if tensors with wrong shapes actually fail when used with CoreML models

use anyhow::Result;
use candle_coreml::{QwenConfig, get_builtin_config, CoreMLModel, Config as CoreMLConfig};

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn test_wrong_shapes_cause_failures() -> Result<()> {
    println!("🔍 TESTING: Do wrong tensor shapes actually cause failures?");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    // Get the configuration
    let model_config = get_builtin_config("mazhewitt/qwen-typo-fixer")
        .ok_or_else(|| anyhow::anyhow!("No built-in config found"))?;
    let config = QwenConfig::from_model_config(model_config);
    
    println!("\n🧪 TEST 1: Create tensors with CORRECT shapes for prefill");
    
    // Create CORRECT tensors for prefill (should work)
    let correct_position_ids = config.create_position_ids_with_mode_detection(&[0, 1, 2, 3, 4], true)?;
    let correct_causal_mask = config.create_causal_mask_with_mode_detection(5, 256, true)?;
    
    println!("✅ Correct prefill position_ids shape: {:?}", correct_position_ids.shape());
    println!("✅ Correct prefill causal_mask shape: {:?}", correct_causal_mask.shape());
    
    println!("\n🧪 TEST 2: Create tensors with WRONG shapes for prefill");
    
    // Create WRONG tensors for prefill (should fail when used)
    let wrong_position_ids = config.create_position_ids_with_mode_detection(&[0], false)?; // Using INFER shape for PREFILL
    let wrong_causal_mask = config.create_causal_mask_with_mode_detection(0, 256, false)?; // Using INFER shape for PREFILL
    
    println!("⚠️ Wrong prefill position_ids shape: {:?}", wrong_position_ids.shape());
    println!("⚠️ Wrong prefill causal_mask shape: {:?}", wrong_causal_mask.shape());
    
    println!("\n🧪 TEST 3: Try to use tensors with the actual prefill model");
    
    // Load the prefill model directly
    let prefill_path = model_path.join("qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage");
    
    let prefill_config = CoreMLConfig {
        input_names: vec![
            "hidden_states".to_string(),
            "position_ids".to_string(),
            "causal_mask".to_string(),
            "current_pos".to_string(),
        ],
        output_name: "output_hidden_states".to_string(),
        max_sequence_length: 64,
        vocab_size: 1024,
        model_type: "qwen-ffn-prefill".to_string(),
    };
    
    let mut prefill_model = CoreMLModel::load_with_function(&prefill_path, &prefill_config, "prefill")?;
    println!("✅ Prefill model loaded");
    
    // Create some dummy hidden_states and current_pos for testing
    let hidden_states = candle_core::Tensor::zeros((1, 64, 1024), candle_core::DType::F32, &candle_core::Device::Cpu)?;
    let current_pos = candle_core::Tensor::from_vec(vec![5_i64], (1,), &candle_core::Device::Cpu)?;
    
    println!("\n🧪 TEST 4: Try CORRECT tensors with prefill model");
    let correct_inputs = [&hidden_states, &correct_position_ids, &correct_causal_mask, &current_pos];
    
    let mut state = prefill_model.make_state()?;
    match prefill_model.predict_with_state(&correct_inputs, &mut state) {
        Ok(output) => {
            println!("✅ SUCCESS: Correct shapes worked! Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("❌ FAILED with correct shapes: {}", e);
            if e.to_string().contains("MultiArray shape") {
                println!("🎯 Even CORRECT shapes fail - model file issue!");
            }
        }
    }
    
    println!("\n🧪 TEST 5: Try WRONG tensors with prefill model");
    let wrong_inputs = [&hidden_states, &wrong_position_ids, &wrong_causal_mask, &current_pos];
    
    let mut wrong_state = prefill_model.make_state()?;
    match prefill_model.predict_with_state(&wrong_inputs, &mut wrong_state) {
        Ok(output) => {
            println!("😱 UNEXPECTED: Wrong shapes somehow worked! Output shape: {:?}", output.shape());
        }
        Err(e) => {
            println!("✅ EXPECTED: Wrong shapes failed as expected: {}", e);
            if e.to_string().contains("MultiArray shape") {
                println!("🎯 Shape validation is working!");
            }
        }
    }
    
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore]
async fn test_actual_model_file_expectations() -> Result<()> {
    println!("🔍 TESTING: What does the prefill model file actually expect?");
    
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let model_path = std::path::Path::new(working_model_path);
    
    let prefill_path = model_path.join("qwen-typo-fixer_FFN_PF_lut4_chunk_01of01.mlpackage");
    
    // Try loading with different configurations to see which one works
    let configs_to_try = vec![
        ("Config 1: Our current config (expects 64)", CoreMLConfig {
            input_names: vec![
                "hidden_states".to_string(),
                "position_ids".to_string(), 
                "causal_mask".to_string(),
                "current_pos".to_string(),
            ],
            output_name: "output_hidden_states".to_string(),
            max_sequence_length: 64,
            vocab_size: 1024,
            model_type: "qwen-ffn-prefill".to_string(),
        }),
        ("Config 2: Single token config (expects 1)", CoreMLConfig {
            input_names: vec![
                "hidden_states".to_string(),
                "position_ids".to_string(),
                "causal_mask".to_string(), 
                "current_pos".to_string(),
            ],
            output_name: "output_hidden_states".to_string(),
            max_sequence_length: 1,
            vocab_size: 1024,
            model_type: "qwen-ffn-prefill".to_string(),
        }),
    ];
    
    for (config_name, config) in configs_to_try {
        println!("\n🧪 Trying: {}", config_name);
        
        match CoreMLModel::load_with_function(&prefill_path, &config, "prefill") {
            Ok(model) => {
                println!("✅ {} - Model loads successfully", config_name);
                
                // Try to run inference with minimal inputs to see what shapes it expects
                println!("   Testing with minimal shapes...");
                
                // Try shape [1] inputs
                let hidden_states_1 = candle_core::Tensor::zeros((1, 1, 1024), candle_core::DType::F32, &candle_core::Device::Cpu)?;
                let position_ids_1 = candle_core::Tensor::from_vec(vec![0_i64], (1,), &candle_core::Device::Cpu)?;
                let causal_mask_1 = candle_core::Tensor::zeros((1, 1, 1, 256), candle_core::DType::F32, &candle_core::Device::Cpu)?;
                let current_pos_1 = candle_core::Tensor::from_vec(vec![0_i64], (1,), &candle_core::Device::Cpu)?;
                
                let inputs_1 = [&hidden_states_1, &position_ids_1, &causal_mask_1, &current_pos_1];
                
                let mut state1 = model.make_state()?;
                match model.predict_with_state(&inputs_1, &mut state1) {
                    Ok(_) => println!("   ✅ Works with shape [1] inputs!"),
                    Err(e) => {
                        println!("   ❌ Fails with shape [1]: {}", e);
                        if e.to_string().contains("MultiArray shape") {
                            println!("   📐 Shape error with [1] inputs");
                        }
                    }
                }
                
                // Try shape [64] inputs  
                let hidden_states_64 = candle_core::Tensor::zeros((1, 64, 1024), candle_core::DType::F32, &candle_core::Device::Cpu)?;
                let position_ids_64 = candle_core::Tensor::from_vec((0..64_i64).collect::<Vec<_>>(), (64,), &candle_core::Device::Cpu)?;
                let causal_mask_64 = candle_core::Tensor::zeros((1, 1, 64, 256), candle_core::DType::F32, &candle_core::Device::Cpu)?;
                let current_pos_64 = candle_core::Tensor::from_vec(vec![63_i64], (1,), &candle_core::Device::Cpu)?;
                
                let inputs_64 = [&hidden_states_64, &position_ids_64, &causal_mask_64, &current_pos_64];
                
                let mut state64 = model.make_state()?;
                match model.predict_with_state(&inputs_64, &mut state64) {
                    Ok(_) => println!("   ✅ Works with shape [64] inputs!"),
                    Err(e) => {
                        println!("   ❌ Fails with shape [64]: {}", e);
                        if e.to_string().contains("MultiArray shape") {
                            println!("   📐 Shape error with [64] inputs");
                        }
                    }
                }
            }
            Err(e) => {
                println!("❌ {} - Failed to load: {}", config_name, e);
            }
        }
    }
    
    Ok(())
}