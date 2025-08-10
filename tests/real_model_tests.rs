//! Integration tests with real typo-fixer functionality

use anyhow::Result;
use tokio;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually since it downloads models
async fn test_actual_typo_fixing_simple_case() -> Result<()> {
    use typo_fixer_cli::TypoFixerLib;

    println!("🧪 Testing actual typo fixing with simple case");
    
    // Try to create the typo fixer with the local working model
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    let mut typo_fixer = match TypoFixerLib::new_from_local(
        working_model_path.to_string(), 
        true
    ).await {
        Ok(fixer) => {
            println!("✅ TypoFixerLib created successfully");
            fixer
        }
        Err(e) => {
            println!("⚠️  Failed to create TypoFixerLib: {}", e);
            // For now, let's return a controlled failure so we can debug
            return Err(anyhow::anyhow!("Model loading failed - this is expected while we debug the tokenizer issue: {}", e));
        }
    };

    // Test basic typo fixing
    let input = "this sentance has typoos";
    println!("🔤 Input: '{}'", input);
    
    let corrected = typo_fixer.fix_typos(input).await?;
    println!("✅ Corrected: '{}'", corrected);
    
    // Basic validation - the output should be different and fix at least one typo
    assert_ne!(input, corrected, "Output should be different from input");
    
    // Check that some common typos are fixed
    let corrected_lower = corrected.to_lowercase();
    assert!(
        corrected_lower.contains("sentence") || corrected_lower.contains("this sentence"), 
        "Should fix 'sentance' to 'sentence'. Got: {}", corrected
    );
    
    println!("🎉 Typo fixing test completed successfully");
    Ok(())
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually
async fn test_try_working_model_first() -> Result<()> {
    // Using TypoFixerLib now (imported below)
    
    println!("🧪 Testing with a known working Qwen model first");
    
    // Use the local working model provided by the user
    let working_model_path = "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane";
    
    println!("📁 Using local working model: {}", working_model_path);
    
    let model_path = std::path::Path::new(working_model_path);
    println!("✅ Model downloaded to: {:?}", model_path);
    
    // Use the same code path as the main app via TypoFixerLib
    use typo_fixer_cli::TypoFixerLib;
    
    let mut typo_fixer = TypoFixerLib::new_from_local(working_model_path.to_string(), true).await?;
    println!("✅ TypoFixerLib loaded successfully");
    
    // Test basic typo fixing (same as main app does)
    let test_text = "The quick brown fox";
    println!("🔤 Testing typo fixing with: '{}'", test_text);
    
    let result = typo_fixer.fix_typos_with_options(test_text, 0.0, Some(10)).await?;
    println!("🎯 Fixed text: '{}'", result);
    
    println!("🎉 Working model test completed - this proves candle-coreml integration works");
    Ok(())
}

// Helper test to diagnose the tokenizer issue
#[test]
fn test_diagnose_tokenizer_files() {
    use std::path::PathBuf;
    
    let model_path = PathBuf::from("/Users/mazdahewitt/Library/Caches/candle-coreml/clean-mazhewitt--qwen-typo-fixer");
    
    if !model_path.exists() {
        println!("⏭️ Model not downloaded yet, skipping tokenizer diagnosis");
        return;
    }
    
    let tokenizer_json = model_path.join("tokenizer.json");
    let tokenizer_config = model_path.join("tokenizer_config.json");
    let vocab_json = model_path.join("vocab.json");
    
    println!("🔍 Diagnosing tokenizer files:");
    
    if tokenizer_json.exists() {
        let metadata = std::fs::metadata(&tokenizer_json).unwrap();
        println!("  📄 tokenizer.json: {} bytes", metadata.len());
        
        if metadata.len() < 1000 {
            println!("    ⚠️  File is suspiciously small - likely an LFS pointer");
            if let Ok(content) = std::fs::read_to_string(&tokenizer_json) {
                println!("    📝 Content preview: {}", content.lines().take(3).collect::<Vec<_>>().join(" "));
            }
        }
    } else {
        println!("  ❌ tokenizer.json missing");
    }
    
    if tokenizer_config.exists() {
        let metadata = std::fs::metadata(&tokenizer_config).unwrap();
        println!("  📄 tokenizer_config.json: {} bytes", metadata.len());
    } else {
        println!("  ❌ tokenizer_config.json missing");
    }
    
    if vocab_json.exists() {
        let metadata = std::fs::metadata(&vocab_json).unwrap();
        println!("  📄 vocab.json: {} bytes", metadata.len());
    } else {
        println!("  ❌ vocab.json missing");
    }
}