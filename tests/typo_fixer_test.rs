use anyhow::Result;
use typo_fixer_cli::TypoFixerLib;

#[tokio::test]
#[ignore]
async fn test_typo_fixer_with_working_model() -> Result<()> {
    // Use the HuggingFace model instead of hardcoded local path
    // This replaces: "/Users/mazdahewitt/projects/train-typo-fixer/models/qwen-typo-fixer-ane-flex"
    let model_id = "mazhewitt/qwen-typo-fixer-coreml";
    
    let mut typo_fixer = TypoFixerLib::new(Some(model_id.to_string()), true).await?;
    
    // Test basic typo correction
    let result = typo_fixer.fix_typos("this sentance has typoos").await?;
    
    // The model should fix the typos
    assert!(result.contains("sentence"), "Should fix 'sentance' to 'sentence'");
    assert!(result.contains("typos"), "Should fix 'typoos' to 'typos'");
    
    println!("✅ Test passed: '{}' -> '{}'", "this sentance has typoos", result);
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_typo_fixer_multiple_corrections() -> Result<()> {
    // Use HuggingFace model instead of hardcoded path
    let model_id = "mazhewitt/qwen-typo-fixer-coreml";
    
    let mut typo_fixer = TypoFixerLib::new(Some(model_id.to_string()), true).await?;
    
    let test_cases = vec![
        ("recieve the package", "receive"),
        ("seperate the items", "separate"), 
        ("the quik brown fox", "quick"),
        ("i cant beleive it", "can't"),
        ("definately wrong", "definitely"),
    ];
    
    for (input, expected_fix) in test_cases {
        let result = typo_fixer.fix_typos(input).await?;
        // Check if the result contains the expected fix (case-insensitive)
        assert!(result.to_lowercase().contains(expected_fix), 
               "Expected '{}' to contain '{}' (case-insensitive), got '{}'", input, expected_fix, result);
        println!("✅ '{}' -> '{}'", input, result);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_diagnose_typo_fixer_model_issue() -> Result<()> {
    // This test specifically looks at what happens when we try to load the problematic model
    let problematic_model = "mazhewitt/qwen-typo-fixer";
    
    let result = TypoFixerLib::new(Some(problematic_model.to_string()), true).await;
    
    match result {
        Ok(_) => {
            println!("✅ Surprising: mazhewitt/qwen-typo-fixer model loaded successfully!");
        }
        Err(e) => {
            println!("❌ Expected error loading mazhewitt/qwen-typo-fixer:");
            println!("   Error: {}", e);
            
            // Check if it's the CoreML parsing error we expect
            let error_str = e.to_string();
            if error_str.contains("wireType 6") || error_str.contains("Field number 14") {
                println!("🔍 This is the expected CoreML parsing error");
                println!("💡 Recommendation: The model may need to be re-exported or use different CoreML version");
            } else if error_str.contains("Failed to compile CoreML model") {
                println!("🔍 This is a CoreML compilation error");
            } else {
                println!("🔍 This is a different error type: {}", error_str);
            }
        }
    }
    
    Ok(())
}

#[tokio::test]
#[ignore]
async fn test_typo_fixer_with_custom_temperature() -> Result<()> {
    // Use HuggingFace model instead of hardcoded path
    let model_id = "mazhewitt/qwen-typo-fixer-coreml";
    
    let mut typo_fixer = TypoFixerLib::new(Some(model_id.to_string()), true).await?;
    
    // Test with deterministic generation (temperature = 0.0)
    let result1 = typo_fixer.fix_typos_with_options("this sentance has typoos", 0.0, Some(30)).await?;
    let result2 = typo_fixer.fix_typos_with_options("this sentance has typoos", 0.0, Some(30)).await?;
    
    // With temperature 0, results should be identical (deterministic)
    assert_eq!(result1, result2, "Temperature 0.0 should produce identical results");
    
    println!("✅ Deterministic generation test passed: '{}'", result1);
    
    Ok(())
}