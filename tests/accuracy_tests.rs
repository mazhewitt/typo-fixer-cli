//! Accuracy and integration tests for typo fixing

use anyhow::Result;
use serde_json::Value;
use std::process::Command;
use tokio;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually since it uses models
async fn test_cli_integration_with_working_model() -> Result<()> {
    println!("🧪 Testing CLI integration with working Qwen model");
    
    let working_model = "anemll/anemll-Qwen-Qwen3-0.6B-ctx512_0.3.4";
    
    // Test 1: Simple text completion (demonstrating the infrastructure works)
    let output = Command::new("cargo")
        .args([
            "run", 
            "--", 
            "The quick brown fox",
            "--model", working_model,
            "--max-tokens", "5",
            "--temperature", "0.0",
            "--output", "json"
        ])
        .output()
        .expect("Failed to execute command");
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        println!("❌ CLI failed: {}", stderr);
        return Err(anyhow::anyhow!("CLI execution failed: {}", stderr));
    }
    
    let stdout = String::from_utf8_lossy(&output.stdout);
    println!("✅ CLI output: {}", stdout);
    
    // Parse JSON output
    let json: Value = serde_json::from_str(&stdout)
        .map_err(|e| anyhow::anyhow!("Failed to parse JSON output: {}", e))?;
    
    // Validate JSON structure
    assert!(json["input"].is_string(), "Should have input field");
    assert!(json["output"].is_string(), "Should have output field");
    assert!(json["model"].is_string(), "Should have model field");
    
    let input = json["input"].as_str().unwrap();
    let output_text = json["output"].as_str().unwrap();
    
    println!("📝 Input: '{}'", input);
    println!("📝 Output: '{}'", output_text);
    
    // Basic validation - output should be different (completion)
    assert_ne!(input, output_text, "Output should be different from input");
    
    println!("✅ CLI integration test passed");
    Ok(())
}

#[tokio::test]
async fn test_typo_fixing_prompt_engineering() {
    use typo_fixer_cli::prompt::PromptTemplate;
    
    println!("🧪 Testing typo fixing prompt engineering");
    
    let template = PromptTemplate::new();
    let prompt = template.create_correction_prompt("this sentance has typoos");
    
    println!("📝 Generated prompt:\n{}", prompt);
    
    // Validate prompt structure
    assert!(prompt.contains("Fix typos in these sentences:"), "Should contain instruction");
    assert!(prompt.contains("Input: the quik brown fox"), "Should contain example input");
    assert!(prompt.contains("Output: the quick brown fox"), "Should contain example output");
    assert!(prompt.contains("Input: this sentance has typoos"), "Should contain our input");
    assert!(prompt.ends_with("Output:"), "Should end with Output:");
    
    // Test with multiple examples
    let mut custom_template = PromptTemplate::new();
    custom_template.add_example(
        "definately wrong".to_string(),
        "definitely wrong".to_string()
    );
    
    let custom_prompt = custom_template.create_correction_prompt("test");
    assert!(custom_prompt.contains("definately wrong"), "Should contain custom example");
    assert!(custom_prompt.contains("definitely wrong"), "Should contain custom correction");
    
    println!("✅ Prompt engineering tests passed");
}

#[test]
fn test_expected_typo_corrections() {
    // Test cases similar to what we'd expect from your Python demo
    let test_cases = vec![
        ("this sentance has typoos", vec!["sentence", "typos"]),
        ("the quik brown fox", vec!["quick"]),
        ("i cant beleive it", vec!["can't", "believe"]),
        ("recieve the package", vec!["receive"]),
        ("seperate the items", vec!["separate"]),
        ("it occured yesterday", vec!["occurred"]),
    ];
    
    println!("🧪 Testing expected typo correction patterns");
    
    for (input, expected_fixes) in test_cases {
        println!("📝 Test case: '{}' should contain: {:?}", input, expected_fixes);
        
        // This is a validation of our test cases themselves
        // In a real test with the model, we'd validate that the model output contains these fixes
        for expected in expected_fixes {
            assert!(
                !input.to_lowercase().contains(&expected.to_lowercase()),
                "Input '{}' should not already contain correct word '{}'", input, expected
            );
        }
    }
    
    println!("✅ Test case structure validation passed");
    println!("💡 These test cases are ready for use with the actual typo-fixer model");
}

#[test]
fn test_cli_argument_combinations() {
    use typo_fixer_cli::cli::{Args, OutputFormat};
    
    println!("🧪 Testing various CLI argument combinations");
    
    // Test verbose JSON output
    let args = Args {
        input: Some("test input".to_string()),
        stdin: false,
        model: "test-model".to_string(),
        temperature: 0.0,
        max_tokens: 25,
        output: OutputFormat::Json,
        verbose: true,
        batch: false,
    };
    
    assert!(args.validate().is_ok(), "Verbose JSON args should be valid");
    
    // Test batch mode
    let args = Args {
        input: Some("line1\nline2\nline3".to_string()),
        stdin: false,
        model: "test-model".to_string(),
        temperature: 0.5,
        max_tokens: 50,
        output: OutputFormat::Verbose,
        verbose: false,
        batch: true,
    };
    
    assert!(args.validate().is_ok(), "Batch mode args should be valid");
    
    // Test edge cases
    let args = Args {
        input: Some("test".to_string()),
        stdin: false,
        model: "test-model".to_string(),
        temperature: 2.0, // Maximum allowed
        max_tokens: 512,  // Maximum allowed
        output: OutputFormat::Text,
        verbose: false,
        batch: false,
    };
    
    assert!(args.validate().is_ok(), "Edge case args should be valid");
    
    println!("✅ CLI argument validation tests passed");
}

// Performance benchmark placeholder
#[test]
fn test_performance_expectations() {
    println!("🧪 Testing performance expectations and benchmarks");
    
    // This test documents our performance expectations
    // Once we have the typo-fixer model working, we can add actual benchmarks
    
    let expected_latency_ms = 1000; // Target: < 1 second per correction
    let expected_accuracy = 0.90;   // Target: > 90% accuracy (from Python demo)
    let max_model_size_gb = 3.0;    // Target: < 3GB total model size
    
    println!("📊 Performance targets:");
    println!("   • Latency: < {} ms per correction", expected_latency_ms);
    println!("   • Accuracy: > {:.0}% correction rate", expected_accuracy * 100.0);
    println!("   • Model size: < {:.1} GB", max_model_size_gb);
    
    // These will be real benchmarks once the model is working
    assert!(expected_latency_ms > 0);
    assert!(expected_accuracy > 0.0 && expected_accuracy <= 1.0);
    assert!(max_model_size_gb > 0.0);
    
    println!("✅ Performance expectations documented");
}

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Will be enabled once typo-fixer model downloads complete
async fn test_actual_typo_fixer_model() -> Result<()> {
    use typo_fixer_cli::TypoFixerLib;
    
    println!("🧪 Testing with actual typo-fixer model (once download completes)");
    
    let mut typo_fixer = TypoFixerLib::new(
        Some("mazhewitt/qwen-typo-fixer".to_string()),
        true
    ).await?;
    
    // Test cases from your Python accuracy test
    let test_cases = vec![
        "this sentance has typoos",
        "the quik brown fox jumps",
        "i cant beleive this happend", 
        "recieve the seperate package",
    ];
    
    for input in test_cases {
        println!("🔤 Testing: '{}'", input);
        
        let corrected = typo_fixer.fix_typos(input).await?;
        println!("✅ Corrected: '{}'", corrected);
        
        // Validate that some correction happened
        assert_ne!(input, corrected, "Should produce some correction for: {}", input);
    }
    
    println!("🎉 Typo-fixer model test completed successfully");
    Ok(())
}