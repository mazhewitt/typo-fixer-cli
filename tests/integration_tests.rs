//! Integration tests for the typo fixer CLI
//!
//! These tests will verify the complete end-to-end functionality

use typo_fixer_cli::cli::{Args, OutputFormat};

/// Test that CLI argument validation works correctly
#[test]
fn test_cli_args_validation() {
    // Test valid args with input text
    let args = Args {
        input: Some("test input".to_string()),
        stdin: false,
        model: "mazhewitt/qwen-typo-fixer".to_string(),
        local_path: None,
        config: None,
        temperature: 0.1,
        max_tokens: 50,
        output: OutputFormat::Text,
        verbose: false,
        batch: false,
    };
    assert!(args.validate().is_ok());

    // Test valid args with stdin
    let args = Args {
        input: None,
        stdin: true,
        model: "mazhewitt/qwen-typo-fixer".to_string(),
        local_path: None,
        config: None,
        temperature: 0.1,
        max_tokens: 50,
        output: OutputFormat::Text,
        verbose: false,
        batch: false,
    };
    assert!(args.validate().is_ok());
}

/// Test that invalid CLI arguments are rejected
#[test]
fn test_cli_args_validation_failures() {
    // Test no input provided
    let args = Args {
        input: None,
        stdin: false,
        model: "test".to_string(),
        local_path: None,
        config: None,
        temperature: 0.1,
        max_tokens: 50,
        output: OutputFormat::Text,
        verbose: false,
        batch: false,
    };
    assert!(args.validate().is_err());

    // Test invalid temperature
    let args = Args {
        input: Some("test".to_string()),
        stdin: false,
        model: "test".to_string(),
        local_path: None,
        config: None,
        temperature: 3.0, // Too high
        max_tokens: 50,
        output: OutputFormat::Text,
        verbose: false,
        batch: false,
    };
    assert!(args.validate().is_err());
}

// Note: We'll add more integration tests as we implement the actual model loading
// For now, these test the CLI argument parsing which is already implemented