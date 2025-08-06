//! CLI argument parsing and handling

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about = "Fix typos using fine-tuned Qwen models", long_about = None)]
pub struct Args {
    /// Input text to fix typos in
    #[arg(value_name = "TEXT")]
    pub input: Option<String>,

    /// Read input from stdin instead of command line argument
    #[arg(short, long)]
    pub stdin: bool,

    /// Model ID from HuggingFace Hub
    #[arg(short, long, default_value = "mazhewitt/qwen-typo-fixer")]
    pub model: String,

    /// Sampling temperature (0.0 = deterministic, higher = more creative)
    #[arg(short, long, default_value = "0.1")]
    pub temperature: f32,

    /// Maximum number of tokens to generate
    #[arg(long, default_value = "50")]
    pub max_tokens: usize,

    /// Output format: text, json, or verbose
    #[arg(short, long, default_value = "text")]
    pub output: OutputFormat,

    /// Enable verbose output
    #[arg(short, long)]
    pub verbose: bool,

    /// Batch mode: process multiple lines
    #[arg(short, long)]
    pub batch: bool,
}

#[derive(clap::ValueEnum, Clone, Debug)]
pub enum OutputFormat {
    /// Plain corrected text
    Text,
    /// JSON format with metadata
    Json,
    /// Verbose format with before/after comparison
    Verbose,
}

impl Args {
    /// Validate CLI arguments
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.input.is_none() && !self.stdin {
            return Err(anyhow::anyhow!(
                "Must provide either input text or use --stdin flag"
            ));
        }

        if self.input.is_some() && self.stdin {
            return Err(anyhow::anyhow!(
                "Cannot use both input text and --stdin flag"
            ));
        }

        if !(0.0..=2.0).contains(&self.temperature) {
            return Err(anyhow::anyhow!(
                "Temperature must be between 0.0 and 2.0, got {}",
                self.temperature
            ));
        }

        if self.max_tokens == 0 || self.max_tokens > 512 {
            return Err(anyhow::anyhow!(
                "max_tokens must be between 1 and 512, got {}",
                self.max_tokens
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_args_validation_with_input() {
        let args = Args {
            input: Some("test input".to_string()),
            stdin: false,
            model: "test-model".to_string(),
            temperature: 0.5,
            max_tokens: 50,
            output: OutputFormat::Text,
            verbose: false,
            batch: false,
        };

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_args_validation_with_stdin() {
        let args = Args {
            input: None,
            stdin: true,
            model: "test-model".to_string(),
            temperature: 0.5,
            max_tokens: 50,
            output: OutputFormat::Text,
            verbose: false,
            batch: false,
        };

        assert!(args.validate().is_ok());
    }

    #[test]
    fn test_args_validation_no_input() {
        let args = Args {
            input: None,
            stdin: false,
            model: "test-model".to_string(),
            temperature: 0.5,
            max_tokens: 50,
            output: OutputFormat::Text,
            verbose: false,
            batch: false,
        };

        assert!(args.validate().is_err());
    }

    #[test]
    fn test_args_validation_both_input_and_stdin() {
        let args = Args {
            input: Some("test".to_string()),
            stdin: true,
            model: "test-model".to_string(),
            temperature: 0.5,
            max_tokens: 50,
            output: OutputFormat::Text,
            verbose: false,
            batch: false,
        };

        assert!(args.validate().is_err());
    }

    #[test]
    fn test_temperature_validation() {
        let mut args = Args {
            input: Some("test".to_string()),
            stdin: false,
            model: "test-model".to_string(),
            temperature: 3.0, // Invalid
            max_tokens: 50,
            output: OutputFormat::Text,
            verbose: false,
            batch: false,
        };

        assert!(args.validate().is_err());

        args.temperature = 0.5; // Valid
        assert!(args.validate().is_ok());
    }
}