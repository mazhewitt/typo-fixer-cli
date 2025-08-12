//! # Typo Fixer CLI Library
//!
//! A production-ready library for fixing typos using fine-tuned Qwen models via candle-coreml.
//! 
//! This library demonstrates real-world usage of candle-coreml with:
//! - Multi-component CoreML model orchestration (embeddings, FFN, LM head)
//! - Single-token sequential architecture support
//! - Apple Neural Engine acceleration via CoreML
//! - Comprehensive configuration management with JSON-based ModelConfig
//! - Local and remote model loading capabilities
//!
//! ## Quick Start
//!
//! ```rust
//! use typo_fixer_cli::TypoFixerLib;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     // Create with local model and configuration
//!     let mut fixer = TypoFixerLib::new_with_config_file(
//!         "./configs/qwen-typo-fixer-ane.json",
//!         "/path/to/model",
//!         true // verbose
//!     ).await?;
//!
//!     // Fix typos with custom parameters
//!     let corrected = fixer.fix_typos_with_options(
//!         "helo wrold, how ar you?",
//!         0.1,        // temperature
//!         Some(50)    // max_tokens
//!     ).await?;
//!
//!     println!("Fixed: {}", corrected);
//!     Ok(())
//! }
//! ```
//!
//! ## Features
//!
//! - **Multi-format Output**: Text, JSON, and verbose modes
//! - **Performance Control**: Temperature and token limit configuration
//! - **CoreML Acceleration**: Full Apple Neural Engine integration
//! - **Error Handling**: Comprehensive error reporting with context

pub mod cli;
pub mod prompt;
pub mod typo_fixer;

pub use typo_fixer::TypoFixer;

use anyhow::Result;

/// Main library interface for fixing typos using candle-coreml.
///
/// This struct provides a high-level API for typo correction using fine-tuned Qwen models
/// with CoreML acceleration. It handles model loading, configuration, and text generation.
///
/// # Examples
///
/// ## Basic Usage
/// ```rust
/// # use typo_fixer_cli::TypoFixerLib;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let mut fixer = TypoFixerLib::new(Some("model-id".to_string()), false).await?;
/// let result = fixer.fix_typos("helo wrold").await?;
/// println!("Fixed: {}", result);
/// # Ok(())
/// # }
/// ```
///
/// ## With Configuration File
/// ```rust
/// # use typo_fixer_cli::TypoFixerLib;
/// # #[tokio::main]
/// # async fn main() -> anyhow::Result<()> {
/// let mut fixer = TypoFixerLib::new_with_config_file(
///     "./configs/model-config.json",
///     "/path/to/model",
///     true // verbose logging
/// ).await?;
/// let result = fixer.fix_typos("seperate the itmes").await?;
/// # Ok(())
/// # }
/// ```
pub struct TypoFixerLib {
    typo_fixer: TypoFixer,
}

impl TypoFixerLib {
    /// Create a new TypoFixerLib instance by downloading a model from HuggingFace Hub.
    ///
    /// # Arguments
    /// * `model_id` - HuggingFace model identifier (e.g., "mazhewitt/qwen-typo-fixer")
    /// * `verbose` - Enable verbose logging for model loading and inference
    ///
    /// # Returns
    /// A new TypoFixerLib instance ready for typo correction
    ///
    /// # Errors
    /// Returns an error if model download fails or model loading fails
    pub async fn new(model_id: Option<String>, verbose: bool) -> Result<Self> {
        let model_id = model_id.unwrap_or_else(|| {
            "mazhewitt/qwen-typo-fixer".to_string()
        });
        
        let typo_fixer = TypoFixer::new(&model_id, verbose).await?;
        
        Ok(Self { typo_fixer })
    }

    /// Create a new TypoFixerLib instance from a local model directory.
    ///
    /// This method loads a model from a local directory using auto-detection
    /// for configuration files or falls back to default settings.
    ///
    /// # Arguments
    /// * `model_path` - Path to the local model directory containing .mlpackage files
    /// * `verbose` - Enable verbose logging for model loading and inference
    ///
    /// # Returns
    /// A new TypoFixerLib instance ready for typo correction
    ///
    /// # Errors
    /// Returns an error if the model path doesn't exist or model loading fails
    pub async fn new_from_local(model_path: String, verbose: bool) -> Result<Self> {
        let typo_fixer = TypoFixer::new_from_local(&model_path, verbose).await?;
        
        Ok(Self { typo_fixer })
    }

    /// Create a new TypoFixerLib instance with an explicit configuration file.
    ///
    /// This is the recommended method for production use as it provides full control
    /// over model component loading and shapes.
    ///
    /// # Arguments
    /// * `config_path` - Path to the JSON configuration file (ModelConfig format)
    /// * `model_path` - Path to the local model directory containing .mlpackage files  
    /// * `verbose` - Enable verbose logging for model loading and inference
    ///
    /// # Returns
    /// A new TypoFixerLib instance with explicit configuration
    ///
    /// # Errors
    /// Returns an error if configuration loading fails or model loading fails
    pub async fn new_with_config_file(config_path: &str, model_path: &str, verbose: bool) -> Result<Self> {
        let typo_fixer = TypoFixer::new_with_config_file(config_path, model_path, verbose).await?;
        
        Ok(Self { typo_fixer })
    }
    
    /// Fix typos in the given text using default settings.
    ///
    /// Uses temperature=0.1 and max_tokens=50 for balanced quality and speed.
    ///
    /// # Arguments
    /// * `text` - Input text containing potential typos
    ///
    /// # Returns
    /// Corrected text with typos fixed
    ///
    /// # Errors
    /// Returns an error if text generation fails
    pub async fn fix_typos(&mut self, text: &str) -> Result<String> {
        self.typo_fixer.fix_text(text).await
    }
    
    /// Fix typos with custom generation parameters.
    ///
    /// Provides full control over the text generation process for fine-tuning
    /// performance vs quality trade-offs.
    ///
    /// # Arguments
    /// * `text` - Input text containing potential typos
    /// * `temperature` - Sampling temperature (0.0=deterministic, higher=more creative)
    /// * `max_tokens` - Maximum number of tokens to generate (affects speed and context)
    ///
    /// # Returns
    /// Corrected text with typos fixed using specified parameters
    ///
    /// # Errors
    /// Returns an error if text generation fails
    ///
    /// # Performance Notes
    /// - `temperature=0.0`: Deterministic, fastest, most consistent
    /// - `temperature=0.1-0.3`: Good balance of quality and consistency
    /// - `max_tokens=10-20`: Fast inference, suitable for short corrections
    /// - `max_tokens=50+`: Better context, slower inference
    pub async fn fix_typos_with_options(
        &mut self, 
        text: &str, 
        temperature: f32,
        max_tokens: Option<usize>
    ) -> Result<String> {
        self.typo_fixer.fix_text_with_options(text, temperature, max_tokens).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_library_creation() {
        // This is a smoke test to ensure the library structure works
        // We'll add more comprehensive tests as we implement the functionality
        let result = TypoFixerLib::new(Some("test-model".to_string()), false).await;
        
        // For now, we expect this to fail since we haven't implemented TypoFixer yet
        assert!(result.is_err(), "Expected error since TypoFixer is not implemented yet");
    }
}