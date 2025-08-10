//! Typo Fixer CLI Library
//!
//! A library for fixing typos using fine-tuned Qwen models via candle-coreml.
//! This serves as a test case for candle-coreml as a consumer library.

pub mod cli;
pub mod prompt;
pub mod typo_fixer;

pub use typo_fixer::TypoFixer;

use anyhow::Result;

/// Main library interface for fixing typos
pub struct TypoFixerLib {
    typo_fixer: TypoFixer,
}

impl TypoFixerLib {
    /// Create a new TypoFixerLib instance by loading the model
    pub async fn new(model_id: Option<String>, verbose: bool) -> Result<Self> {
        let model_id = model_id.unwrap_or_else(|| {
            "mazhewitt/qwen-typo-fixer".to_string()
        });
        
        let typo_fixer = TypoFixer::new(&model_id, verbose).await?;
        
        Ok(Self { typo_fixer })
    }

    /// Create a new TypoFixerLib instance from a local model directory
    pub async fn new_from_local(model_path: String, verbose: bool) -> Result<Self> {
        let typo_fixer = TypoFixer::new_from_local(&model_path, verbose).await?;
        
        Ok(Self { typo_fixer })
    }

    /// Create a new TypoFixerLib instance with a configuration file
    pub async fn new_with_config_file(config_path: &str, model_path: &str, verbose: bool) -> Result<Self> {
        let typo_fixer = TypoFixer::new_with_config_file(config_path, model_path, verbose).await?;
        
        Ok(Self { typo_fixer })
    }
    
    /// Fix typos in the given text
    pub async fn fix_typos(&mut self, text: &str) -> Result<String> {
        self.typo_fixer.fix_text(text).await
    }
    
    /// Fix typos with additional options
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