//! Core typo fixing functionality using candle-coreml

use anyhow::{Result, Context};
use candle_coreml::{QwenModel, QwenConfig, ModelNamingConfig};
use crate::prompt::PromptTemplate;

/// Core typo fixer that uses a fine-tuned Qwen model via candle-coreml
pub struct TypoFixer {
    model: QwenModel,
    prompt_template: PromptTemplate,
    verbose: bool,
}

impl TypoFixer {
    /// Create a new TypoFixer by loading the model from HuggingFace Hub
    pub async fn new(model_id: &str, verbose: bool) -> Result<Self> {
        Self::new_with_config(model_id, verbose, None).await
    }

    /// Create a new TypoFixer with custom naming configuration
    pub async fn new_with_config(model_id: &str, verbose: bool, custom_config: Option<QwenConfig>) -> Result<Self> {
        if verbose {
            println!("🔄 Loading typo-fixing model: {}", model_id);
        }

        // Note: This will need to be implemented once we test model loading
        // For now, we'll create a placeholder that will fail in a controlled way
        Self::load_model_placeholder(model_id, verbose, custom_config).await
    }

    /// Load the model from HuggingFace Hub using candle-coreml
    async fn load_model_placeholder(model_id: &str, verbose: bool, custom_config: Option<QwenConfig>) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            use candle_coreml::{model_downloader, QwenConfig};

            if verbose {
                println!("📥 Downloading model from HuggingFace Hub...");
            }

            // Download the model from HuggingFace Hub
            let model_path = model_downloader::ensure_model_downloaded(model_id, verbose)
                .context("Failed to download model from HuggingFace Hub")?;

            if verbose {
                println!("📦 Model downloaded to: {:?}", model_path);
                println!("🔧 Loading QwenModel components...");
            }

            // Use custom config if provided, otherwise auto-detect naming patterns
            let config = custom_config.unwrap_or_else(|| {
                if model_id.contains("typo-fixer") {
                    if verbose {
                        println!("🔧 Using typo-fixer naming patterns");
                    }
                    QwenConfig::for_typo_fixer()
                } else if model_id.contains("bobs-qwen") {
                    if verbose {
                        println!("🔧 Using Bob's custom naming patterns");
                    }
                    QwenConfig::for_bobs_model()
                } else {
                    if verbose {
                        println!("🔧 Using default naming patterns (supports both qwen_ and qwen-typo-fixer_)");
                    }
                    QwenConfig::default()
                }
            });

            // Load the QwenModel from the downloaded components
            let model = QwenModel::load_from_directory(&model_path, Some(config))
                .context("Failed to load QwenModel from downloaded components")?;

            if verbose {
                println!("✅ QwenModel loaded successfully");
            }

            // Create the prompt template with examples suitable for typo fixing
            let prompt_template = PromptTemplate::new();

            Ok(Self {
                model,
                prompt_template,
                verbose,
            })
        }

        #[cfg(not(target_os = "macos"))]
        {
            // On non-macOS platforms, return an appropriate error
            let _ = (model_id, verbose); // Use parameters to avoid warnings
            Err(anyhow::anyhow!(
                "Model loading is only supported on macOS with CoreML. Current platform is not supported."
            ))
        }
    }

    /// Fix typos in the given text using default settings
    pub async fn fix_text(&mut self, text: &str) -> Result<String> {
        self.fix_text_with_options(text, 0.1, None).await
    }

    /// Fix typos with custom temperature and max tokens
    pub async fn fix_text_with_options(
        &mut self,
        text: &str,
        temperature: f32,
        max_tokens: Option<usize>,
    ) -> Result<String> {
        if self.verbose {
            println!("🔧 Fixing typos in: '{}'", text);
            println!("   Temperature: {}, Max tokens: {:?}", temperature, max_tokens);
        }

        // Create the prompt using few-shot examples
        let prompt = self.prompt_template.create_correction_prompt(text);
        
        if self.verbose {
            println!("📝 Generated prompt:\n{}", prompt);
        }

        // Generate the correction using the model
        let max_tokens = max_tokens.unwrap_or(50);
        
        let corrected_text = if temperature == 0.0 {
            // Use deterministic generation (greedy)
            self.generate_deterministic(&prompt, max_tokens).await?
        } else {
            // Use temperature sampling
            self.generate_with_temperature(&prompt, temperature, max_tokens).await?
        };

        // Post-process the output to extract just the corrected text
        let cleaned = self.extract_correction(&corrected_text)?;

        if self.verbose {
            println!("✅ Corrected: '{}' -> '{}'", text, cleaned);
        }

        Ok(cleaned)
    }

    /// Generate text using deterministic (greedy) sampling
    async fn generate_deterministic(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        if self.verbose {
            println!("🎯 Using deterministic (greedy) generation");
        }

        // Use candle-coreml's generate_text method with temperature 0.0
        let generated_text = self.model.generate_text(prompt, max_tokens, 0.0)
            .context("Failed to generate tokens deterministically")?;

        Ok(generated_text)
    }

    /// Generate text using temperature sampling
    async fn generate_with_temperature(
        &mut self, 
        prompt: &str, 
        temperature: f32, 
        max_tokens: usize
    ) -> Result<String> {
        if self.verbose {
            println!("🎲 Using temperature sampling (temperature: {})", temperature);
        }

        // Use candle-coreml's generate_text method with specified temperature
        let generated_text = self.model.generate_text(prompt, max_tokens, temperature)
            .context("Failed to generate tokens with temperature")?;

        Ok(generated_text)
    }

    /// Extract the corrected text from the model's output
    fn extract_correction(&self, raw_output: &str) -> Result<String> {
        // The model should output the correction after "Output:"
        // We need to parse and clean this up
        
        // Look for lines that might contain the correction
        let lines: Vec<&str> = raw_output.lines().collect();
        
        for line in lines {
            let trimmed = line.trim();
            // Skip empty lines and lines that look like prompts
            if !trimmed.is_empty() && 
               !trimmed.starts_with("Input:") && 
               !trimmed.starts_with("Output:") &&
               !trimmed.starts_with("Fix typos") {
                return Ok(trimmed.to_string());
            }
        }

        // If we can't find a clear correction, return the raw output cleaned up
        let cleaned = raw_output.trim();
        if cleaned.is_empty() {
            return Err(anyhow::anyhow!("Model produced empty output"));
        }

        Ok(cleaned.to_string())
    }

    /// Get mutable reference to the prompt template for customization
    pub fn prompt_template_mut(&mut self) -> &mut PromptTemplate {
        &mut self.prompt_template
    }

    /// Get reference to the prompt template
    pub fn prompt_template(&self) -> &PromptTemplate {
        &self.prompt_template
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_typo_fixer_creation_with_invalid_model() {
        // Test that TypoFixer creation fails gracefully with an invalid model
        let result = TypoFixer::new("nonexistent/invalid-model", false).await;
        assert!(result.is_err(), "Expected TypoFixer::new to fail with invalid model");
        
        // Test that the error indicates download or model loading failure
        if let Err(e) = result {
            let error_msg = e.to_string();
            // Should contain information about download failure
            assert!(
                error_msg.contains("Failed to download") || 
                error_msg.contains("Failed to load") ||
                error_msg.contains("not supported"),
                "Error should indicate download or loading failure. Got: {}", error_msg
            );
        }
    }

    // Note: We'll add tests for extract_correction once we have a proper way to 
    // create test instances. For now, we focus on testing what we can safely test.
    
    #[test]
    fn test_extract_correction_logic_standalone() {
        // Test the extraction logic without needing a TypoFixer instance
        
        // Test simple case
        let raw_output = "the quick brown fox";
        let result = extract_correction_standalone(raw_output);
        assert_eq!(result, "the quick brown fox");
        
        // Test case with noise  
        let raw_output = "Input: test\nOutput: \nthe quick brown fox\n";
        let result = extract_correction_standalone(raw_output);
        assert_eq!(result, "the quick brown fox");
    }
    
    // Standalone function to test extraction logic
    fn extract_correction_standalone(raw_output: &str) -> String {
        // Look for lines that might contain the correction
        let lines: Vec<&str> = raw_output.lines().collect();
        
        for line in lines {
            let trimmed = line.trim();
            // Skip empty lines and lines that look like prompts
            if !trimmed.is_empty() && 
               !trimmed.starts_with("Input:") && 
               !trimmed.starts_with("Output:") &&
               !trimmed.starts_with("Fix typos") {
                return trimmed.to_string();
            }
        }

        // If we can't find a clear correction, return the raw output cleaned up
        raw_output.trim().to_string()
    }
}