//! Core typo fixing functionality using candle-coreml

use anyhow::{Result, Context};
use candle_coreml::{QwenModel, QwenConfig, ModelConfig, get_builtin_config, model_downloader};
use crate::prompt::PromptTemplate;
use std::path::Path;

/// Core typo fixer that uses QwenModel from candle-coreml
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

    /// Create a new TypoFixer by loading the model from a local directory
    pub async fn new_from_local(model_path: &str, verbose: bool) -> Result<Self> {
        Self::load_local_model(model_path, verbose, None).await
    }

    /// Create a new TypoFixer by loading the model with a configuration file
    pub async fn new_with_config_file(config_path: &str, model_path: &str, verbose: bool) -> Result<Self> {
        if verbose {
            println!("🔧 Loading model configuration from: {}", config_path);
        }
        
        let model_config = ModelConfig::load_from_file(config_path)
            .context("Failed to load model configuration file")?;
        
        let qwen_config = QwenConfig::from_model_config(model_config);
        
        Self::load_local_model(model_path, verbose, Some(qwen_config)).await
    }

    /// Create a new TypoFixer with custom naming configuration
    pub async fn new_with_config(model_id: &str, verbose: bool, custom_config: Option<QwenConfig>) -> Result<Self> {
        if verbose {
            println!("🔄 Loading typo-fixing model: {}", model_id);
        }

        Self::load_model(model_id, verbose, custom_config).await
    }

    /// Load the model from HuggingFace Hub using candle-coreml
    async fn load_model(model_id: &str, verbose: bool, custom_config: Option<QwenConfig>) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
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

            // Use custom config if provided, otherwise try to get config by model ID
            let config = custom_config.unwrap_or_else(|| {
                if let Some(mut model_config) = get_builtin_config(model_id) {
                    if verbose { println!("🔧 Using built-in configuration for model: {}", model_id); }
                    // Attempt to patch component file paths relative to downloaded directory
                    if let Some(path_str) = model_path.to_str() {
                        let base = std::path::Path::new(path_str);
                        // Reuse the same helper logic as local load
                        #[allow(unused)]
                        fn adjust(model_config: &mut ModelConfig, base_dir: &std::path::Path, verbose: bool) {
                            use std::path::PathBuf;
                            for (name, comp) in model_config.components.iter_mut() {
                                if let Some(fp) = comp.file_path.clone() {
                                    let original = PathBuf::from(&fp);
                                    if original.exists() { continue; }
                                    if let Some(fname) = original.file_name() {
                                        let candidate = base_dir.join(fname);
                                        if candidate.exists() {
                                            comp.file_path = Some(candidate.to_string_lossy().to_string());
                                            if verbose { println!("🔄 Updated {} component path -> {}", name, candidate.display()); }
                                        }
                                    }
                                }
                            }
                        }
                        adjust(&mut model_config, base, verbose);
                    }
                    QwenConfig::from_model_config(model_config)
                } else {
                    if verbose { println!("🔧 Using default configuration"); }
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
            let _ = (model_id, verbose, custom_config); // Use parameters to avoid warnings
            Err(anyhow::anyhow!(
                "Model loading is only supported on macOS with CoreML. Current platform is not supported."
            ))
        }
    }

    /// Load the model from a local directory using candle-coreml
    async fn load_local_model(model_path: &str, verbose: bool, custom_config: Option<QwenConfig>) -> Result<Self> {
        #[cfg(target_os = "macos")]
        {
            use std::path::Path;
            
            if verbose {
                println!("📁 Loading model from local path: {}", model_path);
            }

            let path = Path::new(model_path);
            if !path.exists() {
                return Err(anyhow::anyhow!("Model path does not exist: {}", model_path));
            }

            // Use custom config if provided, otherwise try to auto-detect from known CLI configs
            let config = match custom_config {
                Some(cfg) => cfg,
                None => {
                    // 1. Look for explicit candle ModelConfig in model directory
                    if let Some(cfg) = Self::try_load_model_dir_config(path, verbose) { cfg } else if let Some(cfg) = Self::try_discover_cli_config(path, verbose) { cfg } else {
                        if verbose { println!("⚠️ No explicit config found. Falling back to default QwenConfig (may lack component file paths)"); }
                        QwenConfig::default()
                    }
                }
            };

            // Load the QwenModel from the local directory
            let model = QwenModel::load_from_directory(path, Some(config))
                .context("Failed to load QwenModel from local directory")?;

            if verbose {
                println!("✅ QwenModel loaded successfully from local path");
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
            let _ = (model_path, verbose, custom_config); // Use parameters to avoid warnings
            Err(anyhow::anyhow!(
                "Model loading is only supported on macOS with CoreML. Current platform is not supported."
            ))
        }
    }

    #[cfg(target_os = "macos")]
    fn try_load_model_dir_config(model_dir: &Path, verbose: bool) -> Option<QwenConfig> {
        let candidate = model_dir.join("config.json");
        if candidate.exists() {
            if verbose { println!("🔧 Found config.json in model directory, attempting to load..."); }
            match ModelConfig::load_from_file(&candidate) {
                Ok(mut mc) => {
                    if verbose { println!("✅ Loaded configuration from model directory"); }
                    Self::adjust_component_paths(&mut mc, model_dir, verbose);
                    Some(QwenConfig::from_model_config(mc))
                }
                Err(e) => {
                    if verbose { println!("⚠️ Failed to parse model directory config: {e}"); }
                    None
                }
            }
        } else {
            None
        }
    }

    #[cfg(target_os = "macos")]
    fn try_discover_cli_config(model_dir: &Path, verbose: bool) -> Option<QwenConfig> {
        use std::fs;
        use std::env;
        let cwd = env::current_dir().ok();
        let mut search_roots = vec![];
        if let Some(c) = &cwd { search_roots.push(c.clone()); }
        // If running from workspace root, also add typo-fixer-cli subfolder
        if let Some(c) = &cwd { search_roots.push(c.join("typo-fixer-cli")); }

        let model_hint = model_dir.file_name().and_then(|s| s.to_str()).unwrap_or("");
        let name_candidates = [
            "qwen-typo-fixer-ane",
            "qwen-typo-fixer",
            "typo-fixer"
        ];

        for root in search_roots {
            let configs_dir = root.join("configs");
            if !configs_dir.is_dir() { continue; }
            if verbose { println!("🔍 Searching for CLI config in {}", configs_dir.display()); }
            if let Ok(entries) = fs::read_dir(&configs_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) != Some("json") { continue; }
                    let file_name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
                    // Must match candidate names and hint
                    if name_candidates.iter().any(|cand| file_name.contains(cand) || model_hint.contains(cand)) {
                        if verbose { println!("🔧 Found candidate CLI config: {}", path.display()); }
                        match ModelConfig::load_from_file(&path) {
                            Ok(mut mc) => {
                                if verbose { println!("✅ Loaded CLI configuration: {}", path.display()); }
                                Self::adjust_component_paths(&mut mc, model_dir, verbose);
                                return Some(QwenConfig::from_model_config(mc));
                            }
                            Err(e) => {
                                if verbose { println!("⚠️ Failed to parse CLI config {}: {e}", path.display()); }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    #[cfg(target_os = "macos")]
    fn adjust_component_paths(model_config: &mut ModelConfig, base_dir: &Path, verbose: bool) {
        use std::path::PathBuf;
        for (name, comp) in model_config.components.iter_mut() {
            if let Some(fp) = comp.file_path.clone() {
                let original = PathBuf::from(&fp);
                if original.exists() { continue; }
                if let Some(fname) = original.file_name() {
                    let candidate = base_dir.join(fname);
                    if candidate.exists() {
                        comp.file_path = Some(candidate.to_string_lossy().to_string());
                        if verbose { println!("🔄 Updated {} component path -> {}", name, candidate.display()); }
                    }
                }
            }
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