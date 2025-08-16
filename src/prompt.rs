//! Prompt engineering for typo fixing
//! 
//! Based on the few-shot prompting approach from the Python demo

/// Template for typo fixing prompts with few-shot examples
pub struct PromptTemplate {
    examples: Vec<(String, String)>, // (input with typos, corrected output)
}

impl PromptTemplate {
    /// Create a new prompt template with optimized few-shot examples (88.5% accuracy)
    pub fn new() -> Self {
        let examples = vec![
            ("I beleive this is teh answer.".to_string(), "I believe this is the answer.".to_string()),
            ("She recieved her degre yesterday.".to_string(), "She received her degree yesterday.".to_string()),
            ("The resturant serves good food.".to_string(), "The restaurant serves good food.".to_string()),
        ];

        Self { examples }
    }

    /// Create a typo correction prompt using the simple format that works with the model
    pub fn create_correction_prompt(&self, input_text: &str) -> String {
    // Default to richer few-shot format (backward compat kept via create_few_shot_prompt)
    self.create_few_shot_prompt(input_text)
    }

    /// Create a typo correction prompt with few-shot examples (alternative format)
    pub fn create_few_shot_prompt(&self, input_text: &str) -> String {
        let mut prompt = String::new();
        
        prompt.push_str("Fix typos in these sentences:\n\n");
        
        // Add few-shot examples
        for (typo_text, correct_text) in &self.examples {
            prompt.push_str(&format!("Input: {}\nOutput: {}\n\n", typo_text, correct_text));
        }
        
        // Add the actual input to be corrected
        prompt.push_str(&format!("Input: {}\nOutput:", input_text));
        
        prompt
    }

    /// Add a custom example to the prompt template
    pub fn add_example(&mut self, input_with_typos: String, corrected_output: String) {
        self.examples.push((input_with_typos, corrected_output));
    }

    /// Clear all examples (for custom prompting)
    pub fn clear_examples(&mut self) {
        self.examples.clear();
    }

    /// Get the number of examples currently in the template
    pub fn example_count(&self) -> usize {
        self.examples.len()
    }
}

impl Default for PromptTemplate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompt_template_creation() {
        let template = PromptTemplate::new();
        assert!(!template.examples.is_empty());
        assert_eq!(template.example_count(), 3); // Optimized examples count
    }

    #[test]
    fn test_create_correction_prompt() {
        let template = PromptTemplate::new();
        let prompt = template.create_correction_prompt("this sentance has typoos");
        
        // Check that prompt contains few-shot examples
        assert!(prompt.contains("Fix typos in these sentences:"));
        assert!(prompt.contains("Input: I beleive this is teh answer."));
        assert!(prompt.contains("Output: I believe this is the answer."));
        
        // Check that it contains our input
        assert!(prompt.contains("Input: this sentance has typoos"));
        assert!(prompt.ends_with("Output:"));
    }

    #[test]
    fn test_add_custom_example() {
        let mut template = PromptTemplate::new();
        let initial_count = template.example_count();
        
        template.add_example(
            "definately wrong".to_string(),
            "definitely wrong".to_string()
        );
        
        assert_eq!(template.example_count(), initial_count + 1);
        
        let prompt = template.create_correction_prompt("test");
    // few-shot prompt includes examples before replacement; ensure custom example present
    assert!(prompt.contains("definately wrong"));
    assert!(prompt.contains("definitely wrong"));
    }

    #[test]
    fn test_clear_examples() {
        let mut template = PromptTemplate::new();
        assert!(!template.examples.is_empty());
        
        template.clear_examples();
        assert_eq!(template.example_count(), 0);
        
        let prompt = template.create_correction_prompt("test");
    // With no examples, still returns few-shot header + our input
    assert!(prompt.starts_with("Fix typos in these sentences:"));
    assert!(prompt.contains("Input: test"));
    assert!(prompt.ends_with("Output:"));
    }

    #[test]
    fn test_prompt_structure() {
        let template = PromptTemplate::new();
        let prompt = template.create_correction_prompt("hello wrold");
        
        // Check basic structure
        let lines: Vec<&str> = prompt.lines().collect();
    assert_eq!(lines[0], "Fix typos in these sentences:");
    assert_eq!(lines[1], "");
        
        // Should end with our input
        assert!(prompt.ends_with("Input: hello wrold\nOutput:"));
    }
}