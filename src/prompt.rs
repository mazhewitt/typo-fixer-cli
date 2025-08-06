//! Prompt engineering for typo fixing
//! 
//! Based on the few-shot prompting approach from the Python demo

/// Template for typo fixing prompts with few-shot examples
pub struct PromptTemplate {
    examples: Vec<(String, String)>, // (input with typos, corrected output)
}

impl PromptTemplate {
    /// Create a new prompt template with default few-shot examples
    pub fn new() -> Self {
        let examples = vec![
            ("the quik brown fox".to_string(), "the quick brown fox".to_string()),
            ("i cant beleive it".to_string(), "i can't believe it".to_string()),
            ("recieve the package".to_string(), "receive the package".to_string()),
            ("seperate the items".to_string(), "separate the items".to_string()),
            ("occured yesterday".to_string(), "occurred yesterday".to_string()),
        ];

        Self { examples }
    }

    /// Create a typo correction prompt with few-shot examples
    pub fn create_correction_prompt(&self, input_text: &str) -> String {
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
        assert_eq!(template.example_count(), 5); // Default examples count
    }

    #[test]
    fn test_create_correction_prompt() {
        let template = PromptTemplate::new();
        let prompt = template.create_correction_prompt("this sentance has typoos");
        
        // Check that prompt contains few-shot examples
        assert!(prompt.contains("Fix typos in these sentences:"));
        assert!(prompt.contains("Input: the quik brown fox"));
        assert!(prompt.contains("Output: the quick brown fox"));
        
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
        // Should still have basic structure but no examples
        assert!(prompt.contains("Fix typos in these sentences:"));
        assert!(prompt.contains("Input: test"));
        assert!(prompt.ends_with("Output:"));
    }

    #[test]
    fn test_prompt_structure() {
        let template = PromptTemplate::new();
        let prompt = template.create_correction_prompt("hello wrold");
        
        // Check basic structure
        let lines: Vec<&str> = prompt.lines().collect();
        assert!(lines[0] == "Fix typos in these sentences:");
        assert!(lines[1] == "");
        
        // Should end with our input
        assert!(prompt.ends_with("Input: hello wrold\nOutput:"));
    }
}