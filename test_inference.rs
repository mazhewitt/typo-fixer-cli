use anyhow::Result;
use candle_coreml::UnifiedModelLoader;

#[tokio::main]
async fn main() -> Result<()> {
    let loader = UnifiedModelLoader::new()?;
    let mut model = loader.load_model("anemll/anemll-Qwen-Qwen3-0.6B-LUT888-ctx512_0.3.4")?;
    
    let prompt = "The quick brown fox jumps over the lazy";
    println!("Prompt: {}", prompt);
    
    // Generate first token
    let token1 = model.forward_text(prompt)?;
    println!("Token 1: {}", token1);
    if let Ok(decoded) = model.tokenizer().decode(&[token1 as u32], false) {
        println!("Decoded 1: '{}'", decoded);
    }
    
    // Generate second token with appended first token
    let prompt2 = format!("{}{}", prompt, 
        model.tokenizer().decode(&[token1 as u32], false).unwrap_or_default());
    println!("\nPrompt 2: {}", prompt2);
    let token2 = model.forward_text(&prompt2)?;
    println!("Token 2: {}", token2);
    if let Ok(decoded) = model.tokenizer().decode(&[token2 as u32], false) {
        println!("Decoded 2: '{}'", decoded);
    }
    
    Ok(())
}
