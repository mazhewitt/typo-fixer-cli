//! Test to complete the typo-fixer model download

use anyhow::Result;

#[cfg(target_os = "macos")]
#[tokio::test]
#[ignore] // Run manually to complete download
async fn test_complete_typo_fixer_download() -> Result<()> {
    use candle_coreml::model_downloader;
    
    println!("🔄 Attempting to complete download of typo-fixer model");
    
    let model_id = "mazhewitt/qwen-typo-fixer";
    
    // Use the candle-coreml downloader with verbose output
    let model_path = model_downloader::ensure_model_downloaded(model_id, true)?;
    
    println!("✅ Download completed to: {:?}", model_path);
    
    // Verify the tokenizer.json file is now properly downloaded
    let tokenizer_path = model_path.join("tokenizer.json");
    if tokenizer_path.exists() {
        let metadata = std::fs::metadata(&tokenizer_path)?;
        println!("📄 tokenizer.json size: {} bytes", metadata.len());
        
        if metadata.len() > 1000 {
            println!("✅ tokenizer.json appears to be properly downloaded");
        } else {
            println!("⚠️  tokenizer.json is still an LFS pointer");
        }
    }
    
    Ok(())
}