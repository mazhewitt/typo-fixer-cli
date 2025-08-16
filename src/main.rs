use anyhow::Result;
use clap::Parser;
use std::io::{self, Read};
use serde_json::json;
use tracing::debug;
use tracing_subscriber::fmt;
use typo_fixer_cli::{TypoFixerLib, cli::{Args, OutputFormat}};

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    
    // Validate CLI arguments
    args.validate()?;
    
    if args.verbose {
        println!("🚀 Typo Fixer CLI starting...");
        println!("   Model: {}", args.model);
        println!("   Temperature: {}", args.temperature);
        println!("   Max tokens: {}", args.max_tokens);
    }

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter("debug") // Adjust the filter level as needed (e.g., "info", "warn", etc.)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    if args.verbose {
        debug!("Tracing subscriber set up with env filter 'debug'");
    }
    
    // Initialize the typo fixer
    let mut typo_fixer = if let Some(local_path) = &args.local_path {
        if args.verbose {
            println!("   Using local model path: {}", local_path);
        }
        
        if let Some(config_path) = &args.config {
            if args.verbose {
                println!("   Using configuration file: {}", config_path);
            }
            TypoFixerLib::new_with_config_file(config_path, local_path, args.verbose).await?
        } else {
            TypoFixerLib::new_from_local(local_path.clone(), args.verbose).await?
        }
    } else {
        TypoFixerLib::new(Some(args.model.clone()), args.verbose).await?
    };
    
    // Get input text
    let input_text = if args.stdin {
        read_from_stdin()?
    } else {
        args.input.clone().unwrap() // Safe because we validated
    };
    
    if args.batch {
        process_batch(&mut typo_fixer, &input_text, &args).await?;
    } else {
        process_single(&mut typo_fixer, &input_text, &args).await?;
    }
    
    Ok(())
}

async fn process_single(
    typo_fixer: &mut TypoFixerLib, 
    input: &str, 
    args: &Args
) -> Result<()> {
    let corrected = typo_fixer.fix_typos_with_options(
        input, 
        args.temperature, 
        Some(args.max_tokens)
    ).await?;
    
    match args.output {
        OutputFormat::Text => {
            println!("{}", corrected);
        },
        OutputFormat::Json => {
            let output = json!({
                "input": input,
                "output": corrected,
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        },
        OutputFormat::Verbose => {
            println!("Input:  {}", input);
            println!("Output: {}", corrected);
            println!("Model:  {}", args.model);
        }
    }
    
    Ok(())
}

async fn process_batch(
    typo_fixer: &mut TypoFixerLib,
    input: &str,
    args: &Args
) -> Result<()> {
    let lines: Vec<&str> = input.lines().collect();
    let mut results = Vec::new();
    
    if args.verbose {
        println!("📋 Processing {} lines in batch mode", lines.len());
    }
    
    for (i, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        
        if args.verbose {
            println!("Processing line {}/{}: {}", i + 1, lines.len(), line);
        }
        
        let corrected = typo_fixer.fix_typos_with_options(
            line, 
            args.temperature, 
            Some(args.max_tokens)
        ).await?;
        
        results.push((line, corrected));
    }
    
    // Output results
    match args.output {
        OutputFormat::Text => {
            for (_original, corrected) in results {
                println!("{}", corrected);
            }
        },
        OutputFormat::Json => {
            let output = json!({
                "results": results.iter().map(|(original, corrected)| {
                    json!({
                        "input": original,
                        "output": corrected
                    })
                }).collect::<Vec<_>>(),
                "model": args.model,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens
            });
            println!("{}", serde_json::to_string_pretty(&output)?);
        },
        OutputFormat::Verbose => {
            for (original, corrected) in results {
                println!("Input:  {}", original);
                println!("Output: {}", corrected);
                println!("---");
            }
            println!("Processed {} lines with model: {}", lines.len(), args.model);
        }
    }
    
    Ok(())
}

fn read_from_stdin() -> Result<String> {
    let mut buffer = String::new();
    io::stdin().read_to_string(&mut buffer)?;
    Ok(buffer.trim().to_string())
}
