use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use pegainfer_dsv4f::Dsv4Config;
use pegainfer_dsv4f::Dsv4Manifest;

#[derive(Debug, Parser)]
#[command(about = "Validate a DeepSeek V4 Flash checkpoint without starting CUDA")]
struct Args {
    /// Raw Hugging Face checkpoint directory.
    model_path: PathBuf,

    /// Write the complete tensor ledger and validation summary as JSON.
    #[arg(long)]
    ledger_json: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let config = Dsv4Config::load(&args.model_path)?;
    let report = Dsv4Manifest::inspect(&args.model_path, &config)?;

    println!("DSV4F G0 validation passed");
    println!("  model: {}", report.model_path);
    println!("  shards: {}", report.shard_count);
    println!(
        "  tensors: {} target + {} DSpark skip = {}",
        report.target_tensor_count, report.dspark_skip_tensor_count, report.tensor_count
    );
    println!(
        "  payload: {} target + {} DSpark skip = {} bytes",
        report.target_source_bytes, report.dspark_skip_source_bytes, report.source_bytes
    );
    println!("  load actions: {:?}", report.load_action_counts);

    if let Some(path) = args.ledger_json {
        let bytes = serde_json::to_vec_pretty(&report).context("serialize DSV4F G0 report")?;
        std::fs::write(&path, bytes)
            .with_context(|| format!("write DSV4F tensor ledger to {}", path.display()))?;
        println!("  ledger: {}", path.display());
    }
    Ok(())
}
