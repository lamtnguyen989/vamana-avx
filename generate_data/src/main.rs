use std::path::{Path, PathBuf};

use clap::Parser;
use rayon::prelude::*;

#[derive(Parser, Debug)]
struct Args 
{
    /// Output path: (Internally default to <PROJECT_ROOT>/data/<output_path>)
    out_path: PathBuf, 

    /// Total number of vectors to be generated
    #[arg(long = "n-vectors", default_value_t = 50_000)]
    num_vectors: usize, 
    
    /// Dimension of the vector space
    #[arg(long, default_value_t = 128)] 
    dim: usize,

    /// Number of vector clusters
    #[arg(long = "n-clusters", default_value_t = 200)]
    num_clusters: usize,

    /// Standard deviations for each vector blobs
    #[arg(long = "cluster-std-dev", default_value_t = 1.5)] 
    cluster_std_dev: f64, 
    
    /// RNG seed
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    /// Batch size for vector writes
    #[arg(long, default_value_t = 1000)] 
    batch_size: usize, 
}

fn resolve_output_path(user_path: &Path) -> std::io::Result<PathBuf> {
    // Obey the absolute path outputs
    if user_path.is_absolute() {
        if let Some(parent) = user_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        return Ok(user_path.to_path_buf());
    }

    // Relative path processing
    // Note the default is to write to <PROJECT_ROOT>/data/<data_name>.vecf
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);

    let data_dir = project_root.join("data");
    std::fs::create_dir_all(&data_dir)?;

    let filename = user_path.file_name().unwrap_or(user_path.as_os_str());
    Ok(data_dir.join(filename))
}

fn main() -> std::io::Result<()> {
    // Parsing CLI
    let mut args = Args::parse();

    // Resolving output path
    args.out_path = resolve_output_path(&args.out_path)?;

    // Generate data and serialize to file


    println!("Wrote data to {:?}", args.out_path);

    Ok(())
}
