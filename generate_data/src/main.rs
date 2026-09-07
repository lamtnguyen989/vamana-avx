use std::{io::{BufWriter, Write}, path::{Path, PathBuf}};

use clap::Parser;
use rayon::prelude::*;

const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;   // 8 MB for now

#[derive(Parser, Debug)]
struct Args 
{
    /// Output path: [Internally default relative paths to <PROJECT_ROOT>/<output_file>]
    output_file: PathBuf, 

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

    /// Number of threads to be used in generation concurrently
    #[arg(long = "threads", default_value_t = 4)] 
    threads: usize, 
    
    /// RNG seed
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    /// Batch size for vector writes
    #[arg(long, default_value_t = 2000)] 
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
    // Note the default is to write to <PROJECT_ROOT>/<output_file>
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);
    let full_path = project_root.join(user_path);

    // Create all path leading up to the `user_path`
    if let Some(parent) = full_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    Ok(full_path)
}

fn generate_vecfile(args: &Args) -> std::io::Result<()> {
    // Open up file for serializing
    let vecfile = std::fs::File::create(&args.output_file)?;

    // Initialize the buffered writer (mmap possibility later but now safety first)
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, vecfile);

    // Writing header
    writer.write_all(&(args.num_vectors as u32).to_le_bytes())?;
    writer.write_all(&(args.dim as u32).to_le_bytes())?;

    todo!();
}

fn main() -> std::io::Result<()> {
    // Parsing CLI
    let mut args = Args::parse();

    // Resolving output path
    args.output_file = resolve_output_path(&args.output_file)?;

    // Setting up Rayon threadpool
    let rayon_setup = rayon::ThreadPoolBuilder::new()
                        .num_threads(args.threads)
                        .build_global();
    match rayon_setup {
        Ok(_) => println!("Global threadpool initialized successfully."),
        Err(e) => panic!("Failed to Initialize global threadpool. Error: {}", e)
    }

    // Generate data and serialize to file
    generate_vecfile(&args)?;


    println!("Wrote data to {:?}", args.output_file);

    Ok(())
}
