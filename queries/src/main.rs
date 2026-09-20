use std::path::{Path, PathBuf};

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

const HEADER_BYTES: usize = 8;
const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;


#[derive(Copy, Clone, Debug, ValueEnum)]
enum Metric {
    L2,
    Cosine,
}

#[derive(Debug, Parser)]
struct Args
{
    /// Data file path: [Internally default relative paths to <PROJECT_ROOT>/<output_file>]
    data_path: PathBuf,

    /// Queries output path
    queries_ouut_path: PathBuf,

    /// Output path for the ground truth CSV
    ground_truth_csv: PathBuf,

    /// Number of threads to be used in generation concurrently
    #[arg(short, long = "threads", default_value_t = 4)] 
    threads: usize, 
}

/// Resolving user output file paths
/// Absolute paths are honered as-is
/// Realtive paths are default to <PROJECT_ROOT>/<output_file>
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

/// Resolving user input file paths
/// Absolute paths are honered as-is
/// Realtive paths are default to <PROJECT_ROOT>/<output_file>
fn resolve_input_path(user_path: &Path) -> std::io::Result<PathBuf> {
    if user_path.is_absolute() {
        return Ok(user_path.to_path_buf());
    }
 
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);
    Ok(project_root.join(user_path))
}

fn main() {
    // Parsing CLI
    let mut args = Args::parse();

    // Setting up Rayon threadpool
    if args.threads < 1 {
        eprintln!("Thread count can not be less than 1! Default thread count to 1.");
        args.threads = 1;
    }
    rayon::ThreadPoolBuilder::new()
                        .num_threads(args.threads)
                        .build_global()
                        .unwrap_or_else(|e| panic!("Failed to initialize global threadpool. Error: {}", e));

    println!("Hello, world!");
}
