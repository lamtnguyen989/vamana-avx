mod query;

use std::{collections::BinaryHeap, path::{Path, PathBuf}};

use clap::{Parser, ValueEnum};
use rayon::prelude::*;

use crate::query::QueryItem;

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
    /// Data file path: [Internally default relative paths to <PROJECT_ROOT>/<data_path>]
    data_path: PathBuf,

    /// Queries output path [Internally default relative paths to <PROJECT_ROOT>/<queries_out_path>]
    queries_out_path: PathBuf,

    /// Output path for the ground truth CSV [Internally default relative paths to <PROJECT_ROOT>/<ground_truth_csv>]
    ground_truth_csv: PathBuf,

    /// Number of queries to generate
    #[arg(short = 'n', long = "n-queries", default_value_t = 1000)]
    n_queries: usize,

    /// Number of nearest neighbors to compute per query
    #[arg(short = 'k', long, default_value_t = 100)]
    neighbors: usize,

    /// Standard deviation of the Gaussian noise added to sampled base points
    #[arg(long = "std-dev", default_value_t = 0.5)]
    std_dev: f64,

    /// Number of threads to be used in generation concurrently
    #[arg(short, long = "threads", default_value_t = 4)] 
    threads: usize, 

    /// RNG seed
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    /// Batch size for vector writes
    #[arg(short, long, default_value_t = 2000)] 
    batch_size: usize, 

    /// Distance metric used for ground truth computation
    #[arg(long, value_enum, default_value_t = Metric::L2)]
    metric: Metric,

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
/// Realtive paths are default to <PROJECT_ROOT>/<input_file>
fn resolve_input_path(user_path: &Path) -> std::io::Result<PathBuf> {
    if user_path.is_absolute() {
        return Ok(user_path.to_path_buf());
    }
 
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let project_root = crate_dir.parent().unwrap_or(&crate_dir);

    Ok(project_root.join(user_path))
}

/// Sorting Query Heap in to pairs of Vecs
fn sort_query(heap: BinaryHeap<QueryItem>) -> (Vec<u32>, Vec<f32>) {
    let mut neighbors = heap.into_vec();
    neighbors.sort();

    let indices = neighbors.iter().map(|neighbor| neighbor.id).collect();
    let distances = neighbors.iter().map(|neighbor| neighbor.dist).collect();

    return (indices, distances);
}

/// Computing L2 nearest neighbors with respect to a single query
fn knn_l2(query: &[f32], data: &[f32], n_vectors: usize, dim: usize, k: usize) -> (Vec<u32>, Vec<f32>)
{
    let mut heap: BinaryHeap<QueryItem> = BinaryHeap::with_capacity(k+1);

    for i in 0..n_vectors {
        // Computing L2 distance from query vector to the data vector
        let data_vector = &data[k*dim..(i+1)*dim];
        let mut diff_sq = 0.0_f32;
        for j in 0..dim {
            diff_sq = (data_vector[j] - query[j]) * (data_vector[j] - query[j]);
        }
        let dist = diff_sq.sqrt();

        // Push Item onto the max-binary heap
        if heap.len() < k {
            heap.push(QueryItem { dist: dist, id: i as u32 });
        }
        else if let Some(worst) = heap.peek() {
            if dist < worst.dist {
                heap.pop();
                heap.push(QueryItem { dist: dist, id: i as u32 });
            }
        }
    }

    return sort_query(heap);
}

/// Computing ground truths with respect to L2-metric
fn ground_truths_l2()
{
    todo!();
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

}
