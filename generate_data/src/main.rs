use clap::Parser;

#[derive(Parser, Debug)]
struct Args 
{
    out_path: String, 
    #[arg(long = "n-vectors", default_value_t = 50_000)]
    num_vectors: usize, 
    
    #[arg(long, default_value_t = 128)] 
    dim: usize,

    #[arg(long = "n-clusters", default_value_t = 200)]
    num_clusters: usize,

    #[arg(long = "cluster-std-dev", default_value_t = 1.5)] 
    cluster_std_dev: f64, 
    
    #[arg(long, default_value_t = 69)] 
    seed: u64, 
    
    #[arg(long, default_value_t = 1000)] 
    batch_size: usize, 
}

fn main() {
    println!("Hello, world!");
}
