use std::{io::{BufWriter, Write}, path::Path};

use memmap2::Mmap;

const HEADER_BYTES: usize = 8;
pub const WRITE_BUFFER_SIZE: usize = 16 * 1024 * 1024;

/// Mmap'd .vecf file representation
pub struct Vecf {
    mmap: Mmap,
    pub n_vectors: u32,
    pub dim: u32,
}

impl Vecf
{
    /// Loading vecfile via mmap
    pub fn open(path: &Path) -> std::io::Result<Self> {
        // Setting up the file mmap
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Very crude header file size error checking
        if mmap.len() < HEADER_BYTES {
            return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "File size is smaller than header size")
                );
        }
        // Decode the headers
        let n_vectors = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        let dim = u32::from_le_bytes(mmap[4..8].try_into().unwrap());

        // Content size checking
        let expected_bytes = HEADER_BYTES + (n_vectors as usize)*(dim as usize)*4;
        if mmap.len() < expected_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("File contents size mismatch: expected {expected_bytes} bytes, found {}", mmap.len())
            ));
        }
        
        Ok(Vecf { mmap, n_vectors, dim })

    }

    /// Viewing actual vectors data contents from file
    pub fn data(&self) -> &[f32] {
        // Getting the pointer to the contents
        let data = &self.mmap[HEADER_BYTES..];
        let ptr = data.as_ptr() as  *const f32;

        // Return the data as raw (immutable) slice
        let len = data.len() / 4;
        return unsafe { std::slice::from_raw_parts(ptr, len) };
    }

}

/// Writing .vecf serialization
pub fn write_vecf(path: &Path, vectors_data: &[f32], n_vectors: u32, dim: u32) -> std::io::Result<()> 
{
    let file = std::fs::File::create(path)?;
    let mut writer = BufWriter::with_capacity(WRITE_BUFFER_SIZE, file);
 
    writer.write_all(&n_vectors.to_le_bytes())?;
    writer.write_all(&dim.to_le_bytes())?;
 
    let mut byte_buf = Vec::with_capacity(vectors_data.len() * 4);
    for &float in vectors_data {
        byte_buf.extend_from_slice(&float.to_le_bytes());
    }
    writer.write_all(&byte_buf)?;
 
    writer.flush()?;
    Ok(())
}

