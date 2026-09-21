use std::path::Path;

use memmap2::Mmap;

pub const HEADER_BYTES: usize = 8;

/// Mmap'd .vecf file representation
pub struct Vecf {
    pub mmap: Mmap,
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

    /// Viewing vectors data from file
    pub fn data(&self) -> &[f32] {
        // Getting the pointer to the contents
        let data = &self.mmap[HEADER_BYTES..];
        let ptr = data.as_ptr() as  *const f32;

        // Return the data as raw (immutable) slice
        let len = data.len() / 4;
        return unsafe { std::slice::from_raw_parts(ptr, len) };
    }

}
