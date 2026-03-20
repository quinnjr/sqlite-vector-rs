use std::fmt;

use bytemuck::cast_slice;
use half::f16;

use crate::distance::{DistanceMetric, vtype_to_scalar_kind};
use crate::types::VectorType;

/// Optional HNSW tuning parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
        }
    }
}

#[derive(Debug)]
pub struct IndexError(pub String);

impl fmt::Display for IndexError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "index error: {}", self.0)
    }
}

impl std::error::Error for IndexError {}

/// Wrapper around usearch::Index providing a typed interface.
pub struct HnswIndex {
    inner: usearch::Index,
    dim: usize,
    vtype: VectorType,
}

impl HnswIndex {
    /// Create a new empty HNSW index.
    pub fn new(
        dim: usize,
        vtype: VectorType,
        metric: DistanceMetric,
        params: Option<HnswParams>,
    ) -> Result<Self, IndexError> {
        let p = params.unwrap_or_default();
        let opts = usearch::IndexOptions {
            dimensions: dim,
            metric: metric.to_usearch(),
            quantization: vtype_to_scalar_kind(vtype),
            connectivity: p.m,
            expansion_add: p.ef_construction,
            expansion_search: p.ef_search,
            multi: false,
        };
        let inner = usearch::Index::new(&opts)
            .map_err(|e| IndexError(e.to_string()))?;
        Ok(Self { inner, dim, vtype })
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Add a vector to the index. The blob must match the index's type and dimension.
    pub fn add(&self, key: u64, blob: &[u8]) -> Result<(), IndexError> {
        self.reserve_if_needed()?;
        match self.vtype {
            VectorType::Float4 => {
                let v: &[f32] = cast_slice(blob);
                self.inner.add(key, v).map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Float8 => {
                let v: &[f64] = cast_slice(blob);
                self.inner.add(key, v).map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int1 => {
                let v: &[i8] = cast_slice(blob);
                self.inner.add(key, v).map_err(|e| IndexError(e.to_string()))
            }
            // Float2 (f16), Int2 (i16), Int4 (i32) are not natively supported by the usearch
            // generic VectorType trait as half::f16 — convert to f32 for index operations.
            VectorType::Float2 => {
                let v: &[f16] = cast_slice(blob);
                let f: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
                self.inner.add(key, &f).map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int2 => {
                let v: &[i16] = cast_slice(blob);
                let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                self.inner.add(key, &f).map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int4 => {
                let v: &[i32] = cast_slice(blob);
                let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                self.inner.add(key, &f).map_err(|e| IndexError(e.to_string()))
            }
        }
    }

    /// Search for k nearest neighbors. Returns vec of (key, distance) pairs
    /// sorted by distance ascending.
    pub fn search(&self, query_blob: &[u8], k: usize) -> Result<Vec<(u64, f32)>, IndexError> {
        if self.is_empty() {
            return Ok(Vec::new());
        }

        let matches = match self.vtype {
            VectorType::Float4 => {
                let q: &[f32] = cast_slice(query_blob);
                self.inner.search(q, k)
            }
            VectorType::Float8 => {
                let q: &[f64] = cast_slice(query_blob);
                self.inner.search(q, k)
            }
            VectorType::Int1 => {
                let q: &[i8] = cast_slice(query_blob);
                self.inner.search(q, k)
            }
            VectorType::Float2 => {
                let q: &[f16] = cast_slice(query_blob);
                let f: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
                self.inner.search(&f, k)
            }
            VectorType::Int2 => {
                let q: &[i16] = cast_slice(query_blob);
                let f: Vec<f32> = q.iter().map(|x| *x as f32).collect();
                self.inner.search(&f, k)
            }
            VectorType::Int4 => {
                let q: &[i32] = cast_slice(query_blob);
                let f: Vec<f32> = q.iter().map(|x| *x as f32).collect();
                self.inner.search(&f, k)
            }
        }.map_err(|e| IndexError(e.to_string()))?;

        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    /// Remove a vector by key (soft delete).
    pub fn remove(&self, key: u64) -> Result<(), IndexError> {
        self.inner.remove(key)
            .map(|_| ())
            .map_err(|e| IndexError(e.to_string()))
    }

    /// Serialize the index to a byte buffer.
    pub fn save_to_buffer(&self) -> Result<Vec<u8>, IndexError> {
        let len = self.inner.serialized_length();
        let mut buf = vec![0u8; len];
        self.inner.save_to_buffer(&mut buf)
            .map_err(|e| IndexError(e.to_string()))?;
        Ok(buf)
    }

    /// Load index state from a byte buffer. Replaces current index contents.
    pub fn load_from_buffer(&self, buf: &[u8]) -> Result<(), IndexError> {
        self.inner.load_from_buffer(buf)
            .map_err(|e| IndexError(e.to_string()))
    }

    /// Reserve capacity if needed (doubles current capacity).
    fn reserve_if_needed(&self) -> Result<(), IndexError> {
        if self.inner.size() >= self.inner.capacity() {
            let new_cap = (self.inner.capacity() * 2).max(64);
            self.inner.reserve(new_cap)
                .map_err(|e| IndexError(e.to_string()))?;
        }
        Ok(())
    }
}
