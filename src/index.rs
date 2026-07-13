use std::fmt;

use half::f16;

use crate::distance::{DistanceMetric, vtype_to_scalar_kind};
use crate::types::{VectorType, cast_blob};

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
        let inner = usearch::Index::new(&opts).map_err(|e| IndexError(e.to_string()))?;
        Ok(Self { inner, vtype })
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if the index contains a vector for the given key.
    pub fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }

    /// Add a vector to the index. The blob must match the index's type and dimension.
    pub fn add(&self, key: u64, blob: &[u8]) -> Result<(), IndexError> {
        self.reserve_if_needed()?;
        match self.vtype {
            VectorType::Float4 => {
                let v = cast_blob::<f32>(blob);
                self.inner
                    .add(key, &v[..])
                    .map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Float8 => {
                let v = cast_blob::<f64>(blob);
                self.inner
                    .add(key, &v[..])
                    .map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int1 => {
                let v = cast_blob::<i8>(blob);
                self.inner
                    .add(key, &v[..])
                    .map_err(|e| IndexError(e.to_string()))
            }
            // Float2 (f16), Int2 (i16), Int4 (i32) are not natively supported by the usearch
            // generic VectorType trait as half::f16 — convert to f32 for index operations.
            VectorType::Float2 => {
                let v = cast_blob::<f16>(blob);
                let f: Vec<f32> = v.iter().map(|x| x.to_f32()).collect();
                self.inner
                    .add(key, &f)
                    .map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int2 => {
                let v = cast_blob::<i16>(blob);
                let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                self.inner
                    .add(key, &f)
                    .map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Int4 => {
                let v = cast_blob::<i32>(blob);
                let f: Vec<f32> = v.iter().map(|x| *x as f32).collect();
                self.inner
                    .add(key, &f)
                    .map_err(|e| IndexError(e.to_string()))
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
                let q = cast_blob::<f32>(query_blob);
                self.inner.search(&q[..], k)
            }
            VectorType::Float8 => {
                let q = cast_blob::<f64>(query_blob);
                self.inner.search(&q[..], k)
            }
            VectorType::Int1 => {
                let q = cast_blob::<i8>(query_blob);
                self.inner.search(&q[..], k)
            }
            VectorType::Float2 => {
                let q = cast_blob::<f16>(query_blob);
                let f: Vec<f32> = q.iter().map(|x| x.to_f32()).collect();
                self.inner.search(&f, k)
            }
            VectorType::Int2 => {
                let q = cast_blob::<i16>(query_blob);
                let f: Vec<f32> = q.iter().map(|x| *x as f32).collect();
                self.inner.search(&f, k)
            }
            VectorType::Int4 => {
                let q = cast_blob::<i32>(query_blob);
                let f: Vec<f32> = q.iter().map(|x| *x as f32).collect();
                self.inner.search(&f, k)
            }
        }
        .map_err(|e| IndexError(e.to_string()))?;

        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    /// Remove a vector by key (soft delete).
    pub fn remove(&self, key: u64) -> Result<(), IndexError> {
        self.inner
            .remove(key)
            .map(|_| ())
            .map_err(|e| IndexError(e.to_string()))
    }

    /// Serialize the index to a byte buffer.
    pub fn save_to_buffer(&self) -> Result<Vec<u8>, IndexError> {
        let len = self.inner.serialized_length();
        let mut buf = vec![0u8; len];
        self.inner
            .save_to_buffer(&mut buf)
            .map_err(|e| IndexError(e.to_string()))?;
        Ok(buf)
    }

    /// Load index state from a byte buffer. Replaces current index contents.
    pub fn load_from_buffer(&self, buf: &[u8]) -> Result<(), IndexError> {
        self.inner
            .load_from_buffer(buf)
            .map_err(|e| IndexError(e.to_string()))
    }

    /// Set the expansion factor for search.
    pub fn set_ef_search(&self, n: usize) {
        self.inner.change_expansion_search(n);
    }

    /// Get the current expansion factor for search.
    pub fn ef_search(&self) -> usize {
        self.inner.expansion_search()
    }

    /// Reserve capacity if needed (doubles current capacity).
    fn reserve_if_needed(&self) -> Result<(), IndexError> {
        if self.inner.size() >= self.inner.capacity() {
            let new_cap = (self.inner.capacity() * 2).max(64);
            self.inner
                .reserve(new_cap)
                .map_err(|e| IndexError(e.to_string()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::cast_slice;

    // ----------------------------------------------------------------
    // Helper: build a Float4 blob from a &[f32]
    // ----------------------------------------------------------------

    fn f32_blob(values: &[f32]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    fn f64_blob(values: &[f64]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    // ----------------------------------------------------------------
    // HnswParams::default
    // ----------------------------------------------------------------

    #[test]
    fn hnsw_params_default_values() {
        let p = HnswParams::default();
        assert_eq!(p.m, 16);
        assert_eq!(p.ef_construction, 200);
        assert_eq!(p.ef_search, 64);
    }

    // ----------------------------------------------------------------
    // HnswIndex::new
    // ----------------------------------------------------------------

    #[test]
    fn new_float4_l2_does_not_error() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None);
        assert!(idx.is_ok(), "expected Ok, got {:?}", idx.err());
    }

    #[test]
    fn new_float8_cosine_does_not_error() {
        let idx = HnswIndex::new(4, VectorType::Float8, DistanceMetric::Cosine, None);
        assert!(idx.is_ok(), "expected Ok, got {:?}", idx.err());
    }

    #[test]
    fn new_with_custom_params_does_not_error() {
        let params = HnswParams {
            m: 8,
            ef_construction: 64,
            ef_search: 32,
        };
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, Some(params));
        assert!(idx.is_ok(), "expected Ok, got {:?}", idx.err());
    }

    // ----------------------------------------------------------------
    // len / is_empty
    // ----------------------------------------------------------------

    #[test]
    fn len_empty_index_is_zero() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn len_increases_after_add() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();

        idx.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        assert_eq!(idx.len(), 1);
        assert!(!idx.is_empty());

        idx.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        assert_eq!(idx.len(), 2);

        idx.add(3, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();
        assert_eq!(idx.len(), 3);
    }

    // ----------------------------------------------------------------
    // add + search — orthogonal Float4 unit vectors
    // ----------------------------------------------------------------

    #[test]
    fn search_nearest_orthogonal_float4() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();

        idx.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        idx.add(3, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();

        // Query is close to [1, 0, 0] — key 1 must be the nearest neighbour.
        let results = idx.search(&f32_blob(&[0.9, 0.1, 0.0]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 1,
            "expected key 1 ([1,0,0]) as nearest, got key {}",
            results[0].0
        );
    }

    #[test]
    fn search_returns_empty_on_empty_index() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
        let results = idx.search(&f32_blob(&[1.0, 0.0, 0.0]), 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_k_larger_than_index_returns_all_vectors() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
        idx.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();

        // k=10 > index size 2; usearch returns at most size() results.
        let results = idx.search(&f32_blob(&[1.0, 0.0, 0.0]), 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    // ----------------------------------------------------------------
    // remove
    // ----------------------------------------------------------------

    #[test]
    fn remove_decreases_len() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();

        idx.add(10, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(20, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        idx.add(30, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();
        assert_eq!(idx.len(), 3);

        idx.remove(20).unwrap();
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn remove_key_no_longer_returned_by_search() {
        let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();

        idx.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        idx.add(3, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();

        // Remove the vector that would otherwise be the nearest to [0, 1, 0].
        idx.remove(2).unwrap();

        let results = idx.search(&f32_blob(&[0.0, 1.0, 0.0]), 3).unwrap();
        let returned_keys: Vec<u64> = results.iter().map(|(k, _)| *k).collect();
        assert!(
            !returned_keys.contains(&2),
            "removed key 2 should not appear in search results, got {:?}",
            returned_keys
        );
    }

    // ----------------------------------------------------------------
    // save_to_buffer / load_from_buffer — round-trip (Float4)
    // ----------------------------------------------------------------

    #[test]
    fn save_load_roundtrip_float4() {
        // Build and populate the source index.
        let src = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
        src.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        src.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        src.add(3, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();

        let buf = src.save_to_buffer().unwrap();
        assert!(!buf.is_empty(), "serialized buffer must not be empty");

        // Load into a fresh index with identical configuration.
        let dst = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
        dst.load_from_buffer(&buf).unwrap();

        // The loaded index must contain the same number of vectors.
        assert_eq!(dst.len(), src.len());

        // Search must still return the correct nearest neighbour.
        let results = dst.search(&f32_blob(&[0.9, 0.1, 0.0]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 1,
            "post-load search should return key 1, got {}",
            results[0].0
        );
    }

    // ----------------------------------------------------------------
    // Float8 (f64) type — add + search
    // ----------------------------------------------------------------

    #[test]
    fn add_search_float8() {
        let idx = HnswIndex::new(3, VectorType::Float8, DistanceMetric::L2, None).unwrap();

        idx.add(1, &f64_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(2, &f64_blob(&[0.0, 1.0, 0.0])).unwrap();
        idx.add(3, &f64_blob(&[0.0, 0.0, 1.0])).unwrap();

        let results = idx.search(&f64_blob(&[0.1, 0.0, 0.9]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 3,
            "expected key 3 ([0,0,1]) as nearest, got key {}",
            results[0].0
        );
    }

    #[test]
    fn save_load_roundtrip_float8() {
        let src = HnswIndex::new(3, VectorType::Float8, DistanceMetric::L2, None).unwrap();
        src.add(1, &f64_blob(&[1.0, 0.0, 0.0])).unwrap();
        src.add(2, &f64_blob(&[0.0, 1.0, 0.0])).unwrap();
        src.add(3, &f64_blob(&[0.0, 0.0, 1.0])).unwrap();

        let buf = src.save_to_buffer().unwrap();

        let dst = HnswIndex::new(3, VectorType::Float8, DistanceMetric::L2, None).unwrap();
        dst.load_from_buffer(&buf).unwrap();

        assert_eq!(dst.len(), 3);

        let results = dst.search(&f64_blob(&[0.0, 0.9, 0.1]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 2,
            "post-load search should return key 2, got {}",
            results[0].0
        );
    }

    // ----------------------------------------------------------------
    // Custom HnswParams
    // ----------------------------------------------------------------

    #[test]
    fn custom_params_index_behaves_correctly() {
        let params = HnswParams {
            m: 4,
            ef_construction: 32,
            ef_search: 16,
        };
        let idx =
            HnswIndex::new(3, VectorType::Float4, DistanceMetric::Cosine, Some(params)).unwrap();

        idx.add(1, &f32_blob(&[1.0, 0.0, 0.0])).unwrap();
        idx.add(2, &f32_blob(&[0.0, 1.0, 0.0])).unwrap();
        idx.add(3, &f32_blob(&[0.0, 0.0, 1.0])).unwrap();

        assert_eq!(idx.len(), 3);

        let results = idx.search(&f32_blob(&[0.0, 0.1, 0.9]), 1).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].0, 3,
            "expected key 3 ([0,0,1]) as nearest under cosine, got {}",
            results[0].0
        );
    }
}
