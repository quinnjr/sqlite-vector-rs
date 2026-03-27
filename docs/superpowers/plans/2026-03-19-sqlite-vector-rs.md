# sqlite-vector-rs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust SQLite extension providing PGVector-like native vector types with HNSW indexing for nearest-neighbor search on pre-computed embeddings.

**Architecture:** Dual-output Rust crate (`cdylib` + `rlib`). `sqlite3_ext` (vendored fork) provides the loadable extension framework and virtual table API. `usearch` provides HNSW indexing with built-in persistence and SIMD-accelerated distance computation. Vectors stored as raw typed byte blobs per-row; Arrow IPC for bulk import/export. Six element types (f16, f32, f64, i8, i16, i32) with three distance metrics (L2, cosine, inner product).

**Tech Stack:** Rust (edition 2024), sqlite3_ext, usearch, arrow-array/arrow-ipc/arrow-schema, half, bytemuck, serde_json, rusqlite (dev/testing)

**Spec:** `docs/superpowers/specs/2026-03-19-sqlite-vector-rs-design.md`

---

## File Structure

```
sqlite-vector-rs/
├── Cargo.toml
├── src/
│   ├── lib.rs                  # Crate root, feature gates, public API, extension entry point
│   ├── types.rs                # VectorType enum, element size, byte casting, validation
│   ├── json.rs                 # JSON <-> vector blob conversion
│   ├── distance.rs             # Distance metric enum, computation via usearch metric API
│   ├── index.rs                # HnswIndex wrapper around usearch (create, add, remove, search, serialize)
│   ├── vtab/
│   │   ├── mod.rs              # Virtual table module struct, VTab/CreateVTab/UpdateVTab/TransactionVTab impls
│   │   ├── config.rs           # Parse CREATE VIRTUAL TABLE arguments into VectorTableConfig
│   │   ├── cursor.rs           # VTabCursor impl (scan + KNN modes)
│   │   ├── shadow.rs           # Shadow table creation, read/write for embeddings_data and embeddings_index
│   │   └── transaction.rs      # VTabTransaction impl (sync/commit/rollback with dirty flag)
│   ├── scalar.rs               # Standalone scalar functions (vector_distance, vector_from_json, etc.)
│   └── arrow_io.rs             # Arrow IPC bulk import/export (vector_insert_arrow, vector_export_arrow)
├── tests/
│   ├── common/mod.rs           # Shared test helpers (create in-memory DB, register module, generate random vectors)
│   ├── types_test.rs           # Unit tests for VectorType, byte casting, validation
│   ├── json_test.rs            # Unit tests for JSON parsing/serialization
│   ├── distance_test.rs        # Unit tests for distance computation (known values, all type×metric combos)
│   ├── index_test.rs           # Unit tests for HnswIndex wrapper (add, search, remove, serialize round-trip)
│   ├── vtab_test.rs            # Integration tests for virtual table (CREATE, INSERT, KNN, DELETE, UPDATE)
│   ├── scalar_test.rs          # Integration tests for standalone scalar functions
│   ├── arrow_test.rs           # Integration tests for Arrow bulk I/O
│   └── persistence_test.rs     # Integration tests for index persistence across connections
└── vendor/
    └── sqlite3_ext/            # Vendored fork of CGamesPlay/sqlite3_ext
```

**Important usearch type limitation:** usearch's Rust bindings implement `VectorType` for `f32`, `f64`, `i8`, and its own `f16` type — but NOT for `i16` or `i32`. For `int2` and `int4` element types, vectors must be cast to `f32` before indexing/searching with usearch. The raw `i16`/`i32` bytes are still stored in `embeddings_data`; the cast only applies to the HNSW index operations. Similarly, usearch's `f16` is its own type, so `half::f16` values need conversion when passed to usearch.

---

## Task 1: Project Setup & Dependencies

**Files:**
- Modify: `Cargo.toml`
- Create: `src/lib.rs`
- Create: `vendor/sqlite3_ext/` (git subtree or copy)
- Create: `.gitignore` (update)

- [ ] **Step 1: Vendor sqlite3_ext**

Clone the `CGamesPlay/sqlite3_ext` repo into `vendor/`:

```bash
git clone https://github.com/CGamesPlay/sqlite3_ext.git vendor/sqlite3_ext
rm -rf vendor/sqlite3_ext/.git
```

- [ ] **Step 2: Write Cargo.toml**

```toml
[package]
name = "sqlite-vector-rs"
version = "0.1.0"
edition = "2024"

[lib]
crate-type = ["cdylib", "rlib"]

[features]
default = ["loadable_extension"]
loadable_extension = ["sqlite3_ext/loadable_extension"]
library = ["dep:rusqlite"]

[dependencies]
sqlite3_ext = { path = "vendor/sqlite3_ext" }
usearch = "2"
arrow-array = "58"
arrow-buffer = "58"
arrow-ipc = "58"
arrow-schema = "58"
half = { version = "2", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }
serde_json = "1"
rusqlite = { version = "0.39", features = ["bundled", "vtab"], optional = true }

[dev-dependencies]
rusqlite = { version = "0.39", features = ["bundled", "vtab"] }
rand = "0.9"
```

- [ ] **Step 3: Create minimal src/lib.rs**

```rust
pub mod types;
```

- [ ] **Step 4: Create stub src/types.rs**

```rust
/// Vector element types supported by the extension.
pub enum VectorType {
    Float2,
    Float4,
    Float8,
    Int1,
    Int2,
    Int4,
}
```

- [ ] **Step 5: Verify the project compiles**

Run: `cargo check`
Expected: compiles with no errors (warnings OK at this stage)

- [ ] **Step 6: Update .gitignore**

```
/target
vendor/sqlite3_ext/.git
```

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/lib.rs src/types.rs vendor/sqlite3_ext .gitignore
git commit -m "feat: scaffold project with dependencies and vendored sqlite3_ext

Set up dual-output crate (cdylib + rlib) with feature flags for
loadable_extension and library modes. Vendor sqlite3_ext for the
SQLite extension framework. Add usearch, arrow, half, bytemuck,
and serde_json dependencies."
```

---

## Task 2: Vector Type System

**Files:**
- Create: `src/types.rs` (replace stub)
- Create: `tests/types_test.rs`

- [ ] **Step 1: Write failing tests for VectorType**

Create `tests/types_test.rs`:

```rust
use sqlite_vector_rs::types::{VectorType, VectorTypeError};

#[test]
fn parse_type_names() {
    assert_eq!(VectorType::from_name("float4").unwrap(), VectorType::Float4);
    assert_eq!(VectorType::from_name("float2").unwrap(), VectorType::Float2);
    assert_eq!(VectorType::from_name("float8").unwrap(), VectorType::Float8);
    assert_eq!(VectorType::from_name("int1").unwrap(), VectorType::Int1);
    assert_eq!(VectorType::from_name("int2").unwrap(), VectorType::Int2);
    assert_eq!(VectorType::from_name("int4").unwrap(), VectorType::Int4);
    assert!(VectorType::from_name("float16").is_err());
    assert!(VectorType::from_name("").is_err());
}

#[test]
fn element_size_bytes() {
    assert_eq!(VectorType::Float2.element_size(), 2);
    assert_eq!(VectorType::Float4.element_size(), 4);
    assert_eq!(VectorType::Float8.element_size(), 8);
    assert_eq!(VectorType::Int1.element_size(), 1);
    assert_eq!(VectorType::Int2.element_size(), 2);
    assert_eq!(VectorType::Int4.element_size(), 4);
}

#[test]
fn validate_blob_size() {
    // 3-dim float4 = 12 bytes
    let vtype = VectorType::Float4;
    assert!(vtype.validate_blob(&[0u8; 12], 3).is_ok());
    assert!(vtype.validate_blob(&[0u8; 8], 3).is_err());
    assert!(vtype.validate_blob(&[0u8; 16], 3).is_err());
}

#[test]
fn blob_round_trip_float4() {
    let values: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob = VectorType::Float4.slice_to_blob(&values);
    assert_eq!(blob.len(), 12);
    let restored: &[f32] = VectorType::Float4.blob_to_slice(&blob);
    assert_eq!(restored, &[1.0, 2.0, 3.0]);
}

#[test]
fn blob_round_trip_float2() {
    use half::f16;
    let values: Vec<f16> = vec![f16::from_f32(1.0), f16::from_f32(2.0)];
    let blob = VectorType::Float2.slice_to_blob(&values);
    assert_eq!(blob.len(), 4);
    let restored: &[f16] = VectorType::Float2.blob_to_slice(&blob);
    assert_eq!(restored[0].to_f32(), 1.0);
    assert_eq!(restored[1].to_f32(), 2.0);
}

#[test]
fn reject_nan_inf_float4() {
    let with_nan: Vec<f32> = vec![1.0, f32::NAN, 3.0];
    let blob = VectorType::Float4.slice_to_blob(&with_nan);
    assert!(VectorType::Float4.validate_finite(&blob, 3).is_err());

    let with_inf: Vec<f32> = vec![1.0, f32::INFINITY, 3.0];
    let blob = VectorType::Float4.slice_to_blob(&with_inf);
    assert!(VectorType::Float4.validate_finite(&blob, 3).is_err());
}

#[test]
fn validate_finite_skips_integer_types() {
    // Integer types don't have NaN/Inf, validate_finite should always pass
    let blob = VectorType::Int4.slice_to_blob(&[1i32, 2, 3]);
    assert!(VectorType::Int4.validate_finite(&blob, 3).is_ok());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test types_test`
Expected: compilation errors (types not implemented yet)

- [ ] **Step 3: Implement VectorType**

Replace `src/types.rs`:

```rust
use std::fmt;

use bytemuck::{Pod, Zeroable, cast_slice};
use half::f16;

/// Errors from vector type operations.
#[derive(Debug, Clone, PartialEq)]
pub enum VectorTypeError {
    UnknownType(String),
    DimensionMismatch { expected: usize, got: usize },
    NonFiniteValue,
}

impl fmt::Display for VectorTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownType(name) => write!(f, "unknown vector type: {name}"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "expected {expected} dimensions, got {got}")
            }
            Self::NonFiniteValue => write!(f, "vector contains NaN or Inf"),
        }
    }
}

impl std::error::Error for VectorTypeError {}

/// Supported vector element types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorType {
    Float2,
    Float4,
    Float8,
    Int1,
    Int2,
    Int4,
}

impl VectorType {
    /// Parse a SQL type name into a VectorType.
    pub fn from_name(name: &str) -> Result<Self, VectorTypeError> {
        match name {
            "float2" => Ok(Self::Float2),
            "float4" => Ok(Self::Float4),
            "float8" => Ok(Self::Float8),
            "int1" => Ok(Self::Int1),
            "int2" => Ok(Self::Int2),
            "int4" => Ok(Self::Int4),
            other => Err(VectorTypeError::UnknownType(other.to_string())),
        }
    }

    /// Size in bytes of one element.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Float2 => 2,
            Self::Float4 => 4,
            Self::Float8 => 8,
            Self::Int1 => 1,
            Self::Int2 => 2,
            Self::Int4 => 4,
        }
    }

    /// Expected blob size for a given dimension.
    pub fn blob_size(&self, dim: usize) -> usize {
        dim * self.element_size()
    }

    /// Validate that a blob has the correct size for the given dimension.
    pub fn validate_blob(&self, blob: &[u8], dim: usize) -> Result<(), VectorTypeError> {
        let expected = self.blob_size(dim);
        if blob.len() != expected {
            return Err(VectorTypeError::DimensionMismatch {
                expected: dim,
                got: blob.len() / self.element_size(),
            });
        }
        Ok(())
    }

    /// Check that all float values are finite (not NaN or Inf).
    /// No-op for integer types. The `dim` parameter is used to verify
    /// the blob length before casting.
    pub fn validate_finite(&self, blob: &[u8], dim: usize) -> Result<(), VectorTypeError> {
        self.validate_blob(blob, dim)?;
        match self {
            Self::Float2 => {
                let values: &[f16] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float4 => {
                let values: &[f32] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Float8 => {
                let values: &[f64] = cast_slice(blob);
                if values.iter().any(|v| !v.is_finite()) {
                    return Err(VectorTypeError::NonFiniteValue);
                }
            }
            Self::Int1 | Self::Int2 | Self::Int4 => {} // integers are always finite
        }
        Ok(())
    }

    /// Cast a typed slice to a byte blob. Generic helper.
    pub fn slice_to_blob<T: Pod>(values: &[T]) -> Vec<u8> {
        cast_slice(values).to_vec()
    }

    /// Cast a byte blob back to a typed slice. Generic helper.
    /// Caller must ensure the blob was created with the matching type.
    pub fn blob_to_slice<T: Pod>(blob: &[u8]) -> &[T] {
        cast_slice(blob)
    }

    /// Returns true if this is a float type (has NaN/Inf concerns).
    pub fn is_float(&self) -> bool {
        matches!(self, Self::Float2 | Self::Float4 | Self::Float8)
    }

    /// SQL type name string.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Float2 => "float2",
            Self::Float4 => "float4",
            Self::Float8 => "float8",
            Self::Int1 => "int1",
            Self::Int2 => "int2",
            Self::Int4 => "int4",
        }
    }
}
```

Note: `slice_to_blob` and `blob_to_slice` are deliberately standalone generic functions on the type. The caller knows the concrete type `T` at each call site. The `VectorType` enum is used for dispatch/validation, not for calling these generics.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test types_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/types.rs tests/types_test.rs
git commit -m "feat(types): implement VectorType enum with byte casting and validation

Six element types (float2/4/8, int1/2/4) with name parsing, element
size calculation, blob size validation, and NaN/Inf rejection for
float types. Uses bytemuck for zero-copy casting between typed slices
and byte buffers."
```

---

## Task 3: JSON Conversion

**Files:**
- Create: `src/json.rs`
- Create: `tests/json_test.rs`
- Modify: `src/lib.rs` (add module)

- [ ] **Step 1: Write failing tests for JSON conversion**

Create `tests/json_test.rs`:

```rust
use sqlite_vector_rs::json::{json_to_blob, blob_to_json};
use sqlite_vector_rs::types::VectorType;

#[test]
fn json_to_float4_blob() {
    let blob = json_to_blob("[1.0, 2.0, 3.0]", VectorType::Float4).unwrap();
    let values: &[f32] = VectorType::blob_to_slice(&blob);
    assert_eq!(values, &[1.0, 2.0, 3.0]);
}

#[test]
fn json_to_float2_blob() {
    let blob = json_to_blob("[1.0, 2.0]", VectorType::Float2).unwrap();
    assert_eq!(blob.len(), 4); // 2 elements * 2 bytes
}

#[test]
fn json_to_int1_blob() {
    let blob = json_to_blob("[1, 2, -3]", VectorType::Int1).unwrap();
    let values: &[i8] = VectorType::blob_to_slice(&blob);
    assert_eq!(values, &[1, 2, -3]);
}

#[test]
fn blob_to_json_float4() {
    let values: Vec<f32> = vec![1.5, 2.5, 3.5];
    let blob = VectorType::slice_to_blob(&values);
    let json = blob_to_json(&blob, VectorType::Float4).unwrap();
    assert_eq!(json, "[1.5,2.5,3.5]");
}

#[test]
fn json_rejects_non_array() {
    assert!(json_to_blob("{}", VectorType::Float4).is_err());
    assert!(json_to_blob("42", VectorType::Float4).is_err());
}

#[test]
fn json_rejects_non_numeric_elements() {
    assert!(json_to_blob("[1.0, \"hello\", 3.0]", VectorType::Float4).is_err());
}

#[test]
fn json_rejects_nan() {
    // NaN is not valid JSON, but just in case serde allows it
    assert!(json_to_blob("[1.0, null, 3.0]", VectorType::Float4).is_err());
}

#[test]
fn json_empty_array() {
    let blob = json_to_blob("[]", VectorType::Float4).unwrap();
    assert!(blob.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test json_test`
Expected: compilation errors

- [ ] **Step 3: Implement JSON conversion**

Create `src/json.rs`:

```rust
use half::f16;
use serde_json::Value;

use crate::types::{VectorType, VectorTypeError};

/// Errors from JSON conversion.
#[derive(Debug)]
pub enum JsonError {
    Parse(serde_json::Error),
    NotAnArray,
    NonNumericElement(usize),
    Type(VectorTypeError),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "invalid JSON: {e}"),
            Self::NotAnArray => write!(f, "expected a JSON array"),
            Self::NonNumericElement(i) => write!(f, "element {i} is not a number"),
            Self::Type(e) => write!(f, "{e}"),
        }
    }
}

use std::fmt;
impl std::error::Error for JsonError {}

/// Parse a JSON array string into a vector blob of the given type.
pub fn json_to_blob(json: &str, vtype: VectorType) -> Result<Vec<u8>, JsonError> {
    let value: Value = serde_json::from_str(json).map_err(JsonError::Parse)?;
    let arr = value.as_array().ok_or(JsonError::NotAnArray)?;

    match vtype {
        VectorType::Float2 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))?;
                let h = f16::from_f64(n);
                if !h.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(h);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Float4 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))? as f32;
                if !n.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(n);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Float8 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_f64().ok_or(JsonError::NonNumericElement(i))?;
                if !n.is_finite() {
                    return Err(JsonError::Type(VectorTypeError::NonFiniteValue));
                }
                values.push(n);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i8;
                values.push(n);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i16;
                values.push(n);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let mut values = Vec::with_capacity(arr.len());
            for (i, v) in arr.iter().enumerate() {
                let n = v.as_i64().ok_or(JsonError::NonNumericElement(i))? as i32;
                values.push(n);
            }
            Ok(VectorType::slice_to_blob(&values))
        }
    }
}

/// Convert a vector blob back to a JSON array string.
pub fn blob_to_json(blob: &[u8], vtype: VectorType) -> Result<String, JsonError> {
    let values: Vec<Value> = match vtype {
        VectorType::Float2 => {
            let s: &[f16] = VectorType::blob_to_slice(blob);
            s.iter().map(|v| Value::from(v.to_f64())).collect()
        }
        VectorType::Float4 => {
            let s: &[f32] = VectorType::blob_to_slice(blob);
            // Use serde_json's From<f32> to get shortest round-trip representation
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Float8 => {
            let s: &[f64] = VectorType::blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v)).collect()
        }
        VectorType::Int1 => {
            let s: &[i8] = VectorType::blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int2 => {
            let s: &[i16] = VectorType::blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
        VectorType::Int4 => {
            let s: &[i32] = VectorType::blob_to_slice(blob);
            s.iter().map(|v| Value::from(*v as i64)).collect()
        }
    };
    serde_json::to_string(&values).map_err(JsonError::Parse)
}
```

Add to `src/lib.rs`:

```rust
pub mod types;
pub mod json;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test json_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/json.rs src/lib.rs tests/json_test.rs
git commit -m "feat(json): implement JSON-to-blob and blob-to-JSON conversion

Parses JSON arrays into typed vector byte blobs for all six element
types. Validates non-null numeric elements and rejects NaN/Inf for
float types. Round-trip conversion preserves values."
```

---

## Task 4: Distance Computation

**Files:**
- Create: `src/distance.rs`
- Create: `tests/distance_test.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests for distance functions**

Create `tests/distance_test.rs`:

```rust
use sqlite_vector_rs::distance::{DistanceMetric, compute_distance};
use sqlite_vector_rs::types::VectorType;

#[test]
fn l2_identical_vectors_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let dist = compute_distance(&blob_a, &blob_a, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
    assert!((dist - 0.0).abs() < 1e-6);
}

#[test]
fn l2_known_distance_float4() {
    let a: Vec<f32> = vec![1.0, 0.0, 0.0];
    let b: Vec<f32> = vec![0.0, 1.0, 0.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let blob_b = VectorType::slice_to_blob(&b);
    let dist = compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::L2, 3).unwrap();
    // Squared L2: (1-0)^2 + (0-1)^2 + (0-0)^2 = 2.0
    assert!((dist - 2.0).abs() < 1e-6);
}

#[test]
fn cosine_identical_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let dist = compute_distance(&blob_a, &blob_a, VectorType::Float4, DistanceMetric::Cosine, 3).unwrap();
    assert!(dist.abs() < 1e-5);
}

#[test]
fn cosine_orthogonal_float4() {
    let a: Vec<f32> = vec![1.0, 0.0];
    let b: Vec<f32> = vec![0.0, 1.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let blob_b = VectorType::slice_to_blob(&b);
    let dist = compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::Cosine, 2).unwrap();
    // cosine distance = 1 - cos(90°) = 1.0
    assert!((dist - 1.0).abs() < 1e-5);
}

#[test]
fn inner_product_known_float4() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![4.0, 5.0, 6.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let blob_b = VectorType::slice_to_blob(&b);
    let dist = compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::InnerProduct, 3).unwrap();
    // -dot(a,b) = -(1*4 + 2*5 + 3*6) = -32.0
    assert!((dist - (-32.0)).abs() < 1e-6);
}

#[test]
fn parse_metric_names() {
    assert_eq!(DistanceMetric::from_name("l2").unwrap(), DistanceMetric::L2);
    assert_eq!(DistanceMetric::from_name("cosine").unwrap(), DistanceMetric::Cosine);
    assert_eq!(DistanceMetric::from_name("ip").unwrap(), DistanceMetric::InnerProduct);
    assert!(DistanceMetric::from_name("hamming").is_err());
}

#[test]
fn distance_dimension_mismatch() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0];
    let b: Vec<f32> = vec![1.0, 2.0];
    let blob_a = VectorType::slice_to_blob(&a);
    let blob_b = VectorType::slice_to_blob(&b);
    assert!(compute_distance(&blob_a, &blob_b, VectorType::Float4, DistanceMetric::L2, 3).is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test distance_test`
Expected: compilation errors

- [ ] **Step 3: Implement distance computation**

Create `src/distance.rs`:

```rust
use std::fmt;

use bytemuck::cast_slice;
use half::f16;

use crate::types::VectorType;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    L2,
    Cosine,
    InnerProduct,
}

impl DistanceMetric {
    pub fn from_name(name: &str) -> Result<Self, DistanceError> {
        match name {
            "l2" => Ok(Self::L2),
            "cosine" => Ok(Self::Cosine),
            "ip" => Ok(Self::InnerProduct),
            other => Err(DistanceError::UnknownMetric(other.to_string())),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::L2 => "l2",
            Self::Cosine => "cosine",
            Self::InnerProduct => "ip",
        }
    }

    /// Convert to usearch MetricKind.
    pub fn to_usearch(&self) -> usearch::MetricKind {
        match self {
            Self::L2 => usearch::MetricKind::L2sq,
            Self::Cosine => usearch::MetricKind::Cos,
            Self::InnerProduct => usearch::MetricKind::IP,
        }
    }
}

#[derive(Debug)]
pub enum DistanceError {
    UnknownMetric(String),
    DimensionMismatch,
    Usearch(String),
}

impl fmt::Display for DistanceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownMetric(name) => write!(f, "unknown metric: {name}"),
            Self::DimensionMismatch => write!(f, "vector dimensions do not match"),
            Self::Usearch(e) => write!(f, "usearch error: {e}"),
        }
    }
}

impl std::error::Error for DistanceError {}

/// Compute distance between two vector blobs.
///
/// Both blobs must be the same type and dimension. For int2/int4 types,
/// values are cast to f32 before computation since usearch only supports
/// f32, f64, f16, and i8 natively.
pub fn compute_distance(
    a: &[u8],
    b: &[u8],
    vtype: VectorType,
    metric: DistanceMetric,
    dim: usize,
) -> Result<f64, DistanceError> {
    let expected_size = vtype.blob_size(dim);
    if a.len() != expected_size || b.len() != expected_size {
        return Err(DistanceError::DimensionMismatch);
    }

    match vtype {
        VectorType::Float4 => {
            let va: &[f32] = cast_slice(a);
            let vb: &[f32] = cast_slice(b);
            Ok(scalar_distance(va, vb, metric))
        }
        VectorType::Float8 => {
            let va: &[f64] = cast_slice(a);
            let vb: &[f64] = cast_slice(b);
            Ok(scalar_distance_f64(va, vb, metric))
        }
        VectorType::Float2 => {
            // Convert half::f16 to f32 for computation
            let va: &[f16] = cast_slice(a);
            let vb: &[f16] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| v.to_f32()).collect();
            let fb: Vec<f32> = vb.iter().map(|v| v.to_f32()).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int1 => {
            let va: &[i8] = cast_slice(a);
            let vb: &[i8] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int2 => {
            let va: &[i16] = cast_slice(a);
            let vb: &[i16] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
        VectorType::Int4 => {
            let va: &[i32] = cast_slice(a);
            let vb: &[i32] = cast_slice(b);
            let fa: Vec<f32> = va.iter().map(|v| *v as f32).collect();
            let fb: Vec<f32> = vb.iter().map(|v| *v as f32).collect();
            Ok(scalar_distance(&fa, &fb, metric))
        }
    }
}

fn scalar_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() as f64
        }
        DistanceMetric::Cosine => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) as f64 }
        }
        DistanceMetric::InnerProduct => {
            let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            -(dot as f64)
        }
    }
}

fn scalar_distance_f64(a: &[f64], b: &[f64], metric: DistanceMetric) -> f64 {
    match metric {
        DistanceMetric::L2 => {
            a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
        }
        DistanceMetric::Cosine => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            let denom = norm_a * norm_b;
            if denom == 0.0 { 1.0 } else { 1.0 - (dot / denom) }
        }
        DistanceMetric::InnerProduct => {
            let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
            -dot
        }
    }
}

/// Map VectorType to usearch ScalarKind for index creation.
pub fn vtype_to_scalar_kind(vtype: VectorType) -> usearch::ScalarKind {
    match vtype {
        VectorType::Float2 => usearch::ScalarKind::F16,
        VectorType::Float4 => usearch::ScalarKind::F32,
        VectorType::Float8 => usearch::ScalarKind::F64,
        VectorType::Int1 => usearch::ScalarKind::I8,
        // i16/i32 not natively supported by usearch, quantize to f32
        VectorType::Int2 | VectorType::Int4 => usearch::ScalarKind::F32,
    }
}
```

Note: The initial implementation uses scalar (non-SIMD) distance computation for `compute_distance` so we have full control and predictable results. The HNSW index in Task 5 will use usearch's internal SIMD-accelerated computation for search operations. The scalar functions serve as a fallback and reference implementation.

Update `src/lib.rs`:

```rust
pub mod types;
pub mod json;
pub mod distance;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test distance_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/distance.rs src/lib.rs tests/distance_test.rs
git commit -m "feat(distance): implement L2, cosine, and inner product distance computation

Scalar distance computation for all six element types and three
metrics. Non-native usearch types (i16, i32, f16) are cast to f32
for computation. Includes metric name parsing and usearch type
mapping for index creation."
```

---

## Task 5: HNSW Index Wrapper

**Files:**
- Create: `src/index.rs`
- Create: `tests/index_test.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests for HnswIndex**

Create `tests/index_test.rs`:

```rust
use sqlite_vector_rs::index::HnswIndex;
use sqlite_vector_rs::types::VectorType;
use sqlite_vector_rs::distance::DistanceMetric;

#[test]
fn create_empty_index() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    assert_eq!(idx.len(), 0);
    assert!(idx.is_empty());
}

#[test]
fn add_and_search_float4() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    let v3: Vec<f32> = vec![0.0, 0.0, 1.0];
    idx.add(1, &VectorType::slice_to_blob(&v1)).unwrap();
    idx.add(2, &VectorType::slice_to_blob(&v2)).unwrap();
    idx.add(3, &VectorType::slice_to_blob(&v3)).unwrap();

    let query: Vec<f32> = vec![1.0, 0.1, 0.0];
    let results = idx.search(&VectorType::slice_to_blob(&query), 2).unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest should be v1
}

#[test]
fn remove_vector() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    idx.add(1, &VectorType::slice_to_blob(&v1)).unwrap();
    idx.add(2, &VectorType::slice_to_blob(&v2)).unwrap();
    assert_eq!(idx.len(), 2);

    idx.remove(1).unwrap();
    // After soft delete, size may still report 2, but search should not return key 1
    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx.search(&VectorType::slice_to_blob(&query), 2).unwrap();
    // Key 1 should not appear in results
    assert!(!results.iter().any(|(k, _)| *k == 1));
}

#[test]
fn serialize_round_trip() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    let v2: Vec<f32> = vec![0.0, 1.0, 0.0];
    idx.add(1, &VectorType::slice_to_blob(&v1)).unwrap();
    idx.add(2, &VectorType::slice_to_blob(&v2)).unwrap();

    let buf = idx.save_to_buffer().unwrap();
    assert!(!buf.is_empty());

    let idx2 = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    idx2.load_from_buffer(&buf).unwrap();

    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx2.search(&VectorType::slice_to_blob(&query), 1).unwrap();
    assert_eq!(results[0].0, 1);
}

#[test]
fn search_empty_index() {
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::L2, None).unwrap();
    let query: Vec<f32> = vec![1.0, 0.0, 0.0];
    let results = idx.search(&VectorType::slice_to_blob(&query), 10).unwrap();
    assert!(results.is_empty());
}

#[test]
fn custom_hnsw_params() {
    use sqlite_vector_rs::index::HnswParams;
    let params = HnswParams { m: 32, ef_construction: 400, ef_search: 128 };
    let idx = HnswIndex::new(3, VectorType::Float4, DistanceMetric::Cosine, Some(params)).unwrap();
    let v1: Vec<f32> = vec![1.0, 0.0, 0.0];
    idx.add(1, &VectorType::slice_to_blob(&v1)).unwrap();
    assert_eq!(idx.len(), 1);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test index_test`
Expected: compilation errors

- [ ] **Step 3: Implement HnswIndex**

Create `src/index.rs`:

```rust
use std::fmt;

use bytemuck::cast_slice;
use half::f16;

use crate::distance::{DistanceMetric, vtype_to_scalar_kind};
use crate::types::VectorType;

/// Optional HNSW tuning parameters.
#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    pub m: u32,
    pub ef_construction: u32,
    pub ef_search: u32,
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
                // usearch supports i8 natively
                self.inner.add(key, v).map_err(|e| IndexError(e.to_string()))
            }
            VectorType::Float2 => {
                // Convert half::f16 to f32 for usearch
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
        self.inner.remove(key).map_err(|e| IndexError(e.to_string()))?;
        Ok(())
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
            .map_err(|e| IndexError(e.to_string()))?;
        Ok(())
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
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod json;
pub mod distance;
pub mod index;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test index_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/index.rs src/lib.rs tests/index_test.rs
git commit -m "feat(index): implement HnswIndex wrapper around usearch

Typed wrapper handling the usearch API differences for all six element
types. Non-native types (f16, i16, i32) are cast to f32 for index
operations while preserving original storage format. Supports add,
remove (soft delete), search, and buffer-based serialization for
persistence in SQLite shadow tables."
```

---

## Task 6: Virtual Table Config Parsing

**Files:**
- Create: `src/vtab/mod.rs`
- Create: `src/vtab/config.rs`
- Create: `tests/common/mod.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests for config parsing**

Create `tests/common/mod.rs`:

```rust
// Shared test helpers - will grow as needed
```

Add to a new section in `tests/types_test.rs` or create a dedicated test. For simplicity, test config parsing in the vtab test file. Create `tests/vtab_config_test.rs`:

```rust
use sqlite_vector_rs::vtab::config::VectorTableConfig;
use sqlite_vector_rs::types::VectorType;
use sqlite_vector_rs::distance::DistanceMetric;

#[test]
fn parse_basic_args() {
    // sqlite3_ext passes args as: ["module_name", "db_name", "table_name", "dim=3", "type=float4", "metric=l2"]
    let args = vec!["vector", "main", "embeddings", "dim=3", "type=float4", "metric=l2"];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.dim, 3);
    assert_eq!(config.vtype, VectorType::Float4);
    assert_eq!(config.metric, DistanceMetric::L2);
    assert_eq!(config.table_name, "embeddings");
    assert!(config.metadata_columns.is_empty());
}

#[test]
fn parse_with_hnsw_params() {
    let args = vec!["vector", "main", "emb", "dim=768", "type=float4", "metric=cosine", "m=32", "ef_construction=400", "ef_search=128"];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.hnsw_params.m, 32);
    assert_eq!(config.hnsw_params.ef_construction, 400);
    assert_eq!(config.hnsw_params.ef_search, 128);
}

#[test]
fn parse_with_metadata() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2", "metadata=\"label TEXT, category INTEGER\""];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.metadata_columns.len(), 2);
    assert_eq!(config.metadata_columns[0], ("label".to_string(), "TEXT".to_string()));
    assert_eq!(config.metadata_columns[1], ("category".to_string(), "INTEGER".to_string()));
}

#[test]
fn parse_defaults() {
    let args = vec!["vector", "main", "emb", "dim=3"];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.vtype, VectorType::Float4); // default
    assert_eq!(config.metric, DistanceMetric::L2); // default
    assert_eq!(config.hnsw_params.m, 16); // default
}

#[test]
fn parse_missing_dim_fails() {
    let args = vec!["vector", "main", "emb", "type=float4"];
    assert!(VectorTableConfig::parse(&args).is_err());
}

#[test]
fn parse_invalid_dim_fails() {
    let args = vec!["vector", "main", "emb", "dim=0"];
    assert!(VectorTableConfig::parse(&args).is_err());

    let args = vec!["vector", "main", "emb", "dim=-5"];
    assert!(VectorTableConfig::parse(&args).is_err());
}

#[test]
fn generates_create_table_sql() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2"];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = config.vtab_schema();
    assert!(sql.contains("id INTEGER PRIMARY KEY"));
    assert!(sql.contains("vector BLOB"));
    assert!(sql.contains("distance REAL"));
}

#[test]
fn generates_schema_with_metadata() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2", "metadata=\"label TEXT, category INTEGER\""];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = config.vtab_schema();
    assert!(sql.contains("label TEXT"));
    assert!(sql.contains("category INTEGER"));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test vtab_config_test`
Expected: compilation errors

- [ ] **Step 3: Implement VectorTableConfig**

Create `src/vtab/mod.rs`:

```rust
pub mod config;
```

Create `src/vtab/config.rs`:

```rust
use std::fmt;

use crate::distance::DistanceMetric;
use crate::index::HnswParams;
use crate::types::VectorType;

#[derive(Debug)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config error: {}", self.0)
    }
}

impl std::error::Error for ConfigError {}

/// Parsed configuration from CREATE VIRTUAL TABLE arguments.
#[derive(Debug)]
pub struct VectorTableConfig {
    pub db_name: String,
    pub table_name: String,
    pub dim: usize,
    pub vtype: VectorType,
    pub metric: DistanceMetric,
    pub hnsw_params: HnswParams,
    /// Metadata columns as (name, sql_type) pairs.
    pub metadata_columns: Vec<(String, String)>,
}

impl VectorTableConfig {
    /// Parse CREATE VIRTUAL TABLE arguments.
    ///
    /// Args from sqlite3_ext: ["module_name", "db_name", "table_name", ...params]
    pub fn parse(args: &[&str]) -> Result<Self, ConfigError> {
        if args.len() < 3 {
            return Err(ConfigError("expected at least module, db, and table name".into()));
        }

        let db_name = args[1].to_string();
        let table_name = args[2].to_string();

        let mut dim: Option<usize> = None;
        let mut vtype = VectorType::Float4;
        let mut metric = DistanceMetric::L2;
        let mut hnsw_params = HnswParams::default();
        let mut metadata_columns = Vec::new();

        for &arg in &args[3..] {
            let (key, value) = arg.split_once('=')
                .ok_or_else(|| ConfigError(format!("invalid argument: {arg}")))?;
            let key = key.trim();
            let value = value.trim().trim_matches('"');

            match key {
                "dim" => {
                    let d: i64 = value.parse()
                        .map_err(|_| ConfigError(format!("invalid dim: {value}")))?;
                    if d <= 0 {
                        return Err(ConfigError(format!("dim must be positive, got {d}")));
                    }
                    dim = Some(d as usize);
                }
                "type" => {
                    vtype = VectorType::from_name(value)
                        .map_err(|e| ConfigError(e.to_string()))?;
                }
                "metric" => {
                    metric = DistanceMetric::from_name(value)
                        .map_err(|e| ConfigError(e.to_string()))?;
                }
                "m" => {
                    hnsw_params.m = value.parse()
                        .map_err(|_| ConfigError(format!("invalid m: {value}")))?;
                }
                "ef_construction" => {
                    hnsw_params.ef_construction = value.parse()
                        .map_err(|_| ConfigError(format!("invalid ef_construction: {value}")))?;
                }
                "ef_search" => {
                    hnsw_params.ef_search = value.parse()
                        .map_err(|_| ConfigError(format!("invalid ef_search: {value}")))?;
                }
                "metadata" => {
                    metadata_columns = parse_metadata_columns(value)?;
                }
                other => {
                    return Err(ConfigError(format!("unknown parameter: {other}")));
                }
            }
        }

        let dim = dim.ok_or_else(|| ConfigError("dim is required".into()))?;

        Ok(Self {
            db_name,
            table_name,
            dim,
            vtype,
            metric,
            hnsw_params,
            metadata_columns,
        })
    }

    /// Generate the CREATE TABLE schema string for sqlite3_ext's connect/create return.
    pub fn vtab_schema(&self) -> String {
        let mut cols = vec![
            "id INTEGER PRIMARY KEY".to_string(),
            "vector BLOB".to_string(),
        ];
        for (name, sql_type) in &self.metadata_columns {
            cols.push(format!("{name} {sql_type}"));
        }
        cols.push("distance REAL HIDDEN".to_string());
        format!("CREATE TABLE x({})", cols.join(", "))
    }
}

fn parse_metadata_columns(spec: &str) -> Result<Vec<(String, String)>, ConfigError> {
    let mut columns = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let mut tokens = part.split_whitespace();
        let name = tokens.next()
            .ok_or_else(|| ConfigError(format!("empty metadata column definition")))?
            .to_string();
        let sql_type = tokens.next()
            .ok_or_else(|| ConfigError(format!("missing type for metadata column {name}")))?
            .to_string();
        columns.push((name, sql_type));
    }
    Ok(columns)
}
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod json;
pub mod distance;
pub mod index;
pub mod vtab;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test vtab_config_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vtab/ src/lib.rs tests/vtab_config_test.rs
git commit -m "feat(vtab): implement CREATE VIRTUAL TABLE argument parsing

Parses dim, type, metric, HNSW tuning parameters (m, ef_construction,
ef_search), and metadata column definitions from the CREATE VIRTUAL
TABLE arguments. Generates the virtual table schema string. Provides
sensible defaults (float4, l2, m=16, ef_construction=200, ef_search=64)."
```

---

## Task 7: Shadow Table Management

**Files:**
- Create: `src/vtab/shadow.rs`
- Modify: `src/vtab/mod.rs`

This task implements the SQL generation for the `_data` and `_index` shadow tables.

- [ ] **Step 1: Write failing tests for shadow SQL generation**

Create `tests/shadow_test.rs`:

```rust
use sqlite_vector_rs::vtab::config::VectorTableConfig;
use sqlite_vector_rs::vtab::shadow::ShadowOps;

#[test]
fn create_data_table_sql_basic() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2"];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = ShadowOps::create_data_table_sql(&config);
    assert!(sql.contains("\"emb_data\""));
    assert!(sql.contains("id INTEGER PRIMARY KEY AUTOINCREMENT"));
    assert!(sql.contains("vector BLOB NOT NULL"));
}

#[test]
fn create_data_table_sql_with_metadata() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2", "metadata=\"label TEXT, score REAL\""];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = ShadowOps::create_data_table_sql(&config);
    assert!(sql.contains("label TEXT"));
    assert!(sql.contains("score REAL"));
}

#[test]
fn create_index_table_sql() {
    let args = vec!["vector", "main", "emb", "dim=3"];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = ShadowOps::create_index_table_sql(&config);
    assert!(sql.contains("\"emb_index\""));
    assert!(sql.contains("key TEXT PRIMARY KEY"));
    assert!(sql.contains("value BLOB"));
}

#[test]
fn insert_data_sql_with_metadata() {
    let args = vec!["vector", "main", "emb", "dim=3", "type=float4", "metric=l2", "metadata=\"label TEXT\""];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = ShadowOps::insert_data_sql(&config);
    assert!(sql.contains("\"emb_data\""));
    assert!(sql.contains("vector, label"));
    assert!(sql.contains("?, ?"));
}

#[test]
fn drop_shadow_tables() {
    let stmts = ShadowOps::drop_shadow_tables_sql("emb");
    assert_eq!(stmts.len(), 2);
    assert!(stmts[0].contains("\"emb_data\""));
    assert!(stmts[1].contains("\"emb_index\""));
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test shadow_test`
Expected: compilation errors

- [ ] **Step 3: Implement shadow table operations**

Create `src/vtab/shadow.rs`:

```rust
use crate::vtab::config::VectorTableConfig;

/// SQL statements for shadow table management.
///
/// Shadow tables:
/// - `{table}_data`: stores vector blobs and metadata
/// - `{table}_index`: stores serialized HNSW graph and config
pub struct ShadowOps;

impl ShadowOps {
    /// SQL to create the _data shadow table.
    pub fn create_data_table_sql(config: &VectorTableConfig) -> String {
        let mut cols = vec![
            "id INTEGER PRIMARY KEY AUTOINCREMENT".to_string(),
            "vector BLOB NOT NULL".to_string(),
        ];
        for (name, sql_type) in &config.metadata_columns {
            cols.push(format!("{name} {sql_type}"));
        }
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_data\"({})",
            config.table_name,
            cols.join(", ")
        )
    }

    /// SQL to create the _index shadow table.
    pub fn create_index_table_sql(config: &VectorTableConfig) -> String {
        format!(
            "CREATE TABLE IF NOT EXISTS \"{}_index\"(key TEXT PRIMARY KEY, value BLOB)",
            config.table_name
        )
    }

    /// SQL to drop both shadow tables.
    pub fn drop_shadow_tables_sql(table_name: &str) -> Vec<String> {
        vec![
            format!("DROP TABLE IF EXISTS \"{table_name}_data\""),
            format!("DROP TABLE IF EXISTS \"{table_name}_index\""),
        ]
    }

    /// SQL to insert a vector row into _data.
    pub fn insert_data_sql(config: &VectorTableConfig) -> String {
        let mut col_names = vec!["vector".to_string()];
        let mut placeholders = vec!["?".to_string()];
        for (name, _) in &config.metadata_columns {
            col_names.push(name.clone());
            placeholders.push("?".to_string());
        }
        format!(
            "INSERT INTO \"{}_data\"({}) VALUES({})",
            config.table_name,
            col_names.join(", "),
            placeholders.join(", ")
        )
    }

    /// SQL to delete a vector row from _data by id.
    pub fn delete_data_sql(table_name: &str) -> String {
        format!("DELETE FROM \"{table_name}_data\" WHERE id = ?")
    }

    /// SQL to select a vector row from _data by id.
    pub fn select_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\" WHERE id = ?")
    }

    /// SQL to select all vector rows from _data (for full scan or index rebuild).
    pub fn select_all_data_sql(table_name: &str) -> String {
        format!("SELECT * FROM \"{table_name}_data\"")
    }

    /// SQL to upsert a key-value pair into _index.
    pub fn upsert_index_sql(table_name: &str) -> String {
        format!(
            "INSERT OR REPLACE INTO \"{table_name}_index\"(key, value) VALUES(?, ?)"
        )
    }

    /// SQL to read a value from _index by key.
    pub fn select_index_sql(table_name: &str) -> String {
        format!("SELECT value FROM \"{table_name}_index\" WHERE key = ?")
    }
}
```

Update `src/vtab/mod.rs`:

```rust
pub mod config;
pub mod shadow;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test shadow_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/vtab/shadow.rs src/vtab/mod.rs tests/shadow_test.rs
git commit -m "feat(vtab): implement shadow table SQL generation

SQL builders for the _data and _index shadow tables: create, drop,
insert, delete, select, and upsert operations. The _data table
stores vector blobs and metadata columns. The _index table stores
the serialized HNSW graph and configuration as key-value pairs."
```

---

## Task 8: Virtual Table Core (VTab + CreateVTab + UpdateVTab)

**Files:**
- Modify: `src/vtab/mod.rs` (main vtab implementation)
- Create: `src/vtab/cursor.rs`
- Create: `src/vtab/transaction.rs`

This is the largest task — it wires together all previous components into a working SQLite virtual table. Due to the tight coupling with sqlite3_ext's trait system, this task implements the full virtual table in one pass, then tests it in Task 10.

- [ ] **Step 1: Implement the VTabCursor**

Create `src/vtab/cursor.rs`:

```rust
use sqlite3_ext::vtab::{ColumnContext, VTabCursor};
use sqlite3_ext::{Result, ValueRef};

/// Cursor modes for the vector virtual table.
pub enum CursorMode {
    /// Full table scan — iterates all rows from _data.
    Scan {
        rows: Vec<ScanRow>,
        pos: usize,
    },
    /// KNN search — iterates results from HNSW index.
    Knn {
        results: Vec<KnnRow>,
        pos: usize,
    },
}

pub struct ScanRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Option<Vec<u8>>>,
}

pub struct KnnRow {
    pub id: i64,
    pub vector: Vec<u8>,
    pub metadata: Vec<Option<Vec<u8>>>,
    pub distance: f64,
}

pub struct VectorCursor {
    pub mode: CursorMode,
    pub num_metadata_cols: usize,
}

impl VTabCursor for VectorCursor {
    fn filter(
        &mut self,
        index_num: i32,
        _index_str: Option<&str>,
        _args: &mut [&mut ValueRef],
    ) -> Result<()> {
        // index_num == 1: KNN search (populated by best_index)
        // index_num == 0: full scan
        // Actual filtering logic is set up by the VTab before calling filter.
        // The cursor just iterates the pre-populated rows.
        match &mut self.mode {
            CursorMode::Scan { pos, .. } => *pos = 0,
            CursorMode::Knn { pos, .. } => *pos = 0,
        }
        Ok(())
    }

    fn next(&mut self) -> Result<()> {
        match &mut self.mode {
            CursorMode::Scan { pos, .. } => *pos += 1,
            CursorMode::Knn { pos, .. } => *pos += 1,
        }
        Ok(())
    }

    fn eof(&mut self) -> bool {
        match &self.mode {
            CursorMode::Scan { rows, pos } => *pos >= rows.len(),
            CursorMode::Knn { results, pos } => *pos >= results.len(),
        }
    }

    fn column(&mut self, idx: usize, ctx: &ColumnContext) -> Result<()> {
        // Column indices: 0=id, 1=vector, 2..2+N=metadata, last=distance
        match &self.mode {
            CursorMode::Scan { rows, pos } => {
                let row = &rows[*pos];
                match idx {
                    0 => ctx.set_result(row.id),
                    1 => ctx.set_result(&row.vector[..]),
                    i if i >= 2 && i < 2 + self.num_metadata_cols => {
                        if let Some(val) = &row.metadata[i - 2] {
                            ctx.set_result(&val[..]);
                        }
                        // else: NULL (default)
                    }
                    _ => {} // distance column: NULL for scan mode
                }
            }
            CursorMode::Knn { results, pos } => {
                let row = &results[*pos];
                match idx {
                    0 => ctx.set_result(row.id),
                    1 => ctx.set_result(&row.vector[..]),
                    i if i >= 2 && i < 2 + self.num_metadata_cols => {
                        if let Some(val) = &row.metadata[i - 2] {
                            ctx.set_result(&val[..]);
                        }
                    }
                    _ => ctx.set_result(row.distance), // distance column
                }
            }
        }
        Ok(())
    }

    fn rowid(&mut self) -> Result<i64> {
        match &self.mode {
            CursorMode::Scan { rows, pos } => Ok(rows[*pos].id),
            CursorMode::Knn { results, pos } => Ok(results[*pos].id),
        }
    }
}
```

Note: The exact `sqlite3_ext` cursor API types (`ColumnContext::set_result`, `ValueRef`, etc.) may need adjustment when compiling against the vendored fork. The implementer should consult `vendor/sqlite3_ext/src/vtab.rs` for exact method signatures and adapt accordingly. The structure and logic above are correct; only type-level details may differ.

- [ ] **Step 2: Implement the Transaction handler**

Create `src/vtab/transaction.rs`:

```rust
use sqlite3_ext::vtab::VTabTransaction;
use sqlite3_ext::Result;

use std::sync::Arc;
use std::cell::RefCell;

use crate::index::HnswIndex;

/// Shared mutable state for the virtual table's index.
pub struct IndexState {
    pub index: HnswIndex,
    pub dirty: bool,
    /// Serialized index from last commit (for rollback).
    pub last_committed: Option<Vec<u8>>,
}

pub struct VectorTransaction {
    pub state: Arc<RefCell<IndexState>>,
    /// The table name, for shadow table SQL.
    pub table_name: String,
}

impl VTabTransaction for VectorTransaction {
    fn sync(&mut self) -> Result<()> {
        let mut state = self.state.borrow_mut();
        if state.dirty {
            // Serialize the index to the shadow table.
            // The actual SQL execution happens via the VTab's db connection,
            // which the implementer will need to wire through.
            // For now, serialize to the last_committed buffer.
            let buf = state.index.save_to_buffer()
                .map_err(|e| sqlite3_ext::Error::Module(e.to_string()))?;
            state.last_committed = Some(buf);
            state.dirty = false;
        }
        Ok(())
    }

    fn commit(self) -> Result<()> {
        // sync already did the work
        Ok(())
    }

    fn rollback(self) -> Result<()> {
        let mut state = self.state.borrow_mut();
        if let Some(buf) = &state.last_committed {
            state.index.load_from_buffer(buf)
                .map_err(|e| sqlite3_ext::Error::Module(e.to_string()))?;
        }
        state.dirty = false;
        Ok(())
    }

    fn savepoint(&mut self, _n: i32) -> Result<()> {
        Ok(())
    }

    fn release(&mut self, _n: i32) -> Result<()> {
        Ok(())
    }

    fn rollback_to(&mut self, _n: i32) -> Result<()> {
        // For simplicity, reload from last committed state
        let mut state = self.state.borrow_mut();
        if let Some(buf) = &state.last_committed {
            state.index.load_from_buffer(buf)
                .map_err(|e| sqlite3_ext::Error::Module(e.to_string()))?;
        }
        state.dirty = false;
        Ok(())
    }
}
```

- [ ] **Step 3: Implement the main VTab struct**

Update `src/vtab/mod.rs`:

```rust
pub mod config;
pub mod shadow;
pub mod cursor;
pub mod transaction;

use std::cell::RefCell;
use std::sync::Arc;

use sqlite3_ext::vtab::*;
use sqlite3_ext::*;

use crate::distance::DistanceMetric;
use crate::index::HnswIndex;
use crate::types::VectorType;

use config::VectorTableConfig;
use cursor::{CursorMode, VectorCursor, ScanRow, KnnRow};
use transaction::{IndexState, VectorTransaction};

/// The vector virtual table module.
///
/// NOTE: `VTabConnection` has lifetime `'vtab` from connect/create. Storing it
/// requires either a lifetime parameter on this struct (`VectorTable<'vtab>`) or
/// a raw `*const` pointer. Since sqlite3_ext's trait system uses
/// `VTab<'vtab>`, adding a lifetime parameter to VectorTable is natural:
/// `struct VectorTable<'vtab>`. However, if the trait bounds conflict, use a raw
/// pointer with a safety comment documenting that the pointer is valid for the
/// lifetime of the virtual table (guaranteed by SQLite's module lifecycle).
/// The implementer must choose the approach that compiles against the vendored
/// sqlite3_ext fork and adapt accordingly.
pub struct VectorTable {
    config: VectorTableConfig,
    state: Arc<RefCell<IndexState>>,
    // db: *const VTabConnection  // raw pointer to connection, set in connect/create
    // Safety: valid for the lifetime of the virtual table. SQLite guarantees the
    // connection outlives all virtual tables registered on it.
}

impl<'vtab> VTab<'vtab> for VectorTable {
    type Aux = ();
    type Cursor = VectorCursor;

    fn connect(
        db: &'vtab VTabConnection,
        _aux: &'vtab (),
        args: &[&str],
    ) -> Result<(String, Self)> {
        let config = VectorTableConfig::parse(args)
            .map_err(|e| Error::Module(e.to_string()))?;

        let index = HnswIndex::new(
            config.dim,
            config.vtype,
            config.metric,
            Some(config.hnsw_params),
        ).map_err(|e| Error::Module(e.to_string()))?;

        // Try to load existing index from shadow table
        // (connect is called when re-opening an existing table)
        let last_committed = load_index_from_shadow(db, &config.table_name)
            .ok()
            .flatten();

        if let Some(buf) = &last_committed {
            index.load_from_buffer(buf)
                .map_err(|e| Error::Module(e.to_string()))?;
        }

        let schema = config.vtab_schema();
        let state = Arc::new(RefCell::new(IndexState {
            index,
            dirty: false,
            last_committed,
        }));

        Ok((schema, Self { config, state }))
    }

    fn best_index(&'vtab self, info: &mut IndexInfo) -> Result<()> {
        // Detect knn_match constraint. SQLite 3.25.0+ supports
        // SQLITE_INDEX_CONSTRAINT_FUNCTION (value 150+). When knn_match()
        // is registered as an overloaded function on the virtual table,
        // SQLite calls best_index with a function constraint.
        //
        // The knn_match constraint targets the "distance" column (HIDDEN,
        // last column). We identify it by:
        // 1. Column index == distance column index (2 + num_metadata_cols)
        // 2. Constraint op >= 150 (SQLITE_INDEX_CONSTRAINT_FUNCTION)
        //
        // The implementer must also implement `find_function` on the VTab
        // to register knn_match as an overloadable function. See
        // sqlite3_ext's VTab trait for the find_function method, or
        // implement it via raw sqlite3_overload_function.

        let distance_col_idx = 2 + self.config.metadata_columns.len();
        let mut knn_constraint_idx = None;

        for (i, constraint) in info.constraints().enumerate() {
            if constraint.usable()
                && constraint.column() == distance_col_idx as i32
                && constraint.op() >= 150  // SQLITE_INDEX_CONSTRAINT_FUNCTION
            {
                knn_constraint_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = knn_constraint_idx {
            // Mark the knn_match constraint as consumed and pass its
            // argument (the query vector + k) to filter via argv[0]
            info.constraint_usage(idx).set_argv_index(1);
            info.constraint_usage(idx).set_omit(true);
            info.set_index_num(1); // KNN mode
            info.set_estimated_cost(10.0);
            info.set_estimated_rows(10);
            info.set_order_by_consumed(true); // results pre-sorted by distance
        } else {
            info.set_index_num(0); // Full scan mode
            info.set_estimated_cost(1000000.0);
            info.set_estimated_rows(self.state.borrow().index.len() as i64);
        }
        Ok(())
    }

    fn open(&'vtab self) -> Result<VectorCursor> {
        Ok(VectorCursor {
            mode: CursorMode::Scan {
                rows: Vec::new(),
                pos: 0,
            },
            num_metadata_cols: self.config.metadata_columns.len(),
        })
    }
}

impl<'vtab> CreateVTab<'vtab> for VectorTable {
    const SHADOW_NAMES: &'static [&'static str] = &["data", "index"];

    fn create(
        db: &'vtab VTabConnection,
        aux: &'vtab (),
        args: &[&str],
    ) -> Result<(String, Self)> {
        let config = VectorTableConfig::parse(args)
            .map_err(|e| Error::Module(e.to_string()))?;

        // Create shadow tables
        db.execute_batch(&shadow::ShadowOps::create_data_table_sql(&config))?;
        db.execute_batch(&shadow::ShadowOps::create_index_table_sql(&config))?;

        // Save initial config to _index table
        let meta = serde_json::json!({
            "dim": config.dim,
            "type": config.vtype.name(),
            "metric": config.metric.name(),
            "m": config.hnsw_params.m,
            "ef_construction": config.hnsw_params.ef_construction,
            "ef_search": config.hnsw_params.ef_search,
        });
        save_meta_to_shadow(db, &config.table_name, &meta.to_string())?;

        // Delegate to connect for the rest
        Self::connect(db, aux, args)
    }

    fn destroy(self) -> DisconnectResult<Self> {
        // Drop shadow tables
        // Note: need db access here — sqlite3_ext provides it through the
        // destroy mechanism. Implementer should verify exact API.
        Ok(())
    }
}

impl<'vtab> UpdateVTab<'vtab> for VectorTable {
    fn update(&'vtab self, info: &mut ChangeInfo) -> Result<i64> {
        match info.change_type() {
            ChangeType::Insert => {
                // args layout: [rowid, vector_blob, ...metadata, distance(ignored)]
                let vector_blob = info.args()[1].get_blob()?;

                // Validate
                self.config.vtype.validate_blob(vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;
                if self.config.vtype.is_float() {
                    self.config.vtype.validate_finite(vector_blob, self.config.dim)
                        .map_err(|e| Error::Module(e.to_string()))?;
                }

                // Insert into _data shadow table and get rowid
                let rowid = insert_into_data_shadow(
                    &self.config,
                    vector_blob,
                    &info.args()[2..],  // metadata values
                )?;

                // Add to HNSW index
                let mut state = self.state.borrow_mut();
                state.index.add(rowid as u64, vector_blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
                state.dirty = true;

                Ok(rowid)
            }
            ChangeType::Delete => {
                let rowid = info.rowid().get_i64()?;

                // Delete from _data shadow table
                delete_from_data_shadow(&self.config.table_name, rowid)?;

                // Remove from HNSW index (soft delete)
                let mut state = self.state.borrow_mut();
                state.index.remove(rowid as u64)
                    .map_err(|e| Error::Module(e.to_string()))?;
                state.dirty = true;

                Ok(rowid)
            }
            ChangeType::Update => {
                // Delete + re-insert
                let rowid = info.rowid().get_i64()?;
                let vector_blob = info.args()[1].get_blob()?;

                // Validate new vector
                self.config.vtype.validate_blob(vector_blob, self.config.dim)
                    .map_err(|e| Error::Module(e.to_string()))?;
                if self.config.vtype.is_float() {
                    self.config.vtype.validate_finite(vector_blob, self.config.dim)
                        .map_err(|e| Error::Module(e.to_string()))?;
                }

                // Update _data shadow table
                update_data_shadow(&self.config, rowid, vector_blob, &info.args()[2..])?;

                // Remove old, add new in index
                let mut state = self.state.borrow_mut();
                state.index.remove(rowid as u64)
                    .map_err(|e| Error::Module(e.to_string()))?;
                state.index.add(rowid as u64, vector_blob)
                    .map_err(|e| Error::Module(e.to_string()))?;
                state.dirty = true;

                Ok(rowid)
            }
        }
    }
}

impl<'vtab> TransactionVTab<'vtab> for VectorTable {
    type Transaction = VectorTransaction;

    fn begin(&'vtab self) -> Result<VectorTransaction> {
        Ok(VectorTransaction {
            state: self.state.clone(),
            table_name: self.config.table_name.clone(),
        })
    }
}

// Helper functions for shadow table I/O.
// These are stubs — the implementer must wire them to actual SQL execution
// through the VTabConnection. The exact mechanism depends on sqlite3_ext's
// connection API (execute, prepare, etc.).

fn load_index_from_shadow(
    _db: &VTabConnection,
    _table_name: &str,
) -> Result<Option<Vec<u8>>> {
    // SELECT value FROM "{table_name}_index" WHERE key = 'hnsw_graph'
    todo!("wire to actual SQL execution")
}

fn save_meta_to_shadow(
    _db: &VTabConnection,
    _table_name: &str,
    _meta_json: &str,
) -> Result<()> {
    // INSERT OR REPLACE INTO "{table_name}_index"(key, value) VALUES('meta', ?)
    todo!("wire to actual SQL execution")
}

fn insert_into_data_shadow(
    _config: &VectorTableConfig,
    _vector_blob: &[u8],
    _metadata_args: &[&ValueRef],
) -> Result<i64> {
    todo!("wire to actual SQL execution — return new rowid")
}

fn delete_from_data_shadow(
    _table_name: &str,
    _rowid: i64,
) -> Result<()> {
    todo!("wire to actual SQL execution")
}

fn update_data_shadow(
    _config: &VectorTableConfig,
    _rowid: i64,
    _vector_blob: &[u8],
    _metadata_args: &[&ValueRef],
) -> Result<()> {
    todo!("wire to actual SQL execution")
}
```

**Important notes to implementer:**

1. **Shadow table I/O:** The `todo!()` stubs must be replaced with actual SQL execution through `sqlite3_ext`'s `VTabConnection`. The `VTabConnection` derefs to `Connection`, which provides `execute`, `prepare`, and `execute_batch`. Consult `vendor/sqlite3_ext/src/connection.rs` for the exact API. The shadow table SQL strings are already generated by `ShadowOps` in `shadow.rs`.

2. **Interior mutability:** The `UpdateVTab::update` method receives `&self` (immutable) because cursors may be active. All mutable index state goes through the `Arc<RefCell<IndexState>>`. The shadow table writes use the `VTabConnection` which is provided through the `create`/`connect` call — store a raw pointer to the connection in the `VectorTable` struct (see the comment on the struct definition).

3. **knn_match function registration:** For KNN queries to work, `knn_match` must be registered as an overloaded function on the virtual table. This is done via sqlite3_ext's `find_function` method on the VTab trait (if available) or via `sqlite3_overload_function`. The function itself is a no-op — its only purpose is to generate a `SQLITE_INDEX_CONSTRAINT_FUNCTION` constraint in `best_index`. The actual KNN search happens in the cursor's `filter` method when `index_num == 1`. Check `vendor/sqlite3_ext/src/vtab.rs` for whether `find_function` is exposed; if not, register `knn_match` as a regular scalar function that accepts (table_ref, query_blob, k) and returns a boolean — SQLite will still generate the constraint if the virtual table claims the function.

4. **Query-time ef_search override:** The spec allows `knn_match(emb, :query, 10, 128)` with an optional 4th argument for ef_search. The cursor's `filter` method should check if argv contains an extra argument and, if so, temporarily adjust the usearch index's ef_search parameter before running the search.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check`
Expected: compiles (the `todo!()` stubs will panic at runtime but compile fine)

- [ ] **Step 5: Commit**

```bash
git add src/vtab/
git commit -m "feat(vtab): implement virtual table core with cursor and transaction support

VTab, CreateVTab, UpdateVTab, and TransactionVTab trait implementations
for the vector virtual table. Cursor supports scan and KNN modes.
Transaction handler implements dirty-flag serialization on sync and
index reload on rollback. Shadow table I/O stubs marked with todo!()
for wiring to sqlite3_ext's connection API."
```

---

## Task 9: Scalar Functions

**Files:**
- Create: `src/scalar.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Implement scalar function registration**

Create `src/scalar.rs`:

```rust
use sqlite3_ext::*;

use crate::distance::{DistanceMetric, compute_distance};
use crate::json::{json_to_blob, blob_to_json};
use crate::types::VectorType;

/// Register all standalone scalar functions on a connection.
pub fn register_scalar_functions(db: &Connection) -> Result<()> {
    // vector_distance(blob_a, blob_b, metric, type) -> REAL
    db.create_scalar_function(
        "vector_distance",
        &FunctionOptions::default().set_n_args(4).set_deterministic(true),
        |ctx, args| {
            let blob_a = args[0].get_blob()?;
            let blob_b = args[1].get_blob()?;
            let metric_name = args[2].get_str()?;
            let type_name = args[3].get_str()?;

            let vtype = VectorType::from_name(type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let metric = DistanceMetric::from_name(metric_name)
                .map_err(|e| Error::Module(e.to_string()))?;

            let dim = blob_a.len() / vtype.element_size();
            let dist = compute_distance(blob_a, blob_b, vtype, metric, dim)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(dist);
            Ok(())
        },
    )?;

    // vector_from_json(json_text, type) -> BLOB
    db.create_scalar_function(
        "vector_from_json",
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let json_text = args[0].get_str()?;
            let type_name = args[1].get_str()?;

            let vtype = VectorType::from_name(type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let blob = json_to_blob(json_text, vtype)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(&blob[..]);
            Ok(())
        },
    )?;

    // vector_to_json(blob, type) -> TEXT
    db.create_scalar_function(
        "vector_to_json",
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let blob = args[0].get_blob()?;
            let type_name = args[1].get_str()?;

            let vtype = VectorType::from_name(type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let json = blob_to_json(blob, vtype)
                .map_err(|e| Error::Module(e.to_string()))?;

            ctx.set_result(json.as_str());
            Ok(())
        },
    )?;

    // vector_dims(blob, type) -> INTEGER
    db.create_scalar_function(
        "vector_dims",
        &FunctionOptions::default().set_n_args(2).set_deterministic(true),
        |ctx, args| {
            let blob = args[0].get_blob()?;
            let type_name = args[1].get_str()?;

            let vtype = VectorType::from_name(type_name)
                .map_err(|e| Error::Module(e.to_string()))?;
            let dims = blob.len() / vtype.element_size();

            ctx.set_result(dims as i64);
            Ok(())
        },
    )?;

    Ok(())
}
```

Update `src/lib.rs`:

```rust
pub mod types;
pub mod json;
pub mod distance;
pub mod index;
pub mod vtab;
pub mod scalar;
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add src/scalar.rs src/lib.rs
git commit -m "feat(scalar): implement standalone vector SQL functions

Four scalar functions: vector_distance (pairwise distance computation),
vector_from_json (JSON array to typed blob), vector_to_json (blob to
JSON), and vector_dims (dimension count from blob). All are deterministic
and work on plain BLOB columns without requiring a virtual table."
```

---

## Task 10: Extension Entry Point

**Files:**
- Modify: `src/lib.rs`

- [ ] **Step 1: Implement the loadable extension entry point**

Update `src/lib.rs` to its final form:

```rust
pub mod types;
pub mod json;
pub mod distance;
pub mod index;
pub mod vtab;
pub mod scalar;

#[cfg(feature = "loadable_extension")]
use sqlite3_ext::*;

/// Entry point for the loadable SQLite extension.
/// Called when `SELECT load_extension('sqlite_vector_rs')` is executed.
#[cfg(feature = "loadable_extension")]
#[sqlite3_ext_main(persistent)]
fn sqlite3_extension_init(db: &Connection) -> Result<()> {
    // Register the "vector" virtual table module
    db.create_module("vector", StandardModule::<vtab::VectorTable>::new(), ())?;

    // Register standalone scalar functions
    scalar::register_scalar_functions(db)?;

    Ok(())
}

/// Register the extension on a rusqlite connection (library mode).
#[cfg(feature = "library")]
pub fn register(conn: &rusqlite::Connection) -> std::result::Result<(), rusqlite::Error> {
    // Use rusqlite's module registration API
    // The implementer needs to bridge between rusqlite's vtab API and the
    // sqlite3_ext-based VectorTable, OR provide a separate rusqlite-native
    // implementation. For initial scope, this registers scalar functions only.
    //
    // Full virtual table support in library mode requires either:
    // 1. A separate rusqlite VTab implementation sharing core logic, or
    // 2. Loading the cdylib via rusqlite's load_extension_enable/load_extension
    //
    // Option 2 is simpler for the initial implementation.
    todo!("implement library-mode registration")
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check`
Expected: compiles

- [ ] **Step 3: Commit**

```bash
git add src/lib.rs
git commit -m "feat: implement loadable extension entry point

Registers the 'vector' virtual table module and all standalone scalar
functions when the extension is loaded via load_extension(). Library
mode registration is stubbed for future implementation."
```

---

## Task 11: Arrow Bulk I/O

**Files:**
- Create: `src/arrow_io.rs`
- Create: `tests/arrow_test.rs`
- Modify: `src/lib.rs`

- [ ] **Step 1: Write failing tests for Arrow conversion**

Create `tests/arrow_test.rs`:

```rust
use sqlite_vector_rs::arrow_io::{vectors_to_arrow_ipc, arrow_ipc_to_vectors};
use sqlite_vector_rs::types::VectorType;

#[test]
fn round_trip_float4_vectors() {
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ];
    let blobs: Vec<Vec<u8>> = vectors.iter()
        .map(|v| VectorType::slice_to_blob(v))
        .collect();

    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
    assert!(!ipc.is_empty());

    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 3).unwrap();
    assert_eq!(restored.len(), 3);
    assert_eq!(restored[0], blobs[0]);
    assert_eq!(restored[1], blobs[1]);
    assert_eq!(restored[2], blobs[2]);
}

#[test]
fn round_trip_int1_vectors() {
    let vectors: Vec<Vec<i8>> = vec![
        vec![1, 2, 3, 4],
        vec![5, 6, 7, 8],
    ];
    let blobs: Vec<Vec<u8>> = vectors.iter()
        .map(|v| VectorType::slice_to_blob(v))
        .collect();

    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Int1, 4).unwrap();
    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Int1, 4).unwrap();
    assert_eq!(restored, blobs);
}

#[test]
fn empty_vectors_round_trip() {
    let blobs: Vec<Vec<u8>> = vec![];
    let ipc = vectors_to_arrow_ipc(&blobs, VectorType::Float4, 3).unwrap();
    let restored = arrow_ipc_to_vectors(&ipc, VectorType::Float4, 3).unwrap();
    assert!(restored.is_empty());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --test arrow_test`
Expected: compilation errors

- [ ] **Step 3: Implement Arrow I/O**

Create `src/arrow_io.rs`:

```rust
use std::io::Cursor;
use std::sync::Arc;

use arrow_array::{ArrayRef, FixedSizeListArray, RecordBatch};
use arrow_array::types::*;
use arrow_array::*;
use arrow_buffer::Buffer;
use arrow_ipc::reader::StreamReader;
use arrow_ipc::writer::StreamWriter;
use arrow_schema::{DataType, Field, Schema};

use crate::types::VectorType;

#[derive(Debug)]
pub struct ArrowError(pub String);

impl std::fmt::Display for ArrowError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "arrow error: {}", self.0)
    }
}

impl std::error::Error for ArrowError {}

impl From<arrow_schema::ArrowError> for ArrowError {
    fn from(e: arrow_schema::ArrowError) -> Self {
        Self(e.to_string())
    }
}

/// Convert a list of raw vector blobs into an Arrow IPC byte buffer.
///
/// The output RecordBatch has a single column "vector" of type
/// FixedSizeList(element_type, dim).
pub fn vectors_to_arrow_ipc(
    blobs: &[Vec<u8>],
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<u8>, ArrowError> {
    let (inner_field, values_array) = build_values_array(blobs, vtype, dim)?;

    let field = Arc::new(Field::new("item", inner_field, true));
    let list_array = FixedSizeListArray::new(field, dim as i32, values_array, None);

    let schema = Schema::new(vec![Field::new(
        "vector",
        list_array.data_type().clone(),
        false,
    )]);
    let batch = RecordBatch::try_new(
        Arc::new(schema.clone()),
        vec![Arc::new(list_array)],
    ).map_err(|e| ArrowError(e.to_string()))?;

    let mut buf = Vec::new();
    let mut writer = StreamWriter::try_new(&mut buf, &schema)
        .map_err(|e| ArrowError(e.to_string()))?;
    writer.write(&batch).map_err(|e| ArrowError(e.to_string()))?;
    writer.finish().map_err(|e| ArrowError(e.to_string()))?;
    drop(writer);

    Ok(buf)
}

/// Parse an Arrow IPC byte buffer back into raw vector blobs.
pub fn arrow_ipc_to_vectors(
    ipc_bytes: &[u8],
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<Vec<u8>>, ArrowError> {
    let reader = StreamReader::try_new(Cursor::new(ipc_bytes), None)
        .map_err(|e| ArrowError(e.to_string()))?;

    let mut all_blobs = Vec::new();
    for batch_result in reader {
        let batch = batch_result.map_err(|e| ArrowError(e.to_string()))?;
        let list_col = batch.column(0)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| ArrowError("expected FixedSizeListArray".into()))?;

        for i in 0..list_col.len() {
            let sub = list_col.value(i);
            let blob = extract_blob_from_array(&sub, vtype, dim)?;
            all_blobs.push(blob);
        }
    }

    Ok(all_blobs)
}

fn build_values_array(
    blobs: &[Vec<u8>],
    vtype: VectorType,
    dim: usize,
) -> Result<(DataType, ArrayRef), ArrowError> {
    // Concatenate all blobs into a flat buffer, then wrap in the appropriate Arrow array
    let total_elements = blobs.len() * dim;
    match vtype {
        VectorType::Float4 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[f32] = VectorType::blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Float32, Arc::new(Float32Array::from(flat))))
        }
        VectorType::Float8 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[f64] = VectorType::blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Float64, Arc::new(Float64Array::from(flat))))
        }
        VectorType::Float2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[half::f16] = VectorType::blob_to_slice(blob);
                flat.extend(v.iter().map(|x| *x));
            }
            Ok((DataType::Float16, Arc::new(Float16Array::from(flat))))
        }
        VectorType::Int1 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i8] = VectorType::blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Int8, Arc::new(Int8Array::from(flat))))
        }
        VectorType::Int2 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i16] = VectorType::blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Int16, Arc::new(Int16Array::from(flat))))
        }
        VectorType::Int4 => {
            let mut flat = Vec::with_capacity(total_elements);
            for blob in blobs {
                let v: &[i32] = VectorType::blob_to_slice(blob);
                flat.extend_from_slice(v);
            }
            Ok((DataType::Int32, Arc::new(Int32Array::from(flat))))
        }
    }
}

fn extract_blob_from_array(
    array: &ArrayRef,
    vtype: VectorType,
    dim: usize,
) -> Result<Vec<u8>, ArrowError> {
    match vtype {
        VectorType::Float4 => {
            let a = array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| ArrowError("expected Float32Array".into()))?;
            let values: Vec<f32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Float8 => {
            let a = array.as_any().downcast_ref::<Float64Array>()
                .ok_or_else(|| ArrowError("expected Float64Array".into()))?;
            let values: Vec<f64> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Float2 => {
            let a = array.as_any().downcast_ref::<Float16Array>()
                .ok_or_else(|| ArrowError("expected Float16Array".into()))?;
            let values: Vec<half::f16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int1 => {
            let a = array.as_any().downcast_ref::<Int8Array>()
                .ok_or_else(|| ArrowError("expected Int8Array".into()))?;
            let values: Vec<i8> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int2 => {
            let a = array.as_any().downcast_ref::<Int16Array>()
                .ok_or_else(|| ArrowError("expected Int16Array".into()))?;
            let values: Vec<i16> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
        VectorType::Int4 => {
            let a = array.as_any().downcast_ref::<Int32Array>()
                .ok_or_else(|| ArrowError("expected Int32Array".into()))?;
            let values: Vec<i32> = (0..dim).map(|i| a.value(i)).collect();
            Ok(VectorType::slice_to_blob(&values))
        }
    }
}
```

Update `src/lib.rs` to add the module:

```rust
pub mod arrow_io;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --test arrow_test`
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/arrow_io.rs src/lib.rs tests/arrow_test.rs
git commit -m "feat(arrow): implement Arrow IPC bulk import/export

Converts between raw vector blobs and Arrow IPC streaming format.
Vectors are represented as FixedSizeList(element_type, dim) in the
Arrow schema. Supports all six element types. Round-trip tested."
```

---

## Task 12: Integration Tests (Virtual Table)

**Files:**
- Create: `tests/common/mod.rs`
- Create: `tests/vtab_test.rs`
- Create: `tests/scalar_test.rs`
- Create: `tests/persistence_test.rs`

These tests use the loadable extension loaded into a `rusqlite` connection. They exercise the full stack end-to-end.

- [ ] **Step 1: Create test helpers**

Create `tests/common/mod.rs`:

```rust
use rusqlite::Connection;
use std::path::Path;

/// Create an in-memory SQLite connection with the vector extension loaded.
pub fn open_with_extension() -> Connection {
    let conn = Connection::open_in_memory().unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        // The cdylib name depends on the platform:
        // Linux: libsqlite_vector_rs.so
        // macOS: libsqlite_vector_rs.dylib
        // Windows: sqlite_vector_rs.dll
        let ext_path = find_extension_path();
        conn.load_extension(ext_path, None).unwrap();
        conn.load_extension_disable().unwrap();
    }
    conn
}

/// Create a file-backed SQLite connection (for persistence tests).
pub fn open_file_with_extension(path: &Path) -> Connection {
    let conn = Connection::open(path).unwrap();
    unsafe {
        conn.load_extension_enable().unwrap();
        let ext_path = find_extension_path();
        conn.load_extension(ext_path, None).unwrap();
        conn.load_extension_disable().unwrap();
    }
    conn
}

fn find_extension_path() -> &'static str {
    // cargo builds the cdylib in target/debug/
    if cfg!(target_os = "linux") {
        "target/debug/libsqlite_vector_rs"
    } else if cfg!(target_os = "macos") {
        "target/debug/libsqlite_vector_rs"
    } else {
        "target/debug/sqlite_vector_rs"
    }
}

/// Generate a random f32 vector of given dimension.
pub fn random_f32_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::rng();
    (0..dim).map(|_| rng.random_range(-1.0..1.0)).collect()
}
```

- [ ] **Step 2: Write virtual table integration tests**

Create `tests/vtab_test.rs`:

```rust
mod common;

use common::{open_with_extension, random_f32_vector};
use sqlite_vector_rs::types::VectorType;

#[test]
fn create_virtual_table() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE test_emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();
}

#[test]
fn insert_and_query_knn() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    // Insert 3 vectors
    let v1 = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let v2 = VectorType::slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let v3 = VectorType::slice_to_blob(&[0.0f32, 0.0, 1.0]);

    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v1]).unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v2]).unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v3]).unwrap();

    // KNN query: find 2 nearest to [1, 0.1, 0]
    let query = VectorType::slice_to_blob(&[1.0f32, 0.1, 0.0]);
    let mut stmt = conn.prepare(
        "SELECT id, distance FROM emb WHERE knn_match(emb, ?, 2)"
    ).unwrap();
    let results: Vec<(i64, f64)> = stmt.query_map([&query], |row| {
        Ok((row.get(0)?, row.get(1)?))
    }).unwrap().collect::<Result<_, _>>().unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1); // closest should be v1
}

#[test]
fn delete_vector() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let v1 = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let v2 = VectorType::slice_to_blob(&[0.0f32, 1.0, 0.0]);
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v1]).unwrap();
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v2]).unwrap();

    conn.execute("DELETE FROM emb WHERE id = 1", []).unwrap();

    let query = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let mut stmt = conn.prepare(
        "SELECT id FROM emb WHERE knn_match(emb, ?, 10)"
    ).unwrap();
    let ids: Vec<i64> = stmt.query_map([&query], |row| row.get(0))
        .unwrap().collect::<Result<_, _>>().unwrap();

    assert!(!ids.contains(&1));
}

#[test]
fn update_vector() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let v1 = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v1]).unwrap();

    let v1_new = VectorType::slice_to_blob(&[0.0f32, 0.0, 1.0]);
    conn.execute("UPDATE emb SET vector = ? WHERE id = 1", [&v1_new]).unwrap();

    let query = VectorType::slice_to_blob(&[0.0f32, 0.0, 1.0]);
    let mut stmt = conn.prepare(
        "SELECT id, distance FROM emb WHERE knn_match(emb, ?, 1)"
    ).unwrap();
    let results: Vec<(i64, f64)> = stmt.query_map([&query], |row| {
        Ok((row.get(0)?, row.get(1)?))
    }).unwrap().collect::<Result<_, _>>().unwrap();

    assert_eq!(results[0].0, 1);
    assert!(results[0].1 < 0.01); // should be very close
}

#[test]
fn metadata_columns() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2, metadata=\"label TEXT\")"
    ).unwrap();

    let v1 = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    conn.execute("INSERT INTO emb(vector, label) VALUES(?, 'cat')", [&v1]).unwrap();

    let query = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let label: String = conn.query_row(
        "SELECT label FROM emb WHERE knn_match(emb, ?, 1)",
        [&query],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(label, "cat");
}

#[test]
fn reject_wrong_dimension() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let wrong = VectorType::slice_to_blob(&[1.0f32, 0.0]); // 2-dim, expected 3
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [&wrong]);
    assert!(result.is_err());
}

#[test]
fn reject_nan() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let with_nan = VectorType::slice_to_blob(&[1.0f32, f32::NAN, 3.0]);
    let result = conn.execute("INSERT INTO emb(vector) VALUES(?)", [&with_nan]);
    assert!(result.is_err());
}

#[test]
fn empty_table_knn() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
    ).unwrap();

    let query = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let mut stmt = conn.prepare(
        "SELECT id FROM emb WHERE knn_match(emb, ?, 10)"
    ).unwrap();
    let results: Vec<i64> = stmt.query_map([&query], |row| row.get(0))
        .unwrap().collect::<Result<_, _>>().unwrap();
    assert!(results.is_empty());
}
```

- [ ] **Step 3: Write scalar function integration tests**

Create `tests/scalar_test.rs`:

```rust
mod common;

use common::open_with_extension;
use sqlite_vector_rs::types::VectorType;

#[test]
fn vector_from_json_and_back() {
    let conn = open_with_extension();
    let json: String = conn.query_row(
        "SELECT vector_to_json(vector_from_json('[1.0, 2.0, 3.0]', 'float4'), 'float4')",
        [],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(json, "[1.0,2.0,3.0]");
}

#[test]
fn vector_distance_l2() {
    let conn = open_with_extension();
    let a = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
    let b = VectorType::slice_to_blob(&[0.0f32, 1.0, 0.0]);
    let dist: f64 = conn.query_row(
        "SELECT vector_distance(?, ?, 'l2', 'float4')",
        [&a, &b],
        |row| row.get(0),
    ).unwrap();
    assert!((dist - 2.0).abs() < 1e-6); // squared L2
}

#[test]
fn vector_dims() {
    let conn = open_with_extension();
    let v = VectorType::slice_to_blob(&[1.0f32, 2.0, 3.0, 4.0]);
    let dims: i64 = conn.query_row(
        "SELECT vector_dims(?, 'float4')",
        [&v],
        |row| row.get(0),
    ).unwrap();
    assert_eq!(dims, 4);
}
```

- [ ] **Step 4: Write persistence integration tests**

Create `tests/persistence_test.rs`:

```rust
mod common;

use common::{open_file_with_extension};
use sqlite_vector_rs::types::VectorType;
use std::path::PathBuf;

#[test]
fn index_survives_reconnect() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("test.db");

    // Insert vectors in first connection
    {
        let conn = open_file_with_extension(&db_path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE emb USING vector(dim=3, type=float4, metric=l2)"
        ).unwrap();

        let v1 = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
        let v2 = VectorType::slice_to_blob(&[0.0f32, 1.0, 0.0]);
        conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v1]).unwrap();
        conn.execute("INSERT INTO emb(vector) VALUES(?)", [&v2]).unwrap();
    }
    // Connection dropped, DB closed

    // Reopen and verify KNN still works
    {
        let conn = open_file_with_extension(&db_path);
        let query = VectorType::slice_to_blob(&[1.0f32, 0.0, 0.0]);
        let mut stmt = conn.prepare(
            "SELECT id, distance FROM emb WHERE knn_match(emb, ?, 1)"
        ).unwrap();
        let results: Vec<(i64, f64)> = stmt.query_map([&query], |row| {
            Ok((row.get(0)?, row.get(1)?))
        }).unwrap().collect::<Result<_, _>>().unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1); // v1 should be closest
        assert!(results[0].1 < 0.01);
    }
}
```

Add `tempfile` to dev-dependencies in `Cargo.toml`:

```toml
[dev-dependencies]
rusqlite = { version = "0.39", features = ["bundled", "vtab"] }
rand = "0.9"
tempfile = "3"
```

- [ ] **Step 5: Build the extension, then run integration tests**

Run:

```bash
cargo build
cargo test --test vtab_test --test scalar_test --test persistence_test
```

Expected: all tests PASS

Note: If tests fail due to sqlite3_ext API mismatches, consult `vendor/sqlite3_ext/src/` for exact method signatures and adjust the implementation accordingly. The most likely areas of friction are: `ColumnContext::set_result` argument types, `ValueRef` accessor methods, and `ChangeInfo` argument layout.

- [ ] **Step 6: Commit**

```bash
git add tests/ Cargo.toml
git commit -m "test: add integration tests for virtual table, scalar functions, and persistence

End-to-end tests loading the extension via rusqlite: virtual table
CRUD, KNN queries, metadata columns, dimension/NaN validation, empty
table edge cases, scalar function round-trips, and index persistence
across database reconnections."
```

---

## Task 13: Wire Shadow Table I/O (Replace todo!() Stubs)

**Files:**
- Modify: `src/vtab/mod.rs`

This task replaces the `todo!()` stubs from Task 8 with actual SQL execution through `sqlite3_ext`'s `VTabConnection`.

- [ ] **Step 1: Study the sqlite3_ext Connection API**

Read: `vendor/sqlite3_ext/src/connection.rs`

Identify the methods available for executing SQL: `execute`, `prepare`, `execute_batch`, and how to bind parameters and read results.

- [ ] **Step 2: Store a connection reference in VectorTable**

The `VTabConnection` reference from `connect`/`create` has lifetime `'vtab`. Store it in the `VectorTable` struct:

```rust
pub struct VectorTable {
    config: VectorTableConfig,
    state: Arc<RefCell<IndexState>>,
    db: &'vtab VTabConnection,  // or a raw pointer if lifetime constraints require it
}
```

The implementer must handle the lifetime carefully. If `sqlite3_ext` does not allow storing the reference (due to self-referential struct issues), use a raw `*const` pointer with appropriate safety documentation.

- [ ] **Step 3: Implement the shadow table helper functions**

Replace each `todo!()` in `src/vtab/mod.rs` with actual SQL execution using the stored connection reference and the SQL strings from `ShadowOps`.

Key patterns:

```rust
fn load_index_from_shadow(db: &VTabConnection, table_name: &str) -> Result<Option<Vec<u8>>> {
    let sql = shadow::ShadowOps::select_index_sql(table_name);
    // Use db.prepare(&sql) to create a statement
    // Bind key = "hnsw_graph"
    // Execute and read the BLOB result
    // Return None if no row found
}

fn insert_into_data_shadow(
    db: &VTabConnection,
    config: &VectorTableConfig,
    vector_blob: &[u8],
    metadata_values: &[&ValueRef],
) -> Result<i64> {
    let sql = shadow::ShadowOps::insert_data_sql(config);
    // Use db.prepare(&sql) to create a statement
    // Bind vector blob and metadata values
    // Execute and return last_insert_rowid
}
```

- [ ] **Step 4: Wire the sync hook to persist the index**

In `VectorTransaction::sync`, execute the upsert SQL to write the serialized index buffer into the `_index` shadow table. This requires access to the `VTabConnection`, which the implementer must pass through (e.g., via a raw pointer stored in `IndexState` or `VectorTransaction`).

- [ ] **Step 5: Run integration tests to verify**

Run: `cargo test --test vtab_test --test persistence_test`
Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/vtab/mod.rs
git commit -m "feat(vtab): wire shadow table I/O to sqlite3_ext connection

Replace todo!() stubs with actual SQL execution through VTabConnection.
Shadow table reads/writes now flow through sqlite3_ext's prepare/execute
API. Index serialization happens in the xSync hook via upsert into the
_index shadow table."
```

---

## Task 14: vector_rebuild_index and vector_insert_arrow / vector_export_arrow

**Files:**
- Modify: `src/scalar.rs`
- Modify: `src/vtab/mod.rs` (expose index rebuild)

- [ ] **Step 1: Implement vector_rebuild_index scalar function**

Add to `src/scalar.rs`:

```rust
// vector_rebuild_index(table_name) -> NULL
// Rebuilds the HNSW index from scratch using all vectors in _data.
db.create_scalar_function(
    "vector_rebuild_index",
    &FunctionOptions::default().set_n_args(1),
    |ctx, args| {
        let table_name = args[0].get_str()?;
        // Read all vectors from {table_name}_data
        // Create a fresh HnswIndex
        // Add all vectors
        // Serialize to {table_name}_index
        // This requires database access from within a scalar function,
        // which sqlite3_ext provides through the function context.
        todo!("implement rebuild using context's db connection")
    },
)?;
```

Note: Implementing `vector_rebuild_index`, `vector_insert_arrow`, and `vector_export_arrow` as scalar functions that need database access is complex. The implementer should explore whether `sqlite3_ext`'s `Context` provides a connection handle, or whether these need to be implemented as table-valued functions instead. Consult `vendor/sqlite3_ext/src/function.rs` for the context API.

- [ ] **Step 2: Add vector_insert_arrow and vector_export_arrow**

These follow the same pattern — scalar functions that need database access for reading/writing the shadow tables. Wire them to the `arrow_io` module functions from Task 11.

- [ ] **Step 3: Run all tests**

Run: `cargo test`
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/scalar.rs src/vtab/mod.rs
git commit -m "feat(scalar): implement vector_rebuild_index and Arrow bulk I/O functions

vector_rebuild_index reconstructs the HNSW graph from all vectors in
the _data shadow table. vector_insert_arrow and vector_export_arrow
provide Arrow IPC bulk import/export through scalar SQL functions."
```

---

## Task 15: Final Polish & All-Type Test Matrix

**Files:**
- Create: `tests/type_matrix_test.rs`

- [ ] **Step 1: Write the 18-combination type×metric test**

Create `tests/type_matrix_test.rs`:

```rust
mod common;

use common::open_with_extension;
use sqlite_vector_rs::types::VectorType;

/// Test all 6 types × 3 metrics = 18 combinations for basic insert + KNN.
#[test]
fn all_type_metric_combinations() {
    let types = ["float2", "float4", "float8", "int1", "int2", "int4"];
    let metrics = ["l2", "cosine", "ip"];

    for type_name in &types {
        for metric in &metrics {
            let conn = open_with_extension();
            let create_sql = format!(
                "CREATE VIRTUAL TABLE emb USING vector(dim=4, type={type_name}, metric={metric})"
            );
            conn.execute_batch(&create_sql).unwrap_or_else(|e| {
                panic!("Failed to create table with type={type_name}, metric={metric}: {e}");
            });

            // Insert a vector (use JSON for convenience)
            conn.execute(
                "INSERT INTO emb(vector) VALUES(vector_from_json('[1, 2, 3, 4]', ?))",
                [type_name],
            ).unwrap_or_else(|e| {
                panic!("Failed to insert with type={type_name}, metric={metric}: {e}");
            });

            // KNN query
            let query_blob: Vec<u8> = conn.query_row(
                "SELECT vector_from_json('[1, 2, 3, 4]', ?)",
                [type_name],
                |row| row.get(0),
            ).unwrap();

            let count: i64 = conn.query_row(
                "SELECT COUNT(*) FROM emb WHERE knn_match(emb, ?, 10)",
                [&query_blob],
                |row| row.get(0),
            ).unwrap_or_else(|e| {
                panic!("Failed KNN with type={type_name}, metric={metric}: {e}");
            });

            assert_eq!(count, 1, "Expected 1 result for type={type_name}, metric={metric}");
        }
    }
}
```

- [ ] **Step 2: Run the full test suite**

Run: `cargo test`
Expected: all tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/type_matrix_test.rs
git commit -m "test: add 18-combination type×metric test matrix

Verifies that all six element types (float2/4/8, int1/2/4) work
correctly with all three distance metrics (l2, cosine, ip) for
basic virtual table create, insert, and KNN query operations."
```

---

## Implementation Notes

### sqlite3_ext API Friction

The vendored `sqlite3_ext` crate (v0.1.3, last updated 2022) has sparse documentation. The implementer should:

1. Read `vendor/sqlite3_ext/src/vtab.rs` thoroughly before starting Task 8
2. Study the test files in `vendor/sqlite3_ext/tests/` for working examples
3. Check `vendor/sqlite3_ext/examples/` if they exist
4. Be prepared to modify the vendored fork if the API is insufficient (e.g., missing transaction hooks, incomplete `ChangeInfo` API)

### usearch Type Mapping

usearch's Rust `VectorType` trait is only implemented for `f32`, `f64`, `i8`, and usearch's own `f16`. For `half::f16`, `i16`, and `i32`:

- `half::f16` → convert to `f32` via `to_f32()` before passing to usearch
- `i16` → cast to `f32` before passing to usearch
- `i32` → cast to `f32` before passing to usearch

This conversion happens in `HnswIndex::add` and `HnswIndex::search` (Task 5). The raw bytes stored in the shadow table always use the original type — the conversion is only for index operations.

### Library Mode (Deferred)

The `register()` function for library mode (Task 10) is stubbed. Full library-mode support requires either:

1. A separate `rusqlite`-based `VTab` implementation sharing the core logic from `types.rs`, `json.rs`, `distance.rs`, `index.rs`, and `arrow_io.rs`
2. Loading the built cdylib via `rusqlite`'s `load_extension` (simpler, recommended for initial release)

Option 2 is recommended for the initial implementation. Option 1 can be added later if users need a pure-Rust embedded experience without the cdylib.
