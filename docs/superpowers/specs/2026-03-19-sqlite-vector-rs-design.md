# sqlite-vector-rs Design Spec

A Rust SQLite extension providing PGVector-like native vector types with HNSW
indexing. Stores pre-computed embeddings and performs nearest-neighbor search.

## 1. Crate Structure & Build

Single Rust crate producing two artifacts:

- **`cdylib`** — loadable SQLite extension (`.so`/`.dylib`/`.dll`) for any
  SQLite client
- **`rlib`** — Rust library for embedding into Rust applications via `rusqlite`

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

### Dependencies

| Crate | Purpose |
|-------|---------|
| `sqlite3_ext` | Loadable extension framework, virtual table API |
| `rusqlite` (dev + optional feature) | Library-mode API, testing |
| `usearch` | HNSW indexing with persistence |
| `arrow-array`, `arrow-buffer`, `arrow-ipc`, `arrow-schema` | Vector type system, storage format, bulk I/O |
| `half` (with `bytemuck` feature) | f16 support |
| `bytemuck` | Safe zero-copy casting between typed slices and byte slices |

The `sqlite3_ext` fork lives in `vendor/` within the repo. `usearch` is pulled
from crates.io and handles its own C++ compilation via `cc` in its build script.

### Feature Flags

- `loadable_extension` (default) — builds the cdylib entry point
- `library` — enables the `rusqlite`-based API for embedding

## 2. Type System

Six vector element types, mirroring PGVector naming:

| SQL Name | Rust Type | Arrow Type | Bytes | Use Case |
|----------|-----------|------------|-------|----------|
| `float2` | `half::f16` | `Float16` | 2 | Quantized embeddings, memory-constrained |
| `float4` | `f32` | `Float32` | 4 | Standard embeddings (OpenAI, etc.) |
| `float8` | `f64` | `Float64` | 8 | High-precision scientific |
| `int1` | `i8` | `Int8` | 1 | Quantized/binary-style compact vectors |
| `int2` | `i16` | `Int16` | 2 | Intermediate quantization |
| `int4` | `i32` | `Int32` | 4 | Integer embeddings, sparse features |

### Internal Representation

Each vector is stored as a raw typed byte slice in a SQLite BLOB. A 768-dim
`float4` vector is 3072 bytes with no header or metadata. The type and dimension
are fixed per-table (declared at creation time), so per-row metadata is
unnecessary.

### Rust Dispatch

A `VectorType` enum dispatches over the six element types. All distance
computations, Arrow conversions, and serialization are generic over this enum.
The `bytemuck` crate provides safe zero-copy casting between `&[T]` and `&[u8]`.

```rust
enum VectorType {
    Float2,  // half::f16
    Float4,  // f32
    Float8,  // f64
    Int1,    // i8
    Int2,    // i16
    Int4,    // i32
}
```

### Input Formats Accepted

- **JSON array:** `'[1.0, 2.0, 3.0]'`
- **Raw BLOB:** direct byte buffer of the declared element type
- **Arrow IPC:** for bulk operations

## 3. SQL Interface

### Virtual Table Creation

```sql
CREATE VIRTUAL TABLE embeddings USING vector(
    dim=1536,
    type=float4,
    metric=cosine
);
```

This creates:

- The virtual table `embeddings` with columns: `id INTEGER PRIMARY KEY`,
  `vector BLOB`, `distance REAL` (output-only, populated during KNN queries;
  returns NULL when queried without a KNN predicate; writes to this column are
  ignored)
- Shadow table `embeddings_data` storing the raw vector blobs
- Shadow table `embeddings_index` storing the persisted HNSW graph
- Both shadow tables are registered with SQLite's shadow table protection
  mechanism (`SQLITE_DBCONFIG_DEFENSIVE`, SQLite 3.26+)

### Metadata Columns

```sql
CREATE VIRTUAL TABLE embeddings USING vector(
    dim=1536,
    type=float4,
    metric=cosine,
    metadata="label TEXT, category INTEGER"
);
```

Metadata columns are stored in the shadow data table alongside the vector blob.
They can be used in post-filtering after KNN retrieval.

### KNN Query

```sql
SELECT id, vector, distance FROM embeddings
WHERE knn_match(embeddings, :query_vector, 10);
```

Returns the 10 nearest neighbors to `:query_vector`, ordered by distance. The
`distance` column is populated automatically.

### DML

```sql
-- Insert
INSERT INTO embeddings(vector, label) VALUES (:vec, 'dog');

-- Delete
DELETE FROM embeddings WHERE id = 42;

-- Update (delete + re-insert internally)
UPDATE embeddings SET vector = :new_vec WHERE id = 42;
```

Inserts and deletes update the HNSW index incrementally via usearch.

### Standalone Functions

For use without virtual tables, on plain BLOB columns in regular tables:

```sql
-- Distance between two vectors
SELECT vector_distance(:a, :b, 'cosine', 'float4');

-- Parse JSON array into a typed vector blob
SELECT vector_from_json('[1.0, 2.0, 3.0]', 'float4');

-- Convert a vector blob back to JSON
SELECT vector_to_json(:vec, 'float4');

-- Vector metadata
SELECT vector_dims(:vec, 'float4');
```

## 4. Storage & Arrow Integration

### Per-Row Storage

Individual vectors are stored as raw typed byte blobs in the `embeddings_data`
shadow table. No Arrow overhead per row — just `dim * sizeof(element_type)`
bytes. The raw blob is directly castable to `&[f32]` (or whichever element type)
via `bytemuck` — zero-copy on read.

```
embeddings_data:
+---------+-------------+------------------+
| id (PK) | vector BLOB | ...metadata cols |
+---------+-------------+------------------+
```

### Bulk I/O (Arrow IPC)

For batch insert and batch export, Arrow IPC streaming format is used. The
vector column is represented as `FixedSizeList(element_type, dim)` in the Arrow
schema.

```sql
-- Bulk insert from Arrow IPC blob
SELECT vector_insert_arrow('embeddings', :arrow_ipc_blob);

-- Bulk export to Arrow IPC blob
SELECT vector_export_arrow('embeddings');

-- Bulk export with filter
SELECT vector_export_arrow('embeddings', 'category = 3');
```

**Size limitation:** SQLite's default BLOB size limit is 1GB
(`SQLITE_MAX_LENGTH`). Arrow IPC blobs for large datasets may exceed this. For
example, 1M x 1536 x float4 produces ~6GB of vector data alone. Bulk I/O
operations are intended for moderate batches (tens of thousands of vectors).
For larger datasets, callers should chunk their imports into multiple calls.

Python interop example:

```python
import pyarrow as pa
batch = pa.RecordBatch.from_pydict({"vector": vectors, "label": labels})
blob = batch.serialize().to_pybytes()
cursor.execute("SELECT vector_insert_arrow('embeddings', ?)", [blob])
```

### Index Storage (HNSW Persistence)

The HNSW graph is persisted using usearch's native save/load mechanism into the
`embeddings_index` shadow table as a single large BLOB. On database open, the
index is loaded from this blob into memory.

```
embeddings_index:
+--------------+------------+
| key TEXT (PK)| value BLOB |
+--------------+------------+
```

- Key `"hnsw_graph"` — serialized usearch index
- Key `"meta"` — configuration (dim, type, metric, usearch parameters)

### Write Path

Insert -> append raw vector to `embeddings_data` -> add to in-memory usearch
index -> mark index as dirty.

The index is re-serialized to `embeddings_index` only when the dirty flag is
set, via the virtual table's `xSync` hook (called at transaction commit). This
is a full serialization of the usearch graph. For large indices (e.g., 1M
vectors at M=16 produces ~150-200MB), this is expensive. This is an accepted
tradeoff for the initial implementation — the cost is proportional to index
size, not transaction size. Future optimization: delta-based persistence or
usearch's memory-mapped file mode.

### Read Path

KNN query -> search in-memory usearch index -> retrieve matching rows from
`embeddings_data`.

### Transaction Rollback

If a transaction is rolled back, the in-memory usearch index may contain
vectors that are not in `embeddings_data`. The virtual table's `xRollback` hook
handles this by reloading the index from the last committed state in the
`embeddings_index` shadow table. This is expensive but correct, and rollbacks
are expected to be rare in embedding workloads.

## 5. HNSW Index Management

### Index Lifecycle

1. **Creation** — `CREATE VIRTUAL TABLE` initializes an empty usearch index
   with the declared dimension, element type, and distance metric. Usearch
   parameters (M, ef_construction, ef_search) use sensible defaults.
2. **Inserts** — each `INSERT` adds the vector to the in-memory usearch index
   incrementally. The index is re-serialized to the `embeddings_index` shadow
   table on transaction commit.
3. **Deletes** — usearch's `remove()` performs a lazy/soft delete: the node is
   marked as deleted but its graph edges are not reclaimed. Over time, many
   deletes degrade search quality. Use `vector_rebuild_index()` to reclaim
   space and restore optimal graph quality. The vector is also removed from
   `embeddings_data`.
4. **Updates** — implemented as delete + re-insert.
5. **Database close** — current index state is flushed to the shadow table.
6. **Database open** — index is deserialized from the shadow table into memory.

### Tuning Parameters

Optional, with defaults:

```sql
CREATE VIRTUAL TABLE embeddings USING vector(
    dim=1536,
    type=float4,
    metric=cosine,
    m=16,                -- max connections per node (default: 16)
    ef_construction=200, -- build-time search width (default: 200)
    ef_search=64         -- query-time search width (default: 64)
);
```

Query-time override:

```sql
SELECT id, distance FROM embeddings
WHERE knn_match(embeddings, :query, 10, 128);  -- ef_search=128
```

### Index Rebuild

For cases where incremental inserts have degraded graph quality:

```sql
SELECT vector_rebuild_index('embeddings');
```

Reconstructs the HNSW graph from scratch using all vectors in `embeddings_data`.

### Memory Considerations

The full HNSW graph lives in memory during the connection lifetime. For a
1M-vector, 1536-dim, float4 dataset, usearch's graph overhead is roughly
150-200MB (dominated by the M parameter). The vectors themselves in
`embeddings_data` are ~6GB on disk but not held in memory — only accessed on
demand after KNN retrieval.

## 6. Distance Computation

Three metrics supported across all six element types:

| Metric | Function | Returns |
|--------|----------|---------|
| L2 (Euclidean) | Squared L2 distance | `0.0` = identical |
| Cosine | `1 - cosine_similarity` | `0.0` = identical, `2.0` = opposite |
| Inner Product | `-dot(a, b)` | Lower = more similar (`0.0` = orthogonal) |

Simsimd (bundled via usearch) auto-detects the best SIMD instruction set at
runtime (AVX-512, AVX2, NEON, SVE) and dispatches accordingly. No compile-time
feature flags needed.

For standalone functions (`vector_distance`), the computation uses usearch's
metric function API directly (without building an index), avoiding the need for
a separate simsimd dependency. Usearch exposes its distance functions
independently of the index structure.

### Type Promotion

Both input vectors to `vector_distance` must be the same element type. No
implicit promotion — mismatched types return an error. This keeps the fast path
simple and avoids silent precision loss.

## 7. Error Handling & Validation

### Input Validation (System Boundary)

- **Dimension mismatch** — inserting a vector with wrong dimension returns
  `SQLITE_ERROR` with message like `"expected 1536 dimensions, got 768"`
- **Type mismatch** — blob size must equal `dim * sizeof(element_type)`,
  otherwise error
- **JSON parsing** — malformed JSON arrays or non-numeric elements return
  descriptive errors
- **Invalid parameters** — unknown metric names, zero/negative dimensions, bad
  HNSW tuning values caught at table creation time
- **NaN/Inf** — rejected on insert for float types

### Runtime Errors

- **Index corruption** — if the shadow table blob fails to deserialize, the
  virtual table returns an error on open rather than silently operating without
  an index
- **Out of memory** — usearch allocation failures propagate as `SQLITE_NOMEM`

### Not Validated

- Vector values beyond NaN/Inf — normalization is the caller's responsibility
- Metadata column values — SQLite's type affinity handles this

## 8. Testing Strategy

### Unit Tests (Rust `#[test]`)

- Vector type conversions (byte slicing, Arrow round-trips, JSON parsing)
- Distance computations against known values for each metric x element type
- Dimension/type validation logic

### Integration Tests (via `rusqlite` in library mode)

- Create virtual table, insert vectors, run KNN queries, verify results
- CRUD lifecycle: insert -> query -> update -> delete -> query again
- Metadata columns: insert with metadata, verify post-filter
- Bulk Arrow IPC import/export round-trip
- Index persistence: insert vectors, drop connection, reopen, verify KNN works
- Edge cases: empty table queries, single-vector table, duplicate inserts
- All six element types x three metrics (18 combinations)

### Loadable Extension Tests (via SQLite CLI or Python)

- Load extension via `SELECT load_extension(...)`
- Run the same query patterns as integration tests to verify the cdylib works
- Cross-language verification: Python script using `sqlite3` module to exercise
  the full interface

No benchmarks in initial scope — correctness first. Benchmarks can be added
once the core is stable using `criterion`.
