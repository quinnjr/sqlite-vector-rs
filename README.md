# sqlite-vector-rs

A Rust SQLite extension providing PGVector-style typed vector columns with HNSW
approximate nearest-neighbor search, scalar SQL functions, and Arrow IPC bulk I/O.

Vectors are stored as raw typed byte blobs inside SQLite tables — no external
service required. The HNSW index (powered by [usearch](https://github.com/unum-cloud/usearch))
lives in a shadow table and is persisted across connections.

## Features

- **6 vector types** — `float2` (f16), `float4` (f32), `float8` (f64), `int1` (i8), `int2` (i16), `int4` (i32)
- **3 distance metrics** — L2 (squared Euclidean), cosine, inner product
- **HNSW approximate nearest-neighbor search** via usearch with configurable parameters
- **Arrow IPC bulk import/export** for efficient batch operations
- **Full virtual table** with INSERT, UPDATE, DELETE, and transaction rollback
- **Optional metadata columns** alongside vectors (TEXT, INTEGER, REAL, BLOB)
- **Works three ways** — loadable SQLite extension, Rust library, or standalone CLI

## Quick Start

### Build

```bash
cargo build --release
```

This produces `target/release/libsqlite_vector_rs.so` (Linux),
`.dylib` (macOS), or `.dll` (Windows).

### Load into any SQLite client

```sql
.load target/release/libsqlite_vector_rs

CREATE VIRTUAL TABLE embeddings USING vector(
    dim=384,
    type=float4,
    metric=cosine
);

INSERT INTO embeddings(vector)
VALUES (vector_from_json('[0.1, 0.2, 0.3, ...]', 'float4'));

SELECT rowid, distance
FROM embeddings
WHERE knn_match(distance, vector_from_json('[0.15, 0.25, 0.35, ...]', 'float4'))
LIMIT 10;
```

### Use from Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
sqlite-vector-rs = { path = "../sqlite-vector-rs", features = ["library"] }
rusqlite = { version = "0.39", features = ["bundled"] }
```

```rust
use rusqlite::Connection;

let conn = Connection::open("vectors.db")?;
sqlite_vector_rs::register(&conn)?;

conn.execute_batch("
    CREATE VIRTUAL TABLE embeddings USING vector(
        dim=3, type=float4, metric=cosine
    );
")?;

conn.execute(
    "INSERT INTO embeddings(vector) VALUES (vector_from_json(?, 'float4'))",
    ["[1.0, 0.0, 0.0]"],
)?;
```

### Standalone CLI

```bash
cargo build --features library --bin sqlite3
./target/debug/sqlite3 my_vectors.db
```

```
sqlite3-vector v0.1.0 (SQLite 3.49.1)
Enter ".help" for usage hints.
sqlite3-vector> CREATE VIRTUAL TABLE docs USING vector(dim=3, type=float4, metric=cosine);
sqlite3-vector> INSERT INTO docs(vector) VALUES (vector_from_json('[1,0,0]', 'float4'));
sqlite3-vector> SELECT rowid, distance FROM docs
          ...> WHERE knn_match(distance, vector_from_json('[0.9,0.1,0]', 'float4'))
          ...> LIMIT 5;
rowid  distance
-----  --------
1      0.006116
```

## SQL Reference

### CREATE VIRTUAL TABLE

```sql
CREATE VIRTUAL TABLE <name> USING vector(
    dim=<integer>,               -- vector dimension (required)
    type=<vector_type>,          -- element type (required)
    metric=<distance_metric>,    -- distance metric (required)
    m=<integer>,                 -- HNSW M parameter (default: 16)
    ef_construction=<integer>,   -- HNSW build quality (default: 128)
    ef_search=<integer>,         -- HNSW query quality (default: 64)
    metadata='col1 TYPE, ...'    -- optional metadata columns
);
```

**Vector types:** `float2`, `float4`, `float8`, `int1`, `int2`, `int4`

**Distance metrics:** `l2`, `cosine`, `inner_product`

### KNN Search

```sql
SELECT rowid, distance [, metadata_cols...]
FROM <table>
WHERE knn_match(distance, <query_vector_blob>)
LIMIT <k>;
```

The `distance` column is a hidden virtual column that returns the distance
between each stored vector and the query. `knn_match` activates the HNSW index
for efficient approximate search.

### Scalar Functions

| Function | Description |
|----------|-------------|
| `vector_from_json(json, type)` | Parse a JSON array into a vector blob |
| `vector_to_json(blob, type)` | Convert a vector blob to a JSON array string |
| `vector_distance(blob_a, blob_b, metric, type)` | Compute distance between two vectors |
| `vector_dims(blob, type)` | Return the number of dimensions |
| `vector_rebuild_index(table, type, metric)` | Rebuild the HNSW index from shadow data |
| `vector_export_arrow(table, type)` | Export all vectors as an Arrow IPC blob |
| `vector_insert_arrow(table, type, ipc_blob)` | Import vectors from an Arrow IPC blob |

### Metadata Columns

```sql
CREATE VIRTUAL TABLE docs USING vector(
    dim=384,
    type=float4,
    metric=cosine,
    metadata='title TEXT, source TEXT, page INTEGER'
);

INSERT INTO docs(vector, title, source, page)
VALUES (vector_from_json('[...]', 'float4'), 'Chapter 1', 'book.pdf', 42);

SELECT rowid, distance, title, page
FROM docs
WHERE knn_match(distance, vector_from_json('[...]', 'float4'))
LIMIT 5;
```

### Arrow IPC Bulk Operations

Export all vectors to an Arrow IPC stream, then re-import into another table:

```sql
-- Export
SELECT vector_export_arrow('source_table', 'float4');

-- Import (returns row count)
SELECT vector_insert_arrow('dest_table', 'float4', <ipc_blob>);

-- Rebuild the HNSW index after bulk import
SELECT vector_rebuild_index('dest_table', 'float4', 'cosine');
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│  SQLite                                          │
│  ┌────────────────────────────────────────────┐  │
│  │  vector virtual table module               │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐ │  │
│  │  │ Config   │  │ Cursor   │  │ Txn      │ │  │
│  │  │ parsing  │  │ scan/KNN │  │ rollback │ │  │
│  │  └──────────┘  └──────────┘  └──────────┘ │  │
│  └──────────────────┬─────────────────────────┘  │
│                     │                             │
│  ┌──────────────────▼─────────────────────────┐  │
│  │  Shadow tables                              │  │
│  │  {name}_data  → rowid, vector, metadata     │  │
│  │  {name}_index → serialized HNSW graph       │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │  Scalar functions                            │  │
│  │  vector_from_json, vector_distance, etc.     │  │
│  └─────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────┘
         │
         ▼
┌────────────────┐     ┌────────────────┐
│  usearch HNSW  │     │  Arrow IPC     │
│  (in-memory)   │     │  (bulk I/O)    │
└────────────────┘     └────────────────┘
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| [sqlite3_ext](https://crates.io/crates/sqlite3_ext) | SQLite extension + virtual table API |
| [usearch](https://crates.io/crates/usearch) | HNSW approximate nearest-neighbor index |
| [arrow-*](https://crates.io/crates/arrow) (v58) | Arrow IPC stream encoding for bulk I/O |
| [half](https://crates.io/crates/half) | IEEE 754 half-precision (f16) support |
| [bytemuck](https://crates.io/crates/bytemuck) | Zero-copy byte casting |
| [serde_json](https://crates.io/crates/serde_json) | JSON vector parsing |
| [rusqlite](https://crates.io/crates/rusqlite) | Library-mode API (optional, `library` feature) |

## Testing

```bash
# Build the extension first (required for integration tests)
cargo build

# Run all 271 tests
cargo test
```

The test suite includes unit tests for every module, integration tests for
all SQL interfaces, a Shakespeare PDF ingestion pipeline, and GGUF embedding
tests (which download a small model on first run).

## License

Licensed under either of [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
or [MIT License](http://opensource.org/licenses/MIT), at your option.
