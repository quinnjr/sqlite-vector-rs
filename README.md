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
- **Exact brute-force mode** (`mode=exact`) — streaming top-k with true distances, no index
- **Filtered KNN** — metadata `=`/`<`/`<=`/`>`/`>=` predicates pushed into the search with oversampling
- **Crash-safe lazy index persistence** — the graph persists every `sync_every` changes and reconciles against row data on connect
- **Vector math functions** — normalize, add, sub, scale, slice, int8 quantization
- **Arrow IPC bulk import/export** for efficient batch operations
- **Full virtual table** with INSERT, UPDATE, DELETE, and transaction/savepoint rollback
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
sqlite3-vector v0.3.0 (SQLite 3.49.1)
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
    ef_construction=<integer>,   -- HNSW build quality (default: 200)
    ef_search=<integer>,         -- HNSW query quality (default: 64)
    mode=<hnsw|exact>,           -- index mode (default: hnsw)
    sync_every=<integer>,        -- rows between graph persists (default: 1024)
    metadata='col1 TYPE, ...'    -- optional metadata columns
);
```

**Vector types:** `float2`, `float4`, `float8`, `int1`, `int2`, `int4`

**Distance metrics:** `l2`, `cosine`, `ip` (inner product)

**Index modes:**
- `hnsw` (default) — approximate nearest-neighbor search via a usearch HNSW
  graph, tunable with `m` / `ef_construction` / `ef_search`.
- `exact` — brute-force streaming top-k over the shadow data table. Computes
  true distances with no graph/index machinery at all; useful for small
  tables, ground-truth comparisons, or when approximate recall isn't
  acceptable. `vector_rebuild_index`, `vector_sync_index`, and
  `vector_ef_search` all reject `mode=exact` tables since there is no index
  to rebuild, sync, or tune.

### KNN Search

```sql
SELECT rowid, distance [, metadata_cols...]
FROM <table>
WHERE knn_match(distance, <query_vector_blob>)
LIMIT <k>;
```

The `distance` column is a hidden virtual column that returns the distance
between each stored vector and the query. `knn_match` activates the index
(HNSW or exact, depending on the table's `mode`) for the search.

- If `LIMIT` is omitted, `k` defaults to 100 rows.
- `ORDER BY distance` ascending is served directly by the index/exact scan;
  `ORDER BY distance DESC` (or any other ordering) falls back to SQLite's own
  sort over the returned rows.
- Adding equality/range predicates on metadata columns (`=`, `<`, `<=`, `>`,
  `>=`) is pushed down as a filter: the index is searched with oversampling
  (fetching more than `k` candidates and filtering) until `k` rows satisfying
  the predicate are found or the table is exhausted.

### Scalar Functions

| Function | Description |
|----------|-------------|
| `vector_from_json(json, type)` | Parse a JSON array into a vector blob |
| `vector_to_json(blob, type)` | Convert a vector blob to a JSON array string |
| `vector_distance(blob_a, blob_b, metric, type)` | Compute distance between two vectors |
| `vector_dims(blob, type)` | Return the number of dimensions |
| `vector_rebuild_index(table)` | Rebuild the HNSW index from shadow data, using the table's persisted config (`hnsw` tables only) |
| `vector_sync_index(table)` | Force-persist the in-memory HNSW graph now, regardless of `sync_every` (`hnsw` tables only) |
| `vector_ef_search(table, n)` | Change a live table's `ef_search` and persist it for future reconnects (`hnsw` tables only) |
| `vector_index_info(table)` | Return a JSON object describing the table: `rows`, `dim`, `type`, `metric`, `mode`, `m`, `ef_construction`, `ef_search`, `sync_every`, `changes_since_persist` |
| `vector_normalize(blob, type)` | L2-normalize a vector; errors on a zero vector |
| `vector_add(blob_a, blob_b, type)` | Element-wise vector addition |
| `vector_sub(blob_a, blob_b, type)` | Element-wise vector subtraction |
| `vector_scale(blob, factor, type)` | Multiply every element by a scalar |
| `vector_slice(blob, type, start, end)` | Extract a half-open element range `[start, end)` |
| `vector_quantize_int8(blob, type)` | Quantize to `int1` using symmetric max-abs scaling |
| `vector_export_arrow(table, type)` | Export all vectors as an Arrow IPC blob |
| `vector_insert_arrow(table, type, ipc_blob)` | Import vectors from an Arrow IPC blob |

`vector_sync_index`, `vector_ef_search`, and `vector_index_info` accept either
a bare table name or a `db.table`-qualified name, but currently only tables in
the `main` database are supported — calling them against a table in an
attached database returns an error. See "Known limitations" below.

Notes on the arithmetic/quantization functions:
- `vector_add`/`vector_sub`/`vector_scale` on integer-typed vectors compute in
  `f64` internally and cast back to the element type on output; results
  saturate (clamp) at the type's min/max on overflow rather than wrapping or
  erroring.
- `vector_normalize` on an integer-typed vector returns a `float4` blob — a
  unit-length vector cannot be represented in the integer domain. Float
  inputs are normalized in place at their own element type.
- `vector_quantize_int8` scales by `127 / max(|v|)` (symmetric, based on the
  vector's own max absolute value) and rounds half-away-from-zero, clamping
  to `[-127, 127]`.

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

SQLite splits unquoted `CREATE VIRTUAL TABLE` module arguments on commas, so a
multi-column `metadata` value **must** be quoted as a single argument (as
shown above: `metadata='title TEXT, source TEXT, page INTEGER'`). Passing it
unquoted (`metadata=title TEXT, source TEXT, page INTEGER`) would be parsed
as several separate, invalid module arguments.

### Durability & Concurrency

- The `_data` shadow table (rowid, vector, metadata) is the transactional
  source of truth — every INSERT/UPDATE/DELETE goes through normal SQLite
  transactions, rollback, and savepoints.
- The in-memory HNSW graph is **lazily** persisted to the `_index` shadow
  table: it's written every `sync_every` changes (default 1024), not on every
  write. Call `vector_sync_index(table)` to force a persist immediately (e.g.
  before a backup).
- On connect, the table reconciles the persisted graph against `_data`: rows
  that were added/removed since the last persist are patched into the graph.
  This makes the setup crash-safe — a crash between graph persists loses at
  most the not-yet-persisted graph state, never row data — at the cost of a
  reconcile pass at open time.
- One writer per table at a time is assumed. Concurrent writers are
  last-write-wins **on the graph blob only** (never on `_data` rows, which
  remain fully transactional); other connections observe index changes only
  after they reconnect (or the table is re-opened), not live.

### Known Limitations

- `int2`/`int4` vectors are indexed as `f32` internally, which is lossy for
  integer magnitudes above 2^24.
- `vector_sync_index`, `vector_ef_search`, and `vector_index_info` only
  support tables in the `main` database. Vector tables in attached databases
  currently share `main`'s shadow-table namespace for these operations, so
  calling them against an attached-database table is rejected outright rather
  than risk writing into the wrong (or nonexistent) shadow tables.

### Arrow IPC Bulk Operations

Export all vectors to an Arrow IPC stream, then re-import into another table:

```sql
-- Export
SELECT vector_export_arrow('source_table', 'float4');

-- Import (returns row count)
SELECT vector_insert_arrow('dest_table', 'float4', <ipc_blob>);

-- Rebuild the HNSW index after bulk import
SELECT vector_rebuild_index('dest_table');
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

# Run all 324 tests
cargo test --features library
```

The test suite includes unit tests for every module, integration tests for
all SQL interfaces, a Shakespeare PDF ingestion pipeline, and GGUF embedding
tests (which download a small model on first run).

## License

Licensed under either of [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0)
or [MIT License](http://opensource.org/licenses/MIT), at your option.
