# Changelog

All notable changes to this project will be documented in this file.

## 0.3.1 — 2026-07-13

### Changed

- Packaging: `tests/fixtures/*` (the LFS-tracked Shakespeare PDF) is excluded
  from the published crate, shrinking the package. No code or API changes.

## 0.3.0 — 2026-07-12

### Fixed

- **B1** — `UPDATE` now preserves the rowid instead of delete+reinserting under a
  new id, and re-keys the HNSW entry; partial `UPDATE`s preserve untouched
  columns.
- **B2** — `INSERT` honors an explicitly supplied rowid instead of silently
  assigning the next sequential id.
- **B3** — Transaction rollback on a fresh connection no longer leaves phantom
  index keys ("Duplicate keys not allowed"); the rollback snapshot is primed at
  connect and savepoints use a real savepoint stack.
- **B4** — Metadata columns round-trip with their declared types through the
  cursor (`typeof(label)` is `text`, not `blob`; `WHERE label = 'a'` matches).
- **B5** — `LIMIT`/`ORDER BY` are consumed by the index only when provably safe:
  `ORDER BY distance` ascending is served by the index, `DESC` and other
  orderings fall back to SQLite; `LIMIT 0` returns no rows.
- **B6** — `vector_rebuild_index` rebuilds with the table's real
  dim/type/metric/HNSW parameters from the persisted config instead of
  defaults; its signature is now the 1-argument `vector_rebuild_index(table)`.
- **B7** — All blob cast sites tolerate misaligned input instead of relying on
  allocator alignment (latent `bytemuck` panic).

### Added

- `mode=exact` table option: brute-force streaming top-k with exact distances
  and no index/graph machinery.
- Filtered KNN: metadata `=`/`<`/`<=`/`>`/`>=` constraints are pushed into the
  KNN scan with oversampling until the `LIMIT` is satisfied.
- `sync_every=<n>` table option (default 1024) for lazy graph persistence.
- `vector_sync_index(table)` — force-persist the in-memory HNSW graph.
- `vector_ef_search(table, n)` — change `ef_search` live and persist it for
  future connections.
- `vector_index_info(table)` — JSON snapshot of rows, dim, type, metric, mode,
  m, ef_construction, ef_search, sync_every, and changes_since_persist.
- Vector math scalar functions: `vector_normalize`, `vector_add`,
  `vector_sub`, `vector_scale`, `vector_slice`, `vector_quantize_int8`
  (symmetric max-abs int8 quantization).
- Table configuration is persisted in a `meta` row of the `_index` shadow
  table and reloaded on connect.
- Throughput regression test guarding autocommit insert cost.

### Changed

- Lazy graph persistence: the HNSW graph is written to the `_index` shadow
  table every `sync_every` changes instead of on every dirty commit, with
  crash-safe reconcile-on-connect (`_data` remains the transactional source of
  truth).
- Full scans stream rows from the shadow table instead of materializing the
  whole table in memory.
- Shadow-table SQL is built once per table and the KNN fetch statement is
  reused across rows.
- Distance kernels are single-pass and allocation-free.
- **Library API (breaking):** `VectorType::slice_to_blob` is now the free
  function `types::slice_to_blob`, and `VectorType::blob_to_slice` is removed —
  use the alignment-tolerant `types::cast_blob` to decode a blob. Affects rlib
  consumers only; the loadable-extension SQL surface is unchanged.

### Known limitations

- `vector_sync_index`, `vector_ef_search`, `vector_index_info`, and
  `vector_rebuild_index` accept bare or `db.table`-qualified names but only
  support tables in the `main` database; attached-database vector tables are
  rejected.
- `int2`/`int4` vectors are indexed as `f32` (lossy above 2^24).

## 0.2.0

Initial published release: typed vector virtual table with HNSW search, scalar
functions, Arrow IPC bulk I/O, library mode, and the `sqlite3` REPL. See the
git history prior to `v0.3.0` for details.
