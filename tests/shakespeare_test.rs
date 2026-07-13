mod common;

use common::open_with_extension;
use rusqlite::params;
use sqlite_vector_rs::types::{VectorType, slice_to_blob};

/// Fixed dimension for our character-frequency feature vectors.
/// We use 26 lowercase letter frequencies + 10 digit frequencies + 4 punctuation
/// features (space, comma, period, question mark) = 40 dimensions.
const EMBED_DIM: usize = 40;

/// Build a deterministic "embedding" from text using character frequency.
///
/// This is NOT a real semantic embedding — it's a cheap, deterministic feature
/// vector used purely to exercise the vector store with realistic text data.
/// The vector is L2-normalised so cosine distance is meaningful.
fn text_to_vector(text: &str) -> Vec<f32> {
    let mut counts = [0u32; EMBED_DIM];
    let total = text.len().max(1) as f32;
    for ch in text.chars() {
        let idx = match ch {
            'a'..='z' => (ch as u32 - 'a' as u32) as usize,
            'A'..='Z' => (ch as u32 - 'A' as u32) as usize,
            '0'..='9' => 26 + (ch as u32 - '0' as u32) as usize,
            ' ' => 36,
            ',' => 37,
            '.' => 38,
            '?' => 39,
            _ => continue,
        };
        counts[idx] += 1;
    }
    let mut v: Vec<f32> = counts.iter().map(|&c| c as f32 / total).collect();

    // L2 normalise
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

/// Split text into chunks of roughly `target_len` characters, breaking at
/// whitespace boundaries. Skips chunks that are too short to be meaningful.
fn chunk_text(text: &str, target_len: usize, min_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut start = 0;
    let bytes = text.as_bytes();
    while start < text.len() {
        let mut end = (start + target_len).min(text.len());
        // Walk forward to the next whitespace to avoid splitting mid-word
        if end < text.len() {
            while end < text.len() && bytes[end] != b' ' && bytes[end] != b'\n' {
                end += 1;
            }
        }
        let chunk = text[start..end].trim();
        if chunk.len() >= min_len {
            chunks.push(chunk.to_string());
        }
        start = end;
    }
    chunks
}

#[test]
fn shakespeare_pdf_to_vector_store() {
    // --- 1. Extract text from the PDF ---
    let pdf_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/shakespeare.pdf"
    );
    let text =
        pdf_extract::extract_text(pdf_path).expect("failed to extract text from shakespeare.pdf");
    assert!(
        text.len() > 100_000,
        "expected substantial text from Shakespeare, got {} bytes",
        text.len()
    );

    // --- 2. Chunk into passages ---
    let chunks = chunk_text(&text, 500, 100);
    assert!(
        chunks.len() > 50,
        "expected many chunks from Shakespeare, got {}",
        chunks.len()
    );
    // Cap at 200 chunks for test speed
    let chunks: Vec<&String> = chunks.iter().take(200).collect();

    // --- 3. Create virtual table and insert ---
    let conn = open_with_extension();
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE shakespeare USING vector(dim={EMBED_DIM}, type=float4, metric=cosine)"
    ))
    .unwrap();

    for chunk in &chunks {
        let vec = text_to_vector(chunk);
        let blob = slice_to_blob(&vec);
        conn.execute(
            "INSERT INTO shakespeare(vector) VALUES(?)",
            [blob.as_slice()],
        )
        .unwrap();
    }

    // Verify row count
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shakespeare", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, chunks.len() as i64);

    // --- 4. KNN search for a passage similar to "to be or not to be" ---
    let query_vec = text_to_vector("to be or not to be that is the question");
    let query_blob = slice_to_blob(&query_vec);

    let mut stmt = conn
        .prepare("SELECT id, distance FROM shakespeare WHERE knn_match(distance, ?) LIMIT 5")
        .unwrap();

    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 5, "expected 5 nearest neighbours");

    // Results should be ordered by ascending distance
    for window in results.windows(2) {
        assert!(
            window[0].1 <= window[1].1,
            "results not ordered: {} > {}",
            window[0].1,
            window[1].1
        );
    }

    // The nearest result should have a reasonable distance (< 1.0 for cosine)
    assert!(
        results[0].1 < 1.0,
        "nearest distance {} is too large for cosine metric",
        results[0].1
    );

    // --- 5. Verify we can retrieve vector data for each result ---
    for (id, _dist) in &results {
        let blob: Vec<u8> = conn
            .query_row("SELECT vector FROM shakespeare WHERE id = ?", [id], |row| {
                row.get(0)
            })
            .unwrap();
        let expected_size = VectorType::Float4.blob_size(EMBED_DIM);
        assert_eq!(
            blob.len(),
            expected_size,
            "vector blob for id {id} has wrong size"
        );
    }
}

// ---------------------------------------------------------------------------
// Helper: extract and chunk the Shakespeare PDF (cached via lazy init)
// ---------------------------------------------------------------------------

fn load_shakespeare_chunks() -> Vec<String> {
    let pdf_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/shakespeare.pdf"
    );
    let text =
        pdf_extract::extract_text(pdf_path).expect("failed to extract text from shakespeare.pdf");
    chunk_text(&text, 500, 100)
}

/// Insert `n` chunks into a table named `name` with given type and metric.
/// Returns the chunks that were actually inserted.
fn populate_table(
    conn: &rusqlite::Connection,
    name: &str,
    vtype: VectorType,
    metric: &str,
    chunks: &[String],
    n: usize,
) -> Vec<String> {
    let used: Vec<String> = chunks.iter().take(n).cloned().collect();
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE {name} USING vector(dim={EMBED_DIM}, type={}, metric={metric})",
        vtype.name()
    ))
    .unwrap();
    for chunk in &used {
        let vec = text_to_vector(chunk);
        let blob = slice_to_blob(&vec);
        conn.execute(
            &format!("INSERT INTO {name}(vector) VALUES(?)"),
            [blob.as_slice()],
        )
        .unwrap();
    }
    used
}

// ---------------------------------------------------------------------------
// Tests: different metrics
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_l2_metric() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let used = populate_table(&conn, "shk_l2", VectorType::Float4, "l2", &chunks, 150);

    let query_blob = slice_to_blob(&text_to_vector("Romeo Romeo wherefore art thou Romeo"));
    let mut stmt = conn
        .prepare("SELECT id, distance FROM shk_l2 WHERE knn_match(distance, ?) LIMIT 3")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 3);
    // L2 distances should be non-negative and ordered
    for r in &results {
        assert!(r.1 >= 0.0, "L2 distance must be non-negative, got {}", r.1);
    }
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_l2", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, used.len() as i64);
}

#[test]
fn shakespeare_inner_product_metric() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    populate_table(&conn, "shk_ip", VectorType::Float4, "ip", &chunks, 100);

    let query_blob = slice_to_blob(&text_to_vector("double double toil and trouble fire burn"));
    let mut stmt = conn
        .prepare("SELECT id, distance FROM shk_ip WHERE knn_match(distance, ?) LIMIT 5")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 5);
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1, "IP results not ordered");
    }
}

// ---------------------------------------------------------------------------
// Tests: different vector types
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_float8_vectors() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 80;
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE shk_f8 USING vector(dim={EMBED_DIM}, type=float8, metric=l2)"
    ))
    .unwrap();

    for chunk in chunks.iter().take(n) {
        let f32_vec = text_to_vector(chunk);
        let f64_vec: Vec<f64> = f32_vec.iter().map(|&x| x as f64).collect();
        let blob = slice_to_blob(&f64_vec);
        conn.execute("INSERT INTO shk_f8(vector) VALUES(?)", [blob.as_slice()])
            .unwrap();
    }

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_f8", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, n as i64);

    // KNN search with float8
    let f32_q = text_to_vector("a midsummer nights dream");
    let f64_q: Vec<f64> = f32_q.iter().map(|&x| x as f64).collect();
    let query_blob = slice_to_blob(&f64_q);
    let mut stmt = conn
        .prepare("SELECT id, distance FROM shk_f8 WHERE knn_match(distance, ?) LIMIT 3")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(results.len(), 3);
}

// ---------------------------------------------------------------------------
// Tests: delete and re-search
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_delete_and_search() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let used = populate_table(&conn, "shk_del", VectorType::Float4, "cosine", &chunks, 50);

    // Delete first 10 rows
    for id in 1..=10 {
        conn.execute("DELETE FROM shk_del WHERE id = ?", [id])
            .unwrap();
    }

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_del", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, (used.len() - 10) as i64);

    // KNN search should still work and not return deleted rows
    let query_blob = slice_to_blob(&text_to_vector("friends romans countrymen"));
    let mut stmt = conn
        .prepare("SELECT id, distance FROM shk_del WHERE knn_match(distance, ?) LIMIT 5")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();

    assert_eq!(results.len(), 5);
    for (id, _) in &results {
        assert!(*id > 10, "deleted row {id} appeared in search results");
    }
}

// ---------------------------------------------------------------------------
// Tests: scalar functions on Shakespeare data
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_vector_distance_between_chunks() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();

    // Compute distance between two Shakespeare passages via SQL
    let v1 = slice_to_blob(&text_to_vector(&chunks[0]));
    let v2 = slice_to_blob(&text_to_vector(&chunks[1]));

    let dist: f64 = conn
        .query_row(
            "SELECT vector_distance(?, ?, 'cosine', 'float4')",
            [v1.as_slice(), v2.as_slice()],
            |row| row.get(0),
        )
        .unwrap();

    // Cosine distance between different texts should be > 0
    assert!(dist > 0.0, "expected positive distance, got {dist}");
    // And bounded (cosine distance is in [0, 2])
    assert!(dist <= 2.0, "cosine distance out of range: {dist}");
}

#[test]
fn shakespeare_vector_dims_matches() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let blob = slice_to_blob(&text_to_vector(&chunks[0]));

    let dims: i64 = conn
        .query_row(
            "SELECT vector_dims(?, 'float4')",
            [blob.as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(dims, EMBED_DIM as i64);
}

#[test]
fn shakespeare_self_distance_is_zero() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let blob = slice_to_blob(&text_to_vector(&chunks[5]));

    let dist: f64 = conn
        .query_row(
            "SELECT vector_distance(?, ?, 'l2', 'float4')",
            [blob.as_slice(), blob.as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert!(dist.abs() < 1e-6, "self-distance should be ~0, got {dist}");
}

// ---------------------------------------------------------------------------
// Tests: Arrow export/import round-trip with Shakespeare data
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_arrow_export_import() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 30;
    populate_table(&conn, "shk_arrow", VectorType::Float4, "l2", &chunks, n);

    // Export to Arrow IPC
    let ipc_blob: Vec<u8> = conn
        .query_row(
            "SELECT vector_export_arrow('shk_arrow', 'float4')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert!(!ipc_blob.is_empty());

    // Import into a fresh table
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE shk_arrow2 USING vector(dim={EMBED_DIM}, type=float4, metric=l2)"
    ))
    .unwrap();

    let imported: i64 = conn
        .query_row(
            "SELECT vector_insert_arrow('shk_arrow2', 'float4', ?)",
            [ipc_blob.as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(imported, n as i64);

    // Verify counts match
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_arrow2", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, n as i64);
}

// ---------------------------------------------------------------------------
// Tests: rebuild index on Shakespeare data
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_rebuild_index() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 60;
    populate_table(&conn, "shk_rebuild", VectorType::Float4, "l2", &chunks, n);

    let rebuilt: i64 = conn
        .query_row("SELECT vector_rebuild_index('shk_rebuild')", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(rebuilt, n as i64);

    // Shadow table data should still be intact
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_rebuild", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, n as i64);
}

// ---------------------------------------------------------------------------
// Tests: full scan returns all rows
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_full_scan() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 75;
    populate_table(&conn, "shk_scan", VectorType::Float4, "cosine", &chunks, n);

    // Full table scan (no knn_match constraint) should return all rows
    let mut stmt = conn.prepare("SELECT id, vector FROM shk_scan").unwrap();
    let rows: Vec<(i64, Vec<u8>)> = stmt
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(rows.len(), n);

    // Every vector should have the correct blob size
    let expected_size = VectorType::Float4.blob_size(EMBED_DIM);
    for (id, blob) in &rows {
        assert_eq!(blob.len(), expected_size, "wrong blob size for row {id}");
    }
}

// ---------------------------------------------------------------------------
// Tests: multiple queries return consistent results
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_repeated_knn_is_stable() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    populate_table(
        &conn,
        "shk_stable",
        VectorType::Float4,
        "cosine",
        &chunks,
        100,
    );

    let query_blob = slice_to_blob(&text_to_vector("shall I compare thee to a summers day"));

    // Run the same KNN query twice — results must be identical
    let fetch = |conn: &rusqlite::Connection| -> Vec<(i64, f64)> {
        let mut stmt = conn
            .prepare("SELECT id, distance FROM shk_stable WHERE knn_match(distance, ?) LIMIT 10")
            .unwrap();
        stmt.query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
    };

    let run1 = fetch(&conn);
    let run2 = fetch(&conn);

    assert_eq!(run1.len(), run2.len());
    for (a, b) in run1.iter().zip(run2.iter()) {
        assert_eq!(a.0, b.0, "row ids differ between runs");
        assert!(
            (a.1 - b.1).abs() < 1e-10,
            "distances differ: {} vs {}",
            a.1,
            b.1
        );
    }
}

// ---------------------------------------------------------------------------
// Tests: varying LIMIT values
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_knn_varying_k() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 100;
    populate_table(&conn, "shk_k", VectorType::Float4, "cosine", &chunks, n);

    let query_blob = slice_to_blob(&text_to_vector("the lady doth protest too much"));

    for k in [1, 5, 10, 50] {
        let sql = format!("SELECT id, distance FROM shk_k WHERE knn_match(distance, ?) LIMIT {k}");
        let mut stmt = conn.prepare(&sql).unwrap();
        let results: Vec<(i64, f64)> = stmt
            .query_map(params![query_blob.as_slice()], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();

        assert_eq!(results.len(), k, "LIMIT {k} should return {k} rows");
        for w in results.windows(2) {
            assert!(
                w[0].1 <= w[1].1,
                "k={k}: results not ordered: {} > {}",
                w[0].1,
                w[1].1
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Tests: different Shakespeare quotes find different nearest neighbours
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_different_queries_different_results() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    populate_table(
        &conn,
        "shk_diff",
        VectorType::Float4,
        "cosine",
        &chunks,
        200,
    );

    let queries = [
        "to be or not to be",
        "now is the winter of our discontent",
        "all the worlds a stage and all the men and women merely players",
        "out out brief candle life is but a walking shadow",
    ];

    let mut all_top_ids: Vec<Vec<i64>> = Vec::new();
    for q in &queries {
        let query_blob = slice_to_blob(&text_to_vector(q));
        let mut stmt = conn
            .prepare("SELECT id FROM shk_diff WHERE knn_match(distance, ?) LIMIT 3")
            .unwrap();
        let ids: Vec<i64> = stmt
            .query_map(params![query_blob.as_slice()], |row| row.get(0))
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(ids.len(), 3);
        all_top_ids.push(ids);
    }

    // At least some queries should return different top-1 results
    let top1s: Vec<i64> = all_top_ids.iter().map(|ids| ids[0]).collect();
    let unique_count = {
        let mut s = top1s.clone();
        s.sort();
        s.dedup();
        s.len()
    };
    assert!(
        unique_count >= 2,
        "expected at least 2 distinct top-1 results across 4 queries, got {unique_count}: {top1s:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests: insert, search, delete all, verify empty
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_insert_delete_all_empty() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    let n = 20;
    populate_table(&conn, "shk_empty", VectorType::Float4, "l2", &chunks, n);

    // Delete every row
    for id in 1..=n {
        conn.execute("DELETE FROM shk_empty WHERE id = ?", [id as i64])
            .unwrap();
    }

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_empty", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

// ---------------------------------------------------------------------------
// Tests: file-backed database with Shakespeare data
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_file_backed_persistence() {
    use common::open_file_with_extension;

    let chunks = load_shakespeare_chunks();
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("shakespeare.db");

    // Write phase
    {
        let conn = open_file_with_extension(&db_path);
        populate_table(&conn, "shk_file", VectorType::Float4, "cosine", &chunks, 40);
    }

    // Read phase — reopen the database
    {
        let conn = open_file_with_extension(&db_path);
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM shk_file", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 40);

        // KNN still works after reopen
        let query_blob = slice_to_blob(&text_to_vector("what light through yonder window"));
        let mut stmt = conn
            .prepare("SELECT id, distance FROM shk_file WHERE knn_match(distance, ?) LIMIT 3")
            .unwrap();
        let results: Vec<(i64, f64)> = stmt
            .query_map(params![query_blob.as_slice()], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(results.len(), 3);
    }
}

// ---------------------------------------------------------------------------
// Tests: large batch insert
// ---------------------------------------------------------------------------

#[test]
fn shakespeare_large_batch() {
    let chunks = load_shakespeare_chunks();
    let conn = open_with_extension();
    // Use all available chunks (typically 500+)
    let n = chunks.len().min(500);
    let used = populate_table(&conn, "shk_large", VectorType::Float4, "cosine", &chunks, n);

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM shk_large", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, used.len() as i64);

    // KNN on a bigger table
    let query_blob = slice_to_blob(&text_to_vector("parting is such sweet sorrow"));
    let mut stmt = conn
        .prepare("SELECT id, distance FROM shk_large WHERE knn_match(distance, ?) LIMIT 10")
        .unwrap();
    let results: Vec<(i64, f64)> = stmt
        .query_map(params![query_blob.as_slice()], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(results.len(), 10);
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
}

// ---------------------------------------------------------------------------
// Unit-like tests for helpers
// ---------------------------------------------------------------------------

#[test]
fn text_to_vector_is_deterministic() {
    let a = text_to_vector("Hello, world!");
    let b = text_to_vector("Hello, world!");
    assert_eq!(a, b, "same text must produce same vector");
}

#[test]
fn text_to_vector_is_normalised() {
    let v = text_to_vector("Shall I compare thee to a summer's day?");
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "vector should be L2-normalised, got norm={norm}"
    );
}

#[test]
fn chunk_text_respects_boundaries() {
    let text = "word ".repeat(100);
    let chunks = chunk_text(&text, 25, 5);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        // No chunk should be drastically longer than target + one word
        assert!(chunk.len() <= 35, "chunk too long: {} chars", chunk.len());
    }
}
