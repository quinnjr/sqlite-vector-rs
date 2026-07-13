mod common;

use common::{open_file_with_extension, open_with_extension};

#[test]
fn unpersisted_commits_survive_reconnect_via_reconcile() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);",
        )
        .unwrap();
        for i in 0..10 {
            conn.execute(
                "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
                [format!("[{i}.0, 0.0]")],
            )
            .unwrap();
        }
    } // dropped without ever reaching the persist threshold
    let conn = open_file_with_extension(&path);
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 20",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n, 10,
        "reconcile must recover rows committed after the last persist"
    );
}

#[test]
fn deleted_rows_absent_after_reconnect() {
    // NOTE: this test does NOT exercise the connect-time rebuild-from-scratch
    // branch in `reconcile_index` (`index.len() != count`). Deletes are
    // eagerly persisted at commit (see `IndexState::destructive_since_persist`
    // in src/vtab/transaction.rs), so by the time this connection is dropped,
    // the persisted graph already agrees with `_data`: reconnect finds
    // `index.len() == count` and takes the cheap "already reconciled" path.
    // What this test DOES verify is the simpler, more common case: a DELETE
    // through the vtab (not a raw shadow-table write) is durable across a
    // reconnect. For a test that genuinely forces the rebuild branch by
    // making the persisted graph stale relative to `_data`, see
    // `connect_time_rebuild_from_data_when_graph_stale` below.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);",
        )
        .unwrap();
        for i in 0..5 {
            conn.execute(
                "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
                [format!("[{i}.0, 0.0]")],
            )
            .unwrap();
        }
        conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
            .unwrap(); // graph persisted with 5 keys
        conn.execute("DELETE FROM t WHERE id <= 2", []).unwrap(); // eagerly persisted (destructive_since_persist)
    }
    let conn = open_file_with_extension(&path);
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        ids.len(),
        3,
        "deleted rows must not resurface after reconnect"
    );
    assert!(ids.iter().all(|id| *id >= 3));
}

#[test]
fn connect_time_rebuild_from_data_when_graph_stale() {
    // Forces the genuine rebuild-from-scratch branch in `reconcile_index`
    // (`index.len() != count`): persist a graph with N keys via
    // `vector_sync_index`, then delete a row by writing DIRECTLY to the
    // `_data` shadow table via plain SQL. This bypasses the vtab entirely
    // (no xUpdate call, so no eager persist / `destructive_since_persist`
    // marking), leaving the persisted graph with a stale key that no longer
    // exists in `_data`. `<table>_data` is an ordinary SQLite table (see
    // `CREATE TABLE IF NOT EXISTS "{}_data"` in src/vtab/shadow.rs) — it is
    // not a real sqlite3 "shadow table" gated by module-declared eponymous
    // naming/innocuous flags, so it is fully writable by direct DML.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    let n = 6;
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
            .unwrap();
        for i in 0..n {
            conn.execute(
                "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
                [format!("[{i}.0, 0.0]")],
            )
            .unwrap();
        }
        conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
            .unwrap(); // graph persisted with n keys

        // Bypass the vtab: direct DML on the shadow table itself. The graph
        // still thinks id=1 exists.
        let deleted: i64 = conn
            .execute("DELETE FROM \"t_data\" WHERE id = 1", [])
            .unwrap() as i64;
        assert_eq!(deleted, 1, "direct shadow-table DML must be permitted");
    }
    let conn = open_file_with_extension(&path);
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 100",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        ids.len(),
        (n - 1) as usize,
        "stale graph key must be dropped by connect-time rebuild"
    );
    assert!(
        !ids.contains(&1),
        "row deleted directly from _data must be absent after rebuild"
    );
}

#[test]
fn vector_sync_index_persists_graph_and_state() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
         INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();
    conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
        .unwrap();
    let has_graph: i64 = conn
        .query_row(
            "SELECT count(*) FROM t_index WHERE key = 'hnsw_graph'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let state: String = conn
        .query_row(
            "SELECT value FROM t_index WHERE key = 'graph_state'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(has_graph, 1);
    let v: serde_json::Value = serde_json::from_str(&state).unwrap();
    assert_eq!(v["row_count"], 1);
}

#[test]
fn full_scan_streams_all_rows_in_order() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE s USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
    for i in 0..200 {
        conn.execute(
            "INSERT INTO s(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0]")],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM s")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 200);
    assert_eq!(ids.first(), Some(&1));
    assert_eq!(ids.last(), Some(&200));
}

#[test]
fn autocommit_inserts_persist_lazily_not_per_row() {
    // Deterministic proxy for the O(N)-per-commit regression this test
    // guards, replacing the old wall-clock `* 3` assertion (flaky under
    // load/CI jitter). The regression this guards against: every autocommit
    // INSERT triggering its own eager graph persist (each commit calling
    // `persist_index`, which resets `changes_since_persist` to 0). With the
    // default `sync_every=1024` threshold, 2000 autocommit inserts should
    // persist at most ~2 times, not once per row.
    //
    // The proxy: after fewer than `sync_every` autocommit inserts (well under
    // the persist threshold), `changes_since_persist` (exposed by
    // `vector_index_info`) must be > 0. If persistence had regressed to
    // per-commit, `sync()` would persist (and reset the counter to 0) after
    // every single insert, so this would read 0 or 1 instead of accumulating.
    // A second assertion checks the counter keeps growing across inserts
    // (rather than being reset), which a naive "persist every Nth insert
    // where N resets differently" implementation could otherwise slip past.
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE a USING vector(dim=8, type=float4, metric=l2);")
        .unwrap();

    let read_pending = |c: &rusqlite::Connection| -> i64 {
        let info: String = c
            .query_row("SELECT vector_index_info('a')", [], |r| r.get(0))
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&info).unwrap();
        v["changes_since_persist"].as_i64().unwrap()
    };

    let vec_json = |i: i64| -> String { format!("[{i}.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]") };

    for i in 0..50 {
        conn.execute(
            "INSERT INTO a(vector) VALUES (vector_from_json(?1, 'float4'))",
            [vec_json(i)],
        )
        .unwrap();
        if i == 9 {
            assert!(
                read_pending(&conn) > 1,
                "changes_since_persist must accumulate across autocommit inserts, \
                 not reset to 0/1 per row (would indicate a per-commit persist regression)"
            );
        }
    }
    let pending_after_50 = read_pending(&conn);
    assert!(
        pending_after_50 > 1,
        "expected pending changes to still be accumulating well under sync_every \
         (default 1024); got changes_since_persist={pending_after_50} after 50 inserts"
    );

    // Correctness under autocommit: a plain-transaction table inserting the
    // same rows must end with the same row count and KNN result set.
    let conn2 = open_with_extension();
    conn2
        .execute_batch("CREATE VIRTUAL TABLE b USING vector(dim=8, type=float4, metric=l2);")
        .unwrap();
    conn2.execute_batch("BEGIN").unwrap();
    for i in 0..50 {
        conn2
            .execute(
                "INSERT INTO b(vector) VALUES (vector_from_json(?1, 'float4'))",
                [vec_json(i)],
            )
            .unwrap();
    }
    conn2.execute_batch("COMMIT").unwrap();

    let count_a: i64 = conn
        .query_row("SELECT count(*) FROM a", [], |r| r.get(0))
        .unwrap();
    let count_b: i64 = conn2
        .query_row("SELECT count(*) FROM b", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        count_a, count_b,
        "autocommit and single-transaction inserts must produce identical row counts"
    );

    // Query vector matches row 25 exactly, so a healthy index on both tables
    // must return the identical *set* of nearest ids (not just a matching
    // count) — a desynced/corrupted index would return a different id set
    // even though both tables have 50 rows.
    let query_vec = vec_json(25);
    let knn = |c: &rusqlite::Connection, t: &str| -> Vec<i64> {
        let mut ids: Vec<i64> = c
            .prepare(&format!(
                "SELECT id FROM {t} WHERE knn_match(distance, vector_from_json('{query_vec}', 'float4')) LIMIT 10"
            ))
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        ids.sort_unstable();
        ids
    };
    assert_eq!(
        knn(&conn, "a"),
        knn(&conn2, "b"),
        "autocommit and single-transaction KNN must return the identical id set for a fixed query"
    );
}

#[test]
fn filtered_knn_returns_full_limit_beyond_default_k() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE fk USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT\");",
    )
    .unwrap();
    // 150 near rows labeled 'b' (crowd out DEFAULT_KNN_K=100), then 5 far rows labeled 'a'.
    for i in 0..150 {
        conn.execute(
            "INSERT INTO fk(vector, label) VALUES (vector_from_json(?1, 'float4'), 'b')",
            [format!("[{}, 0.0]", i as f64 * 0.01)],
        )
        .unwrap();
    }
    for i in 0..5 {
        conn.execute(
            "INSERT INTO fk(vector, label) VALUES (vector_from_json(?1, 'float4'), 'a')",
            [format!("[{}.0, 50.0]", i)],
        )
        .unwrap();
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM fk WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
             AND label = 'a' LIMIT 3",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        ids.len(),
        3,
        "oversampling must reach past the crowd of 'b' rows"
    );
}

#[test]
fn knn_limit_zero_returns_no_rows() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE lz USING vector(dim=2, type=float4, metric=l2);
         INSERT INTO lz(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM lz WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 0)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 0, "LIMIT 0 must return no rows");
}

#[test]
fn exact_mode_knn_matches_hnsw_results() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE eh USING vector(dim=2, type=float4, metric=l2);
         CREATE VIRTUAL TABLE ee USING vector(dim=2, type=float4, metric=l2, mode=exact);",
    )
    .unwrap();
    // Coordinates are chosen (i % 13, (i * 3) % 17) rather than the more
    // obvious (i % 7, i % 5) because the latter produces several points
    // exactly equidistant from the query at the LIMIT-5 boundary: squared L2
    // ties there are broken differently by usearch's graph traversal
    // (HNSW, approximate order) versus a plain distance sort (exact, stable
    // scan order), which made the two tables legitimately disagree on which
    // members of a tied group made the cut — not a bug in either search
    // path, just an underspecified expectation for tied inputs. This spread
    // keeps the top candidates at distinct distances so the parity check is
    // well-defined.
    for i in 0..30 {
        for t in ["eh", "ee"] {
            conn.execute(
                &format!("INSERT INTO {t}(vector) VALUES (vector_from_json(?1, 'float4'))"),
                [format!("[{}.0, {}.0]", i % 13, (i * 3) % 17)],
            )
            .unwrap();
        }
    }
    let q = "vector_from_json('[1.0, 9.0]', 'float4')";
    let get = |t: &str| -> Vec<i64> {
        conn.prepare(&format!(
            "SELECT id FROM {t} WHERE knn_match(distance, {q}) LIMIT 5"
        ))
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
    };
    assert_eq!(
        get("eh"),
        get("ee"),
        "exact and hnsw must agree on this small table"
    );
}

#[test]
fn exact_mode_respects_filters_and_limit() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE ef USING vector(dim=2, type=float4, metric=l2, mode=exact, metadata=\"label TEXT\");
         INSERT INTO ef(vector, label) VALUES (vector_from_json('[0.0, 0.0]', 'float4'), 'b');
         INSERT INTO ef(vector, label) VALUES (vector_from_json('[1.0, 1.0]', 'float4'), 'a');
         INSERT INTO ef(vector, label) VALUES (vector_from_json('[2.0, 2.0]', 'float4'), 'a');",
    )
    .unwrap();
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM ef WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
             AND label = 'a' LIMIT 2",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids, vec![2, 3]);
}

#[test]
fn duplicate_metadata_constraints_do_not_crash_best_index() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE dup_hnsw USING vector(dim=2, type=float4, metric=l2, metadata=\"score REAL\");
         CREATE VIRTUAL TABLE dup_exact USING vector(dim=2, type=float4, metric=l2, mode=exact, metadata=\"score REAL\");",
    )
    .unwrap();
    for t in ["dup_hnsw", "dup_exact"] {
        for i in 0..5 {
            conn.execute(
                &format!(
                    "INSERT INTO {t}(vector, score) VALUES (vector_from_json(?1, 'float4'), ?2)"
                ),
                rusqlite::params![format!("[{}.0, 0.0]", i), i as f64],
            )
            .unwrap();
        }
    }
    for t in ["dup_hnsw", "dup_exact"] {
        // Two constraints on the SAME (column, op) pair used to make best_index
        // assign the same argv slot twice, leaving a gap SQLite rejects with
        // "xBestIndex malfunction". The correct result is the intersection:
        // score > 1.0 AND score > 3.0 == score > 3.0, i.e. ids where score is 4.0.
        let ids: Vec<i64> = conn
            .prepare(&format!(
                "SELECT id FROM {t} WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
                 AND score > 1.0 AND score > 3.0 LIMIT 5"
            ))
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(
            ids,
            vec![5],
            "table {t}: expected only the row with score=4.0 (id=5)"
        );
    }
}

#[test]
fn vector_ef_search_and_index_info_accept_qualified_table_names() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE qn USING vector(dim=4, type=float4, metric=cosine, m=8, ef_search=32);
         INSERT INTO qn(vector) VALUES (vector_from_json('[1.0, 0.0, 0.0, 0.0]', 'float4'));",
    )
    .unwrap();

    // Bare name baseline.
    let info_bare: String = conn
        .query_row("SELECT vector_index_info('qn')", [], |r| r.get(0))
        .unwrap();
    let v_bare: serde_json::Value = serde_json::from_str(&info_bare).unwrap();

    // "main.qn"-qualified calls must succeed and match the bare-name results,
    // and must not build shadow-table SQL from the raw "main.qn" argument
    // (which would look for a nonexistent "main.qn_index"/"main.qn_data" table).
    let info_qualified: String = conn
        .query_row("SELECT vector_index_info('main.qn')", [], |r| r.get(0))
        .unwrap();
    let v_qualified: serde_json::Value = serde_json::from_str(&info_qualified).unwrap();
    assert_eq!(v_qualified, v_bare);

    let new_ef: i64 = conn
        .query_row("SELECT vector_ef_search('main.qn', 128)", [], |r| r.get(0))
        .unwrap();
    assert_eq!(new_ef, 128);

    let info_after: String = conn
        .query_row("SELECT vector_index_info('main.qn')", [], |r| r.get(0))
        .unwrap();
    let v_after: serde_json::Value = serde_json::from_str(&info_after).unwrap();
    assert_eq!(v_after["ef_search"], 128);

    // Bare-name view must observe the same update (same registry entry).
    let info_bare_after: String = conn
        .query_row("SELECT vector_index_info('qn')", [], |r| r.get(0))
        .unwrap();
    let v_bare_after: serde_json::Value = serde_json::from_str(&info_bare_after).unwrap();
    assert_eq!(v_bare_after["ef_search"], 128);
}

#[test]
fn index_info_reports_state_and_ef_search_is_adjustable() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE ii USING vector(dim=4, type=float4, metric=cosine, m=8, ef_search=32);
         INSERT INTO ii(vector) VALUES (vector_from_json('[1.0, 0.0, 0.0, 0.0]', 'float4'));",
    )
    .unwrap();
    let info: String = conn
        .query_row("SELECT vector_index_info('ii')", [], |r| r.get(0))
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&info).unwrap();
    assert_eq!(v["rows"], 1);
    assert_eq!(v["dim"], 4);
    assert_eq!(v["metric"], "cosine");
    assert_eq!(v["mode"], "hnsw");
    assert_eq!(v["ef_search"], 32);

    conn.query_row("SELECT vector_ef_search('ii', 128)", [], |r| {
        r.get::<_, i64>(0)
    })
    .unwrap();
    let info2: String = conn
        .query_row("SELECT vector_index_info('ii')", [], |r| r.get(0))
        .unwrap();
    let v2: serde_json::Value = serde_json::from_str(&info2).unwrap();
    assert_eq!(v2["ef_search"], 128);
}

#[test]
fn vector_utility_functions() {
    let conn = open_with_extension();
    // normalize: [3,4] -> [0.6, 0.8]
    let n: String = conn
        .query_row(
            "SELECT vector_to_json(vector_normalize(vector_from_json('[3.0, 4.0]', 'float4'), 'float4'), 'float4')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    let v: Vec<f64> = serde_json::from_str(&n).unwrap();
    assert!((v[0] - 0.6).abs() < 1e-6 && (v[1] - 0.8).abs() < 1e-6);

    // add / sub / scale
    let s: String = conn
        .query_row(
            "SELECT vector_to_json(vector_add(vector_from_json('[1.0, 2.0]', 'float4'), vector_from_json('[3.0, 4.0]', 'float4'), 'float4'), 'float4')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<f64>>(&s).unwrap(),
        vec![4.0, 6.0]
    );

    let d: String = conn
        .query_row(
            "SELECT vector_to_json(vector_scale(vector_from_json('[1.0, -2.0]', 'float4'), 2.5, 'float4'), 'float4')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<f64>>(&d).unwrap(),
        vec![2.5, -5.0]
    );

    // slice: elements [1, 3) of a 4-dim vector
    let sl: String = conn
        .query_row(
            "SELECT vector_to_json(vector_slice(vector_from_json('[0.0, 1.0, 2.0, 3.0]', 'float4'), 'float4', 1, 3), 'float4')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<f64>>(&sl).unwrap(),
        vec![1.0, 2.0]
    );

    // quantize: [0.0, 127-max scaling] — max_abs=2.0 -> scale 63.5
    // Note: -1.0 * 63.5 = -63.5, which rounds to -64 (round-half-away-from-zero)
    let q: String = conn
        .query_row(
            "SELECT vector_to_json(vector_quantize_int8(vector_from_json('[2.0, -1.0]', 'float4'), 'float4'), 'int1')",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Vec<i64>>(&q).unwrap(),
        vec![127, -64]
    );

    // error cases
    assert!(conn
        .query_row(
            "SELECT vector_add(vector_from_json('[1.0]', 'float4'), vector_from_json('[1.0, 2.0]', 'float4'), 'float4')",
            [],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .is_err(), "dimension mismatch must error");
    assert!(
        conn.query_row(
            "SELECT vector_normalize(vector_from_json('[0.0, 0.0]', 'float4'), 'float4')",
            [],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .is_err(),
        "zero vector cannot be normalized"
    );
    assert!(
        conn.query_row(
            "SELECT vector_slice(vector_from_json('[1.0, 2.0]', 'float4'), 'float4', 1, 5)",
            [],
            |r| r.get::<_, Vec<u8>>(0),
        )
        .is_err(),
        "out-of-bounds slice must error"
    );
}

#[test]
fn unpersisted_update_is_not_stale_after_reconnect() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
             INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'));
             INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 1.0]', 'float4'));",
        )
        .unwrap();
        conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
            .unwrap();
        // Unpersisted destructive UPDATE: id 1's vector moves far away.
        conn.execute(
            "UPDATE t SET vector = vector_from_json('[100.0, 100.0]', 'float4') WHERE id = 1",
            [],
        )
        .unwrap();
    } // dropped without ever reaching the persist threshold
    let conn = open_file_with_extension(&path);
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[100.0, 100.0]', 'float4')) LIMIT 10",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        ids.first(),
        Some(&1),
        "reconnect must serve id 1's NEW vector, not the stale graph embedding"
    );
}

#[test]
fn unpersisted_delete_reinsert_same_id_not_stale() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
             INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'));
             INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 1.0]', 'float4'));",
        )
        .unwrap();
        conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
            .unwrap();
        // Unpersisted destructive DELETE + re-INSERT of the same id with a
        // different (far) vector.
        conn.execute("DELETE FROM t WHERE id = 1", []).unwrap();
        conn.execute(
            "INSERT INTO t(id, vector) VALUES (1, vector_from_json('[200.0, 200.0]', 'float4'))",
            [],
        )
        .unwrap();
    } // dropped without ever reaching the persist threshold
    let conn = open_file_with_extension(&path);
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[200.0, 200.0]', 'float4')) LIMIT 10",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(
        ids.first(),
        Some(&1),
        "reconnect must serve id 1's NEW (re-inserted) vector, not the stale deleted embedding"
    );
}

#[test]
fn mid_transaction_sync_then_rollback_recovers() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);",
    )
    .unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'))",
        [],
    )
    .unwrap(); // autocommit, committed row 1

    conn.execute_batch("BEGIN").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 1.0]', 'float4'))",
        [],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[2.0, 2.0]', 'float4'))",
        [],
    )
    .unwrap();
    // Persist mid-transaction (known hazard: persists uncommitted rows into
    // last_committed).
    conn.query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
        .unwrap();
    conn.execute_batch("ROLLBACK").unwrap();

    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1, "rollback must leave exactly the 1 committed row");

    // Insert again and confirm no duplicate-key failure / graph is usable.
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[3.0, 3.0]', 'float4'))",
        [],
    )
    .unwrap();
    let n2: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n2, 2,
        "post-rollback insert must succeed with no duplicate-key error"
    );
}

#[test]
fn mode_mismatch_at_connect_is_rejected() {
    // CREATE VIRTUAL TABLE cannot be re-run against a table that already
    // exists in sqlite_master, so connect-time verification (init()'s
    // verify_against_meta path in src/vtab/mod.rs) can't be exercised by
    // simply re-issuing the original CREATE VIRTUAL TABLE. Instead, corrupt
    // the persisted `meta` row directly via plain SQL on the `_index` shadow
    // table (an ordinary table, not the vtab itself), then reopen the file
    // and reference the table: SQLite invokes xConnect (not xCreate) the
    // first time a table already declared in sqlite_master is referenced by
    // a new connection, which is exactly where the mismatch check runs.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
            .unwrap();
        let meta: String = conn
            .query_row("SELECT value FROM t_index WHERE key = 'meta'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        // The CREATE VIRTUAL TABLE arguments above declare the default mode
        // (hnsw); flip the persisted value to "exact" so it disagrees.
        v["mode"] = serde_json::Value::String("exact".to_string());
        conn.execute(
            "UPDATE t_index SET value = ?1 WHERE key = 'meta'",
            [v.to_string()],
        )
        .unwrap();
    }
    let conn = open_file_with_extension(&path);
    let err = conn
        .query_row("SELECT count(*) FROM t", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("disagrees with persisted meta")
            || msg.contains("disagree with persisted meta"),
        "expected a persisted-meta disagreement error from connect-time verification, got: {msg}"
    );
}

#[test]
fn exact_mode_table_functions_reject_with_mode_exact_message() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE ex USING vector(dim=2, type=float4, metric=l2, mode=exact);
         INSERT INTO ex(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();

    let err = conn
        .query_row("SELECT vector_sync_index('ex')", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    assert!(
        err.to_string().contains("mode=exact"),
        "vector_sync_index on mode=exact table: expected 'mode=exact' in error, got: {err}"
    );

    let err = conn
        .query_row("SELECT vector_rebuild_index('ex')", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("mode=exact"),
        "vector_rebuild_index on mode=exact table: expected 'mode=exact' in error, got: {err}"
    );

    let err = conn
        .query_row("SELECT vector_ef_search('ex', 64)", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("mode=exact"),
        "vector_ef_search on mode=exact table: expected 'mode=exact' in error, got: {err}"
    );
}

#[test]
fn ef_search_survives_file_reconnect() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch(
            "CREATE VIRTUAL TABLE t USING vector(dim=4, type=float4, metric=l2);
             INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 0.0, 0.0, 0.0]', 'float4'));",
        )
        .unwrap();
        let new_ef: i64 = conn
            .query_row("SELECT vector_ef_search('t', 128)", [], |r| r.get(0))
            .unwrap();
        assert_eq!(new_ef, 128);
    }
    let conn = open_file_with_extension(&path);
    // The scalar functions look the table up in the in-process registry,
    // which is only populated by connect()/create(); reference the vtab
    // itself first so SQLite invokes xConnect (and init() picks up the
    // persisted ef_search via VectorTableConfig::params_from_meta) before
    // querying vector_index_info.
    let _: i64 = conn
        .query_row("SELECT count(*) FROM t", [], |r| r.get(0))
        .unwrap();
    let info: String = conn
        .query_row("SELECT vector_index_info('t')", [], |r| r.get(0))
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&info).unwrap();
    assert_eq!(
        v["ef_search"], 128,
        "ef_search set before close must round-trip through the persisted meta on reconnect"
    );
}

#[test]
fn vector_index_info_reports_exact_mode() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE e USING vector(dim=3, type=float4, metric=l2, mode=exact);",
    )
    .unwrap();
    let n = 7;
    for i in 0..n {
        conn.execute(
            "INSERT INTO e(vector) VALUES (vector_from_json(?1, 'float4'))",
            [format!("[{i}.0, 0.0, 0.0]")],
        )
        .unwrap();
    }
    let info: String = conn
        .query_row("SELECT vector_index_info('e')", [], |r| r.get(0))
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&info).unwrap();
    assert_eq!(v["mode"], "exact");
    assert_eq!(v["rows"], n);
    // Exact mode has no live HNSW index, so `vector_index_info`'s `ef_live`
    // branch takes the `None` arm (see src/scalar.rs) and `ef_live` is 0;
    // the reported `ef_search` then falls back to the configured tuning
    // knob (default 64, since this table didn't override it) rather than 0.
    assert!(
        v["ef_search"].is_i64(),
        "ef_search must be present even in exact mode, got: {info}"
    );
    assert_eq!(v["ef_search"], 64);
}

#[test]
fn vector_ef_search_rejects_non_positive_values() {
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();

    let err = conn
        .query_row("SELECT vector_ef_search('t', 0)", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("positive"),
        "vector_ef_search(0) must error mentioning 'positive', got: {err}"
    );

    let err = conn
        .query_row("SELECT vector_ef_search('t', -5)", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_err();
    assert!(
        err.to_string().contains("positive"),
        "vector_ef_search(-5) must error mentioning 'positive', got: {err}"
    );
}

#[test]
fn dim_mismatch_at_connect_is_rejected() {
    // Sibling to `mode_mismatch_at_connect_is_rejected`: corrupts the
    // persisted `meta` row's `dim` field (instead of `mode`) directly via
    // plain SQL, then reopens the file and references the table so SQLite
    // invokes xConnect, exercising the dim/type/metric branch of
    // connect-time verification in src/vtab/mod.rs::init (distinct from the
    // separate `mode` check just below it).
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
            .unwrap();
        let meta: String = conn
            .query_row("SELECT value FROM t_index WHERE key = 'meta'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        v["dim"] = serde_json::Value::from(999);
        conn.execute(
            "UPDATE t_index SET value = ?1 WHERE key = 'meta'",
            [v.to_string()],
        )
        .unwrap();
    }
    let conn = open_file_with_extension(&path);
    let err = conn
        .query_row("SELECT count(*) FROM t", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("disagree with persisted meta"),
        "expected a persisted-meta disagreement error from connect-time dim verification, got: {msg}"
    );
}

#[test]
fn missing_m_in_persisted_meta_is_rejected_at_connect() {
    // `VectorTableConfig::params_from_meta` (src/vtab/config.rs) now hard-errors
    // when the persisted `meta` row is missing `m` (previously it silently
    // defaulted). Sibling to `dim_mismatch_at_connect_is_rejected`: corrupt the
    // persisted meta by removing the `m` key directly via plain SQL, then
    // reopen the file and reference the table so SQLite invokes xConnect,
    // exercising the `meta["m"].as_u64().ok_or_else(...)` branch.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v.db");
    {
        let conn = open_file_with_extension(&path);
        conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
            .unwrap();
        let meta: String = conn
            .query_row("SELECT value FROM t_index WHERE key = 'meta'", [], |r| {
                r.get(0)
            })
            .unwrap();
        let mut v: serde_json::Value = serde_json::from_str(&meta).unwrap();
        v.as_object_mut().unwrap().remove("m");
        conn.execute(
            "UPDATE t_index SET value = ?1 WHERE key = 'meta'",
            [v.to_string()],
        )
        .unwrap();
    }
    let conn = open_file_with_extension(&path);
    let err = conn
        .query_row("SELECT count(*) FROM t", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("meta missing m"),
        "expected a 'meta missing m' error from connect-time verification, got: {msg}"
    );
}

#[test]
fn released_savepoint_does_not_corrupt_later_rollback_to() {
    // Exercises TransactionVTab::release (RELEASE), confirming that
    // releasing an inner savepoint doesn't disturb an outer savepoint's
    // snapshot that a later ROLLBACK TO still depends on.
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();

    // Committed baseline row, present before any savepoint.
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'))",
        [],
    )
    .unwrap();

    conn.execute_batch("SAVEPOINT a").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 1.0]', 'float4'))",
        [],
    )
    .unwrap();
    conn.execute_batch("SAVEPOINT b").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[2.0, 2.0]', 'float4'))",
        [],
    )
    .unwrap();
    conn.execute_batch("RELEASE b").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[3.0, 3.0]', 'float4'))",
        [],
    )
    .unwrap();
    conn.execute_batch("ROLLBACK TO a").unwrap();

    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        n, 1,
        "ROLLBACK TO a must undo everything after savepoint a, \
         despite the intervening RELEASE of the inner savepoint b"
    );

    // The connection must remain usable: release the still-open savepoint a
    // and confirm further inserts still work against a consistent index.
    conn.execute_batch("RELEASE a").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[4.0, 4.0]', 'float4'))",
        [],
    )
    .unwrap();
    let n2: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n2, 2, "post-rollback insert must succeed and be visible");
}

#[test]
fn rollback_to_savepoint_opened_before_first_write_does_not_desync_index() {
    // A savepoint opened before the vtab's first write in a transaction gets
    // no xSavepoint call, so the vtab holds no snapshot for it. ROLLBACK TO
    // that savepoint — while a *deeper*, post-enrollment savepoint sits on the
    // snapshot stack — must still restore the in-memory index to the target
    // state (rebuilding from the shadow data), not the deeper snapshot. A
    // desync leaves a ghost key in the graph that collides with a reused rowid
    // on the next insert.
    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
    // Committed baseline (autocommit), present before any savepoint.
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'))",
        [],
    )
    .unwrap();

    conn.execute_batch("SAVEPOINT s1").unwrap(); // before first write: no xSavepoint
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[1.0, 1.0]', 'float4'))",
        [],
    )
    .unwrap(); // enrolls the vtab
    conn.execute_batch("SAVEPOINT s2").unwrap(); // snapshotted (deeper)
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[2.0, 2.0]', 'float4'))",
        [],
    )
    .unwrap();
    conn.execute_batch("ROLLBACK TO s1").unwrap();

    // Index must reflect only the committed baseline row, matching _data.
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1, "ROLLBACK TO s1 must undo r1 and r2 in the index");

    // The proof the index is not desynced: a subsequent insert must not
    // collide with a ghost key left behind by the rolled-back rows.
    conn.execute_batch("RELEASE s1").unwrap();
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json('[3.0, 3.0]', 'float4'))",
        [],
    )
    .expect("post-rollback insert must not hit a ghost duplicate key");
    let n2: i64 = conn
        .query_row(
            "SELECT count(*) FROM (SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n2, 2, "baseline + the new row");
}

#[test]
fn multiple_knn_match_constraints_are_rejected() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE mk USING vector(dim=2, type=float4, metric=l2);
         INSERT INTO mk(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'));",
    )
    .unwrap();
    let err = conn
        .prepare(
            "SELECT id FROM mk
             WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
             AND knn_match(distance, vector_from_json('[1.0, 1.0]', 'float4'))",
        )
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("knn_match"),
        "error message should mention knn_match, got: {msg}"
    );
}
