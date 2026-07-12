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
fn stale_deletes_trigger_rebuild_on_connect() {
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
        conn.execute("DELETE FROM t WHERE id <= 2", []).unwrap(); // not persisted
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
    assert_eq!(ids.len(), 3, "stale graph keys must be dropped by rebuild");
    assert!(ids.iter().all(|id| *id >= 3));
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
fn autocommit_inserts_within_3x_of_single_transaction() {
    use std::time::Instant;
    let json = "[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]";

    let conn = open_with_extension();
    conn.execute_batch("CREATE VIRTUAL TABLE a USING vector(dim=8, type=float4, metric=l2);")
        .unwrap();
    let t0 = Instant::now();
    for _ in 0..2000 {
        conn.execute(
            "INSERT INTO a(vector) VALUES (vector_from_json(?1, 'float4'))",
            [json],
        )
        .unwrap();
    }
    let auto = t0.elapsed();

    let conn2 = open_with_extension();
    conn2
        .execute_batch("CREATE VIRTUAL TABLE b USING vector(dim=8, type=float4, metric=l2);")
        .unwrap();
    let t1 = Instant::now();
    conn2.execute_batch("BEGIN").unwrap();
    for _ in 0..2000 {
        conn2
            .execute(
                "INSERT INTO b(vector) VALUES (vector_from_json(?1, 'float4'))",
                [json],
            )
            .unwrap();
    }
    conn2.execute_batch("COMMIT").unwrap();
    let txn = t1.elapsed();

    assert!(
        auto < txn * 3 + std::time::Duration::from_millis(200),
        "autocommit {auto:?} must stay within 3x of one-txn {txn:?} (+200ms slack)"
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
