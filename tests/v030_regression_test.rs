mod common;

use common::open_with_extension;
use rusqlite::Connection;

fn create_2d(conn: &Connection) {
    conn.execute_batch("CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2);")
        .unwrap();
}

fn insert_json(conn: &Connection, json: &str) {
    conn.execute(
        "INSERT INTO t(vector) VALUES (vector_from_json(?1, 'float4'))",
        [json],
    )
    .unwrap();
}

#[test]
fn update_preserves_rowid_and_index() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    insert_json(&conn, "[0.0, 1.0]");
    conn.execute_batch(
        "UPDATE t SET vector = vector_from_json('[9.0, 9.0]', 'float4') WHERE id = 1;",
    )
    .unwrap();

    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM t ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids, vec![1, 2], "UPDATE must not renumber rows");

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[9.0, 9.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        nearest, 1,
        "KNN must find the updated vector under its original id"
    );
}

#[test]
fn update_can_change_rowid() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    conn.execute_batch("UPDATE t SET id = 7 WHERE id = 1;")
        .unwrap();
    let id: i64 = conn
        .query_row("SELECT id FROM t", [], |r| r.get(0))
        .unwrap();
    assert_eq!(id, 7);
    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(nearest, 7, "index must be re-keyed when the rowid changes");
}

fn create_2d_with_metadata(conn: &Connection) {
    conn.execute_batch(
        "CREATE VIRTUAL TABLE mu USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT\");",
    )
    .unwrap();
}

fn insert_json_with_label(conn: &Connection, json: &str, label: &str) {
    conn.execute(
        "INSERT INTO mu(vector, label) VALUES (vector_from_json(?1, 'float4'), ?2)",
        rusqlite::params![json, label],
    )
    .unwrap();
}

#[test]
fn update_vector_only_retains_metadata() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch(
        "UPDATE mu SET vector = vector_from_json('[9.0, 9.0]', 'float4') WHERE id = 1;",
    )
    .unwrap();

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "a", "metadata must survive a vector-only UPDATE");
}

#[test]
fn rollback_on_fresh_connection_discards_index_entries() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "BEGIN;
         INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 0.0]', 'float4'));
         INSERT INTO t(vector) VALUES (vector_from_json('[0.1, 0.1]', 'float4'));
         ROLLBACK;",
    )
    .unwrap();
    // Rowids 1..2 are reused after rollback; inserts must not hit duplicate keys.
    insert_json(&conn, "[100.0, 100.0]");
    insert_json(&conn, "[101.0, 101.0]");
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 2, "index must contain exactly the committed rows");
}

#[test]
fn rollback_to_savepoint_restores_index_to_savepoint() {
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 1.0]");
    conn.execute_batch(
        "BEGIN;
         INSERT INTO t(vector) VALUES (vector_from_json('[2.0, 2.0]', 'float4'));
         SAVEPOINT sp1;
         INSERT INTO t(vector) VALUES (vector_from_json('[3.0, 3.0]', 'float4'));
         ROLLBACK TO sp1;
         COMMIT;",
    )
    .unwrap();
    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 10",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 2, "index must reflect rows 1 and 2 only");
}

#[test]
fn update_metadata_only_retains_vector() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET label = 'b' WHERE id = 1;")
        .unwrap();

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM mu WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        nearest, 1,
        "vector must survive a metadata-only UPDATE (KNN still finds original vector)"
    );

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "b");
}

#[test]
fn update_metadata_to_null_sets_null() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET label = NULL WHERE id = 1;")
        .unwrap();

    let is_null: bool = conn
        .query_row("SELECT label IS NULL FROM mu WHERE id = 1", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert!(
        is_null,
        "explicit SET label = NULL must actually null the column"
    );
}

#[test]
fn update_id_only_retains_vector_and_metadata() {
    let conn = open_with_extension();
    create_2d_with_metadata(&conn);
    insert_json_with_label(&conn, "[1.0, 0.0]", "a");

    conn.execute_batch("UPDATE mu SET id = 7 WHERE id = 1;")
        .unwrap();

    let label: String = conn
        .query_row("SELECT CAST(label AS TEXT) FROM mu WHERE id = 7", [], |r| {
            r.get(0)
        })
        .unwrap();
    assert_eq!(label, "a", "metadata must survive an id-only UPDATE");

    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM mu WHERE knn_match(distance, vector_from_json('[1.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        nearest, 7,
        "vector must survive an id-only UPDATE, re-keyed under the new id"
    );
}

#[test]
fn insert_with_explicit_rowid() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (42, vector_from_json('[2.0, 2.0]', 'float4'));",
    )
    .unwrap();
    let id: i64 = conn
        .query_row("SELECT id FROM t", [], |r| r.get(0))
        .unwrap();
    assert_eq!(id, 42);
    let nearest: i64 = conn
        .query_row(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[2.0, 2.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(nearest, 42);
}

#[test]
fn insert_duplicate_rowid_errors() {
    let conn = open_with_extension();
    create_2d(&conn);
    conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (5, vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();
    let err = conn.execute_batch(
        "INSERT INTO t(id, vector) VALUES (5, vector_from_json('[0.0, 1.0]', 'float4'));",
    );
    assert!(
        err.is_err(),
        "duplicate explicit rowid must be a constraint error"
    );
}

#[test]
fn metadata_columns_keep_declared_types() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE m USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT, score REAL\");
         INSERT INTO m(vector, label, score) VALUES (vector_from_json('[0.0, 0.0]', 'float4'), 'a', 1.5);",
    )
    .unwrap();
    let (t_label, t_score): (String, String) = conn
        .query_row("SELECT typeof(label), typeof(score) FROM m", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!((t_label.as_str(), t_score.as_str()), ("text", "real"));

    let n: i64 = conn
        .query_row(
            "SELECT count(*) FROM m WHERE label = 'a' AND score > 1.0",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(n, 1, "metadata filters must match typed values");

    // KNN mode must preserve types too.
    let t_knn: String = conn
        .query_row(
            "SELECT typeof(label) FROM m WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4')) LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(t_knn, "text");
}

#[test]
fn order_by_distance_desc_returns_farthest() {
    let conn = open_with_extension();
    create_2d(&conn);
    for j in ["[0.0, 0.0]", "[1.0, 1.0]", "[2.0, 2.0]", "[3.0, 3.0]"] {
        insert_json(&conn, j);
    }
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM t WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
             ORDER BY distance DESC LIMIT 2",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids, vec![4, 3], "DESC must return the two farthest rows");
}

#[test]
fn rebuild_index_one_arg_uses_persisted_config() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE r USING vector(dim=2, type=float4, metric=cosine, m=8, ef_construction=64);
         INSERT INTO r(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));
         INSERT INTO r(vector) VALUES (vector_from_json('[0.0, 1.0]', 'float4'));",
    )
    .unwrap();
    let n: i64 = conn
        .query_row("SELECT vector_rebuild_index('r')", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 2);
    // Meta row must exist and carry the table's parameters.
    let meta: String = conn
        .query_row("SELECT value FROM r_index WHERE key = 'meta'", [], |r| {
            r.get(0)
        })
        .unwrap();
    let v: serde_json::Value = serde_json::from_str(&meta).unwrap();
    assert_eq!(v["metric"], "cosine");
    assert_eq!(v["m"], 8);

    // Finding 2: a "main."-qualified name must resolve the same way a bare
    // name does — vector_rebuild_index is consistent with the other table
    // functions (vector_sync_index, vector_ef_search, vector_index_info)
    // even though it's not documented to support attached databases.
    let n2: i64 = conn
        .query_row("SELECT vector_rebuild_index('main.r')", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n2, 2);
}

#[test]
fn rebuild_index_rejects_non_main_qualified_name() {
    let conn = open_with_extension();
    conn.execute_batch(
        "ATTACH ':memory:' AS aux;
         CREATE VIRTUAL TABLE aux.r2 USING vector(dim=2, type=float4, metric=l2);
         INSERT INTO aux.r2(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));",
    )
    .unwrap();

    let err = conn
        .query_row("SELECT vector_rebuild_index('aux.r2')", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("attached database") && msg.contains("aux"),
        "expected an attached-database rejection, got: {msg}"
    );
}

#[test]
fn knn_with_metadata_filter_does_not_truncate() {
    let conn = open_with_extension();
    conn.execute_batch(
        "CREATE VIRTUAL TABLE f USING vector(dim=2, type=float4, metric=l2, metadata=\"label TEXT\");
         INSERT INTO f(vector, label) VALUES (vector_from_json('[0.0, 0.0]', 'float4'), 'b');
         INSERT INTO f(vector, label) VALUES (vector_from_json('[0.1, 0.1]', 'float4'), 'b');
         INSERT INTO f(vector, label) VALUES (vector_from_json('[2.0, 2.0]', 'float4'), 'a');
         INSERT INTO f(vector, label) VALUES (vector_from_json('[3.0, 3.0]', 'float4'), 'a');",
    )
    .unwrap();
    let ids: Vec<i64> = conn
        .prepare(
            "SELECT id FROM f WHERE knn_match(distance, vector_from_json('[0.0, 0.0]', 'float4'))
             AND label = 'a' LIMIT 2",
        )
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(ids.len(), 2, "filter must not shrink results below LIMIT");
    assert_eq!(ids, vec![3, 4]);
}

#[test]
fn vector_sync_index_disambiguates_same_named_tables_across_databases() {
    let conn = open_with_extension();
    conn.execute_batch(
        "ATTACH ':memory:' AS aux;
         CREATE VIRTUAL TABLE main.t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
         CREATE VIRTUAL TABLE aux.t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
         INSERT INTO main.t(vector) VALUES (vector_from_json('[1.0, 0.0]', 'float4'));
         INSERT INTO aux.t(vector) VALUES (vector_from_json('[0.0, 1.0]', 'float4'));
         INSERT INTO aux.t(vector) VALUES (vector_from_json('[0.0, 2.0]', 'float4'));",
    )
    .unwrap();

    // Bare name matches both main.t and aux.t: must be rejected as ambiguous,
    // never silently pick one (or worse, write to the wrong shadow table).
    let err = conn
        .query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("ambiguous"),
        "expected an ambiguity error, got: {msg}"
    );

    // Qualified name resolves unambiguously to main.t's registry entry and
    // succeeds (rather than erroring or picking an arbitrary same-named
    // entry).
    let result: i64 = conn
        .query_row("SELECT vector_sync_index('main.t')", [], |r| r.get(0))
        .unwrap();
    assert_eq!(result, 1);
}

#[test]
fn drop_and_recreate_same_name_does_not_leave_stale_registry_entry() {
    // destroy() (DROP TABLE) must remove the registry entry for
    // the dropped table so a same-named table created afterwards is the
    // *only* match for its bare name — a stale dead-Weak entry under the
    // same key would otherwise make the fresh table look ambiguous.
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    conn.execute_batch("DROP TABLE t;").unwrap();

    conn.execute_batch(
        "CREATE VIRTUAL TABLE t USING vector(dim=2, type=float4, metric=l2, sync_every=1000000);
         INSERT INTO t(vector) VALUES (vector_from_json('[0.0, 1.0]', 'float4'));",
    )
    .unwrap();

    // Must succeed with no ambiguity error and no stale state from the
    // dropped table.
    let n: i64 = conn
        .query_row("SELECT vector_sync_index('t')", [], |r| r.get(0))
        .unwrap();
    assert_eq!(n, 1);
}

#[test]
fn drop_main_then_create_aux_same_name_resolves_unambiguously() {
    // dropping main.t must remove its registry entry outright
    // (not just leave a dead Weak on the get() path) so a subsequently
    // created aux.t is the *sole* live entry for the bare name 't' —
    // resolving to aux.t's own "not supported for attached databases"
    // error, not a spurious "ambiguous table name" error.
    let conn = open_with_extension();
    create_2d(&conn);
    insert_json(&conn, "[1.0, 0.0]");
    conn.execute_batch("DROP TABLE t;").unwrap();

    conn.execute_batch(
        "ATTACH ':memory:' AS aux;
         CREATE VIRTUAL TABLE aux.t USING vector(dim=2, type=float4, metric=l2);
         INSERT INTO aux.t(vector) VALUES (vector_from_json('[0.0, 1.0]', 'float4'));",
    )
    .unwrap();

    let err = conn
        .query_row("SELECT vector_sync_index('t')", [], |r| r.get::<_, i64>(0))
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        !msg.contains("ambiguous"),
        "must not report ambiguity once main.t's entry is gone, got: {msg}"
    );
    assert!(
        msg.contains("attached database") && msg.contains("aux"),
        "expected aux.t's own non-main rejection, got: {msg}"
    );
}
