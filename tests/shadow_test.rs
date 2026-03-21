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
    let args = vec![
        "vector",
        "main",
        "emb",
        "dim=3",
        "type=float4",
        "metric=l2",
        "metadata=\"label TEXT, score REAL\"",
    ];
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
    let args = vec![
        "vector",
        "main",
        "emb",
        "dim=3",
        "type=float4",
        "metric=l2",
        "metadata=\"label TEXT\"",
    ];
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
