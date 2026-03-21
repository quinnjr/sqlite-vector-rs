use sqlite_vector_rs::distance::DistanceMetric;
use sqlite_vector_rs::types::VectorType;
use sqlite_vector_rs::vtab::config::VectorTableConfig;

#[test]
fn parse_basic_args() {
    let args = vec![
        "vector",
        "main",
        "embeddings",
        "dim=3",
        "type=float4",
        "metric=l2",
    ];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.dim, 3);
    assert_eq!(config.vtype, VectorType::Float4);
    assert_eq!(config.metric, DistanceMetric::L2);
    assert_eq!(config.table_name, "embeddings");
    assert!(config.metadata_columns.is_empty());
}

#[test]
fn parse_with_hnsw_params() {
    let args = vec![
        "vector",
        "main",
        "emb",
        "dim=768",
        "type=float4",
        "metric=cosine",
        "m=32",
        "ef_construction=400",
        "ef_search=128",
    ];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.hnsw_params.m, 32);
    assert_eq!(config.hnsw_params.ef_construction, 400);
    assert_eq!(config.hnsw_params.ef_search, 128);
}

#[test]
fn parse_with_metadata() {
    let args = vec![
        "vector",
        "main",
        "emb",
        "dim=3",
        "type=float4",
        "metric=l2",
        "metadata=\"label TEXT, category INTEGER\"",
    ];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.metadata_columns.len(), 2);
    assert_eq!(
        config.metadata_columns[0],
        ("label".to_string(), "TEXT".to_string())
    );
    assert_eq!(
        config.metadata_columns[1],
        ("category".to_string(), "INTEGER".to_string())
    );
}

#[test]
fn parse_defaults() {
    let args = vec!["vector", "main", "emb", "dim=3"];
    let config = VectorTableConfig::parse(&args).unwrap();
    assert_eq!(config.vtype, VectorType::Float4);
    assert_eq!(config.metric, DistanceMetric::L2);
    assert_eq!(config.hnsw_params.m, 16);
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
    let args = vec![
        "vector",
        "main",
        "emb",
        "dim=3",
        "type=float4",
        "metric=l2",
        "metadata=\"label TEXT, category INTEGER\"",
    ];
    let config = VectorTableConfig::parse(&args).unwrap();
    let sql = config.vtab_schema();
    assert!(sql.contains("label TEXT"));
    assert!(sql.contains("category INTEGER"));
}
